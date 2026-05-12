//! GPU sumcheck computation for WHIR.
//!
//! Provides:
//! - Product sumcheck: degree-2 polynomial from Σ a[i]·b[i] (used in every WHIR round)
//! - Split-eq fold: fold the eq polynomial after each round
//! - Partial sum reduction

use std::ffi::CString;
use std::sync::Arc;

use koala_bear::{KoalaBear, extension::QuinticExtensionField};
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use field::PrimeCharacteristicRing;

type EF = QuinticExtensionField<KoalaBear>;

fn kb(v: u32) -> KoalaBear { unsafe { std::mem::transmute(v) } }
fn kb_u32(v: KoalaBear) -> u32 { unsafe { std::mem::transmute(v) } }
fn ef(v: [u32; 5]) -> EF { unsafe { std::mem::transmute(v) } }
fn ef_u32(v: EF) -> [u32; 5] { unsafe { std::mem::transmute(v) } }

pub struct GpuSumcheck {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    fn_prod_base_ext: cuda_sys::CUfunction,
    fn_prod_ext_ext: cuda_sys::CUfunction,
    fn_reduce_ext: cuda_sys::CUfunction,
    fn_eq_fold: cuda_sys::CUfunction,
    fn_transpose: cuda_sys::CUfunction,
    fn_eq_expand: cuda_sys::CUfunction,
    fn_eq_accum: cuda_sys::CUfunction,
    fn_eq_accum_offset: cuda_sys::CUfunction,
    fn_air_exec: cuda_sys::CUfunction,
    fn_gkr_sum: cuda_sys::CUfunction,
}

unsafe impl Send for GpuSumcheck {}
unsafe impl Sync for GpuSumcheck {}

impl Drop for GpuSumcheck {
    fn drop(&mut self) {
        unsafe { let _ = cuda_result::module::unload(self.cu_module); }
    }
}

impl GpuSumcheck {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/sumcheck.ptx"));
        let c_src = CString::new(ptx_src).unwrap();
        let cu_module = unsafe { cuda_result::module::load_data(c_src.as_ptr().cast()) }
            .expect("failed to load sumcheck PTX");

        let load = |name: &str| {
            let c = CString::new(name).unwrap();
            unsafe { cuda_result::module::get_function(cu_module, c) }
                .unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };

        Self {
            stream,
            cu_module,
            fn_prod_base_ext: load("product_sumcheck_base_ext_kernel"),
            fn_prod_ext_ext: load("product_sumcheck_ext_ext_kernel"),
            fn_reduce_ext: load("reduce_ext_kernel"),
            fn_eq_fold: load("split_eq_fold_kernel"),
            fn_transpose: load("transpose_packed_ext_kernel"),
            fn_eq_expand: load("eq_expand_step_kernel"),
            fn_eq_accum: load("eq_accumulate_kernel"),
            fn_eq_accum_offset: load("eq_accumulate_offset_kernel"),
            fn_air_exec: load("air_sumcheck_execution_kernel"),
            fn_gkr_sum: load("gkr_sum_quotients_kernel"),
        }
    }

    /// Reduce partial ext-field sums to a single value.
    fn reduce_partials(&self, d_partials: &CudaSlice<u32>, n_blocks: u32) -> [u32; 5] {
        let mut d_result = self.stream.alloc_zeros::<u32>(5).unwrap();
        {
            let (p_ptr, _g1) = d_partials.device_ptr(&self.stream);
            let (r_ptr, _g2) = d_result.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &p_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_blocks as *const _ as *mut _,
            ];
            // Must launch at least 32 threads (one full warp) for warp shuffle reduction.
            let threads = 256u32.max(32);
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_reduce_ext, (1, 1, 1), (threads, 1, 1),
                    (threads / 32 * 5 * 4) as u32,
                    self.stream.cu_stream(), &mut args,
                ).expect("reduce kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        let result = self.stream.memcpy_dtov(&d_result).unwrap();
        result[..5].try_into().unwrap()
    }

    /// Product sumcheck (base × ext): compute c0 and c2.
    /// `pol_a`: n base field elements. `pol_b`: n × 5 ext field elements.
    /// Returns (c0, c2) as ext field elements.
    pub fn product_sumcheck_base_ext(
        &self,
        pol_a: &[u32],
        pol_b: &[u32],
    ) -> ([u32; 5], [u32; 5]) {
        let n = pol_a.len();
        assert_eq!(pol_b.len(), n * 5);
        let half = (n / 2) as u32;

        let d_a = self.stream.memcpy_stod(pol_a).unwrap();
        let d_b = self.stream.memcpy_stod(pol_b).unwrap();

        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;

        let mut d_c0 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();
        let mut d_c2 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();

        {
            let (a_ptr, _g1) = d_a.device_ptr(&self.stream);
            let (b_ptr, _g2) = d_b.device_ptr(&self.stream);
            let (c0_ptr, _g3) = d_c0.device_ptr_mut(&self.stream);
            let (c2_ptr, _g4) = d_c2.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &c0_ptr as *const _ as *mut _,
                &c2_ptr as *const _ as *mut _,
                &half as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_prod_base_ext, (blocks, 1, 1), (threads, 1, 1),
                    smem, self.stream.cu_stream(), &mut args,
                ).expect("product sumcheck kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        let c0 = self.reduce_partials(&d_c0, blocks);
        let c2 = self.reduce_partials(&d_c2, blocks);
        (c0, c2)
    }

    /// Product sumcheck (ext × ext).
    pub fn product_sumcheck_ext_ext(
        &self,
        pol_a: &[u32],
        pol_b: &[u32],
    ) -> ([u32; 5], [u32; 5]) {
        let n = pol_a.len() / 5;
        assert_eq!(pol_b.len(), n * 5);
        let half = (n / 2) as u32;

        let d_a = self.stream.memcpy_stod(pol_a).unwrap();
        let d_b = self.stream.memcpy_stod(pol_b).unwrap();

        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;

        let mut d_c0 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();
        let mut d_c2 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();

        {
            let (a_ptr, _g1) = d_a.device_ptr(&self.stream);
            let (b_ptr, _g2) = d_b.device_ptr(&self.stream);
            let (c0_ptr, _g3) = d_c0.device_ptr_mut(&self.stream);
            let (c2_ptr, _g4) = d_c2.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &c0_ptr as *const _ as *mut _,
                &c2_ptr as *const _ as *mut _,
                &half as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_prod_ext_ext, (blocks, 1, 1), (threads, 1, 1),
                    smem, self.stream.cu_stream(), &mut args,
                ).expect("product sumcheck ext×ext kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        let c0 = self.reduce_partials(&d_c0, blocks);
        let c2 = self.reduce_partials(&d_c2, blocks);
        (c0, c2)
    }

    /// Fold eq polynomial: out[j] = eq[2j] + r * (eq[2j+1] - eq[2j]).
    pub fn eq_fold(&self, eq_data: &[u32], r_ext: &[u32; 5]) -> Vec<u32> {
        let n_pairs = (eq_data.len() / 10) as u32; // 2 * n_pairs * 5

        let d_eq = self.stream.memcpy_stod(eq_data).unwrap();
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self.stream.alloc_zeros::<u32>((n_pairs as usize) * 5).unwrap();

        {
            let (eq_ptr, _g1) = d_eq.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
            let threads = 256u32;
            let blocks = (n_pairs + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &eq_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_eq_fold, (blocks, 1, 1), (threads, 1, 1),
                    0, self.stream.cu_stream(), &mut args,
                ).expect("eq fold kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_out).unwrap()
    }

    /// Transpose packed extension field data to flat scalar layout, ON GPU.
    /// Input: packed[n_packed * dim * width], Output: flat[n_packed * width * dim].
    pub fn transpose_packed_ext(
        &self,
        packed: &[u32],
        n_packed: u32,
        dim: u32,
        width: u32,
    ) -> Vec<u32> {
        let total = (n_packed * width) as usize;
        let d_packed = self.stream.memcpy_stod(packed).unwrap();
        let mut d_flat = self.stream.alloc_zeros::<u32>(total * dim as usize).unwrap();

        {
            let (p_ptr, _g1) = d_packed.device_ptr(&self.stream);
            let (f_ptr, _g2) = d_flat.device_ptr_mut(&self.stream);
            let threads = 256u32;
            let blocks = (total as u32 + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &p_ptr as *const _ as *mut _,
                &f_ptr as *const _ as *mut _,
                &n_packed as *const _ as *mut _,
                &dim as *const _ as *mut _,
                &width as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_transpose, (blocks, 1, 1), (threads, 1, 1),
                    0, self.stream.cu_stream(), &mut args,
                ).expect("transpose kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_flat).unwrap()
    }

    /// Product sumcheck (base × ext) with GPU-side transpose of packed ext data.
    /// `pol_a_packed`: n_packed * width base field elements (already flat).
    /// `pol_b_packed`: n_packed * dim * width ext field elements (packed layout).
    /// Returns (c0, c2).
    pub fn product_sumcheck_base_ext_packed(
        &self,
        pol_a_flat: &[u32],
        pol_b_packed: &[u32],
        n_packed: u32,
        dim: u32,
        width: u32,
    ) -> ([u32; 5], [u32; 5]) {
        let total_scalars = (n_packed * width) as usize;

        // Upload both arrays to GPU.
        let d_a = self.stream.memcpy_stod(pol_a_flat).unwrap();
        let d_packed_b = self.stream.memcpy_stod(pol_b_packed).unwrap();

        // Transpose pol_b on GPU: packed → flat.
        let mut d_flat_b = self.stream.alloc_zeros::<u32>(total_scalars * dim as usize).unwrap();
        {
            let (p_ptr, _g1) = d_packed_b.device_ptr(&self.stream);
            let (f_ptr, _g2) = d_flat_b.device_ptr_mut(&self.stream);
            let threads = 256u32;
            let blocks = (total_scalars as u32 + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &p_ptr as *const _ as *mut _,
                &f_ptr as *const _ as *mut _,
                &n_packed as *const _ as *mut _,
                &dim as *const _ as *mut _,
                &width as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_transpose, (blocks, 1, 1), (threads, 1, 1),
                    0, self.stream.cu_stream(), &mut args,
                ).expect("transpose kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        // Now run product sumcheck on the transposed data.
        let flat_b = self.stream.memcpy_dtov(&d_flat_b).unwrap();
        self.product_sumcheck_base_ext(pol_a_flat, &flat_b)
    }

    // ── Device-resident APIs ──────────────────────────────────────────────
    // These operate on CudaSlice directly, no htod/dtoh per call.

    /// Product sumcheck (base × ext) on device buffers. Returns (c0, c2) on host.
    pub fn product_sumcheck_base_ext_device(
        &self,
        d_pol_a: &CudaSlice<u32>,
        d_pol_b: &CudaSlice<u32>,
        half: u32,
    ) -> ([u32; 5], [u32; 5]) {
        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;

        let mut d_c0 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();
        let mut d_c2 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();

        {
            let (a_ptr, _g1) = d_pol_a.device_ptr(&self.stream);
            let (b_ptr, _g2) = d_pol_b.device_ptr(&self.stream);
            let (c0_ptr, _g3) = d_c0.device_ptr_mut(&self.stream);
            let (c2_ptr, _g4) = d_c2.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &c0_ptr as *const _ as *mut _,
                &c2_ptr as *const _ as *mut _,
                &half as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_prod_base_ext, (blocks, 1, 1), (threads, 1, 1),
                    smem, self.stream.cu_stream(), &mut args,
                ).expect("product sumcheck kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        let c0 = self.reduce_partials(&d_c0, blocks);
        let c2 = self.reduce_partials(&d_c2, blocks);
        (c0, c2)
    }

    /// Product sumcheck (ext × ext) on device buffers.
    pub fn product_sumcheck_ext_ext_device(
        &self,
        d_pol_a: &CudaSlice<u32>,
        d_pol_b: &CudaSlice<u32>,
        half: u32,
    ) -> ([u32; 5], [u32; 5]) {
        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;

        let mut d_c0 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();
        let mut d_c2 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();

        {
            let (a_ptr, _g1) = d_pol_a.device_ptr(&self.stream);
            let (b_ptr, _g2) = d_pol_b.device_ptr(&self.stream);
            let (c0_ptr, _g3) = d_c0.device_ptr_mut(&self.stream);
            let (c2_ptr, _g4) = d_c2.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &c0_ptr as *const _ as *mut _,
                &c2_ptr as *const _ as *mut _,
                &half as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_prod_ext_ext, (blocks, 1, 1), (threads, 1, 1),
                    smem, self.stream.cu_stream(), &mut args,
                ).expect("product sumcheck ext×ext kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        let c0 = self.reduce_partials(&d_c0, blocks);
        let c2 = self.reduce_partials(&d_c2, blocks);
        (c0, c2)
    }

    /// Build eq(point, x) on GPU for all x in {0,1}^n_vars.
    /// `point`: n_vars ext field coordinates (each 5 u32s).
    /// Returns device buffer with 2^n_vars × 5 u32s.
    pub fn eq_polynomial_device(&self, point: &[[u32; 5]]) -> CudaSlice<u32> {
        let n_vars = point.len();
        if n_vars == 0 {
            // Single element = 1 in ext field.
            let one = [0x01FFFFFEu32, 0, 0, 0, 0]; // KB_MONTY_ONE
            return self.stream.memcpy_stod(&one).unwrap();
        }

        // Start with [1] (one ext element on device).
        let init = [0x01FFFFFEu32, 0, 0, 0, 0];
        let mut d_current = self.stream.memcpy_stod(&init).unwrap();
        let mut n_current = 1u32;

        for k in 0..n_vars {
            let d_pk = self.stream.memcpy_stod(&point[k]).unwrap();
            let mut d_next = self.stream.alloc_zeros::<u32>((n_current as usize * 2) * 5).unwrap();

            {
                let (src_ptr, _g1) = d_current.device_ptr(&self.stream);
                let (dst_ptr, _g2) = d_next.device_ptr_mut(&self.stream);
                let (pk_ptr, _g3) = d_pk.device_ptr(&self.stream);

                let threads = 256u32;
                let blocks = (n_current + threads - 1) / threads;
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &src_ptr as *const _ as *mut _,
                    &dst_ptr as *const _ as *mut _,
                    &pk_ptr as *const _ as *mut _,
                    &n_current as *const _ as *mut _,
                ];
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_eq_expand, (blocks, 1, 1), (threads, 1, 1),
                        0, self.stream.cu_stream(), &mut args,
                    ).expect("eq expand kernel failed");
                }
            }
            self.stream.synchronize().unwrap();
            d_current = d_next;
            n_current *= 2;
        }

        d_current
    }

    /// Accumulate: weights[x] += scalar * eq_val[x] for all x. In-place on device.
    pub fn eq_accumulate_device(
        &self,
        d_weights: &mut CudaSlice<u32>,
        d_eq_val: &CudaSlice<u32>,
        scalar: &[u32; 5],
        n: u32,
    ) {
        let d_scalar = self.stream.memcpy_stod(scalar.as_slice()).unwrap();
        {
            let (w_ptr, _g1) = d_weights.device_ptr_mut(&self.stream);
            let (eq_ptr, _g2) = d_eq_val.device_ptr(&self.stream);
            let (s_ptr, _g3) = d_scalar.device_ptr(&self.stream);

            let threads = 256u32;
            let blocks = (n + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &w_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &s_ptr as *const _ as *mut _,
                &n as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_eq_accum, (blocks, 1, 1), (threads, 1, 1),
                    0, self.stream.cu_stream(), &mut args,
                ).expect("eq accumulate kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
    }

    /// Accumulate with offset: weights[offset + j] += scalar * eq_val[j]. In-place on device.
    pub fn eq_accumulate_offset_device(
        &self,
        d_weights: &mut CudaSlice<u32>,
        d_eq_val: &CudaSlice<u32>,
        scalar: &[u32; 5],
        offset: u32,
        n: u32,
    ) {
        let d_scalar = self.stream.memcpy_stod(scalar.as_slice()).unwrap();
        {
            let (w_ptr, _g1) = d_weights.device_ptr_mut(&self.stream);
            let (eq_ptr, _g2) = d_eq_val.device_ptr(&self.stream);
            let (s_ptr, _g3) = d_scalar.device_ptr(&self.stream);

            let threads = 256u32;
            let blocks = (n + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &w_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &s_ptr as *const _ as *mut _,
                &offset as *const _ as *mut _,
                &n as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_eq_accum_offset, (blocks, 1, 1), (threads, 1, 1),
                    0, self.stream.cu_stream(), &mut args,
                ).expect("eq accumulate offset kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
    }

    /// AIR sumcheck for execution table: evaluate 13 constraints across all row pairs.
    /// Returns (z0_sum, z2_sum) as ext field elements (each 5 u32s).
    ///
    /// columns: 20 columns in col-major order (20 * n_rows base elements on device).
    /// down_cols: 2 shifted columns in col-major order (2 * n_rows base elements on device).
    /// eq_factor: n_pairs ext field elements (n_pairs * 5 on device).
    /// alphas: 13 ext field alpha powers (13 * 5 u32s on host).
    pub fn air_sumcheck_execution_device(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq_factor: &CudaSlice<u32>,
        alphas: &[u32],  // 13 * 5 = 65 u32s
        n_rows: u32,
        n_pairs: u32,
    ) -> ([u32; 5], [u32; 5]) {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;

        let mut d_z0 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();
        let mut d_z2 = self.stream.alloc_zeros::<u32>((blocks as usize) * 5).unwrap();

        {
            let (col_ptr, _g1) = d_columns.device_ptr(&self.stream);
            let (down_ptr, _g2) = d_down_cols.device_ptr(&self.stream);
            let (eq_ptr, _g3) = d_eq_factor.device_ptr(&self.stream);
            let (alpha_ptr, _g4) = d_alphas.device_ptr(&self.stream);
            let (z0_ptr, _g5) = d_z0.device_ptr_mut(&self.stream);
            let (z2_ptr, _g6) = d_z2.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &col_ptr as *const _ as *mut _,
                &down_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &alpha_ptr as *const _ as *mut _,
                &z0_ptr as *const _ as *mut _,
                &z2_ptr as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_air_exec, (blocks, 1, 1), (threads, 1, 1),
                    smem, self.stream.cu_stream(), &mut args,
                ).expect("air sumcheck execution kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        let z0 = self.reduce_partials(&d_z0, blocks);
        let z2 = self.reduce_partials(&d_z2, blocks);
        (z0, z2)
    }

    /// GKR quotient sum: reduce pairs (num[2i], den[2i]) + (num[2i+1], den[2i+1])
    /// into (new_num[i], new_den[i]). Device-resident.
    pub fn gkr_sum_quotients_device(
        &self,
        d_nums: &CudaSlice<u32>,
        d_dens: &CudaSlice<u32>,
        n_pairs: u32,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let mut d_new_nums = self.stream.alloc_zeros::<u32>((n_pairs as usize) * 5).unwrap();
        let mut d_new_dens = self.stream.alloc_zeros::<u32>((n_pairs as usize) * 5).unwrap();

        {
            let (nums_ptr, _g1) = d_nums.device_ptr(&self.stream);
            let (dens_ptr, _g2) = d_dens.device_ptr(&self.stream);
            let (nn_ptr, _g3) = d_new_nums.device_ptr_mut(&self.stream);
            let (nd_ptr, _g4) = d_new_dens.device_ptr_mut(&self.stream);

            let threads = 256u32;
            let blocks = (n_pairs + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &nums_ptr as *const _ as *mut _,
                &dens_ptr as *const _ as *mut _,
                &nn_ptr as *const _ as *mut _,
                &nd_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_gkr_sum, (blocks, 1, 1), (threads, 1, 1),
                    0, self.stream.cu_stream(), &mut args,
                ).expect("gkr sum quotients kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        (d_new_nums, d_new_dens)
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
}

// ── CPU references ───────────────────────────────────────────────────────

/// CPU product sumcheck (base × ext): returns (c0, c2).
/// c0 = Σ a_lo * b_lo, c2 = Σ (a_hi - a_lo) * (b_hi - b_lo).
pub fn cpu_product_sumcheck_base_ext(pol_a: &[u32], pol_b: &[u32]) -> ([u32; 5], [u32; 5]) {
    let n = pol_a.len();
    let half = n / 2;
    let mut c0 = EF::ZERO;
    let mut c2 = EF::ZERO;

    for i in 0..half {
        let a_lo = kb(pol_a[i]);
        let a_hi = kb(pol_a[i + half]);
        let b_lo: EF = ef(pol_b[i * 5..(i + 1) * 5].try_into().unwrap());
        let b_hi: EF = ef(pol_b[(i + half) * 5..(i + half + 1) * 5].try_into().unwrap());
        c0 += b_lo * a_lo;
        c2 += (b_hi - b_lo) * (a_hi - a_lo);
    }

    (ef_u32(c0), ef_u32(c2))
}

/// CPU product sumcheck (ext × ext).
/// c0 = Σ a_lo * b_lo, c2 = Σ (a_hi - a_lo) * (b_hi - b_lo).
pub fn cpu_product_sumcheck_ext_ext(pol_a: &[u32], pol_b: &[u32]) -> ([u32; 5], [u32; 5]) {
    let n = pol_a.len() / 5;
    let half = n / 2;
    let mut c0 = EF::ZERO;
    let mut c2 = EF::ZERO;

    for i in 0..half {
        let a_lo: EF = ef(pol_a[i * 5..(i + 1) * 5].try_into().unwrap());
        let a_hi: EF = ef(pol_a[(i + half) * 5..(i + half + 1) * 5].try_into().unwrap());
        let b_lo: EF = ef(pol_b[i * 5..(i + 1) * 5].try_into().unwrap());
        let b_hi: EF = ef(pol_b[(i + half) * 5..(i + half + 1) * 5].try_into().unwrap());
        c0 += a_lo * b_lo;
        c2 += (a_hi - a_lo) * (b_hi - b_lo);
    }

    (ef_u32(c0), ef_u32(c2))
}

/// CPU eq fold.
pub fn cpu_eq_fold(eq_data: &[u32], r_ext: &[u32; 5]) -> Vec<u32> {
    let r = ef(*r_ext);
    let n_pairs = eq_data.len() / 10;
    let mut out = Vec::with_capacity(n_pairs * 5);
    for j in 0..n_pairs {
        let lo: EF = ef(eq_data[j * 10..j * 10 + 5].try_into().unwrap());
        let hi: EF = ef(eq_data[j * 10 + 5..j * 10 + 10].try_into().unwrap());
        let res = lo + (hi - lo) * r;
        out.extend_from_slice(&ef_u32(res));
    }
    out
}
