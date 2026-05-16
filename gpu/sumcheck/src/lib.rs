//! GPU sumcheck computation for WHIR.
//!
//! Provides:
//! - Product sumcheck: degree-2 polynomial from Σ a[i]·b[i] (used in every WHIR round)
//! - Split-eq fold: fold the eq polynomial after each round
//! - Partial sum reduction

use std::ffi::CString;
use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use field::PrimeCharacteristicRing;
use koala_bear::{KoalaBear, extension::QuinticExtensionField};

type EF = QuinticExtensionField<KoalaBear>;

/// Extract raw device pointer from a CudaSlice, safe to use during stream capture.
/// Must be called BEFORE begin_capture(). The returned pointer is valid for the
/// lifetime of the CudaSlice.
pub fn raw_device_ptr<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> cuda_sys::CUdeviceptr {
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr
}
pub fn raw_device_ptr_mut<T>(
    slice: &mut CudaSlice<T>,
    stream: &CudaStream,
) -> cuda_sys::CUdeviceptr {
    let (ptr, _guard) = slice.device_ptr_mut(stream);
    ptr
}

fn upload_or_zero(stream: &Arc<CudaStream>, values: &[u32]) -> CudaSlice<u32> {
    if values.is_empty() {
        stream.alloc_zeros::<u32>(1).unwrap()
    } else {
        stream.memcpy_stod(values).unwrap()
    }
}

fn kb(v: u32) -> KoalaBear {
    unsafe { std::mem::transmute(v) }
}
fn kb_u32(v: KoalaBear) -> u32 {
    unsafe { std::mem::transmute(v) }
}
fn ef(v: [u32; 5]) -> EF {
    unsafe { std::mem::transmute(v) }
}
fn ef_u32(v: EF) -> [u32; 5] {
    unsafe { std::mem::transmute(v) }
}

pub struct GpuSumcheck {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    fn_prod_base_ext: cuda_sys::CUfunction,
    fn_prod_ext_ext: cuda_sys::CUfunction,
    fn_sum_quot_ext: cuda_sys::CUfunction,
    fn_reduce_ext: cuda_sys::CUfunction,
    fn_eq_fold: cuda_sys::CUfunction,
    fn_transpose: cuda_sys::CUfunction,
    fn_eq_expand: cuda_sys::CUfunction,
    fn_expand_univariate_points: cuda_sys::CUfunction,
    fn_expand_sampled_base_query_points: cuda_sys::CUfunction,
    fn_eq_accum: cuda_sys::CUfunction,
    fn_eq_accum_offset: cuda_sys::CUfunction,
    fn_air_exec: cuda_sys::CUfunction,
    fn_gkr_sum: cuda_sys::CUfunction,
    fn_gkr_quot_sc: cuda_sys::CUfunction,
    fn_gkr_round_step: cuda_sys::CUfunction,
    fn_air_exec_mz: cuda_sys::CUfunction,
    fn_gkr_sum_at_bit: cuda_sys::CUfunction,
    fn_gkr_quot_sc_il: cuda_sys::CUfunction,
    fn_logup_fp: cuda_sys::CUfunction,
    fn_logup_prepare_constants: cuda_sys::CUfunction,
    fn_ext_op_mz: cuda_sys::CUfunction,
    fn_poseidon16_mz: cuda_sys::CUfunction,
    fn_exec_mz_ext: cuda_sys::CUfunction,
    fn_ext_op_mz_ext: cuda_sys::CUfunction,
    fn_poseidon16_mz_ext: cuda_sys::CUfunction,
    fn_fold_multi_b2e: cuda_sys::CUfunction,
    fn_fold_multi_ext: cuda_sys::CUfunction,
    fn_fold_single_b2e_many_points: cuda_sys::CUfunction,
    fn_fold_single_ext_many_points: cuda_sys::CUfunction,
    fn_fold_multi_ext_per_point: cuda_sys::CUfunction,
    fn_repeat_ext_value: cuda_sys::CUfunction,
    fn_repeat_base_value_as_ext: cuda_sys::CUfunction,
    fn_bit_reverse: cuda_sys::CUfunction,
    fn_fold_multi_b2e_fb: cuda_sys::CUfunction,
    fn_fold_multi_ext_fb: cuda_sys::CUfunction,
    fn_exec_mz_fb: cuda_sys::CUfunction,
    fn_ext_op_mz_fb: cuda_sys::CUfunction,
    fn_pos16_mz_fb: cuda_sys::CUfunction,
    fn_exec_mz_ext_fb: cuda_sys::CUfunction,
    fn_ext_op_mz_ext_fb: cuda_sys::CUfunction,
    fn_pos16_mz_ext_fb: cuda_sys::CUfunction,
    fn_debug_pos16_ext: cuda_sys::CUfunction,
    fn_air_patch_state: cuda_sys::CUfunction,
    fn_air_patch_state_from_gkr: cuda_sys::CUfunction,
    fn_air_patch_pad_beta: cuda_sys::CUfunction,
    fn_air_patch_pad_evals: cuda_sys::CUfunction,
    fn_air_pad_eval: cuda_sys::CUfunction,
    fn_air_init_sums_from_logup: cuda_sys::CUfunction,
    fn_air_build_bare: cuda_sys::CUfunction,
    fn_protocol_step: cuda_sys::CUfunction,
    fn_prod_round_observe: cuda_sys::CUfunction,
    fn_prod_round_update: cuda_sys::CUfunction,
    fn_ext_affine_combine: cuda_sys::CUfunction,
    fn_ext_add: cuda_sys::CUfunction,
    fn_ext_powers: cuda_sys::CUfunction,
    fn_dense_eq_accumulate_from_points: cuda_sys::CUfunction,
    fn_ext_dot_accumulate: cuda_sys::CUfunction,
    fn_eq_from_flat_points: cuda_sys::CUfunction,
    fn_extract_permuted_suffix_ext: cuda_sys::CUfunction,
    fn_extract_reversed_suffix_ext: cuda_sys::CUfunction,
    fn_next_mle_from_point: cuda_sys::CUfunction,
    fn_reverse_gkr_point: cuda_sys::CUfunction,
    fn_gkr_finalize_layer: cuda_sys::CUfunction,
    fn_test_fiat_shamir: cuda_sys::CUfunction,
    fn_test_lagrange: cuda_sys::CUfunction,
    fn_test_expand: cuda_sys::CUfunction,
    fn_test_qe_inv: cuda_sys::CUfunction,
    fn_debug_pe: cuda_sys::CUfunction,
    fn_challenger_observe: cuda_sys::CUfunction,
    fn_challenger_sample_base: cuda_sys::CUfunction,
    fn_challenger_observe_sample: cuda_sys::CUfunction,
}

#[derive(Clone)]
pub struct LogupPreparedConstants {
    pub d_contrib: CudaSlice<u32>,
    pub d_alphas: CudaSlice<u32>,
    _input_guards: Option<LogupConstantDeviceInputs>,
}

#[derive(Clone)]
pub struct LogupConstantDeviceInputs {
    d_alpha_indices: CudaSlice<u32>,
    d_alpha_negated: CudaSlice<u32>,
    d_contrib_indices: CudaSlice<u32>,
    d_contrib_coeffs: CudaSlice<u32>,
    n_alphas: u32,
    n_contrib: u32,
}

unsafe impl Send for GpuSumcheck {}
unsafe impl Sync for GpuSumcheck {}

impl Drop for GpuSumcheck {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_result::module::unload(self.cu_module);
        }
    }
}

impl GpuSumcheck {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let cubin = include_bytes!(concat!(env!("OUT_DIR"), "/sumcheck.cubin"));
        let cu_module = unsafe { cuda_result::module::load_data(cubin.as_ptr().cast()) }
            .expect("failed to load sumcheck cubin");

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
            fn_sum_quot_ext: load("sum_quotients_ext_kernel"),
            fn_reduce_ext: load("reduce_ext_kernel"),
            fn_eq_fold: load("split_eq_fold_kernel"),
            fn_transpose: load("transpose_packed_ext_kernel"),
            fn_eq_expand: load("eq_expand_step_kernel"),
            fn_expand_univariate_points: load("expand_univariate_points_kernel"),
            fn_expand_sampled_base_query_points: load("expand_sampled_base_query_points_kernel"),
            fn_eq_accum: load("eq_accumulate_kernel"),
            fn_eq_accum_offset: load("eq_accumulate_offset_kernel"),
            fn_air_exec: load("air_sumcheck_execution_kernel"),
            fn_gkr_sum: load("gkr_sum_quotients_kernel"),
            fn_gkr_quot_sc: load("gkr_quotient_sumcheck_kernel"),
            fn_gkr_round_step: load("gkr_round_protocol_step_kernel"),
            fn_air_exec_mz: load("air_execution_multi_z_kernel"),
            fn_gkr_sum_at_bit: load("gkr_sum_quotients_at_bit_kernel"),
            fn_gkr_quot_sc_il: load("gkr_quotient_sc_interleaved_kernel"),
            fn_logup_fp: load("logup_fingerprint_kernel"),
            fn_logup_prepare_constants: load("logup_prepare_constants_kernel"),
            fn_ext_op_mz: load("air_ext_op_multi_z_kernel"),
            fn_poseidon16_mz: load("air_poseidon16_multi_z_kernel"),
            fn_exec_mz_ext: load("air_execution_multi_z_ext_kernel"),
            fn_ext_op_mz_ext: load("air_ext_op_multi_z_ext_kernel"),
            fn_poseidon16_mz_ext: load("air_poseidon16_multi_z_ext_kernel"),
            fn_fold_multi_b2e: load("fold_multi_col_b2e_half_kernel"),
            fn_fold_multi_ext: load("fold_multi_col_ext_half_kernel"),
            fn_fold_single_b2e_many_points: load("fold_single_b2e_many_points_half_kernel"),
            fn_fold_single_ext_many_points: load("fold_single_ext_many_points_half_kernel"),
            fn_fold_multi_ext_per_point: load("fold_multi_col_ext_half_per_point_kernel"),
            fn_repeat_ext_value: load("repeat_ext_value_kernel"),
            fn_repeat_base_value_as_ext: load("repeat_base_value_as_ext_kernel"),
            fn_bit_reverse: load("bit_reverse_within_chunks_kernel"),
            fn_fold_multi_b2e_fb: load("fold_multi_col_b2e_at_bit_kernel"),
            fn_fold_multi_ext_fb: load("fold_multi_col_ext_at_bit_kernel"),
            fn_exec_mz_fb: load("air_execution_multi_z_fb_kernel"),
            fn_ext_op_mz_fb: load("air_ext_op_multi_z_fb_kernel"),
            fn_pos16_mz_fb: load("air_poseidon16_multi_z_fb_kernel"),
            fn_exec_mz_ext_fb: load("air_execution_multi_z_ext_fb_kernel"),
            fn_ext_op_mz_ext_fb: load("air_ext_op_multi_z_ext_fb_kernel"),
            fn_pos16_mz_ext_fb: load("air_poseidon16_multi_z_ext_fb_kernel"),
            fn_debug_pos16_ext: load("debug_poseidon16_ext_eval_kernel"),
            fn_air_patch_state: load("air_sumcheck_patch_state_kernel"),
            fn_air_patch_state_from_gkr: load("air_sumcheck_patch_state_from_gkr_kernel"),
            fn_air_patch_pad_beta: load("air_sumcheck_patch_pad_beta_kernel"),
            fn_air_patch_pad_evals: load("air_sumcheck_patch_pad_evals_kernel"),
            fn_air_pad_eval: load("air_sumcheck_pad_eval_kernel"),
            fn_air_init_sums_from_logup: load("air_sumcheck_init_sums_from_logup_kernel"),
            fn_air_build_bare: load("air_sumcheck_build_bare_kernel"),
            fn_protocol_step: load("air_sumcheck_protocol_step_kernel"),
            fn_prod_round_observe: load("product_sumcheck_observe_round_poly_kernel"),
            fn_prod_round_update: load("product_sumcheck_update_sum_kernel"),
            fn_ext_affine_combine: load("ext_affine_combine_kernel"),
            fn_ext_add: load("ext_add_kernel"),
            fn_ext_powers: load("extension_powers_kernel"),
            fn_dense_eq_accumulate_from_points: load("dense_eq_accumulate_from_points_kernel"),
            fn_ext_dot_accumulate: load("ext_dot_accumulate_kernel"),
            fn_eq_from_flat_points: load("eq_polynomial_from_flat_points_kernel"),
            fn_extract_permuted_suffix_ext: load("extract_permuted_suffix_ext_kernel"),
            fn_extract_reversed_suffix_ext: load("extract_reversed_suffix_ext_kernel"),
            fn_next_mle_from_point: load("next_mle_from_point_kernel"),
            fn_reverse_gkr_point: load("reverse_gkr_challenges_into_point_kernel"),
            fn_gkr_finalize_layer: load("gkr_finalize_layer_kernel"),
            fn_test_fiat_shamir: load("test_fiat_shamir_kernel"),
            fn_test_lagrange: load("test_lagrange_interp_kernel"),
            fn_test_expand: load("test_expand_bare_to_full_kernel"),
            fn_test_qe_inv: load("test_qe_inv_kernel"),
            fn_debug_pe: load("debug_compute_pe_kernel"),
            fn_challenger_observe: load("challenger_observe_scalars_kernel"),
            fn_challenger_sample_base: load("challenger_sample_base_scalars_kernel"),
            fn_challenger_observe_sample: load("challenger_observe_and_sample_kernel"),
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
                    self.fn_reduce_ext,
                    (1, 1, 1),
                    (threads, 1, 1),
                    (threads / 32 * 5 * 4) as u32,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("reduce kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        let result = self.stream.memcpy_dtov(&d_result).unwrap();
        result[..5].try_into().unwrap()
    }

    fn reduce_partials_device(&self, d_partials: &CudaSlice<u32>, n_blocks: u32) -> CudaSlice<u32> {
        let mut d_result = self.stream.alloc_zeros::<u32>(5).unwrap();
        self.reduce_partials_device_into_async(d_partials, &mut d_result, n_blocks);
        self.stream.synchronize().unwrap();
        d_result
    }

    fn reduce_partials_device_into_async(
        &self,
        d_partials: &CudaSlice<u32>,
        d_result: &mut CudaSlice<u32>,
        n_blocks: u32,
    ) {
        {
            let (p_ptr, _g1) = d_partials.device_ptr(&self.stream);
            let (r_ptr, _g2) = d_result.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &p_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_blocks as *const _ as *mut _,
            ];
            let threads = 256u32.max(32);
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_reduce_ext,
                    (1, 1, 1),
                    (threads, 1, 1),
                    (threads / 32 * 5 * 4) as u32,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("reduce kernel failed");
            }
        }
    }

    /// Product sumcheck (base × ext): compute c0 and c2.
    /// `pol_a`: n base field elements. `pol_b`: n × 5 ext field elements.
    /// Returns (c0, c2) as ext field elements.
    pub fn product_sumcheck_base_ext(&self, pol_a: &[u32], pol_b: &[u32]) -> ([u32; 5], [u32; 5]) {
        let n = pol_a.len();
        assert_eq!(pol_b.len(), n * 5);
        let half = (n / 2) as u32;

        let d_a = self.stream.memcpy_stod(pol_a).unwrap();
        let d_b = self.stream.memcpy_stod(pol_b).unwrap();

        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;

        let mut d_c0 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();

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
                    self.fn_prod_base_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("product sumcheck kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        let c0 = self.reduce_partials(&d_c0, blocks);
        let c2 = self.reduce_partials(&d_c2, blocks);
        (c0, c2)
    }

    /// Sum extension-field quotients Σ nums[i] / dens[i].
    pub fn sum_quotients_ext_device(
        &self,
        d_nums: &CudaSlice<u32>,
        d_dens: &CudaSlice<u32>,
        n: u32,
    ) -> [u32; 5] {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let smem = (threads / 32 * 5 * 4) as u32;
        let mut d_partials = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();

        {
            let (nums_ptr, _g1) = d_nums.device_ptr(&self.stream);
            let (dens_ptr, _g2) = d_dens.device_ptr(&self.stream);
            let (part_ptr, _g3) = d_partials.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &nums_ptr as *const _ as *mut _,
                &dens_ptr as *const _ as *mut _,
                &part_ptr as *const _ as *mut _,
                &n as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_sum_quot_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("sum quotients kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_partials(&d_partials, blocks)
    }

    /// Product sumcheck (ext × ext).
    pub fn product_sumcheck_ext_ext(&self, pol_a: &[u32], pol_b: &[u32]) -> ([u32; 5], [u32; 5]) {
        let n = pol_a.len() / 5;
        assert_eq!(pol_b.len(), n * 5);
        let half = (n / 2) as u32;

        let d_a = self.stream.memcpy_stod(pol_a).unwrap();
        let d_b = self.stream.memcpy_stod(pol_b).unwrap();

        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;

        let mut d_c0 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();

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
                    self.fn_prod_ext_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("product sumcheck ext×ext kernel failed");
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
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();

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
                    self.fn_eq_fold,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("eq fold kernel failed");
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
        let mut d_flat = self
            .stream
            .alloc_zeros::<u32>(total * dim as usize)
            .unwrap();

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
                    self.fn_transpose,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("transpose kernel failed");
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
        let mut d_flat_b = self
            .stream
            .alloc_zeros::<u32>(total_scalars * dim as usize)
            .unwrap();
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
                    self.fn_transpose,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("transpose kernel failed");
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
        let (d_c0, d_c2) = self.product_sumcheck_base_ext_device_resident(d_pol_a, d_pol_b, half);
        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let c0 = self.reduce_partials(&d_c0, blocks);
        let c2 = self.reduce_partials(&d_c2, blocks);
        (c0, c2)
    }

    /// Product sumcheck (base × ext) on device buffers.
    /// Returns the reduced c0/c2 coefficients on device.
    pub fn product_sumcheck_base_ext_device_resident(
        &self,
        d_pol_a: &CudaSlice<u32>,
        d_pol_b: &CudaSlice<u32>,
        half: u32,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let mut d_c0_partials = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2_partials = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c0_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_c2_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        self.product_sumcheck_base_ext_device_resident_into_async(
            d_pol_a,
            d_pol_b,
            half,
            &mut d_c0_partials,
            &mut d_c2_partials,
            &mut d_c0_out,
            &mut d_c2_out,
        );
        self.stream.synchronize().unwrap();
        (d_c0_out, d_c2_out)
    }

    pub fn product_sumcheck_base_ext_device_resident_into_async<A, B>(
        &self,
        d_pol_a: &A,
        d_pol_b: &B,
        half: u32,
        d_c0_partials: &mut CudaSlice<u32>,
        d_c2_partials: &mut CudaSlice<u32>,
        d_c0_out: &mut CudaSlice<u32>,
        d_c2_out: &mut CudaSlice<u32>,
    ) where
        A: DevicePtr<u32>,
        B: DevicePtr<u32>,
    {
        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;
        {
            let (a_ptr, _g1) = d_pol_a.device_ptr(&self.stream);
            let (b_ptr, _g2) = d_pol_b.device_ptr(&self.stream);
            let (c0_ptr, _g3) = d_c0_partials.device_ptr_mut(&self.stream);
            let (c2_ptr, _g4) = d_c2_partials.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &c0_ptr as *const _ as *mut _,
                &c2_ptr as *const _ as *mut _,
                &half as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_prod_base_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("product sumcheck kernel failed");
            }
        }
        self.reduce_partials_device_into_async(d_c0_partials, d_c0_out, blocks);
        self.reduce_partials_device_into_async(d_c2_partials, d_c2_out, blocks);
    }

    /// Product sumcheck (ext × ext) on device buffers.
    pub fn product_sumcheck_ext_ext_device(
        &self,
        d_pol_a: &CudaSlice<u32>,
        d_pol_b: &CudaSlice<u32>,
        half: u32,
    ) -> ([u32; 5], [u32; 5]) {
        let (d_c0, d_c2) = self.product_sumcheck_ext_ext_device_resident(d_pol_a, d_pol_b, half);
        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let c0 = self.reduce_partials(&d_c0, blocks);
        let c2 = self.reduce_partials(&d_c2, blocks);
        (c0, c2)
    }

    /// Product sumcheck (ext × ext) on device buffers.
    /// Returns the reduced c0/c2 coefficients on device.
    pub fn product_sumcheck_ext_ext_device_resident(
        &self,
        d_pol_a: &CudaSlice<u32>,
        d_pol_b: &CudaSlice<u32>,
        half: u32,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let mut d_c0_partials = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2_partials = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c0_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_c2_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        self.product_sumcheck_ext_ext_device_resident_into_async(
            d_pol_a,
            d_pol_b,
            half,
            &mut d_c0_partials,
            &mut d_c2_partials,
            &mut d_c0_out,
            &mut d_c2_out,
        );
        self.stream.synchronize().unwrap();
        (d_c0_out, d_c2_out)
    }

    pub fn product_sumcheck_ext_ext_device_resident_into_async<A, B>(
        &self,
        d_pol_a: &A,
        d_pol_b: &B,
        half: u32,
        d_c0_partials: &mut CudaSlice<u32>,
        d_c2_partials: &mut CudaSlice<u32>,
        d_c0_out: &mut CudaSlice<u32>,
        d_c2_out: &mut CudaSlice<u32>,
    ) where
        A: DevicePtr<u32>,
        B: DevicePtr<u32>,
    {
        let threads = 256u32;
        let blocks = (half + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;
        {
            let (a_ptr, _g1) = d_pol_a.device_ptr(&self.stream);
            let (b_ptr, _g2) = d_pol_b.device_ptr(&self.stream);
            let (c0_ptr, _g3) = d_c0_partials.device_ptr_mut(&self.stream);
            let (c2_ptr, _g4) = d_c2_partials.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &c0_ptr as *const _ as *mut _,
                &c2_ptr as *const _ as *mut _,
                &half as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_prod_ext_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("product sumcheck ext×ext kernel failed");
            }
        }
        self.reduce_partials_device_into_async(d_c0_partials, d_c0_out, blocks);
        self.reduce_partials_device_into_async(d_c2_partials, d_c2_out, blocks);
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
            let mut d_next = self
                .stream
                .alloc_zeros::<u32>((n_current as usize * 2) * 5)
                .unwrap();

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
                        self.fn_eq_expand,
                        (blocks, 1, 1),
                        (threads, 1, 1),
                        0,
                        self.stream.cu_stream(),
                        &mut args,
                    )
                    .expect("eq expand kernel failed");
                }
            }
            self.stream.synchronize().unwrap();
            d_current = d_next;
            n_current *= 2;
        }

        d_current
    }

    /// Build eq(point, x) on GPU for all x in {0,1}^n_vars}, where `point` is
    /// provided as a flat device buffer of `n_vars * 5` words.
    pub fn eq_polynomial_device_from_flat_point_words<P>(
        &self,
        d_point_words: &P,
        n_vars: usize,
    ) -> CudaSlice<u32>
    where
        P: DevicePtr<u32>,
    {
        let d_out = self.eq_polynomial_device_from_flat_point_words_async(d_point_words, n_vars);
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn eq_polynomial_device_from_flat_point_words_async<P>(
        &self,
        d_point_words: &P,
        n_vars: usize,
    ) -> CudaSlice<u32>
    where
        P: DevicePtr<u32>,
    {
        let n_total = 1usize << n_vars;
        let mut d_out = self.stream.alloc_zeros::<u32>(n_total * 5).unwrap();
        self.eq_polynomial_device_from_flat_point_words_into_async(
            d_point_words,
            n_vars as u32,
            &mut d_out,
        );
        d_out
    }

    pub fn eq_polynomial_device_from_flat_point_words_into_async<P>(
        &self,
        d_point_words: &P,
        n_vars: u32,
        d_out: &mut CudaSlice<u32>,
    ) where
        P: DevicePtr<u32>,
    {
        let n_total = 1u32 << n_vars;
        debug_assert!(d_point_words.len() >= (n_vars as usize) * 5);
        debug_assert!(d_out.len() >= (n_total as usize) * 5);
        let threads = 256u32;
        let blocks = (n_total + threads - 1) / threads;
        let (points_ptr, _) = d_point_words.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &points_ptr as *const _ as *mut _,
            &n_vars as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n_total as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_eq_from_flat_points,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("eq polynomial from flat points kernel failed");
        }
    }

    pub fn eq_polynomial_device_from_permuted_suffix_point_words(
        &self,
        d_point_words: &CudaSlice<u32>,
        total_coords: usize,
        suffix_len: usize,
        len: usize,
        pivot: usize,
    ) -> CudaSlice<u32> {
        debug_assert_eq!(d_point_words.len(), total_coords * 5);
        debug_assert!(suffix_len <= total_coords);
        debug_assert!(len <= suffix_len);
        if len == 0 {
            return self
                .stream
                .memcpy_stod(&[0x01FFFFFEu32, 0, 0, 0, 0])
                .unwrap();
        }
        let d_out = self.eq_polynomial_device_from_permuted_suffix_point_words_async(
            d_point_words,
            total_coords,
            suffix_len,
            len,
            pivot,
        );
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn eq_polynomial_device_from_permuted_suffix_point_words_async(
        &self,
        d_point_words: &CudaSlice<u32>,
        total_coords: usize,
        suffix_len: usize,
        len: usize,
        pivot: usize,
    ) -> CudaSlice<u32> {
        let mut d_permuted = self.stream.alloc_zeros::<u32>(len * 5).unwrap();
        let mut d_out = self.stream.alloc_zeros::<u32>((1usize << len) * 5).unwrap();
        self.eq_polynomial_device_from_permuted_suffix_point_words_into_async(
            d_point_words,
            total_coords,
            suffix_len,
            len,
            pivot,
            &mut d_permuted,
            &mut d_out,
        );
        d_out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eq_polynomial_device_from_permuted_suffix_point_words_into_async(
        &self,
        d_point_words: &CudaSlice<u32>,
        total_coords: usize,
        suffix_len: usize,
        len: usize,
        pivot: usize,
        d_permuted: &mut CudaSlice<u32>,
        d_out: &mut CudaSlice<u32>,
    ) {
        debug_assert_eq!(d_point_words.len(), total_coords * 5);
        debug_assert!(suffix_len <= total_coords);
        debug_assert!(len <= suffix_len);
        debug_assert!(d_permuted.len() >= len * 5);
        debug_assert!(d_out.len() >= (1usize << len) * 5);
        assert!(
            len > 0,
            "zero-var permuted suffix eq requires a pre-uploaded device one"
        );

        self.extract_permuted_suffix_point_words_into_async(
            d_point_words,
            total_coords,
            suffix_len,
            len,
            pivot,
            d_permuted,
        );
        self.eq_polynomial_device_from_flat_point_words_into_async(d_permuted, len as u32, d_out);
    }

    pub fn extract_permuted_suffix_point_words_into_async(
        &self,
        d_point_words: &CudaSlice<u32>,
        total_coords: usize,
        suffix_len: usize,
        len: usize,
        pivot: usize,
        d_out: &mut CudaSlice<u32>,
    ) {
        debug_assert_eq!(d_point_words.len(), total_coords * 5);
        debug_assert!(suffix_len <= total_coords);
        debug_assert!(len <= suffix_len);
        debug_assert!(d_out.len() >= len * 5);
        assert!(
            len > 0,
            "zero-var permuted suffix extraction requires no output buffer"
        );

        let total_coords_u32 = total_coords as u32;
        let suffix_len_u32 = suffix_len as u32;
        let len_u32 = len as u32;
        let pivot_u32 = pivot as u32;
        let threads = 256u32;
        let blocks = (len_u32 + threads - 1) / threads;
        let (point_ptr, _) = d_point_words.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &point_ptr as *const _ as *mut _,
            &total_coords_u32 as *const _ as *mut _,
            &suffix_len_u32 as *const _ as *mut _,
            &len_u32 as *const _ as *mut _,
            &pivot_u32 as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_extract_permuted_suffix_ext,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("extract permuted suffix point kernel failed");
        }
    }

    pub fn reversed_suffix_point_words(
        &self,
        d_point_words: &CudaSlice<u32>,
        total_coords: usize,
        suffix_len: usize,
    ) -> CudaSlice<u32> {
        debug_assert_eq!(d_point_words.len(), total_coords * 5);
        debug_assert!(suffix_len <= total_coords);
        if suffix_len == 0 {
            return self.stream.alloc_zeros::<u32>(0).unwrap();
        }

        let mut d_out = self.stream.alloc_zeros::<u32>(suffix_len * 5).unwrap();
        self.reversed_suffix_point_words_into_async(
            d_point_words,
            total_coords,
            suffix_len,
            &mut d_out,
        );
        d_out
    }

    pub fn reversed_suffix_point_words_into_async(
        &self,
        d_point_words: &CudaSlice<u32>,
        total_coords: usize,
        suffix_len: usize,
        d_out: &mut CudaSlice<u32>,
    ) {
        debug_assert_eq!(d_point_words.len(), total_coords * 5);
        debug_assert!(suffix_len <= total_coords);
        debug_assert!(d_out.len() >= suffix_len * 5);
        if suffix_len == 0 {
            return;
        }

        let total_coords_u32 = total_coords as u32;
        let suffix_len_u32 = suffix_len as u32;
        let threads = 256u32;
        let blocks = (suffix_len_u32 + threads - 1) / threads;
        let (point_ptr, _) = d_point_words.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &point_ptr as *const _ as *mut _,
            &total_coords_u32 as *const _ as *mut _,
            &suffix_len_u32 as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_extract_reversed_suffix_ext,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("extract reversed suffix point kernel failed");
        }
    }

    pub fn next_mle_device_from_point_words<P>(
        &self,
        d_point_words: &P,
        n_vars: usize,
    ) -> CudaSlice<u32>
    where
        P: DevicePtr<u32>,
    {
        debug_assert_eq!(d_point_words.len(), n_vars * 5);
        let n_total = 1usize << n_vars;
        let mut d_out = self.stream.alloc_zeros::<u32>(n_total * 5).unwrap();
        self.next_mle_device_from_point_words_into_async(d_point_words, n_vars, &mut d_out);
        d_out
    }

    pub fn next_mle_device_from_point_words_into_async<P>(
        &self,
        d_point_words: &P,
        n_vars: usize,
        d_out: &mut CudaSlice<u32>,
    ) where
        P: DevicePtr<u32>,
    {
        debug_assert_eq!(d_point_words.len(), n_vars * 5);
        let n_vars_u32 = n_vars as u32;
        let n_total_u32 = 1u32 << n_vars_u32;
        debug_assert!(d_out.len() >= (n_total_u32 as usize) * 5);
        let threads = 256u32;
        let blocks = (n_total_u32 + threads - 1) / threads;
        let (point_ptr, _) = d_point_words.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &point_ptr as *const _ as *mut _,
            &n_vars_u32 as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n_total_u32 as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_next_mle_from_point,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("next mle from point kernel failed");
        }
    }

    pub fn reverse_gkr_challenges_into_point_async(
        &self,
        d_round_challenges: &CudaSlice<u32>,
        n_rounds: u32,
        d_out: &mut CudaSlice<u32>,
    ) {
        debug_assert!(d_round_challenges.len() >= (n_rounds as usize + 1) * 5);
        debug_assert!(d_out.len() >= (n_rounds as usize + 1) * 5);
        let total_coords = n_rounds + 1;
        let threads = 64u32;
        let blocks = (total_coords + threads - 1) / threads;
        let (src_ptr, _) = d_round_challenges.device_ptr(&self.stream);
        let (dst_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &n_rounds as *const _ as *mut _,
            &dst_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_reverse_gkr_point,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("reverse gkr point kernel failed");
        }
    }

    pub fn expand_univariate_points_device(
        &self,
        d_univariate_points: &CudaSlice<u32>,
        n_points: u32,
        n_vars: u32,
    ) -> CudaSlice<u32> {
        let d_out =
            self.expand_univariate_points_device_async(d_univariate_points, n_points, n_vars);
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn expand_univariate_points_device_async(
        &self,
        d_univariate_points: &CudaSlice<u32>,
        n_points: u32,
        n_vars: u32,
    ) -> CudaSlice<u32> {
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_points as usize) * (n_vars as usize) * 5)
            .unwrap();
        self.expand_univariate_points_device_into_async(
            d_univariate_points,
            n_points,
            n_vars,
            &mut d_out,
        );
        d_out
    }

    pub fn expand_univariate_points_device_into_async(
        &self,
        d_univariate_points: &CudaSlice<u32>,
        n_points: u32,
        n_vars: u32,
        d_out: &mut CudaSlice<u32>,
    ) {
        assert_eq!(
            d_out.len(),
            (n_points as usize) * (n_vars as usize) * 5,
            "expanded univariate point output has wrong length"
        );
        if n_points == 0 || n_vars == 0 {
            return;
        }
        {
            let (src_ptr, _g1) = d_univariate_points.device_ptr(&self.stream);
            let (dst_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let threads = 256u32;
            let blocks = (n_points + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &src_ptr as *const _ as *mut _,
                &dst_ptr as *const _ as *mut _,
                &n_points as *const _ as *mut _,
                &n_vars as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_expand_univariate_points,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("expand univariate points kernel failed");
            }
        }
    }

    pub fn expand_sampled_base_query_points_device(
        &self,
        d_sampled_words: &CudaSlice<u32>,
        n_samples: u32,
        bits: u32,
        domain_gen: u32,
        n_vars: u32,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let (d_points, d_indices) = self.expand_sampled_base_query_points_device_async(
            d_sampled_words,
            n_samples,
            bits,
            domain_gen,
            n_vars,
        );
        self.stream.synchronize().unwrap();
        (d_points, d_indices)
    }

    pub fn expand_sampled_base_query_points_device_async(
        &self,
        d_sampled_words: &CudaSlice<u32>,
        n_samples: u32,
        bits: u32,
        domain_gen: u32,
        n_vars: u32,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let mut d_points = self
            .stream
            .alloc_zeros::<u32>((n_samples as usize) * (n_vars as usize) * 5)
            .unwrap();
        let mut d_indices = self.stream.alloc_zeros::<u32>(n_samples as usize).unwrap();
        self.expand_sampled_base_query_points_device_into_async(
            d_sampled_words,
            n_samples,
            bits,
            domain_gen,
            n_vars,
            &mut d_points,
            &mut d_indices,
        );
        (d_points, d_indices)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn expand_sampled_base_query_points_device_into_async(
        &self,
        d_sampled_words: &CudaSlice<u32>,
        n_samples: u32,
        bits: u32,
        domain_gen: u32,
        n_vars: u32,
        d_points: &mut CudaSlice<u32>,
        d_indices: &mut CudaSlice<u32>,
    ) {
        assert!(d_sampled_words.len() >= n_samples as usize);
        assert!(d_points.len() >= (n_samples as usize) * (n_vars as usize) * 5);
        assert!(d_indices.len() >= n_samples as usize);
        if n_samples == 0 {
            return;
        }
        {
            let (sample_ptr, _g1) = d_sampled_words.device_ptr(&self.stream);
            let (points_ptr, _g2) = d_points.device_ptr_mut(&self.stream);
            let (indices_ptr, _g3) = d_indices.device_ptr_mut(&self.stream);
            let threads = 256u32;
            let blocks = (n_samples + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &sample_ptr as *const _ as *mut _,
                &points_ptr as *const _ as *mut _,
                &indices_ptr as *const _ as *mut _,
                &n_samples as *const _ as *mut _,
                &bits as *const _ as *mut _,
                &domain_gen as *const _ as *mut _,
                &n_vars as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_expand_sampled_base_query_points,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("expand sampled base query points kernel failed");
            }
        }
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
        self.eq_accumulate_device_scalar_device(d_weights, d_eq_val, &d_scalar, n);
        self.stream.synchronize().unwrap();
    }

    pub fn eq_accumulate_device_scalar_device<E, S>(
        &self,
        d_weights: &mut CudaSlice<u32>,
        d_eq_val: &E,
        d_scalar: &S,
        n: u32,
    ) where
        E: DevicePtr<u32>,
        S: DevicePtr<u32>,
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
                self.fn_eq_accum,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("eq accumulate kernel failed");
        }
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
        self.eq_accumulate_offset_device_scalar_device(d_weights, d_eq_val, &d_scalar, offset, n);
        self.stream.synchronize().unwrap();
    }

    pub fn eq_accumulate_offset_device_scalar_device<E, S>(
        &self,
        d_weights: &mut CudaSlice<u32>,
        d_eq_val: &E,
        d_scalar: &S,
        offset: u32,
        n: u32,
    ) where
        E: DevicePtr<u32>,
        S: DevicePtr<u32>,
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
                self.fn_eq_accum_offset,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("eq accumulate offset kernel failed");
        }
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
        alphas: &[u32], // 13 * 5 = 65 u32s
        n_rows: u32,
        n_pairs: u32,
    ) -> ([u32; 5], [u32; 5]) {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let smem = (threads / 32 * 2 * 5 * 4) as u32;

        let mut d_z0 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_z2 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();

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
                    self.fn_air_exec,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("air sumcheck execution kernel failed");
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
        let mut d_new_nums = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        let mut d_new_dens = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();

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
                    self.fn_gkr_sum,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("gkr sum quotients kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        (d_new_nums, d_new_dens)
    }

    /// GKR quotient sumcheck: compute (c0_num, c2_num, c0_den, c2_den) for one round.
    ///
    /// All four arrays are ext field on GPU. eq_vals has `half` elements.
    /// Returns 4 ext field values: (c0_num, c2_num, c0_den, c2_den).
    /// Multi-z Execution AIR constraint evaluation.
    /// Evaluates at z=0,2,3,4 (degree 5). Returns 4 ext field sums.
    pub fn air_execution_multi_z_device(
        &self,
        d_columns: &CudaSlice<u32>,   // 20 * n_rows base (col-major)
        d_down_cols: &CudaSlice<u32>, // 2 * n_rows base
        d_eq_factor: &CudaSlice<u32>, // n_pairs * 5 ext
        alphas: &[u32],               // 13 * 5 = 65 u32s
        n_rows: u32,
        n_pairs: u32,
    ) -> ([u32; 5], [u32; 5], [u32; 5], [u32; 5]) {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let smem = (4 * (threads / 32) * 5 * 4) as u32;

        let mut d_z0 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_z2 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_z3 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_z4 = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();

        {
            let (c_ptr, _) = d_columns.device_ptr(&self.stream);
            let (dc_ptr, _) = d_down_cols.device_ptr(&self.stream);
            let (eq_ptr, _) = d_eq_factor.device_ptr(&self.stream);
            let (al_ptr, _) = d_alphas.device_ptr(&self.stream);
            let (z0_ptr, _) = d_z0.device_ptr_mut(&self.stream);
            let (z2_ptr, _) = d_z2.device_ptr_mut(&self.stream);
            let (z3_ptr, _) = d_z3.device_ptr_mut(&self.stream);
            let (z4_ptr, _) = d_z4.device_ptr_mut(&self.stream);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c_ptr as *const _ as *mut _,
                &dc_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &al_ptr as *const _ as *mut _,
                &z0_ptr as *const _ as *mut _,
                &z2_ptr as *const _ as *mut _,
                &z3_ptr as *const _ as *mut _,
                &z4_ptr as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_air_exec_mz,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("air execution multi-z kernel failed");
            }
        }

        let z0 = self.reduce_partials(&d_z0, blocks);
        let z2 = self.reduce_partials(&d_z2, blocks);
        let z3 = self.reduce_partials(&d_z3, blocks);
        let z4 = self.reduce_partials(&d_z4, blocks);
        (z0, z2, z3, z4)
    }

    /// GKR quotient sum with fold at arbitrary bit. Device-resident.
    /// Pairs at stride 2^fold_bit. Output has n_pairs elements.
    pub fn gkr_sum_quotients_at_bit_device(
        &self,
        d_nums: &CudaSlice<u32>,
        d_dens: &CudaSlice<u32>,
        n_pairs: u32,
        fold_bit: u32,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let mut d_new_nums = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        let mut d_new_dens = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
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
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_gkr_sum_at_bit,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("gkr sum quotients at bit kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        (d_new_nums, d_new_dens)
    }

    /// GKR quotient sumcheck with parameterized fold bit.
    ///
    /// `fold_bit`: which bit to fold on (0 = adjacent pairs, k = stride 2^k).
    /// `n_pairs`: number of pair positions (elements with fold_bit clear).
    pub fn gkr_quotient_sumcheck_device(
        &self,
        d_nums_l: &CudaSlice<u32>,
        d_nums_r: &CudaSlice<u32>,
        d_dens_l: &CudaSlice<u32>,
        d_dens_r: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        n_pairs: u32,
        fold_bit: u32,
    ) -> ([u32; 5], [u32; 5], [u32; 5], [u32; 5]) {
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let smem_size = (4 * (threads / 32) * 5 * 4) as u32;

        let mut d_c0n = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2n = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c0d = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2d = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();

        {
            let (nl_ptr, _g1) = d_nums_l.device_ptr(&self.stream);
            let (nr_ptr, _g2) = d_nums_r.device_ptr(&self.stream);
            let (dl_ptr, _g3) = d_dens_l.device_ptr(&self.stream);
            let (dr_ptr, _g4) = d_dens_r.device_ptr(&self.stream);
            let (eq_ptr, _g5) = d_eq.device_ptr(&self.stream);
            let (c0n_ptr, _g6) = d_c0n.device_ptr_mut(&self.stream);
            let (c2n_ptr, _g7) = d_c2n.device_ptr_mut(&self.stream);
            let (c0d_ptr, _g8) = d_c0d.device_ptr_mut(&self.stream);
            let (c2d_ptr, _g9) = d_c2d.device_ptr_mut(&self.stream);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &nl_ptr as *const _ as *mut _,
                &nr_ptr as *const _ as *mut _,
                &dl_ptr as *const _ as *mut _,
                &dr_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &c0n_ptr as *const _ as *mut _,
                &c2n_ptr as *const _ as *mut _,
                &c0d_ptr as *const _ as *mut _,
                &c2d_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_gkr_quot_sc,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem_size,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("gkr quotient sumcheck kernel failed");
            }
        }

        let c0_num = self.reduce_partials(&d_c0n, blocks);
        let c2_num = self.reduce_partials(&d_c2n, blocks);
        let c0_den = self.reduce_partials(&d_c0d, blocks);
        let c2_den = self.reduce_partials(&d_c2d, blocks);
        (c0_num, c2_num, c0_den, c2_den)
    }

    pub fn gkr_quotient_sumcheck_device_resident(
        &self,
        d_nums_l: &CudaSlice<u32>,
        d_nums_r: &CudaSlice<u32>,
        d_dens_l: &CudaSlice<u32>,
        d_dens_r: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        n_pairs: u32,
        fold_bit: u32,
    ) -> (
        CudaSlice<u32>,
        CudaSlice<u32>,
        CudaSlice<u32>,
        CudaSlice<u32>,
    ) {
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let smem_size = (4 * (threads / 32) * 5 * 4) as u32;

        let mut d_c0n = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2n = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c0d = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2d = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c0n_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_c2n_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_c0d_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_c2d_out = self.stream.alloc_zeros::<u32>(5).unwrap();

        self.gkr_quotient_sumcheck_device_resident_into_async(
            d_nums_l,
            d_nums_r,
            d_dens_l,
            d_dens_r,
            d_eq,
            n_pairs,
            fold_bit,
            &mut d_c0n,
            &mut d_c2n,
            &mut d_c0d,
            &mut d_c2d,
            &mut d_c0n_out,
            &mut d_c2n_out,
            &mut d_c0d_out,
            &mut d_c2d_out,
        );
        self.stream.synchronize().unwrap();

        (d_c0n_out, d_c2n_out, d_c0d_out, d_c2d_out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gkr_quotient_sumcheck_device_resident_into_async(
        &self,
        d_nums_l: &CudaSlice<u32>,
        d_nums_r: &CudaSlice<u32>,
        d_dens_l: &CudaSlice<u32>,
        d_dens_r: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        n_pairs: u32,
        fold_bit: u32,
        d_c0n_partials: &mut CudaSlice<u32>,
        d_c2n_partials: &mut CudaSlice<u32>,
        d_c0d_partials: &mut CudaSlice<u32>,
        d_c2d_partials: &mut CudaSlice<u32>,
        d_c0n_out: &mut CudaSlice<u32>,
        d_c2n_out: &mut CudaSlice<u32>,
        d_c0d_out: &mut CudaSlice<u32>,
        d_c2d_out: &mut CudaSlice<u32>,
    ) {
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let smem_size = (4 * (threads / 32) * 5 * 4) as u32;

        {
            let (nl_ptr, _g1) = d_nums_l.device_ptr(&self.stream);
            let (nr_ptr, _g2) = d_nums_r.device_ptr(&self.stream);
            let (dl_ptr, _g3) = d_dens_l.device_ptr(&self.stream);
            let (dr_ptr, _g4) = d_dens_r.device_ptr(&self.stream);
            let (eq_ptr, _g5) = d_eq.device_ptr(&self.stream);
            let (c0n_ptr, _g6) = d_c0n_partials.device_ptr_mut(&self.stream);
            let (c2n_ptr, _g7) = d_c2n_partials.device_ptr_mut(&self.stream);
            let (c0d_ptr, _g8) = d_c0d_partials.device_ptr_mut(&self.stream);
            let (c2d_ptr, _g9) = d_c2d_partials.device_ptr_mut(&self.stream);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &nl_ptr as *const _ as *mut _,
                &nr_ptr as *const _ as *mut _,
                &dl_ptr as *const _ as *mut _,
                &dr_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &c0n_ptr as *const _ as *mut _,
                &c2n_ptr as *const _ as *mut _,
                &c0d_ptr as *const _ as *mut _,
                &c2d_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_gkr_quot_sc,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem_size,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("gkr quotient sumcheck kernel failed");
            }
        }
        self.reduce_partials_device_into_async(d_c0n_partials, d_c0n_out, blocks);
        self.reduce_partials_device_into_async(d_c2n_partials, d_c2n_out, blocks);
        self.reduce_partials_device_into_async(d_c0d_partials, d_c0d_out, blocks);
        self.reduce_partials_device_into_async(d_c2d_partials, d_c2d_out, blocks);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gkr_round_protocol_step(
        &self,
        d_c0_num: &CudaSlice<u32>,
        d_c2_num: &CudaSlice<u32>,
        d_c0_den: &CudaSlice<u32>,
        d_c2_den: &CudaSlice<u32>,
        d_alpha: &CudaSlice<u32>,
        eq_alpha: &[u32; 5],
        eq_prefix: &[[u32; 5]],
        active_pairs: u32,
        d_sum: &mut CudaSlice<u32>,
        d_mmf: &mut CudaSlice<u32>,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let d_eq_alpha = self.stream.memcpy_stod(eq_alpha.as_slice()).unwrap();
        let d_eq_prefix = if eq_prefix.is_empty() {
            self.stream.alloc_zeros::<u32>(1).unwrap()
        } else {
            let words: Vec<u32> = eq_prefix
                .iter()
                .flat_map(|coord| coord.iter().copied())
                .collect();
            self.stream.memcpy_stod(&words).unwrap()
        };
        let eq_prefix_len = eq_prefix.len() as u32;
        let mut d_r = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_transcript_tail = self.stream.alloc_zeros::<u32>(10).unwrap();
        self.gkr_round_protocol_step_precomputed_async(
            d_c0_num,
            d_c2_num,
            d_c0_den,
            d_c2_den,
            d_alpha,
            &d_eq_alpha,
            &d_eq_prefix,
            eq_prefix_len,
            active_pairs,
            d_sum,
            d_mmf,
            &mut d_r,
            &mut d_transcript_tail,
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
        );
        self.stream.synchronize().unwrap();
        (d_r, d_transcript_tail)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gkr_round_protocol_step_precomputed_async(
        &self,
        d_c0_num: &CudaSlice<u32>,
        d_c2_num: &CudaSlice<u32>,
        d_c0_den: &CudaSlice<u32>,
        d_c2_den: &CudaSlice<u32>,
        d_alpha: &CudaSlice<u32>,
        d_eq_alpha: &CudaSlice<u32>,
        d_eq_prefix: &CudaSlice<u32>,
        eq_prefix_len: u32,
        active_pairs: u32,
        d_sum: &mut CudaSlice<u32>,
        d_mmf: &mut CudaSlice<u32>,
        d_r: &mut CudaSlice<u32>,
        d_transcript_tail: &mut CudaSlice<u32>,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
    ) {
        {
            let (c0n_ptr, _) = d_c0_num.device_ptr(&self.stream);
            let (c2n_ptr, _) = d_c2_num.device_ptr(&self.stream);
            let (c0d_ptr, _) = d_c0_den.device_ptr(&self.stream);
            let (c2d_ptr, _) = d_c2_den.device_ptr(&self.stream);
            let (al_ptr, _) = d_alpha.device_ptr(&self.stream);
            let (ea_ptr, _) = d_eq_alpha.device_ptr(&self.stream);
            let (ep_ptr, _) = d_eq_prefix.device_ptr(&self.stream);
            let (sum_ptr, _) = d_sum.device_ptr_mut(&self.stream);
            let (mmf_ptr, _) = d_mmf.device_ptr_mut(&self.stream);
            let (r_ptr, _) = d_r.device_ptr_mut(&self.stream);
            let (tt_ptr, _) = d_transcript_tail.device_ptr_mut(&self.stream);
            let (cs_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c0n_ptr as *const _ as *mut _,
                &c2n_ptr as *const _ as *mut _,
                &c0d_ptr as *const _ as *mut _,
                &c2d_ptr as *const _ as *mut _,
                &al_ptr as *const _ as *mut _,
                &ea_ptr as *const _ as *mut _,
                &ep_ptr as *const _ as *mut _,
                &eq_prefix_len as *const _ as *mut _,
                &active_pairs as *const _ as *mut _,
                &sum_ptr as *const _ as *mut _,
                &mmf_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &tt_ptr as *const _ as *mut _,
                &cs_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_gkr_round_step,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("gkr round protocol step kernel failed");
            }
        }
    }

    /// GKR quotient sumcheck on interleaved data (single array, not split).
    ///
    /// Data layout within each chunk: [left_lo | left_hi | right_lo | right_hi].
    /// `chunk_size` = 2^chunk_log, `n_chunks` = n_total / chunk_size.
    /// Returns (c0_num, c2_num, c0_den, c2_den).
    pub fn gkr_quotient_sc_interleaved_device(
        &self,
        d_nums: &CudaSlice<u32>,
        d_dens: &CudaSlice<u32>,
        d_eq_within: &CudaSlice<u32>, // quarter * 5 ext
        d_eq_outer: &CudaSlice<u32>,  // n_chunks * 5 ext
        n_total: u32,
        chunk_size: u32,
        n_chunks: u32,
    ) -> ([u32; 5], [u32; 5], [u32; 5], [u32; 5]) {
        let quarter = chunk_size / 4;
        let total_pairs = n_chunks * quarter;
        let threads = 256u32;
        let blocks = (total_pairs + threads - 1) / threads;
        let smem_size = (4 * (threads / 32) * 5 * 4) as u32;

        let mut d_c0n = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2n = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c0d = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();
        let mut d_c2d = self
            .stream
            .alloc_zeros::<u32>((blocks as usize) * 5)
            .unwrap();

        {
            let (n_ptr, _) = d_nums.device_ptr(&self.stream);
            let (d_ptr, _) = d_dens.device_ptr(&self.stream);
            let (ew_ptr, _) = d_eq_within.device_ptr(&self.stream);
            let (eo_ptr, _) = d_eq_outer.device_ptr(&self.stream);
            let (c0n_ptr, _) = d_c0n.device_ptr_mut(&self.stream);
            let (c2n_ptr, _) = d_c2n.device_ptr_mut(&self.stream);
            let (c0d_ptr, _) = d_c0d.device_ptr_mut(&self.stream);
            let (c2d_ptr, _) = d_c2d.device_ptr_mut(&self.stream);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &n_ptr as *const _ as *mut _,
                &d_ptr as *const _ as *mut _,
                &ew_ptr as *const _ as *mut _,
                &eo_ptr as *const _ as *mut _,
                &c0n_ptr as *const _ as *mut _,
                &c2n_ptr as *const _ as *mut _,
                &c0d_ptr as *const _ as *mut _,
                &c2d_ptr as *const _ as *mut _,
                &n_total as *const _ as *mut _,
                &chunk_size as *const _ as *mut _,
                &n_chunks as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_gkr_quot_sc_il,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem_size,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("gkr quotient sc interleaved kernel failed");
            }
        }

        let c0_num = self.reduce_partials(&d_c0n, blocks);
        let c2_num = self.reduce_partials(&d_c2n, blocks);
        let c0_den = self.reduce_partials(&d_c0d, blocks);
        let c2_den = self.reduce_partials(&d_c2d, blocks);
        (c0_num, c2_num, c0_den, c2_den)
    }

    /// Logup fingerprint: compute denominators for a section of the logup argument.
    ///
    /// denom[i] = c - fingerprint(contrib, columns[*][i], alphas)
    /// where fingerprint = contrib + Σ col[k][i] * alpha[k] + index_i * alpha[n_cols]
    ///
    /// columns: col-major base field (n_cols * n_rows u32s)
    /// Returns device-resident ext field denominators (n_rows * 5 u32s).
    pub fn logup_fingerprint_device(
        &self,
        d_columns: &CudaSlice<u32>, // n_cols * n_rows base elements
        c_ext: &[u32; 5],           // random challenge
        contrib: &[u32; 5],         // domain sep contribution
        alphas: &[[u32; 5]],        // (n_cols + 1) alpha values
        n_rows: u32,
        n_cols: u32,
    ) -> CudaSlice<u32> {
        let d_c = self.stream.memcpy_stod(&c_ext[..]).unwrap();
        let d_contrib = self.stream.memcpy_stod(&contrib[..]).unwrap();
        let alphas_flat: Vec<u32> = alphas.iter().flat_map(|a| a.iter().copied()).collect();
        let d_alphas = self.stream.memcpy_stod(&alphas_flat).unwrap();
        self.logup_fingerprint_device_with_constants(
            d_columns, &d_c, &d_contrib, &d_alphas, n_rows, n_cols,
        )
    }

    pub fn logup_fingerprint_device_with_constants(
        &self,
        d_columns: &CudaSlice<u32>,
        d_c: &CudaSlice<u32>,
        d_contrib: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        n_rows: u32,
        n_cols: u32,
    ) -> CudaSlice<u32> {
        let d_denoms = self.logup_fingerprint_device_with_constants_async(
            d_columns, d_c, d_contrib, d_alphas, n_rows, n_cols,
        );
        self.stream.synchronize().unwrap();
        d_denoms
    }

    pub fn logup_fingerprint_device_with_constants_async(
        &self,
        d_columns: &CudaSlice<u32>,
        d_c: &CudaSlice<u32>,
        d_contrib: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        n_rows: u32,
        n_cols: u32,
    ) -> CudaSlice<u32> {
        debug_assert!(d_c.len() >= 5);
        debug_assert!(d_contrib.len() >= 5);
        debug_assert!(d_alphas.len() >= ((n_cols as usize) + 1) * 5);
        let mut d_denoms = self
            .stream
            .alloc_zeros::<u32>((n_rows as usize) * 5)
            .unwrap();
        self.logup_fingerprint_device_with_constants_into_async(
            d_columns,
            d_c,
            d_contrib,
            d_alphas,
            n_rows,
            n_cols,
            &mut d_denoms,
        );
        d_denoms
    }

    pub fn logup_fingerprint_device_with_constants_into_async(
        &self,
        d_columns: &CudaSlice<u32>,
        d_c: &CudaSlice<u32>,
        d_contrib: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        n_rows: u32,
        n_cols: u32,
        d_denoms: &mut CudaSlice<u32>,
    ) {
        debug_assert!(d_c.len() >= 5);
        debug_assert!(d_contrib.len() >= 5);
        debug_assert!(d_alphas.len() >= ((n_cols as usize) + 1) * 5);
        assert!(d_denoms.len() >= (n_rows as usize) * 5);

        let (cols_ptr, _g1) = d_columns.device_ptr(&self.stream);
        let (c_ptr, _g2) = d_c.device_ptr(&self.stream);
        let (co_ptr, _g3) = d_contrib.device_ptr(&self.stream);
        let (al_ptr, _g4) = d_alphas.device_ptr(&self.stream);
        let (dn_ptr, _g5) = d_denoms.device_ptr_mut(&self.stream);

        let threads = 256u32;
        let blocks = (n_rows + threads - 1) / threads;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &cols_ptr as *const _ as *mut _,
            &c_ptr as *const _ as *mut _,
            &co_ptr as *const _ as *mut _,
            &al_ptr as *const _ as *mut _,
            &dn_ptr as *const _ as *mut _,
            &n_rows as *const _ as *mut _,
            &n_cols as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_logup_fp,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("logup fingerprint kernel failed");
        }
    }

    pub fn logup_prepare_constants_device_async(
        &self,
        d_alpha_eq: &CudaSlice<u32>,
        alpha_indices: &[u32],
        alpha_negated: &[u32],
        contrib_indices: &[u32],
        contrib_coeffs: &[u32],
    ) -> LogupPreparedConstants {
        assert_eq!(alpha_indices.len(), alpha_negated.len());
        assert_eq!(contrib_indices.len(), contrib_coeffs.len());
        debug_assert_eq!(d_alpha_eq.len() % 5, 0);
        let inputs = self.upload_logup_constant_inputs(
            alpha_indices,
            alpha_negated,
            contrib_indices,
            contrib_coeffs,
        );
        let mut d_contrib = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_alphas = self
            .stream
            .alloc_zeros::<u32>((inputs.n_alphas as usize * 5).max(1))
            .unwrap();
        self.logup_prepare_constants_from_device_inputs_into_async(
            d_alpha_eq,
            &inputs,
            &mut d_contrib,
            &mut d_alphas,
        );
        LogupPreparedConstants {
            d_contrib,
            d_alphas,
            _input_guards: Some(inputs),
        }
    }

    pub fn upload_logup_constant_inputs(
        &self,
        alpha_indices: &[u32],
        alpha_negated: &[u32],
        contrib_indices: &[u32],
        contrib_coeffs: &[u32],
    ) -> LogupConstantDeviceInputs {
        assert_eq!(alpha_indices.len(), alpha_negated.len());
        assert_eq!(contrib_indices.len(), contrib_coeffs.len());
        let d_alpha_indices = upload_or_zero(&self.stream, alpha_indices);
        let d_alpha_negated = upload_or_zero(&self.stream, alpha_negated);
        let d_contrib_indices = upload_or_zero(&self.stream, contrib_indices);
        let d_contrib_coeffs = upload_or_zero(&self.stream, contrib_coeffs);
        LogupConstantDeviceInputs {
            d_alpha_indices,
            d_alpha_negated,
            d_contrib_indices,
            d_contrib_coeffs,
            n_alphas: alpha_indices.len() as u32,
            n_contrib: contrib_indices.len() as u32,
        }
    }

    pub fn alloc_logup_prepared_constants(
        &self,
        inputs: &LogupConstantDeviceInputs,
    ) -> LogupPreparedConstants {
        let d_contrib = self.stream.alloc_zeros::<u32>(5).unwrap();
        let d_alphas = self
            .stream
            .alloc_zeros::<u32>((inputs.n_alphas as usize * 5).max(1))
            .unwrap();
        LogupPreparedConstants {
            d_contrib,
            d_alphas,
            _input_guards: None,
        }
    }

    pub fn logup_prepare_constants_from_device_inputs_async(
        &self,
        d_alpha_eq: &CudaSlice<u32>,
        inputs: &LogupConstantDeviceInputs,
    ) -> LogupPreparedConstants {
        debug_assert_eq!(d_alpha_eq.len() % 5, 0);
        let mut d_contrib = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_alphas = self
            .stream
            .alloc_zeros::<u32>((inputs.n_alphas as usize * 5).max(1))
            .unwrap();
        self.logup_prepare_constants_from_device_inputs_into_async(
            d_alpha_eq,
            inputs,
            &mut d_contrib,
            &mut d_alphas,
        );
        LogupPreparedConstants {
            d_contrib,
            d_alphas,
            _input_guards: None,
        }
    }

    pub fn logup_prepare_constants_from_device_inputs_into_prepared_async(
        &self,
        d_alpha_eq: &CudaSlice<u32>,
        inputs: &LogupConstantDeviceInputs,
        prepared: &mut LogupPreparedConstants,
    ) {
        self.logup_prepare_constants_from_device_inputs_into_async(
            d_alpha_eq,
            inputs,
            &mut prepared.d_contrib,
            &mut prepared.d_alphas,
        );
    }

    pub fn logup_prepare_constants_from_device_inputs_into_async(
        &self,
        d_alpha_eq: &CudaSlice<u32>,
        inputs: &LogupConstantDeviceInputs,
        d_contrib: &mut CudaSlice<u32>,
        d_alphas: &mut CudaSlice<u32>,
    ) {
        assert!(inputs.d_alpha_indices.len() >= inputs.n_alphas.max(1) as usize);
        assert!(inputs.d_alpha_negated.len() >= inputs.n_alphas.max(1) as usize);
        assert!(inputs.d_contrib_indices.len() >= inputs.n_contrib.max(1) as usize);
        assert!(inputs.d_contrib_coeffs.len() >= inputs.n_contrib.max(1) as usize);
        assert!(d_contrib.len() >= 5);
        assert!(d_alphas.len() >= (inputs.n_alphas as usize * 5).max(1));

        let (alpha_eq_ptr, _) = d_alpha_eq.device_ptr(&self.stream);
        let (alpha_indices_ptr, _) = inputs.d_alpha_indices.device_ptr(&self.stream);
        let (alpha_negated_ptr, _) = inputs.d_alpha_negated.device_ptr(&self.stream);
        let (contrib_indices_ptr, _) = inputs.d_contrib_indices.device_ptr(&self.stream);
        let (contrib_coeffs_ptr, _) = inputs.d_contrib_coeffs.device_ptr(&self.stream);
        let (contrib_ptr, _) = d_contrib.device_ptr_mut(&self.stream);
        let (alphas_ptr, _) = d_alphas.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &alpha_eq_ptr as *const _ as *mut _,
            &alpha_indices_ptr as *const _ as *mut _,
            &alpha_negated_ptr as *const _ as *mut _,
            &contrib_indices_ptr as *const _ as *mut _,
            &contrib_coeffs_ptr as *const _ as *mut _,
            &contrib_ptr as *const _ as *mut _,
            &alphas_ptr as *const _ as *mut _,
            &inputs.n_alphas as *const _ as *mut _,
            &inputs.n_contrib as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_logup_prepare_constants,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("logup prepare constants kernel failed");
        }
    }

    pub fn logup_prepare_constants_device(
        &self,
        d_alpha_eq: &CudaSlice<u32>,
        alpha_indices: &[u32],
        alpha_negated: &[u32],
        contrib_indices: &[u32],
        contrib_coeffs: &[u32],
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let prepared = self.logup_prepare_constants_device_async(
            d_alpha_eq,
            alpha_indices,
            alpha_negated,
            contrib_indices,
            contrib_coeffs,
        );
        self.stream.synchronize().unwrap();
        let LogupPreparedConstants {
            d_contrib,
            d_alphas,
            ..
        } = prepared;
        (d_contrib, d_alphas)
    }

    /// Reduce a multi-z partial sums buffer.
    /// Input: n_z * n_blocks * 5 u32s. Returns Vec of n_z ext field values.
    fn reduce_multi_z(
        &self,
        d_partials: &CudaSlice<u32>,
        n_z: usize,
        n_blocks: u32,
    ) -> Vec<[u32; 5]> {
        let mut results = Vec::with_capacity(n_z);
        for z in 0..n_z {
            let offset = (z as u32) * n_blocks * 5;
            // Create a view into the partial sums at the right offset.
            // We'll reduce each z-slice independently.
            let mut d_result = self.stream.alloc_zeros::<u32>(5).unwrap();
            {
                let (p_ptr_raw, _g1) = d_partials.device_ptr(&self.stream);
                let p_ptr = p_ptr_raw + (offset as u64) * 4; // byte offset
                let (r_ptr, _g2) = d_result.device_ptr_mut(&self.stream);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &p_ptr as *const _ as *mut _,
                    &r_ptr as *const _ as *mut _,
                    &n_blocks as *const _ as *mut _,
                ];
                let threads = 256u32.max(32);
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_reduce_ext,
                        (1, 1, 1),
                        (threads, 1, 1),
                        (threads / 32 * 5 * 4) as u32,
                        self.stream.cu_stream(),
                        &mut args,
                    )
                    .expect("reduce kernel failed");
                }
            }
            self.stream.synchronize().unwrap();
            let result = self.stream.memcpy_dtov(&d_result).unwrap();
            results.push(result[..5].try_into().unwrap());
        }
        results
    }

    /// Multi-z ExtensionOp AIR constraint evaluation (base field columns, 5 z-points).
    /// Evaluates at z=0,2,3,4,5 (degree 6). Returns 5 ext field sums.
    pub fn air_ext_op_multi_z_device(
        &self,
        d_columns: &CudaSlice<u32>,   // 29 * n_rows base
        d_down_cols: &CudaSlice<u32>, // 13 * n_rows base
        d_eq_factor: &CudaSlice<u32>, // n_pairs * 5 ext
        alphas: &[u32],               // 33 * 5 = 165 u32s
        n_rows: u32,
        n_pairs: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 5u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;

        let mut d_partials = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c_ptr, _) = d_columns.device_ptr(&self.stream);
            let (dc_ptr, _) = d_down_cols.device_ptr(&self.stream);
            let (eq_ptr, _) = d_eq_factor.device_ptr(&self.stream);
            let (al_ptr, _) = d_alphas.device_ptr(&self.stream);
            let (ps_ptr, _) = d_partials.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c_ptr as *const _ as *mut _,
                &dc_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &al_ptr as *const _ as *mut _,
                &ps_ptr as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_ext_op_mz,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("air ext_op multi-z kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_partials, n_z as usize, blocks)
    }

    /// Multi-z Poseidon16 AIR constraint evaluation (base field columns, 10 z-points).
    pub fn air_poseidon16_multi_z_device(
        &self,
        d_columns: &CudaSlice<u32>,   // 100 * n_rows base
        d_eq_factor: &CudaSlice<u32>, // n_pairs * 5 ext
        alphas: &[u32],               // 81 * 5 = 405 u32s
        d_rc: &CudaSlice<u32>,        // 448 base field round constants
        d_mds: &CudaSlice<u32>,       // 16 base field MDS column
        d_sparse: &CudaSlice<u32>,    // sparse matrix data
        n_rows: u32,
        n_pairs: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let threads = 128u32; // lower thread count for register-heavy kernel
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 10u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;

        let mut d_partials = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c_ptr, _) = d_columns.device_ptr(&self.stream);
            let (eq_ptr, _) = d_eq_factor.device_ptr(&self.stream);
            let (al_ptr, _) = d_alphas.device_ptr(&self.stream);
            let (rc_ptr, _) = d_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_sparse.device_ptr(&self.stream);
            let (ps_ptr, _) = d_partials.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &al_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &ps_ptr as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_poseidon16_mz,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("air poseidon16 multi-z kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_partials, n_z as usize, blocks)
    }

    /// Ext-field multi-z Execution (4 z-points). Columns are ext field.
    pub fn air_execution_multi_z_ext_device(
        &self,
        d_columns: &CudaSlice<u32>,   // 20 * n_elems * 5 ext
        d_down_cols: &CudaSlice<u32>, // 2 * n_elems * 5 ext
        d_eq_factor: &CudaSlice<u32>, // n_pairs * 5 ext
        alphas: &[u32],               // 13 * 5 u32s
        n_elems: u32,
        n_pairs: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let threads = 128u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 4u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;

        let mut d_partials = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c_ptr, _) = d_columns.device_ptr(&self.stream);
            let (dc_ptr, _) = d_down_cols.device_ptr(&self.stream);
            let (eq_ptr, _) = d_eq_factor.device_ptr(&self.stream);
            let (al_ptr, _) = d_alphas.device_ptr(&self.stream);
            let (ps_ptr, _) = d_partials.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c_ptr as *const _ as *mut _,
                &dc_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &al_ptr as *const _ as *mut _,
                &ps_ptr as *const _ as *mut _,
                &n_elems as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_exec_mz_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("air execution ext multi-z kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_partials, n_z as usize, blocks)
    }

    /// Ext-field multi-z ExtensionOp (5 z-points). Columns are ext field.
    pub fn air_ext_op_multi_z_ext_device(
        &self,
        d_columns: &CudaSlice<u32>,   // 29 * n_elems * 5 ext
        d_down_cols: &CudaSlice<u32>, // 13 * n_elems * 5 ext
        d_eq_factor: &CudaSlice<u32>, // n_pairs * 5 ext
        alphas: &[u32],               // 33 * 5 u32s
        n_elems: u32,
        n_pairs: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let threads = 128u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 5u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;

        let mut d_partials = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c_ptr, _) = d_columns.device_ptr(&self.stream);
            let (dc_ptr, _) = d_down_cols.device_ptr(&self.stream);
            let (eq_ptr, _) = d_eq_factor.device_ptr(&self.stream);
            let (al_ptr, _) = d_alphas.device_ptr(&self.stream);
            let (ps_ptr, _) = d_partials.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c_ptr as *const _ as *mut _,
                &dc_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &al_ptr as *const _ as *mut _,
                &ps_ptr as *const _ as *mut _,
                &n_elems as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_ext_op_mz_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("air ext_op ext multi-z kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_partials, n_z as usize, blocks)
    }

    /// Ext-field multi-z Poseidon16 (10 z-points). Columns are ext field.
    pub fn air_poseidon16_multi_z_ext_device(
        &self,
        d_columns: &CudaSlice<u32>,   // 100 * n_elems * 5 ext
        d_eq_factor: &CudaSlice<u32>, // n_pairs * 5 ext
        alphas: &[u32],               // 81 * 5 u32s
        d_rc: &CudaSlice<u32>,
        d_mds: &CudaSlice<u32>,
        d_sparse: &CudaSlice<u32>,
        n_elems: u32,
        n_pairs: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let threads = 64u32; // very register-heavy
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 10u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;

        let mut d_partials = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c_ptr, _) = d_columns.device_ptr(&self.stream);
            let (eq_ptr, _) = d_eq_factor.device_ptr(&self.stream);
            let (al_ptr, _) = d_alphas.device_ptr(&self.stream);
            let (rc_ptr, _) = d_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_sparse.device_ptr(&self.stream);
            let (ps_ptr, _) = d_partials.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c_ptr as *const _ as *mut _,
                &eq_ptr as *const _ as *mut _,
                &al_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &ps_ptr as *const _ as *mut _,
                &n_elems as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_poseidon16_mz_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("air poseidon16 ext multi-z kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_partials, n_z as usize, blocks)
    }

    /// Batch fold all columns (base→ext) at half. Col-major layout.
    /// Input: n_cols * n_rows base elements. Output: n_cols * n_pairs * 5 ext elements.
    pub fn fold_multi_col_base_to_ext_device(
        &self,
        d_data: &CudaSlice<u32>,
        n_rows: u32,
        n_pairs: u32,
        n_cols: u32,
        r_ext: &[u32; 5],
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_cols as usize * n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _) = d_data.device_ptr(&self.stream);
            let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _) = d_r.device_ptr(&self.stream);
            let total = n_cols * n_pairs;
            let threads = 256u32;
            let blocks = (total + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &n_cols as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_fold_multi_b2e,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("batch fold b2e kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Async batch fold: base→ext, N columns at device-resident challenge.
    /// Input layout: col-major, n_cols * n_rows base elements.
    /// Output layout: col-major, n_cols * n_pairs * 5 ext elements.
    pub fn fold_multi_col_b2e_device_into_async<R>(
        &self,
        d_data: &CudaSlice<u32>,
        d_out: &mut CudaSlice<u32>,
        n_rows: u32,
        n_pairs: u32,
        n_cols: u32,
        d_r: &R,
    ) where
        R: DevicePtr<u32>,
    {
        let total = n_cols * n_pairs;
        let threads = 256u32;
        let blocks = (total + threads - 1) / threads;
        let (data_ptr, _) = d_data.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let (r_ptr, _) = d_r.device_ptr(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &data_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &r_ptr as *const _ as *mut _,
            &n_rows as *const _ as *mut _,
            &n_pairs as *const _ as *mut _,
            &n_cols as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_fold_multi_b2e,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("async batch fold b2e kernel failed");
        }
    }

    /// Async batch fold: ext→ext, N columns at device-resident challenge.
    /// Input layout: col-major, n_cols * n_elems * 5 ext elements.
    /// Output layout: col-major, n_cols * n_pairs * 5 ext elements.
    pub fn fold_multi_col_ext_device_into_async<R>(
        &self,
        d_data: &CudaSlice<u32>,
        d_out: &mut CudaSlice<u32>,
        n_elems: u32,
        n_pairs: u32,
        n_cols: u32,
        d_r: &R,
    ) where
        R: DevicePtr<u32>,
    {
        let total = n_cols * n_pairs;
        let threads = 256u32;
        let blocks = (total + threads - 1) / threads;
        let (data_ptr, _) = d_data.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let (r_ptr, _) = d_r.device_ptr(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &data_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &r_ptr as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &n_pairs as *const _ as *mut _,
            &n_cols as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_fold_multi_ext,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("async batch fold ext kernel failed");
        }
    }

    /// Batch fold all columns (ext→ext) at half. Col-major layout.
    /// Input: n_cols * n_elems * 5 ext elements. Output: n_cols * n_pairs * 5 ext elements.
    pub fn fold_multi_col_ext_device(
        &self,
        d_data: &CudaSlice<u32>,
        n_elems: u32,
        n_pairs: u32,
        n_cols: u32,
        r_ext: &[u32; 5],
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_cols as usize * n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _) = d_data.device_ptr(&self.stream);
            let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _) = d_r.device_ptr(&self.stream);
            let total = n_cols * n_pairs;
            let threads = 256u32;
            let blocks = (total + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_elems as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &n_cols as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_fold_multi_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("batch fold ext kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// First half-fold of one base polynomial into many point columns.
    /// Input: one base polynomial of `n_elems` elements.
    /// Output: `n_points` columns, each with `n_pairs` ext elements.
    pub fn fold_single_base_to_many_points_device(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
    ) -> CudaSlice<u32> {
        let d_out = self.fold_single_base_to_many_points_device_async(
            d_data,
            d_points,
            point_stride_words,
            coord_idx,
            n_elems,
            n_pairs,
            n_points,
        );
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn fold_single_base_to_many_points_device_async(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
    ) -> CudaSlice<u32> {
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_points as usize * n_pairs as usize) * 5)
            .unwrap();
        self.fold_single_base_to_many_points_device_into_async(
            d_data,
            d_points,
            point_stride_words,
            coord_idx,
            n_elems,
            n_pairs,
            n_points,
            &mut d_out,
        );
        d_out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold_single_base_to_many_points_device_into_async<D>(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
        d_out: &mut D,
    ) where
        D: DevicePtrMut<u32>,
    {
        assert!(d_out.len() >= (n_points as usize * n_pairs as usize) * 5);
        let total = n_points * n_pairs;
        if total == 0 {
            return;
        }
        let threads = 256u32;
        let blocks = (total + threads - 1) / threads;
        let (data_ptr, _) = d_data.device_ptr(&self.stream);
        let (points_ptr, _) = d_points.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &data_ptr as *const _ as *mut _,
            &points_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &point_stride_words as *const _ as *mut _,
            &coord_idx as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &n_pairs as *const _ as *mut _,
            &n_points as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_fold_single_b2e_many_points,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("fold single base to many points kernel failed");
        }
    }

    /// First half-fold of one ext polynomial into many point columns.
    /// Input: one ext polynomial of `n_elems` elements.
    /// Output: `n_points` columns, each with `n_pairs` ext elements.
    pub fn fold_single_ext_to_many_points_device(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
    ) -> CudaSlice<u32> {
        let d_out = self.fold_single_ext_to_many_points_device_async(
            d_data,
            d_points,
            point_stride_words,
            coord_idx,
            n_elems,
            n_pairs,
            n_points,
        );
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn fold_single_ext_to_many_points_device_async(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
    ) -> CudaSlice<u32> {
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_points as usize * n_pairs as usize) * 5)
            .unwrap();
        self.fold_single_ext_to_many_points_device_into_async(
            d_data,
            d_points,
            point_stride_words,
            coord_idx,
            n_elems,
            n_pairs,
            n_points,
            &mut d_out,
        );
        d_out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold_single_ext_to_many_points_device_into_async<D>(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
        d_out: &mut D,
    ) where
        D: DevicePtrMut<u32>,
    {
        assert!(d_out.len() >= (n_points as usize * n_pairs as usize) * 5);
        let total = n_points * n_pairs;
        if total == 0 {
            return;
        }
        let threads = 256u32;
        let blocks = (total + threads - 1) / threads;
        let (data_ptr, _) = d_data.device_ptr(&self.stream);
        let (points_ptr, _) = d_points.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &data_ptr as *const _ as *mut _,
            &points_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &point_stride_words as *const _ as *mut _,
            &coord_idx as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &n_pairs as *const _ as *mut _,
            &n_points as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_fold_single_ext_many_points,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("fold single ext to many points kernel failed");
        }
    }

    /// Subsequent half-folds of many ext point columns with per-point challenges.
    pub fn fold_multi_col_ext_per_point_device(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
    ) -> CudaSlice<u32> {
        let d_out = self.fold_multi_col_ext_per_point_device_async(
            d_data,
            d_points,
            point_stride_words,
            coord_idx,
            n_elems,
            n_pairs,
            n_points,
        );
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn fold_multi_col_ext_per_point_device_async(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
    ) -> CudaSlice<u32> {
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_points as usize * n_pairs as usize) * 5)
            .unwrap();
        self.fold_multi_col_ext_per_point_device_into_async(
            d_data,
            d_points,
            point_stride_words,
            coord_idx,
            n_elems,
            n_pairs,
            n_points,
            &mut d_out,
        );
        d_out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold_multi_col_ext_per_point_device_into_async<D>(
        &self,
        d_data: &CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        point_stride_words: u32,
        coord_idx: u32,
        n_elems: u32,
        n_pairs: u32,
        n_points: u32,
        d_out: &mut D,
    ) where
        D: DevicePtrMut<u32>,
    {
        assert!(d_out.len() >= (n_points as usize * n_pairs as usize) * 5);
        let total = n_points * n_pairs;
        if total == 0 {
            return;
        }
        let threads = 256u32;
        let blocks = (total + threads - 1) / threads;
        let (data_ptr, _) = d_data.device_ptr(&self.stream);
        let (points_ptr, _) = d_points.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &data_ptr as *const _ as *mut _,
            &points_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &point_stride_words as *const _ as *mut _,
            &coord_idx as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &n_pairs as *const _ as *mut _,
            &n_points as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_fold_multi_ext_per_point,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("fold multi ext per-point kernel failed");
        }
    }

    pub fn repeat_ext_value_device_async<S>(&self, d_value: &S, n_values: u32) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        assert!(d_value.len() >= 5);
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_values as usize) * 5)
            .unwrap();
        self.repeat_ext_value_device_into_async(d_value, n_values, &mut d_out);
        d_out
    }

    pub fn repeat_ext_value_device_into_async<S, D>(
        &self,
        d_value: &S,
        n_values: u32,
        d_out: &mut D,
    ) where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(d_value.len() >= 5);
        assert!(d_out.len() >= (n_values as usize) * 5);
        if n_values == 0 {
            return;
        }
        let (value_ptr, _) = d_value.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let threads = 256u32;
        let blocks = (n_values + threads - 1) / threads;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &value_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n_values as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_repeat_ext_value,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("repeat ext value kernel failed");
        }
    }

    pub fn repeat_base_value_as_ext_device_async<S>(
        &self,
        d_value: &S,
        n_values: u32,
    ) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        assert!(d_value.len() >= 1);
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_values as usize) * 5)
            .unwrap();
        self.repeat_base_value_as_ext_device_into_async(d_value, n_values, &mut d_out);
        d_out
    }

    pub fn repeat_base_value_as_ext_device_into_async<S, D>(
        &self,
        d_value: &S,
        n_values: u32,
        d_out: &mut D,
    ) where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(d_value.len() >= 1);
        assert!(d_out.len() >= (n_values as usize) * 5);
        if n_values == 0 {
            return;
        }
        let (value_ptr, _) = d_value.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let threads = 256u32;
        let blocks = (n_values + threads - 1) / threads;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &value_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n_values as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_repeat_base_value_as_ext,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("repeat base value as ext kernel failed");
        }
    }

    /// Bit-reverse elements within chunks of 2^chunk_log. Device-resident.
    pub fn bit_reverse_within_chunks_device<S>(
        &self,
        d_data: &S,
        n: u32,
        chunk_log: u32,
    ) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        let d_out = self.bit_reverse_within_chunks_device_async(d_data, n, chunk_log);
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn bit_reverse_within_chunks_device_async<S>(
        &self,
        d_data: &S,
        n: u32,
        chunk_log: u32,
    ) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        let mut d_out = self.stream.alloc_zeros::<u32>(n as usize).unwrap();
        self.bit_reverse_within_chunks_device_into_async(d_data, &mut d_out, n, chunk_log);
        d_out
    }

    pub fn bit_reverse_within_chunks_device_into_async<S, D>(
        &self,
        d_data: &S,
        d_out: &mut D,
        n: u32,
        chunk_log: u32,
    ) where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(d_data.len() >= n as usize);
        assert!(d_out.len() >= n as usize);
        let (data_ptr, _) = d_data.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &data_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
            &chunk_log as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_bit_reverse,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("bit_reverse kernel failed");
        }
    }

    /// Batch fold at arbitrary bit (base→ext). Col-major layout.
    pub fn fold_multi_col_b2e_at_bit_device(
        &self,
        d_data: &CudaSlice<u32>,
        n_rows: u32,
        n_pairs: u32,
        n_cols: u32,
        r_ext: &[u32; 5],
        fold_bit: u32,
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_cols as usize * n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _) = d_data.device_ptr(&self.stream);
            let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _) = d_r.device_ptr(&self.stream);
            let total = n_cols * n_pairs;
            let threads = 256u32;
            let blocks = (total + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &n_cols as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_fold_multi_b2e_fb,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("batch fold b2e at bit kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Batch fold at arbitrary bit (ext→ext). Col-major layout.
    pub fn fold_multi_col_ext_at_bit_device(
        &self,
        d_data: &CudaSlice<u32>,
        n_elems: u32,
        n_pairs: u32,
        n_cols: u32,
        r_ext: &[u32; 5],
        fold_bit: u32,
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_cols as usize * n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _) = d_data.device_ptr(&self.stream);
            let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _) = d_r.device_ptr(&self.stream);
            let total = n_cols * n_pairs;
            let threads = 256u32;
            let blocks = (total + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_elems as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &n_cols as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_fold_multi_ext_fb,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("batch fold ext at bit kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Multi-z Execution with fold_bit (base field, 4 z-points).
    /// Multi-z Execution with fold_bit (base field, 5 z-points) + bus constraint.
    pub fn air_execution_multi_z_fb_device(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        alphas: &[u32],
        logup_alphas: &[u32], // 5 * 5 = 25 u32s (logup alpha eq poly, first 5 ext values)
        bus_beta: &[u32; 5],  // ext field
        n_rows: u32,
        n_pairs: u32,
        fold_bit: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let d_la = self.stream.memcpy_stod(logup_alphas).unwrap();
        let d_bb = self.stream.memcpy_stod(bus_beta.as_slice()).unwrap();
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 5u32; // degree_air=5 → 5 z-points: z=0,2,3,4,5
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;
        let mut d_ps = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c, _) = d_columns.device_ptr(&self.stream);
            let (dc, _) = d_down_cols.device_ptr(&self.stream);
            let (eq, _) = d_eq.device_ptr(&self.stream);
            let (al, _) = d_alphas.device_ptr(&self.stream);
            let (la, _) = d_la.device_ptr(&self.stream);
            let (bb, _) = d_bb.device_ptr(&self.stream);
            let (ps, _) = d_ps.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c as *const _ as *mut _,
                &dc as *const _ as *mut _,
                &eq as *const _ as *mut _,
                &al as *const _ as *mut _,
                &la as *const _ as *mut _,
                &bb as *const _ as *mut _,
                &ps as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_exec_mz_fb,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("exec mz fb failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_ps, n_z as usize, blocks)
    }

    /// Multi-z ExtensionOp with fold_bit (base field, 6 z-points).
    pub fn air_ext_op_multi_z_fb_device(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        alphas: &[u32],
        logup_alphas: &[u32],
        bus_beta: &[u32; 5],
        n_rows: u32,
        n_pairs: u32,
        fold_bit: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let d_la = self.stream.memcpy_stod(logup_alphas).unwrap();
        let d_bb = self.stream.memcpy_stod(bus_beta.as_slice()).unwrap();
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 6u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;
        let mut d_ps = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c, _) = d_columns.device_ptr(&self.stream);
            let (dc, _) = d_down_cols.device_ptr(&self.stream);
            let (eq, _) = d_eq.device_ptr(&self.stream);
            let (al, _) = d_alphas.device_ptr(&self.stream);
            let (la, _) = d_la.device_ptr(&self.stream);
            let (bb, _) = d_bb.device_ptr(&self.stream);
            let (ps, _) = d_ps.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c as *const _ as *mut _,
                &dc as *const _ as *mut _,
                &eq as *const _ as *mut _,
                &al as *const _ as *mut _,
                &la as *const _ as *mut _,
                &bb as *const _ as *mut _,
                &ps as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_ext_op_mz_fb,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("ext_op mz fb failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_ps, n_z as usize, blocks)
    }

    /// Multi-z Poseidon16 with fold_bit (base field, 10 z-points).
    pub fn air_poseidon16_multi_z_fb_device(
        &self,
        d_columns: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        alphas: &[u32],
        d_rc: &CudaSlice<u32>,
        d_mds: &CudaSlice<u32>,
        d_sparse: &CudaSlice<u32>,
        logup_alphas: &[u32],
        bus_beta: &[u32; 5],
        n_rows: u32,
        n_pairs: u32,
        fold_bit: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let d_la = self.stream.memcpy_stod(logup_alphas).unwrap();
        let d_bb = self.stream.memcpy_stod(bus_beta.as_slice()).unwrap();
        let threads = 128u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 10u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;
        let mut d_ps = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c, _) = d_columns.device_ptr(&self.stream);
            let (eq, _) = d_eq.device_ptr(&self.stream);
            let (al, _) = d_alphas.device_ptr(&self.stream);
            let (rc, _) = d_rc.device_ptr(&self.stream);
            let (md, _) = d_mds.device_ptr(&self.stream);
            let (sp, _) = d_sparse.device_ptr(&self.stream);
            let (la, _) = d_la.device_ptr(&self.stream);
            let (bb, _) = d_bb.device_ptr(&self.stream);
            let (ps, _) = d_ps.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c as *const _ as *mut _,
                &eq as *const _ as *mut _,
                &al as *const _ as *mut _,
                &rc as *const _ as *mut _,
                &md as *const _ as *mut _,
                &sp as *const _ as *mut _,
                &la as *const _ as *mut _,
                &bb as *const _ as *mut _,
                &ps as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_pos16_mz_fb,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("pos16 mz fb failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_ps, n_z as usize, blocks)
    }

    /// Ext-field multi-z Execution with fold_bit (4 z-points).
    pub fn air_execution_multi_z_ext_fb_device(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        alphas: &[u32],
        logup_alphas: &[u32],
        bus_beta: &[u32; 5],
        n_elems: u32,
        n_pairs: u32,
        fold_bit: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let d_la = self.stream.memcpy_stod(logup_alphas).unwrap();
        let d_bb = self.stream.memcpy_stod(bus_beta.as_slice()).unwrap();
        let threads = 128u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 5u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;
        let mut d_ps = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c, _) = d_columns.device_ptr(&self.stream);
            let (dc, _) = d_down_cols.device_ptr(&self.stream);
            let (eq, _) = d_eq.device_ptr(&self.stream);
            let (al, _) = d_alphas.device_ptr(&self.stream);
            let (la, _) = d_la.device_ptr(&self.stream);
            let (bb, _) = d_bb.device_ptr(&self.stream);
            let (ps, _) = d_ps.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c as *const _ as *mut _,
                &dc as *const _ as *mut _,
                &eq as *const _ as *mut _,
                &al as *const _ as *mut _,
                &la as *const _ as *mut _,
                &bb as *const _ as *mut _,
                &ps as *const _ as *mut _,
                &n_elems as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_exec_mz_ext_fb,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("exec ext mz fb failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_ps, n_z as usize, blocks)
    }

    /// Ext-field multi-z ExtensionOp with fold_bit (6 z-points).
    pub fn air_ext_op_multi_z_ext_fb_device(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        alphas: &[u32],
        logup_alphas: &[u32],
        bus_beta: &[u32; 5],
        n_elems: u32,
        n_pairs: u32,
        fold_bit: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let d_la = self.stream.memcpy_stod(logup_alphas).unwrap();
        let d_bb = self.stream.memcpy_stod(bus_beta.as_slice()).unwrap();
        let threads = 128u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 6u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;
        let mut d_ps = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c, _) = d_columns.device_ptr(&self.stream);
            let (dc, _) = d_down_cols.device_ptr(&self.stream);
            let (eq, _) = d_eq.device_ptr(&self.stream);
            let (al, _) = d_alphas.device_ptr(&self.stream);
            let (la, _) = d_la.device_ptr(&self.stream);
            let (bb, _) = d_bb.device_ptr(&self.stream);
            let (ps, _) = d_ps.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c as *const _ as *mut _,
                &dc as *const _ as *mut _,
                &eq as *const _ as *mut _,
                &al as *const _ as *mut _,
                &la as *const _ as *mut _,
                &bb as *const _ as *mut _,
                &ps as *const _ as *mut _,
                &n_elems as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_ext_op_mz_ext_fb,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("ext_op ext mz fb failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_ps, n_z as usize, blocks)
    }

    /// Ext-field multi-z Poseidon16 with fold_bit (10 z-points).
    pub fn air_poseidon16_multi_z_ext_fb_device(
        &self,
        d_columns: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        alphas: &[u32],
        d_rc: &CudaSlice<u32>,
        d_mds: &CudaSlice<u32>,
        d_sparse: &CudaSlice<u32>,
        logup_alphas: &[u32],
        bus_beta: &[u32; 5],
        n_elems: u32,
        n_pairs: u32,
        fold_bit: u32,
    ) -> Vec<[u32; 5]> {
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let d_la = self.stream.memcpy_stod(logup_alphas).unwrap();
        let d_bb = self.stream.memcpy_stod(bus_beta.as_slice()).unwrap();
        let threads = 64u32;
        let blocks = (n_pairs + threads - 1) / threads;
        let n_z = 10u32;
        let smem = (n_z * (threads / 32) * 5 * 4) as u32;
        let mut d_ps = self
            .stream
            .alloc_zeros::<u32>((n_z as usize * blocks as usize) * 5)
            .unwrap();
        {
            let (c, _) = d_columns.device_ptr(&self.stream);
            let (eq, _) = d_eq.device_ptr(&self.stream);
            let (al, _) = d_alphas.device_ptr(&self.stream);
            let (rc, _) = d_rc.device_ptr(&self.stream);
            let (md, _) = d_mds.device_ptr(&self.stream);
            let (sp, _) = d_sparse.device_ptr(&self.stream);
            let (la, _) = d_la.device_ptr(&self.stream);
            let (bb, _) = d_bb.device_ptr(&self.stream);
            let (ps, _) = d_ps.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c as *const _ as *mut _,
                &eq as *const _ as *mut _,
                &al as *const _ as *mut _,
                &rc as *const _ as *mut _,
                &md as *const _ as *mut _,
                &sp as *const _ as *mut _,
                &la as *const _ as *mut _,
                &bb as *const _ as *mut _,
                &ps as *const _ as *mut _,
                &n_elems as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &fold_bit as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_pos16_mz_ext_fb,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    smem,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("pos16 ext mz fb failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.reduce_multi_z(&d_ps, n_z as usize, blocks)
    }

    /// Debug: evaluate Poseidon16 ext-field constraints on given values.
    /// Returns [nonbus_w(5), total_w(5), bus_data(25)] = 35 u32s.
    pub fn debug_poseidon16_ext_eval(
        &self,
        up_vals: &[u32],
        alphas: &[u32],
        d_rc: &CudaSlice<u32>,
        d_mds: &CudaSlice<u32>,
        d_sparse: &CudaSlice<u32>,
        logup_alphas: &[u32],
        bus_beta: &[u32; 5],
    ) -> Vec<u32> {
        let d_up = self.stream.memcpy_stod(up_vals).unwrap();
        let d_alpha = self.stream.memcpy_stod(alphas).unwrap();
        let d_la = self.stream.memcpy_stod(logup_alphas).unwrap();
        let d_bb = self.stream.memcpy_stod(bus_beta.as_slice()).unwrap();
        let mut d_result = self.stream.alloc_zeros::<u32>(35).unwrap();
        {
            let (up_ptr, _) = d_up.device_ptr(&self.stream);
            let (al_ptr, _) = d_alpha.device_ptr(&self.stream);
            let (rc_ptr, _) = d_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_sparse.device_ptr(&self.stream);
            let (la_ptr, _) = d_la.device_ptr(&self.stream);
            let (bb_ptr, _) = d_bb.device_ptr(&self.stream);
            let (res_ptr, _) = d_result.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &up_ptr as *const _ as *mut _,
                &al_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &la_ptr as *const _ as *mut _,
                &bb_ptr as *const _ as *mut _,
                &res_ptr as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_debug_pos16_ext,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("debug pos16 ext eval kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_result).unwrap()
    }

    /// Run the protocol step kernel (single GPU thread):
    /// pe construction, Lagrange interpolation, expand_bare_to_full,
    /// cc accumulation, Fiat-Shamir observe+sample, state update.
    /// Returns (cc, challenge, transcript_chunk).
    #[allow(clippy::too_many_arguments)]
    pub fn protocol_step(
        &self,
        d_ze_all: &CudaSlice<u32>,
        d_ze_offsets: &CudaSlice<u32>,
        d_table_state: &mut CudaSlice<u32>,
        n_tables: u32,
        mfd: u32,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
    ) -> (Vec<u32>, [u32; 5], Vec<u32>) {
        let cc_size = (mfd as usize + 1) * 5;
        let mut d_cc = self.stream.alloc_zeros::<u32>(cc_size).unwrap();
        let mut d_bare_scratch = self
            .stream
            .alloc_zeros::<u32>((n_tables as usize * mfd as usize * 5).max(1))
            .unwrap();
        let mut d_challenge = self.stream.alloc_zeros::<u32>(5).unwrap();
        let max_transcript = cc_size; // upper bound
        let mut d_transcript = self.stream.alloc_zeros::<u32>(max_transcript).unwrap();
        let mut d_tlen = self.stream.alloc_zeros::<u32>(1).unwrap();
        self.build_air_bare_coeffs_async(
            d_ze_all,
            d_ze_offsets,
            d_table_state,
            n_tables,
            mfd,
            &mut d_bare_scratch,
        );
        {
            let (ze_ptr, _) = d_ze_all.device_ptr(&self.stream);
            let (zo_ptr, _) = d_ze_offsets.device_ptr(&self.stream);
            let (ts_ptr, _) = d_table_state.device_ptr_mut(&self.stream);
            let (cc_ptr, _) = d_cc.device_ptr_mut(&self.stream);
            let (bs_ptr, _) = d_bare_scratch.device_ptr_mut(&self.stream);
            let (ch_ptr, _) = d_challenge.device_ptr_mut(&self.stream);
            let (tr_ptr, _) = d_transcript.device_ptr_mut(&self.stream);
            let (tl_ptr, _) = d_tlen.device_ptr_mut(&self.stream);
            let (cs_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &ze_ptr as *const _ as *mut _,
                &zo_ptr as *const _ as *mut _,
                &ts_ptr as *const _ as *mut _,
                &n_tables as *const _ as *mut _,
                &mfd as *const _ as *mut _,
                &cc_ptr as *const _ as *mut _,
                &bs_ptr as *const _ as *mut _,
                &ch_ptr as *const _ as *mut _,
                &tr_ptr as *const _ as *mut _,
                &tl_ptr as *const _ as *mut _,
                &cs_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_protocol_step,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("protocol step kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        let cc = self.stream.memcpy_dtov(&d_cc).unwrap();
        let ch_v = self.stream.memcpy_dtov(&d_challenge).unwrap();
        let challenge: [u32; 5] = ch_v[..5].try_into().unwrap();
        let tlen_v = self.stream.memcpy_dtov(&d_tlen).unwrap();
        let tlen = tlen_v[0] as usize;
        let transcript = self.stream.memcpy_dtov(&d_transcript).unwrap()[..tlen].to_vec();
        (cc, challenge, transcript)
    }

    /// Async device-to-device copy (safe during graph capture).
    pub fn memcpy_d2d_async<S>(
        &self,
        src: &S,
        src_offset_words: usize,
        dst: &mut CudaSlice<u32>,
        dst_offset_words: usize,
        n_words: usize,
    ) where
        S: DevicePtr<u32>,
    {
        let (src_ptr, _) = src.device_ptr(&self.stream);
        let (dst_ptr, _) = dst.device_ptr_mut(&self.stream);
        let src_off = src_ptr + (src_offset_words as u64) * 4;
        let dst_off = dst_ptr + (dst_offset_words as u64) * 4;
        unsafe {
            cuda_result::memcpy_dtod_async(dst_off, src_off, n_words * 4, self.stream.cu_stream())
                .expect("d2d memcpy failed");
        }
    }

    /// Debug: compute pe from ze+state like the protocol step kernel
    pub fn debug_compute_pe(
        &self,
        d_ze: &CudaSlice<u32>,
        d_zo: &CudaSlice<u32>,
        d_ts: &CudaSlice<u32>,
        table_idx: u32,
    ) -> (Vec<u32>, u32) {
        let mut d_pe = self.stream.alloc_zeros::<u32>(13 * 5).unwrap();
        let mut d_nz = self.stream.alloc_zeros::<u32>(1).unwrap();
        let (ze_ptr, _) = d_ze.device_ptr(&self.stream);
        let (zo_ptr, _) = d_zo.device_ptr(&self.stream);
        let (ts_ptr, _) = d_ts.device_ptr(&self.stream);
        let (pe_ptr, _) = d_pe.device_ptr_mut(&self.stream);
        let (nz_ptr, _) = d_nz.device_ptr_mut(&self.stream);
        let func = self.fn_debug_pe;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &ze_ptr as *const _ as *mut _,
            &zo_ptr as *const _ as *mut _,
            &ts_ptr as *const _ as *mut _,
            &table_idx as *const _ as *mut _,
            &pe_ptr as *const _ as *mut _,
            &nz_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                func,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("debug compute pe failed");
        }
        self.stream.synchronize().unwrap();
        let pe = self.stream.memcpy_dtov(&d_pe).unwrap();
        let nz = self.stream.memcpy_dtov(&d_nz).unwrap()[0];
        (pe, nz)
    }

    /// Standalone (non-graph) protocol step — synchronous version for testing.
    #[allow(clippy::too_many_arguments)]
    pub fn protocol_step_standalone(
        &self,
        d_ze: &CudaSlice<u32>,
        d_zo: &CudaSlice<u32>,
        d_ts: &mut CudaSlice<u32>,
        n_tables: u32,
        mfd: u32,
        d_cc: &mut CudaSlice<u32>,
        d_ch: &mut CudaSlice<u32>,
        ch_off: usize,
        d_tr: &mut CudaSlice<u32>,
        tr_off: usize,
        d_tl: &mut CudaSlice<u32>,
        d_cs: &mut CudaSlice<u32>,
        d_rc: &CudaSlice<u32>,
        d_mds: &CudaSlice<u32>,
        d_sp: &CudaSlice<u32>,
    ) {
        let mut d_bare_scratch = self
            .stream
            .alloc_zeros::<u32>((n_tables as usize * mfd as usize * 5).max(1))
            .unwrap();
        self.build_air_bare_coeffs_async(d_ze, d_zo, d_ts, n_tables, mfd, &mut d_bare_scratch);
        self.protocol_step_async(
            d_ze,
            d_zo,
            d_ts,
            n_tables,
            mfd,
            d_cc,
            &mut d_bare_scratch,
            d_ch,
            ch_off,
            d_tr,
            tr_off,
            d_tl,
            d_cs,
            d_rc,
            d_mds,
            d_sp,
        );
        self.stream.synchronize().unwrap();
    }

    /// Test GPU qe_inv
    pub fn test_qe_inv(&self, a: &[u32; 5]) -> [u32; 5] {
        let d_a = self.stream.memcpy_stod(a.as_slice()).unwrap();
        let mut d_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        let (a_ptr, _) = d_a.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> =
            vec![&a_ptr as *const _ as *mut _, &out_ptr as *const _ as *mut _];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_test_qe_inv,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("qe_inv test failed");
        }
        self.stream.synchronize().unwrap();
        let out = self.stream.memcpy_dtov(&d_out).unwrap();
        out[..5].try_into().unwrap()
    }

    /// Test GPU expand_bare_to_full
    pub fn test_expand_bare_to_full(
        &self,
        bare_words: &[u32],
        n: u32,
        alpha: &[u32; 5],
    ) -> Vec<u32> {
        let d_bare = self.stream.memcpy_stod(bare_words).unwrap();
        let d_alpha = self.stream.memcpy_stod(alpha.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n as usize + 1) * 5)
            .unwrap();
        let (bare_ptr, _) = d_bare.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let (alpha_ptr, _) = d_alpha.device_ptr(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &bare_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
            &alpha_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_test_expand,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("test expand kernel failed");
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_out).unwrap()
    }

    /// Test GPU Lagrange interpolation
    pub fn test_lagrange_interp(&self, d_pe: &CudaSlice<u32>, n: u32, d_out: &mut CudaSlice<u32>) {
        let (pe_ptr, _) = d_pe.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &pe_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_test_lagrange,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("test lagrange kernel failed");
        }
        self.stream.synchronize().unwrap();
    }

    /// Test GPU Fiat-Shamir: observe scalars, sample, compare with CPU.
    pub fn test_fiat_shamir(
        &self,
        d_rc: &CudaSlice<u32>,
        d_mds: &CudaSlice<u32>,
        d_sparse: &CudaSlice<u32>,
        input_scalars: &[u32],
    ) -> ([u32; 8], [u32; 5]) {
        let d_input = self.stream.memcpy_stod(input_scalars).unwrap();
        let mut d_state = self.stream.alloc_zeros::<u32>(8).unwrap();
        let mut d_sample = self.stream.alloc_zeros::<u32>(5).unwrap();
        let n = input_scalars.len() as u32;
        {
            let (inp_ptr, _) = d_input.device_ptr(&self.stream);
            let (rc_ptr, _) = d_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_sparse.device_ptr(&self.stream);
            let (st_ptr, _) = d_state.device_ptr_mut(&self.stream);
            let (sa_ptr, _) = d_sample.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &inp_ptr as *const _ as *mut _,
                &n as *const _ as *mut _,
                &st_ptr as *const _ as *mut _,
                &sa_ptr as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_test_fiat_shamir,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("test fiat shamir kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        let st = self.stream.memcpy_dtov(&d_state).unwrap();
        let sa = self.stream.memcpy_dtov(&d_sample).unwrap();
        (st[..8].try_into().unwrap(), sa[..5].try_into().unwrap())
    }

    pub fn challenger_observe_and_sample_exts(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        observe_scalars: &[u32],
        n_sample_exts: u32,
    ) -> Vec<[u32; 5]> {
        let d_observe = if observe_scalars.is_empty() {
            self.stream.alloc_zeros::<u32>(1).unwrap()
        } else {
            self.stream.memcpy_stod(observe_scalars).unwrap()
        };
        let n_observe_scalars = observe_scalars.len() as u32;
        let mut d_samples = self
            .stream
            .alloc_zeros::<u32>((n_sample_exts as usize) * 5)
            .unwrap();
        {
            let (st_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let (ob_ptr, _) = d_observe.device_ptr(&self.stream);
            let (sa_ptr, _) = d_samples.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &st_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &ob_ptr as *const _ as *mut _,
                &n_observe_scalars as *const _ as *mut _,
                &sa_ptr as *const _ as *mut _,
                &n_sample_exts as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_challenger_observe_sample,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("challenger observe+sample kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        let sample_words = self.stream.memcpy_dtov(&d_samples).unwrap();
        sample_words
            .chunks_exact(5)
            .map(|chunk| chunk.try_into().unwrap())
            .collect()
    }

    pub fn challenger_observe_scalars(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        observe_scalars: &[u32],
    ) {
        if observe_scalars.is_empty() {
            return;
        }
        let d_observe = self.stream.memcpy_stod(observe_scalars).unwrap();
        let n_observe_scalars = observe_scalars.len() as u32;
        {
            let (st_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let (ob_ptr, _) = d_observe.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &st_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &ob_ptr as *const _ as *mut _,
                &n_observe_scalars as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_challenger_observe,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("challenger observe kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
    }

    pub fn challenger_observe_device_scalars<O>(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        d_observe: &O,
        n_observe_scalars: u32,
    ) where
        O: DevicePtr<u32>,
    {
        self.challenger_observe_device_scalars_async(
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            d_observe,
            n_observe_scalars,
        );
        self.stream.synchronize().unwrap();
    }

    pub fn challenger_observe_device_scalars_async<O>(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        d_observe: &O,
        n_observe_scalars: u32,
    ) where
        O: DevicePtr<u32>,
    {
        if n_observe_scalars == 0 {
            return;
        }
        {
            let (st_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let (ob_ptr, _) = d_observe.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &st_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &ob_ptr as *const _ as *mut _,
                &n_observe_scalars as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_challenger_observe,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("challenger observe kernel failed");
            }
        }
    }

    pub fn challenger_sample_exts_device(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        n_sample_exts: u32,
    ) -> CudaSlice<u32> {
        let d_observe = self.stream.alloc_zeros::<u32>(1).unwrap();
        let n_observe_scalars = 0u32;
        let mut d_samples = self
            .stream
            .alloc_zeros::<u32>((n_sample_exts as usize) * 5)
            .unwrap();
        self.challenger_sample_exts_device_into_async(
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            n_sample_exts,
            &d_observe,
            &mut d_samples,
        );
        self.stream.synchronize().unwrap();
        d_samples
    }

    pub fn challenger_sample_exts_device_into_async(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        n_sample_exts: u32,
        d_observe: &CudaSlice<u32>,
        d_samples: &mut CudaSlice<u32>,
    ) {
        let n_observe_scalars = 0u32;
        {
            let (st_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let (ob_ptr, _) = d_observe.device_ptr(&self.stream);
            let (sa_ptr, _) = d_samples.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &st_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &ob_ptr as *const _ as *mut _,
                &n_observe_scalars as *const _ as *mut _,
                &sa_ptr as *const _ as *mut _,
                &n_sample_exts as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_challenger_observe_sample,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("challenger sample-ext kernel failed");
            }
        }
    }

    pub fn challenger_sample_base_scalars(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        n_base_samples: u32,
    ) -> Vec<u32> {
        if n_base_samples == 0 {
            return Vec::new();
        }
        let d_samples = self.challenger_sample_base_scalars_device(
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            n_base_samples,
        );
        self.stream.memcpy_dtov(&d_samples).unwrap()
    }

    pub fn challenger_sample_base_scalars_device(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        n_base_samples: u32,
    ) -> CudaSlice<u32> {
        let d_samples = self.challenger_sample_base_scalars_device_async(
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            n_base_samples,
        );
        self.stream.synchronize().unwrap();
        d_samples
    }

    pub fn challenger_sample_base_scalars_device_async(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        n_base_samples: u32,
    ) -> CudaSlice<u32> {
        if n_base_samples == 0 {
            return self.stream.alloc_zeros::<u32>(0).unwrap();
        }
        let mut d_samples = self
            .stream
            .alloc_zeros::<u32>(n_base_samples as usize)
            .unwrap();
        self.challenger_sample_base_scalars_device_into_async(
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            n_base_samples,
            &mut d_samples,
        );
        d_samples
    }

    pub fn challenger_sample_base_scalars_device_into_async(
        &self,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        n_base_samples: u32,
        d_samples: &mut CudaSlice<u32>,
    ) {
        assert!(d_samples.len() >= n_base_samples as usize);
        if n_base_samples == 0 {
            return;
        }
        {
            let (st_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let (sa_ptr, _) = d_samples.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &st_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
                &sa_ptr as *const _ as *mut _,
                &n_base_samples as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_challenger_sample_base,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("challenger sample-base kernel failed");
            }
        }
    }

    pub fn product_sumcheck_observe_round_poly(
        &self,
        d_c0: &CudaSlice<u32>,
        d_c2: &CudaSlice<u32>,
        d_sum: &CudaSlice<u32>,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        let mut d_poly = self.stream.alloc_zeros::<u32>(15).unwrap();
        let mut d_transcript_tail = self.stream.alloc_zeros::<u32>(10).unwrap();
        self.product_sumcheck_observe_round_poly_async(
            d_c0,
            d_c2,
            d_sum,
            &mut d_poly,
            &mut d_transcript_tail,
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
        );
        self.stream.synchronize().unwrap();
        (d_poly, d_transcript_tail)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn product_sumcheck_observe_round_poly_async(
        &self,
        d_c0: &CudaSlice<u32>,
        d_c2: &CudaSlice<u32>,
        d_sum: &CudaSlice<u32>,
        d_poly: &mut CudaSlice<u32>,
        d_transcript_tail: &mut CudaSlice<u32>,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
    ) {
        {
            let (c0_ptr, _) = d_c0.device_ptr(&self.stream);
            let (c2_ptr, _) = d_c2.device_ptr(&self.stream);
            let (sum_ptr, _) = d_sum.device_ptr(&self.stream);
            let (poly_ptr, _) = d_poly.device_ptr_mut(&self.stream);
            let (tail_ptr, _) = d_transcript_tail.device_ptr_mut(&self.stream);
            let (st_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &c0_ptr as *const _ as *mut _,
                &c2_ptr as *const _ as *mut _,
                &sum_ptr as *const _ as *mut _,
                &poly_ptr as *const _ as *mut _,
                &tail_ptr as *const _ as *mut _,
                &st_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_prod_round_observe,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("product round observe kernel failed");
            }
        }
    }

    pub fn product_sumcheck_update_sum(
        &self,
        d_poly: &CudaSlice<u32>,
        d_challenge: &CudaSlice<u32>,
        d_sum: &mut CudaSlice<u32>,
    ) {
        self.product_sumcheck_update_sum_async(d_poly, d_challenge, d_sum);
        self.stream.synchronize().unwrap();
    }

    pub fn product_sumcheck_update_sum_async(
        &self,
        d_poly: &CudaSlice<u32>,
        d_challenge: &CudaSlice<u32>,
        d_sum: &mut CudaSlice<u32>,
    ) {
        let (poly_ptr, _) = d_poly.device_ptr(&self.stream);
        let (ch_ptr, _) = d_challenge.device_ptr(&self.stream);
        let (sum_ptr, _) = d_sum.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &poly_ptr as *const _ as *mut _,
            &ch_ptr as *const _ as *mut _,
            &sum_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_prod_round_update,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("product round update kernel failed");
        }
    }

    pub fn ext_affine_combine_device(
        &self,
        d_a: &CudaSlice<u32>,
        d_b: &CudaSlice<u32>,
        d_alpha: &CudaSlice<u32>,
    ) -> CudaSlice<u32> {
        let mut d_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        self.ext_affine_combine_into_async(d_a, d_b, d_alpha, &mut d_out);
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn ext_add_device_async(
        &self,
        d_a: &CudaSlice<u32>,
        d_b: &CudaSlice<u32>,
    ) -> CudaSlice<u32> {
        let mut d_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        self.ext_add_into_async(d_a, d_b, &mut d_out);
        d_out
    }

    pub fn ext_add_into_async(
        &self,
        d_a: &CudaSlice<u32>,
        d_b: &CudaSlice<u32>,
        d_out: &mut CudaSlice<u32>,
    ) {
        assert!(d_out.len() >= 5);
        let (a_ptr, _) = d_a.device_ptr(&self.stream);
        let (b_ptr, _) = d_b.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &a_ptr as *const _ as *mut _,
            &b_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_ext_add,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("ext add kernel failed");
        }
    }

    pub fn ext_affine_combine_into_async(
        &self,
        d_a: &CudaSlice<u32>,
        d_b: &CudaSlice<u32>,
        d_alpha: &CudaSlice<u32>,
        d_out: &mut CudaSlice<u32>,
    ) {
        let (a_ptr, _) = d_a.device_ptr(&self.stream);
        let (b_ptr, _) = d_b.device_ptr(&self.stream);
        let (alpha_ptr, _) = d_alpha.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &a_ptr as *const _ as *mut _,
            &b_ptr as *const _ as *mut _,
            &alpha_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_ext_affine_combine,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("ext affine combine kernel failed");
        }
    }

    pub fn extension_powers_device(
        &self,
        d_base: &CudaSlice<u32>,
        n_powers: u32,
    ) -> CudaSlice<u32> {
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_powers as usize) * 5)
            .unwrap();
        self.extension_powers_device_into_async(d_base, n_powers, &mut d_out);
        d_out
    }

    pub fn extension_powers_device_into_async(
        &self,
        d_base: &CudaSlice<u32>,
        n_powers: u32,
        d_out: &mut CudaSlice<u32>,
    ) {
        debug_assert!(d_out.len() >= (n_powers as usize) * 5);
        if n_powers == 0 {
            return;
        }
        let (base_ptr, _) = d_base.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &base_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n_powers as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_ext_powers,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("extension powers kernel failed");
        }
    }

    pub fn dense_eq_accumulate_from_points_device<S>(
        &self,
        d_weights: &mut CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        d_scalars: &S,
        n_points: u32,
        n_vars: u32,
        n_total: u32,
    ) where
        S: DevicePtr<u32>,
    {
        self.dense_eq_accumulate_from_points_device_async(
            d_weights, d_points, d_scalars, n_points, n_vars, n_total,
        );
        self.stream.synchronize().unwrap();
    }

    pub fn dense_eq_accumulate_from_points_device_async<S>(
        &self,
        d_weights: &mut CudaSlice<u32>,
        d_points: &CudaSlice<u32>,
        d_scalars: &S,
        n_points: u32,
        n_vars: u32,
        n_total: u32,
    ) where
        S: DevicePtr<u32>,
    {
        if n_points == 0 || n_total == 0 {
            return;
        }
        let (weights_ptr, _) = d_weights.device_ptr_mut(&self.stream);
        let (points_ptr, _) = d_points.device_ptr(&self.stream);
        let (scalars_ptr, _) = d_scalars.device_ptr(&self.stream);
        let threads = 256u32;
        let blocks = (n_total + threads - 1) / threads;
        // Shared memory: cache points+scalars when they fit in 12 KB.
        // The kernel checks the same threshold at runtime.
        let smem_words = (n_points * n_vars * 5 + n_points * 5) as usize;
        let smem_bytes = smem_words * std::mem::size_of::<u32>();
        let smem_launch = if smem_bytes <= 12288 { smem_bytes } else { 0 };
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &weights_ptr as *const _ as *mut _,
            &points_ptr as *const _ as *mut _,
            &scalars_ptr as *const _ as *mut _,
            &n_points as *const _ as *mut _,
            &n_vars as *const _ as *mut _,
            &n_total as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_dense_eq_accumulate_from_points,
                (blocks, 1, 1),
                (threads, 1, 1),
                smem_launch as u32,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("dense eq accumulate from points kernel failed");
        }
    }

    pub fn ext_dot_accumulate_device<V, S>(
        &self,
        d_base_sum: &CudaSlice<u32>,
        d_values: &V,
        d_scalars: &S,
        n_terms: u32,
    ) -> CudaSlice<u32>
    where
        V: DevicePtr<u32>,
        S: DevicePtr<u32>,
    {
        let d_out = self.ext_dot_accumulate_device_async(d_base_sum, d_values, d_scalars, n_terms);
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn ext_dot_accumulate_device_async<V, S>(
        &self,
        d_base_sum: &CudaSlice<u32>,
        d_values: &V,
        d_scalars: &S,
        n_terms: u32,
    ) -> CudaSlice<u32>
    where
        V: DevicePtr<u32>,
        S: DevicePtr<u32>,
    {
        let mut d_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        self.ext_dot_accumulate_into_async(d_base_sum, d_values, d_scalars, n_terms, &mut d_out);
        d_out
    }

    pub fn ext_dot_accumulate_into_async<V, S>(
        &self,
        d_base_sum: &CudaSlice<u32>,
        d_values: &V,
        d_scalars: &S,
        n_terms: u32,
        d_out: &mut CudaSlice<u32>,
    ) where
        V: DevicePtr<u32>,
        S: DevicePtr<u32>,
    {
        assert!(d_out.len() >= 5);
        if n_terms == 0 {
            self.memcpy_d2d_async(d_base_sum, 0, d_out, 0, 5);
            return;
        }
        let (base_ptr, _) = d_base_sum.device_ptr(&self.stream);
        let (values_ptr, _) = d_values.device_ptr(&self.stream);
        let (scalars_ptr, _) = d_scalars.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &base_ptr as *const _ as *mut _,
            &values_ptr as *const _ as *mut _,
            &scalars_ptr as *const _ as *mut _,
            &n_terms as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_ext_dot_accumulate,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("ext dot accumulate kernel failed");
        }
    }

    pub fn gkr_finalize_layer(
        &self,
        d_nl: &CudaSlice<u32>,
        d_nr: &CudaSlice<u32>,
        d_dl: &CudaSlice<u32>,
        d_dr: &CudaSlice<u32>,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
    ) -> (
        CudaSlice<u32>,
        CudaSlice<u32>,
        CudaSlice<u32>,
        CudaSlice<u32>,
    ) {
        let mut d_beta = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_claim_num = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_claim_den = self.stream.alloc_zeros::<u32>(5).unwrap();
        let mut d_transcript = self.stream.alloc_zeros::<u32>(20).unwrap();
        self.gkr_finalize_layer_async(
            d_nl,
            d_nr,
            d_dl,
            d_dr,
            &mut d_beta,
            &mut d_claim_num,
            &mut d_claim_den,
            &mut d_transcript,
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
        );
        self.stream.synchronize().unwrap();
        (d_beta, d_claim_num, d_claim_den, d_transcript)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gkr_finalize_layer_async(
        &self,
        d_nl: &CudaSlice<u32>,
        d_nr: &CudaSlice<u32>,
        d_dl: &CudaSlice<u32>,
        d_dr: &CudaSlice<u32>,
        d_beta: &mut CudaSlice<u32>,
        d_claim_num: &mut CudaSlice<u32>,
        d_claim_den: &mut CudaSlice<u32>,
        d_transcript: &mut CudaSlice<u32>,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
    ) {
        {
            let (nl_ptr, _) = d_nl.device_ptr(&self.stream);
            let (nr_ptr, _) = d_nr.device_ptr(&self.stream);
            let (dl_ptr, _) = d_dl.device_ptr(&self.stream);
            let (dr_ptr, _) = d_dr.device_ptr(&self.stream);
            let (beta_ptr, _) = d_beta.device_ptr_mut(&self.stream);
            let (cn_ptr, _) = d_claim_num.device_ptr_mut(&self.stream);
            let (cd_ptr, _) = d_claim_den.device_ptr_mut(&self.stream);
            let (tr_ptr, _) = d_transcript.device_ptr_mut(&self.stream);
            let (cs_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
            let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
            let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
            let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &nl_ptr as *const _ as *mut _,
                &nr_ptr as *const _ as *mut _,
                &dl_ptr as *const _ as *mut _,
                &dr_ptr as *const _ as *mut _,
                &beta_ptr as *const _ as *mut _,
                &cn_ptr as *const _ as *mut _,
                &cd_ptr as *const _ as *mut _,
                &tr_ptr as *const _ as *mut _,
                &cs_ptr as *const _ as *mut _,
                &rc_ptr as *const _ as *mut _,
                &mds_ptr as *const _ as *mut _,
                &sp_ptr as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_gkr_finalize_layer,
                    (1, 1, 1),
                    (1, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("gkr finalize layer kernel failed");
            }
        }
    }

    /// Fold eq polynomial on GPU (in-place fold at half).
    /// Input: n * 5 ext elements. Output: n/2 * 5 ext elements.
    pub fn eq_fold_device(
        &self,
        d_eq: &CudaSlice<u32>,
        n_pairs: u32,
        r_ext: &[u32; 5],
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
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
                    self.fn_eq_fold,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("eq fold device kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn patch_air_table_state_async(
        &self,
        d_round_meta: &CudaSlice<u32>,
        d_table_state: &mut CudaSlice<u32>,
        n_tables: u32,
    ) {
        let threads = 32u32;
        let blocks = n_tables.div_ceil(threads);
        let (meta_ptr, _) = d_round_meta.device_ptr(&self.stream);
        let (state_ptr, _) = d_table_state.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &meta_ptr as *const _ as *mut _,
            &state_ptr as *const _ as *mut _,
            &n_tables as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_air_patch_state,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("air patch state kernel failed");
        }
    }

    pub fn patch_air_table_state_from_gkr_async(
        &self,
        d_round_meta: &CudaSlice<u32>,
        d_round_extra: &CudaSlice<u32>,
        d_point_words: &CudaSlice<u32>,
        total_coords: u32,
        d_table_state: &mut CudaSlice<u32>,
        n_tables: u32,
    ) {
        debug_assert!(d_round_meta.len() >= (n_tables as usize) * 32);
        debug_assert!(d_round_extra.len() >= (n_tables as usize) * 4);
        debug_assert!(d_point_words.len() >= (total_coords as usize) * 5);
        debug_assert!(d_table_state.len() >= (n_tables as usize) * 32);
        let threads = 32u32;
        let blocks = n_tables.div_ceil(threads);
        let (meta_ptr, _) = d_round_meta.device_ptr(&self.stream);
        let (extra_ptr, _) = d_round_extra.device_ptr(&self.stream);
        let (point_ptr, _) = d_point_words.device_ptr(&self.stream);
        let (state_ptr, _) = d_table_state.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &meta_ptr as *const _ as *mut _,
            &extra_ptr as *const _ as *mut _,
            &point_ptr as *const _ as *mut _,
            &total_coords as *const _ as *mut _,
            &state_ptr as *const _ as *mut _,
            &n_tables as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_air_patch_state_from_gkr,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("air patch state from gkr kernel failed");
        }
    }

    pub fn patch_air_round_pad_beta_async(
        &self,
        d_pad_coeffs: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_round_meta: &mut CudaSlice<u32>,
        n_tables: u32,
    ) {
        debug_assert!(d_pad_coeffs.len() >= (n_tables as usize) * 5);
        debug_assert!(d_bus_beta.len() >= 5);
        debug_assert!(d_round_meta.len() >= (n_tables as usize) * 32);
        let threads = 32u32;
        let blocks = n_tables.div_ceil(threads);
        let (coeff_ptr, _) = d_pad_coeffs.device_ptr(&self.stream);
        let (beta_ptr, _) = d_bus_beta.device_ptr(&self.stream);
        let (meta_ptr, _) = d_round_meta.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &coeff_ptr as *const _ as *mut _,
            &beta_ptr as *const _ as *mut _,
            &meta_ptr as *const _ as *mut _,
            &n_tables as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_air_patch_pad_beta,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("air patch pad beta kernel failed");
        }
    }

    pub fn patch_air_round_pad_evals_async(
        &self,
        d_pad_evals: &CudaSlice<u32>,
        d_round_extra: &CudaSlice<u32>,
        d_round_meta: &mut CudaSlice<u32>,
        n_tables: u32,
    ) {
        debug_assert!(d_pad_evals.len() >= (n_tables as usize) * 5);
        debug_assert!(d_round_extra.len() >= (n_tables as usize) * 4);
        debug_assert!(d_round_meta.len() >= (n_tables as usize) * 32);
        let threads = 32u32;
        let blocks = n_tables.div_ceil(threads);
        let (pad_ptr, _) = d_pad_evals.device_ptr(&self.stream);
        let (extra_ptr, _) = d_round_extra.device_ptr(&self.stream);
        let (meta_ptr, _) = d_round_meta.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &pad_ptr as *const _ as *mut _,
            &extra_ptr as *const _ as *mut _,
            &meta_ptr as *const _ as *mut _,
            &n_tables as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_air_patch_pad_evals,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("air patch pad evals kernel failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn air_table_pad_eval_device(
        &self,
        d_all_cols: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        d_logup_alphas: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        table_index: u32,
        n_rows: u32,
    ) -> CudaSlice<u32> {
        let mut d_out = self.stream.alloc_zeros::<u32>(5).unwrap();
        self.air_table_pad_eval_device_into_async(
            d_all_cols,
            d_alphas,
            d_logup_alphas,
            d_bus_beta,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            table_index,
            n_rows,
            &mut d_out,
            0,
        );
        d_out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn air_table_pad_eval_device_into_async(
        &self,
        d_all_cols: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        d_logup_alphas: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
        table_index: u32,
        n_rows: u32,
        d_out: &mut CudaSlice<u32>,
        out_offset_words: usize,
    ) {
        debug_assert!(n_rows > 0);
        debug_assert!(d_alphas.len() >= 5);
        debug_assert!(d_logup_alphas.len() >= 25);
        debug_assert!(d_bus_beta.len() >= 5);
        debug_assert!(d_out.len() >= out_offset_words + 5);
        let (cols_ptr, _) = d_all_cols.device_ptr(&self.stream);
        let (alphas_ptr, _) = d_alphas.device_ptr(&self.stream);
        let (logup_ptr, _) = d_logup_alphas.device_ptr(&self.stream);
        let (beta_ptr, _) = d_bus_beta.device_ptr(&self.stream);
        let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
        let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
        let (sparse_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
        let (out_ptr_raw, _) = d_out.device_ptr_mut(&self.stream);
        let out_ptr = out_ptr_raw + (out_offset_words as u64) * 4;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &cols_ptr as *const _ as *mut _,
            &alphas_ptr as *const _ as *mut _,
            &logup_ptr as *const _ as *mut _,
            &beta_ptr as *const _ as *mut _,
            &rc_ptr as *const _ as *mut _,
            &mds_ptr as *const _ as *mut _,
            &sparse_ptr as *const _ as *mut _,
            &table_index as *const _ as *mut _,
            &n_rows as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_air_pad_eval,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("air pad eval kernel failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init_air_table_sums_from_logup_device_async(
        &self,
        d_bus_numerators: &CudaSlice<u32>,
        d_bus_denominators: &CudaSlice<u32>,
        d_logup_c: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_bus_directions: &CudaSlice<u32>,
        d_table_state: &mut CudaSlice<u32>,
        n_tables: u32,
    ) {
        debug_assert!(d_bus_numerators.len() >= (n_tables as usize) * 5);
        debug_assert!(d_bus_denominators.len() >= (n_tables as usize) * 5);
        debug_assert!(d_logup_c.len() >= 5);
        debug_assert!(d_bus_beta.len() >= 5);
        debug_assert!(d_bus_directions.len() >= n_tables as usize);
        debug_assert!(d_table_state.len() >= (n_tables as usize) * 32);
        let threads = 32u32;
        let blocks = n_tables.div_ceil(threads);
        let (num_ptr, _) = d_bus_numerators.device_ptr(&self.stream);
        let (den_ptr, _) = d_bus_denominators.device_ptr(&self.stream);
        let (c_ptr, _) = d_logup_c.device_ptr(&self.stream);
        let (beta_ptr, _) = d_bus_beta.device_ptr(&self.stream);
        let (dir_ptr, _) = d_bus_directions.device_ptr(&self.stream);
        let (state_ptr, _) = d_table_state.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &num_ptr as *const _ as *mut _,
            &den_ptr as *const _ as *mut _,
            &c_ptr as *const _ as *mut _,
            &beta_ptr as *const _ as *mut _,
            &dir_ptr as *const _ as *mut _,
            &state_ptr as *const _ as *mut _,
            &n_tables as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_air_init_sums_from_logup,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("air init sums from logup kernel failed");
        }
    }

    pub fn build_air_bare_coeffs_async(
        &self,
        d_ze_all: &CudaSlice<u32>,
        d_ze_offsets: &CudaSlice<u32>,
        d_table_state: &CudaSlice<u32>,
        n_tables: u32,
        mfd: u32,
        d_bare_scratch: &mut CudaSlice<u32>,
    ) {
        let threads = 32u32;
        let blocks = n_tables.div_ceil(threads);
        let (ze_ptr, _) = d_ze_all.device_ptr(&self.stream);
        let (zo_ptr, _) = d_ze_offsets.device_ptr(&self.stream);
        let (ts_ptr, _) = d_table_state.device_ptr(&self.stream);
        let (bs_ptr, _) = d_bare_scratch.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &ze_ptr as *const _ as *mut _,
            &zo_ptr as *const _ as *mut _,
            &ts_ptr as *const _ as *mut _,
            &n_tables as *const _ as *mut _,
            &mfd as *const _ as *mut _,
            &bs_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_air_build_bare,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("air build bare kernel failed");
        }
    }

    pub fn launch_reduce_ext_into_async(
        &self,
        d_partials: &CudaSlice<u32>,
        src_offset_words: usize,
        d_result: &mut CudaSlice<u32>,
        dst_offset_words: usize,
        n_blocks: u32,
    ) {
        let (src_ptr_raw, _) = d_partials.device_ptr(&self.stream);
        let src_ptr = src_ptr_raw + (src_offset_words as u64) * 4;
        let (dst_ptr_raw, _) = d_result.device_ptr_mut(&self.stream);
        let dst_ptr = dst_ptr_raw + (dst_offset_words as u64) * 4;
        let threads = 256u32.max(32);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &dst_ptr as *const _ as *mut _,
            &n_blocks as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_reduce_ext,
                (1, 1, 1),
                (threads, 1, 1),
                (threads / 32 * 5 * 4) as u32,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("reduce kernel failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_air_execution_multi_z_fb_into(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        d_logup_alphas: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_partial_sums: &mut CudaSlice<u32>,
        n_rows: u32,
        active_pairs: u32,
        fold_bit: u32,
    ) {
        let threads = 256u32;
        let blocks = active_pairs.div_ceil(threads);
        let smem = (5u32 * (threads / 32) * 5 * 4) as u32;
        let (c, _) = d_columns.device_ptr(&self.stream);
        let (dc, _) = d_down_cols.device_ptr(&self.stream);
        let (eq, _) = d_eq.device_ptr(&self.stream);
        let (al, _) = d_alphas.device_ptr(&self.stream);
        let (la, _) = d_logup_alphas.device_ptr(&self.stream);
        let (bb, _) = d_bus_beta.device_ptr(&self.stream);
        let (ps, _) = d_partial_sums.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &c as *const _ as *mut _,
            &dc as *const _ as *mut _,
            &eq as *const _ as *mut _,
            &al as *const _ as *mut _,
            &la as *const _ as *mut _,
            &bb as *const _ as *mut _,
            &ps as *const _ as *mut _,
            &n_rows as *const _ as *mut _,
            &active_pairs as *const _ as *mut _,
            &fold_bit as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_exec_mz_fb,
                (blocks, 1, 1),
                (threads, 1, 1),
                smem,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("exec mz fb failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_air_ext_op_multi_z_fb_into(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        d_logup_alphas: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_partial_sums: &mut CudaSlice<u32>,
        n_rows: u32,
        active_pairs: u32,
        fold_bit: u32,
    ) {
        let threads = 256u32;
        let blocks = active_pairs.div_ceil(threads);
        let smem = (6u32 * (threads / 32) * 5 * 4) as u32;
        let (c, _) = d_columns.device_ptr(&self.stream);
        let (dc, _) = d_down_cols.device_ptr(&self.stream);
        let (eq, _) = d_eq.device_ptr(&self.stream);
        let (al, _) = d_alphas.device_ptr(&self.stream);
        let (la, _) = d_logup_alphas.device_ptr(&self.stream);
        let (bb, _) = d_bus_beta.device_ptr(&self.stream);
        let (ps, _) = d_partial_sums.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &c as *const _ as *mut _,
            &dc as *const _ as *mut _,
            &eq as *const _ as *mut _,
            &al as *const _ as *mut _,
            &la as *const _ as *mut _,
            &bb as *const _ as *mut _,
            &ps as *const _ as *mut _,
            &n_rows as *const _ as *mut _,
            &active_pairs as *const _ as *mut _,
            &fold_bit as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_ext_op_mz_fb,
                (blocks, 1, 1),
                (threads, 1, 1),
                smem,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("ext_op mz fb failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_air_poseidon16_multi_z_fb_into(
        &self,
        d_columns: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        d_rc: &CudaSlice<u32>,
        d_mds: &CudaSlice<u32>,
        d_sparse: &CudaSlice<u32>,
        d_logup_alphas: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_partial_sums: &mut CudaSlice<u32>,
        n_rows: u32,
        active_pairs: u32,
        fold_bit: u32,
    ) {
        let threads = 128u32;
        let blocks = active_pairs.div_ceil(threads);
        let smem = (10u32 * (threads / 32) * 5 * 4) as u32;
        let (c, _) = d_columns.device_ptr(&self.stream);
        let (eq, _) = d_eq.device_ptr(&self.stream);
        let (al, _) = d_alphas.device_ptr(&self.stream);
        let (rc, _) = d_rc.device_ptr(&self.stream);
        let (md, _) = d_mds.device_ptr(&self.stream);
        let (sp, _) = d_sparse.device_ptr(&self.stream);
        let (la, _) = d_logup_alphas.device_ptr(&self.stream);
        let (bb, _) = d_bus_beta.device_ptr(&self.stream);
        let (ps, _) = d_partial_sums.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &c as *const _ as *mut _,
            &eq as *const _ as *mut _,
            &al as *const _ as *mut _,
            &rc as *const _ as *mut _,
            &md as *const _ as *mut _,
            &sp as *const _ as *mut _,
            &la as *const _ as *mut _,
            &bb as *const _ as *mut _,
            &ps as *const _ as *mut _,
            &n_rows as *const _ as *mut _,
            &active_pairs as *const _ as *mut _,
            &fold_bit as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_pos16_mz_fb,
                (blocks, 1, 1),
                (threads, 1, 1),
                smem,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("pos16 mz fb failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_air_execution_multi_z_ext_fb_into(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        d_logup_alphas: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_partial_sums: &mut CudaSlice<u32>,
        n_elems: u32,
        active_pairs: u32,
        fold_bit: u32,
    ) {
        let threads = 128u32;
        let blocks = active_pairs.div_ceil(threads);
        let smem = (5u32 * (threads / 32) * 5 * 4) as u32;
        let (c, _) = d_columns.device_ptr(&self.stream);
        let (dc, _) = d_down_cols.device_ptr(&self.stream);
        let (eq, _) = d_eq.device_ptr(&self.stream);
        let (al, _) = d_alphas.device_ptr(&self.stream);
        let (la, _) = d_logup_alphas.device_ptr(&self.stream);
        let (bb, _) = d_bus_beta.device_ptr(&self.stream);
        let (ps, _) = d_partial_sums.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &c as *const _ as *mut _,
            &dc as *const _ as *mut _,
            &eq as *const _ as *mut _,
            &al as *const _ as *mut _,
            &la as *const _ as *mut _,
            &bb as *const _ as *mut _,
            &ps as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &active_pairs as *const _ as *mut _,
            &fold_bit as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_exec_mz_ext_fb,
                (blocks, 1, 1),
                (threads, 1, 1),
                smem,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("exec ext mz fb failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_air_ext_op_multi_z_ext_fb_into(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        d_logup_alphas: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_partial_sums: &mut CudaSlice<u32>,
        n_elems: u32,
        active_pairs: u32,
        fold_bit: u32,
    ) {
        let threads = 128u32;
        let blocks = active_pairs.div_ceil(threads);
        let smem = (6u32 * (threads / 32) * 5 * 4) as u32;
        let (c, _) = d_columns.device_ptr(&self.stream);
        let (dc, _) = d_down_cols.device_ptr(&self.stream);
        let (eq, _) = d_eq.device_ptr(&self.stream);
        let (al, _) = d_alphas.device_ptr(&self.stream);
        let (la, _) = d_logup_alphas.device_ptr(&self.stream);
        let (bb, _) = d_bus_beta.device_ptr(&self.stream);
        let (ps, _) = d_partial_sums.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &c as *const _ as *mut _,
            &dc as *const _ as *mut _,
            &eq as *const _ as *mut _,
            &al as *const _ as *mut _,
            &la as *const _ as *mut _,
            &bb as *const _ as *mut _,
            &ps as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &active_pairs as *const _ as *mut _,
            &fold_bit as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_ext_op_mz_ext_fb,
                (blocks, 1, 1),
                (threads, 1, 1),
                smem,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("ext_op ext mz fb failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_air_poseidon16_multi_z_ext_fb_into(
        &self,
        d_columns: &CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        d_alphas: &CudaSlice<u32>,
        d_rc: &CudaSlice<u32>,
        d_mds: &CudaSlice<u32>,
        d_sparse: &CudaSlice<u32>,
        d_logup_alphas: &CudaSlice<u32>,
        d_bus_beta: &CudaSlice<u32>,
        d_partial_sums: &mut CudaSlice<u32>,
        n_elems: u32,
        active_pairs: u32,
        fold_bit: u32,
    ) {
        let threads = 64u32;
        let blocks = active_pairs.div_ceil(threads);
        let smem = (10u32 * (threads / 32) * 5 * 4) as u32;
        let (c, _) = d_columns.device_ptr(&self.stream);
        let (eq, _) = d_eq.device_ptr(&self.stream);
        let (al, _) = d_alphas.device_ptr(&self.stream);
        let (rc, _) = d_rc.device_ptr(&self.stream);
        let (md, _) = d_mds.device_ptr(&self.stream);
        let (sp, _) = d_sparse.device_ptr(&self.stream);
        let (la, _) = d_logup_alphas.device_ptr(&self.stream);
        let (bb, _) = d_bus_beta.device_ptr(&self.stream);
        let (ps, _) = d_partial_sums.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &c as *const _ as *mut _,
            &eq as *const _ as *mut _,
            &al as *const _ as *mut _,
            &rc as *const _ as *mut _,
            &md as *const _ as *mut _,
            &sp as *const _ as *mut _,
            &la as *const _ as *mut _,
            &bb as *const _ as *mut _,
            &ps as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &active_pairs as *const _ as *mut _,
            &fold_bit as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_pos16_mz_ext_fb,
                (blocks, 1, 1),
                (threads, 1, 1),
                smem,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("pos16 ext mz fb failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn protocol_step_async(
        &self,
        d_ze_all: &CudaSlice<u32>,
        d_ze_offsets: &CudaSlice<u32>,
        d_table_state: &mut CudaSlice<u32>,
        n_tables: u32,
        mfd: u32,
        d_cc_scratch: &mut CudaSlice<u32>,
        d_bare_scratch: &mut CudaSlice<u32>,
        d_challenges: &mut CudaSlice<u32>,
        challenge_offset_words: usize,
        d_transcript: &mut CudaSlice<u32>,
        transcript_offset_words: usize,
        d_transcript_len: &mut CudaSlice<u32>,
        d_challenger_state: &mut CudaSlice<u32>,
        d_p16_rc: &CudaSlice<u32>,
        d_p16_mds: &CudaSlice<u32>,
        d_p16_sparse: &CudaSlice<u32>,
    ) {
        let (ze_ptr, _) = d_ze_all.device_ptr(&self.stream);
        let (zo_ptr, _) = d_ze_offsets.device_ptr(&self.stream);
        let (ts_ptr, _) = d_table_state.device_ptr_mut(&self.stream);
        let (cc_ptr, _) = d_cc_scratch.device_ptr_mut(&self.stream);
        let (bs_ptr, _) = d_bare_scratch.device_ptr_mut(&self.stream);
        let (ch_ptr_raw, _) = d_challenges.device_ptr_mut(&self.stream);
        let ch_ptr = ch_ptr_raw + (challenge_offset_words as u64) * 4;
        let (tr_ptr_raw, _) = d_transcript.device_ptr_mut(&self.stream);
        let tr_ptr = tr_ptr_raw + (transcript_offset_words as u64) * 4;
        let (tl_ptr, _) = d_transcript_len.device_ptr_mut(&self.stream);
        let (cs_ptr, _) = d_challenger_state.device_ptr_mut(&self.stream);
        let (rc_ptr, _) = d_p16_rc.device_ptr(&self.stream);
        let (mds_ptr, _) = d_p16_mds.device_ptr(&self.stream);
        let (sp_ptr, _) = d_p16_sparse.device_ptr(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &ze_ptr as *const _ as *mut _,
            &zo_ptr as *const _ as *mut _,
            &ts_ptr as *const _ as *mut _,
            &n_tables as *const _ as *mut _,
            &mfd as *const _ as *mut _,
            &cc_ptr as *const _ as *mut _,
            &bs_ptr as *const _ as *mut _,
            &ch_ptr as *const _ as *mut _,
            &tr_ptr as *const _ as *mut _,
            &tl_ptr as *const _ as *mut _,
            &cs_ptr as *const _ as *mut _,
            &rc_ptr as *const _ as *mut _,
            &mds_ptr as *const _ as *mut _,
            &sp_ptr as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_protocol_step,
                (1, 1, 1),
                (1, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("protocol step kernel failed");
        }
    }

    pub fn fold_multi_col_b2e_at_bit_into_async(
        &self,
        d_in: &CudaSlice<u32>,
        d_out: &mut CudaSlice<u32>,
        n_rows: u32,
        n_pairs: u32,
        n_cols: u32,
        d_challenges: &CudaSlice<u32>,
        challenge_offset_words: usize,
        fold_bit: u32,
    ) {
        let threads = 256u32;
        let blocks = (n_cols * n_pairs).div_ceil(threads);
        let (in_ptr, _) = d_in.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let (r_ptr_raw, _) = d_challenges.device_ptr(&self.stream);
        let r_ptr = r_ptr_raw + (challenge_offset_words as u64) * 4;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &in_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &r_ptr as *const _ as *mut _,
            &n_rows as *const _ as *mut _,
            &n_pairs as *const _ as *mut _,
            &n_cols as *const _ as *mut _,
            &fold_bit as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_fold_multi_b2e_fb,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("batch fold base->ext at bit kernel failed");
        }
    }

    pub fn fold_multi_col_ext_at_bit_into_async(
        &self,
        d_in: &CudaSlice<u32>,
        d_out: &mut CudaSlice<u32>,
        n_elems: u32,
        n_pairs: u32,
        n_cols: u32,
        d_challenges: &CudaSlice<u32>,
        challenge_offset_words: usize,
        fold_bit: u32,
    ) {
        let threads = 256u32;
        let blocks = (n_cols * n_pairs).div_ceil(threads);
        let (in_ptr, _) = d_in.device_ptr(&self.stream);
        let (out_ptr, _) = d_out.device_ptr_mut(&self.stream);
        let (r_ptr_raw, _) = d_challenges.device_ptr(&self.stream);
        let r_ptr = r_ptr_raw + (challenge_offset_words as u64) * 4;
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &in_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &r_ptr as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &n_pairs as *const _ as *mut _,
            &n_cols as *const _ as *mut _,
            &fold_bit as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_fold_multi_ext_fb,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("batch fold ext at bit kernel failed");
        }
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
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
        let b_hi: EF = ef(pol_b[(i + half) * 5..(i + half + 1) * 5]
            .try_into()
            .unwrap());
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
        let a_hi: EF = ef(pol_a[(i + half) * 5..(i + half + 1) * 5]
            .try_into()
            .unwrap());
        let b_lo: EF = ef(pol_b[i * 5..(i + 1) * 5].try_into().unwrap());
        let b_hi: EF = ef(pol_b[(i + half) * 5..(i + half + 1) * 5]
            .try_into()
            .unwrap());
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
