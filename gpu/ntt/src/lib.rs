//! GPU NTT (Evals DFT) for KoalaBear field.
//!
//! Matches the WHIR DFT convention from leanVM: evaluates a multilinear
//! polynomial (given as evaluations on {0,1}^n) at all points derived from
//! powers of the two-adic generator.
//!
//! The DFT is layer-by-layer radix-2 butterfly, processed from smallest
//! blocks (stride 1) to largest (stride n/2).

use std::ffi::CString;
use std::sync::Arc;

use koala_bear::KoalaBear;
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};

const P: u32 = 0x7F000001;

fn kb(v: u32) -> KoalaBear {
    unsafe { std::mem::transmute(v) }
}

fn kb_u32(v: KoalaBear) -> u32 {
    unsafe { std::mem::transmute(v) }
}

// ── Twiddle table generation ─────────────────────────────────────────────

/// Build the root-of-unity table matching WHIR's convention.
///
/// Returns `log_n` layers. Layer `i` has `n >> (i+1)` twiddle factors.
/// Layer 0 has n/2 twiddles (the full set), layer log_n-1 has 1 twiddle.
///
/// Twiddles are in Montgomery form.
pub fn build_root_table(log_n: usize) -> Vec<Vec<u32>> {
    let n = 1usize << log_n;
    let generator = KoalaBear::two_adic_generator(log_n);

    // nth_roots = [1, g, g², ..., g^(n/2-1)]
    let mut nth_roots = Vec::with_capacity(n / 2);
    let mut acc = KoalaBear::ONE;
    for _ in 0..n / 2 {
        nth_roots.push(kb_u32(acc));
        acc *= generator;
    }

    // Layer i: take every 2^i-th root.
    (0..log_n)
        .map(|i| {
            nth_roots
                .iter()
                .step_by(1 << i)
                .copied()
                .collect()
        })
        .collect()
}

/// Build inverse twiddle table: inv_twiddles[layer][j] = 1 / twiddles[layer][j].
pub fn build_inv_root_table(root_table: &[Vec<u32>]) -> Vec<Vec<u32>> {
    root_table
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|&tw| {
                    if tw == 0 { 0 } else { kb_u32(kb(tw).inverse()) }
                })
                .collect()
        })
        .collect()
}

// ── GPU NTT engine ───────────────────────────────────────────────────────

pub struct GpuNtt {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    fn_dft_layer: cuda_sys::CUfunction,
    fn_idft_layer: cuda_sys::CUfunction,
    fn_dft_fused: cuda_sys::CUfunction,
    fn_idft_fused: cuda_sys::CUfunction,
    fn_prepare_evals: cuda_sys::CUfunction,
}

unsafe impl Send for GpuNtt {}
unsafe impl Sync for GpuNtt {}

impl Drop for GpuNtt {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_result::module::unload(self.cu_module);
        }
    }
}

impl GpuNtt {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/ntt.ptx"));
        let c_src = CString::new(ptx_src).unwrap();
        let cu_module = unsafe { cuda_result::module::load_data(c_src.as_ptr().cast()) }
            .expect("failed to load ntt PTX");

        let load = |name: &str| {
            let c = CString::new(name).unwrap();
            unsafe { cuda_result::module::get_function(cu_module, c) }
                .unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };

        Self {
            stream,
            cu_module,
            fn_dft_layer: load("evals_dft_layer_kernel"),
            fn_idft_layer: load("evals_idft_layer_kernel"),
            fn_dft_fused: load("evals_dft_fused_kernel"),
            fn_idft_fused: load("evals_idft_fused_kernel"),
            fn_prepare_evals: load("prepare_evals_for_fft_kernel"),
        }
    }

    /// In-place evals DFT on a device buffer.
    ///
    /// `d_data`: device buffer of `height * width` u32 elements (row-major).
    /// `root_table`: precomputed twiddle table from `build_root_table`.
    /// `log_height`: log2(height).
    /// `width`: number of columns.
    /// Maximum number of layers to fuse in shared memory.
    /// Limited by shared memory size: 2^FUSED_LOG elements × 4 bytes.
    /// 2^13 = 8192 elements = 32 KB shared memory — fits on all modern GPUs.
    const FUSED_LOG: usize = 13;

    pub fn dft_in_place(
        &self,
        d_data: &mut CudaSlice<u32>,
        root_table: &[Vec<u32>],
        log_height: usize,
        width: usize,
    ) {
        if width == 1 && log_height >= 4 {
            self.dft_in_place_fused(d_data, root_table, log_height);
        } else {
            self.dft_in_place_per_layer(d_data, root_table, log_height, width);
        }
    }

    /// Fused DFT for width=1: process first FUSED_LOG layers in shared memory
    /// (single kernel launch), then remaining layers one at a time.
    fn dft_in_place_fused(
        &self,
        d_data: &mut CudaSlice<u32>,
        root_table: &[Vec<u32>],
        log_height: usize,
    ) {
        let height = 1usize << log_height;
        let threads = 256u32;

        // How many layers to fuse (limited by shared memory).
        let fused = Self::FUSED_LOG.min(log_height);
        let chunk_size = (1u32 << fused) as u32;

        // Build flat twiddle array and offset table for fused layers.
        // Fused layers are the LAST `fused` entries of root_table (processed first = smallest blocks).
        let mut all_twiddles: Vec<u32> = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();
        for k in 0..fused {
            tw_offsets.push(all_twiddles.len() as u32);
            let layer_idx = log_height - 1 - k; // reversed order
            all_twiddles.extend_from_slice(&root_table[layer_idx]);
        }
        tw_offsets.push(all_twiddles.len() as u32);

        let d_tw = self.stream.memcpy_stod(&all_twiddles).unwrap();
        let d_offsets = self.stream.memcpy_stod(&tw_offsets).unwrap();
        let n_elements = height as u32;
        let fused_u32 = fused as u32;
        let n_chunks = (height as u32) / chunk_size;

        {
            let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
            let (tw_ptr, _g2) = d_tw.device_ptr(&self.stream);
            let (off_ptr, _g3) = d_offsets.device_ptr(&self.stream);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &tw_ptr as *const _ as *mut _,
                &off_ptr as *const _ as *mut _,
                &fused_u32 as *const _ as *mut _,
                &chunk_size as *const _ as *mut _,
                &n_elements as *const _ as *mut _,
            ];

            let smem_bytes = chunk_size * 4;
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_dft_fused,
                    (n_chunks, 1, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("fused dft kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        // Remaining layers: single-layer kernel for each.
        let remaining_layers = log_height - fused;
        for k in 0..remaining_layers {
            let layer_idx = remaining_layers - 1 - k; // reversed
            let m = root_table[layer_idx].len() as u32;
            let n_butterflies = (height / 2) as u32;

            let d_tw = self.stream.memcpy_stod(&root_table[layer_idx]).unwrap();
            let w = 1u32;

            {
                let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
                let (tw_ptr, _g2) = d_tw.device_ptr(&self.stream);

                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &data_ptr as *const _ as *mut _,
                    &tw_ptr as *const _ as *mut _,
                    &m as *const _ as *mut _,
                    &w as *const _ as *mut _,
                    &n_butterflies as *const _ as *mut _,
                ];

                let blocks = (n_butterflies + threads - 1) / threads;
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_dft_layer,
                        (blocks, 1, 1),
                        (threads, 1, 1),
                        0,
                        self.stream.cu_stream(),
                        &mut args,
                    )
                    .expect("dft layer kernel failed");
                }
            }
            self.stream.synchronize().unwrap();
        }
    }

    /// Per-layer DFT (original implementation, used for width>1).
    fn dft_in_place_per_layer(
        &self,
        d_data: &mut CudaSlice<u32>,
        root_table: &[Vec<u32>],
        log_height: usize,
        width: usize,
    ) {
        let height = 1usize << log_height;
        let threads = 256u32;

        for layer_idx in (0..log_height).rev() {
            let m = root_table[layer_idx].len() as u32;
            let n_butterflies = ((height / 2) * width) as u32;

            let d_tw = self.stream.memcpy_stod(&root_table[layer_idx]).unwrap();

            {
                let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
                let (tw_ptr, _g2) = d_tw.device_ptr(&self.stream);
                let w = width as u32;

                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &data_ptr as *const _ as *mut _,
                    &tw_ptr as *const _ as *mut _,
                    &m as *const _ as *mut _,
                    &w as *const _ as *mut _,
                    &n_butterflies as *const _ as *mut _,
                ];

                let blocks = (n_butterflies + threads - 1) / threads;
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_dft_layer,
                        (blocks, 1, 1),
                        (threads, 1, 1),
                        0,
                        self.stream.cu_stream(),
                        &mut args,
                    )
                    .expect("dft layer kernel failed");
                }
            }
            self.stream.synchronize().unwrap();
        }
    }

    /// In-place inverse evals DFT.
    pub fn idft_in_place(
        &self,
        d_data: &mut CudaSlice<u32>,
        root_table: &[Vec<u32>],
        inv_root_table: &[Vec<u32>],
        log_height: usize,
        width: usize,
    ) {
        if width == 1 && log_height >= 4 {
            self.idft_in_place_fused(d_data, root_table, inv_root_table, log_height);
        } else {
            self.idft_in_place_per_layer(d_data, root_table, inv_root_table, log_height, width);
        }
    }

    fn idft_in_place_fused(
        &self,
        d_data: &mut CudaSlice<u32>,
        root_table: &[Vec<u32>],
        inv_root_table: &[Vec<u32>],
        log_height: usize,
    ) {
        let height = 1usize << log_height;
        let threads = 256u32;
        let fused = Self::FUSED_LOG.min(log_height);

        // Inverse: large layers first (layer 0 to log_height-fused-1) via single-layer kernel.
        let remaining_layers = log_height - fused;
        for layer_idx in 0..remaining_layers {
            let m = root_table[layer_idx].len() as u32;
            let n_butterflies = (height / 2) as u32;

            let d_tw = self.stream.memcpy_stod(&root_table[layer_idx]).unwrap();
            let d_inv_tw = self.stream.memcpy_stod(&inv_root_table[layer_idx]).unwrap();
            let w = 1u32;

            {
                let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
                let (tw_ptr, _g2) = d_tw.device_ptr(&self.stream);
                let (inv_tw_ptr, _g3) = d_inv_tw.device_ptr(&self.stream);

                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &data_ptr as *const _ as *mut _,
                    &tw_ptr as *const _ as *mut _,
                    &inv_tw_ptr as *const _ as *mut _,
                    &m as *const _ as *mut _,
                    &w as *const _ as *mut _,
                    &n_butterflies as *const _ as *mut _,
                ];

                let blocks = (n_butterflies + threads - 1) / threads;
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_idft_layer,
                        (blocks, 1, 1),
                        (threads, 1, 1),
                        0,
                        self.stream.cu_stream(),
                        &mut args,
                    )
                    .expect("idft layer kernel failed");
                }
            }
            self.stream.synchronize().unwrap();
        }

        // Then fused kernel for the last `fused` small layers.
        let chunk_size = (1u32 << fused) as u32;

        let mut all_tw: Vec<u32> = Vec::new();
        let mut all_inv_tw: Vec<u32> = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();
        for k in 0..fused {
            tw_offsets.push(all_tw.len() as u32);
            let layer_idx = log_height - 1 - k;
            all_tw.extend_from_slice(&root_table[layer_idx]);
            all_inv_tw.extend_from_slice(&inv_root_table[layer_idx]);
        }
        tw_offsets.push(all_tw.len() as u32);

        let d_tw = self.stream.memcpy_stod(&all_tw).unwrap();
        let d_inv_tw = self.stream.memcpy_stod(&all_inv_tw).unwrap();
        let d_offsets = self.stream.memcpy_stod(&tw_offsets).unwrap();
        let n_elements = height as u32;
        let fused_u32 = fused as u32;
        let n_chunks = (height as u32) / chunk_size;

        {
            let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
            let (tw_ptr, _g2) = d_tw.device_ptr(&self.stream);
            let (inv_tw_ptr, _g3) = d_inv_tw.device_ptr(&self.stream);
            let (off_ptr, _g4) = d_offsets.device_ptr(&self.stream);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &tw_ptr as *const _ as *mut _,
                &inv_tw_ptr as *const _ as *mut _,
                &off_ptr as *const _ as *mut _,
                &fused_u32 as *const _ as *mut _,
                &chunk_size as *const _ as *mut _,
                &n_elements as *const _ as *mut _,
            ];

            let smem_bytes = chunk_size * 4;
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_idft_fused,
                    (n_chunks, 1, 1),
                    (threads, 1, 1),
                    smem_bytes,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("fused idft kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
    }

    fn idft_in_place_per_layer(
        &self,
        d_data: &mut CudaSlice<u32>,
        root_table: &[Vec<u32>],
        inv_root_table: &[Vec<u32>],
        log_height: usize,
        width: usize,
    ) {
        let height = 1usize << log_height;
        let threads = 256u32;

        for layer_idx in 0..log_height {
            let m = root_table[layer_idx].len() as u32;
            let n_butterflies = ((height / 2) * width) as u32;

            let d_tw = self.stream.memcpy_stod(&root_table[layer_idx]).unwrap();
            let d_inv_tw = self.stream.memcpy_stod(&inv_root_table[layer_idx]).unwrap();

            {
                let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
                let (tw_ptr, _g2) = d_tw.device_ptr(&self.stream);
                let (inv_tw_ptr, _g3) = d_inv_tw.device_ptr(&self.stream);
                let w = width as u32;

                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &data_ptr as *const _ as *mut _,
                    &tw_ptr as *const _ as *mut _,
                    &inv_tw_ptr as *const _ as *mut _,
                    &m as *const _ as *mut _,
                    &w as *const _ as *mut _,
                    &n_butterflies as *const _ as *mut _,
                ];

                let blocks = (n_butterflies + threads - 1) / threads;
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_idft_layer,
                        (blocks, 1, 1),
                        (threads, 1, 1),
                        0,
                        self.stream.cu_stream(),
                        &mut args,
                    )
                    .expect("idft layer kernel failed");
                }
            }
            self.stream.synchronize().unwrap();
        }
    }

    /// Convenience: DFT on host data, returns host result.
    pub fn dft(&self, data: &[u32], log_height: usize, width: usize) -> Vec<u32> {
        let root_table = build_root_table(log_height);
        let mut d_data = self.stream.memcpy_stod(data).unwrap();
        self.dft_in_place(&mut d_data, &root_table, log_height, width);
        self.stream.memcpy_dtov(&d_data).unwrap()
    }

    /// Convenience: IDFT on host data, returns host result.
    pub fn idft(&self, data: &[u32], log_height: usize, width: usize) -> Vec<u32> {
        let root_table = build_root_table(log_height);
        let inv_table = build_inv_root_table(&root_table);
        let mut d_data = self.stream.memcpy_stod(data).unwrap();
        self.idft_in_place(&mut d_data, &root_table, &inv_table, log_height, width);
        self.stream.memcpy_dtov(&d_data).unwrap()
    }

    /// Reorder evaluations for WHIR DFT convention, on GPU device.
    /// Returns device buffer with reordered data ready for DFT.
    pub fn prepare_evals_for_fft_device(
        &self,
        d_evals: &CudaSlice<u32>,
        n_evals: u32,
        n_cols: u32,
        log_inv_rate: u32,
    ) -> CudaSlice<u32> {
        let n_blocks = n_cols;
        let full_len = (n_evals as u64) << log_inv_rate;
        let block_size = full_len / n_blocks as u64;
        let log_block_size = block_size.trailing_zeros();
        let out_len = (block_size as u32) * n_cols;

        let mut d_out = self.stream.alloc_zeros::<u32>(out_len as usize).unwrap();
        {
            let (evals_ptr, _g1) = d_evals.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let threads = 256u32;
            let blocks = (out_len + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &evals_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &n_evals as *const _ as *mut _,
                &n_cols as *const _ as *mut _,
                &log_block_size as *const _ as *mut _,
                &log_inv_rate as *const _ as *mut _,
                &out_len as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_prepare_evals, (blocks, 1, 1), (threads, 1, 1),
                    0, self.stream.cu_stream(), &mut args,
                ).expect("prepare_evals kernel failed");
            }
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Full reorder → DFT pipeline on device. Returns DFT output as device buffer.
    pub fn reorder_and_dft_device(
        &self,
        d_evals: &CudaSlice<u32>,
        n_evals: u32,
        folding_factor: usize,
        log_inv_rate: usize,
    ) -> CudaSlice<u32> {
        let n_cols = 1u32 << folding_factor;
        let mut d_reordered = self.prepare_evals_for_fft_device(
            d_evals, n_evals, n_cols, log_inv_rate as u32,
        );

        // DFT on the reordered matrix (height = out_len / n_cols, width = n_cols).
        let full_len = (n_evals as u64) << log_inv_rate;
        let block_size = full_len / n_cols as u64;
        let log_height = block_size.trailing_zeros() as usize;
        let root_table = build_root_table(log_height);
        self.dft_in_place(&mut d_reordered, &root_table, log_height, n_cols as usize);
        d_reordered
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

// ── CPU reference ────────────────────────────────────────────────────────

/// CPU evals DFT matching the WHIR convention.
pub fn cpu_evals_dft(data: &mut [u32], root_table: &[Vec<u32>], width: usize) {
    let log_n = root_table.len();

    // Process layers from smallest blocks to largest (rev order).
    for layer_idx in (0..log_n).rev() {
        let twiddles = &root_table[layer_idx];
        let m = twiddles.len();
        let block_size = 2 * m * width;

        for block_start in (0..data.len()).step_by(block_size) {
            for i in 0..m {
                for col in 0..width {
                    let idx_hi = block_start + i * width + col;
                    let idx_lo = block_start + (i + m) * width + col;

                    let x_hi = kb(data[idx_hi]);
                    let x_lo = kb(data[idx_lo]);

                    let (y_hi, y_lo) = if i == 0 {
                        // Twiddle-free: (x_lo, 2*x_hi - x_lo)
                        (x_lo, x_hi.double() - x_lo)
                    } else {
                        // Standard: tmp = (x_lo - x_hi) * tw
                        let tw = kb(twiddles[i]);
                        let tmp = (x_lo - x_hi) * tw;
                        (x_hi + tmp, x_hi - tmp)
                    };

                    data[idx_hi] = kb_u32(y_hi);
                    data[idx_lo] = kb_u32(y_lo);
                }
            }
        }
    }
}

/// CPU inverse evals DFT.
pub fn cpu_evals_idft(data: &mut [u32], root_table: &[Vec<u32>], width: usize) {
    let log_n = root_table.len();
    let inv_table = build_inv_root_table(root_table);

    // Inverse: process layers from largest blocks to smallest.
    for layer_idx in 0..log_n {
        let twiddles = &root_table[layer_idx];
        let inv_twiddles = &inv_table[layer_idx];
        let m = twiddles.len();
        let block_size = 2 * m * width;

        for block_start in (0..data.len()).step_by(block_size) {
            for i in 0..m {
                for col in 0..width {
                    let idx_hi = block_start + i * width + col;
                    let idx_lo = block_start + (i + m) * width + col;

                    let y_hi = kb(data[idx_hi]);
                    let y_lo = kb(data[idx_lo]);

                    let (x_hi, x_lo) = if i == 0 {
                        // Inverse twiddle-free
                        let x_hi = (y_hi + y_lo).halve();
                        (x_hi, y_hi)
                    } else {
                        let x_hi = (y_hi + y_lo).halve();
                        let diff_half = (y_hi - y_lo).halve();
                        let inv_tw = kb(inv_twiddles[i]);
                        (x_hi, x_hi + diff_half * inv_tw)
                    };

                    data[idx_hi] = kb_u32(x_hi);
                    data[idx_lo] = kb_u32(x_lo);
                }
            }
        }
    }
}

/// CPU convenience: DFT returning new vector.
pub fn cpu_dft(data: &[u32], log_height: usize, width: usize) -> Vec<u32> {
    let root_table = build_root_table(log_height);
    let mut out = data.to_vec();
    cpu_evals_dft(&mut out, &root_table, width);
    out
}

/// CPU convenience: IDFT returning new vector.
pub fn cpu_idft(data: &[u32], log_height: usize, width: usize) -> Vec<u32> {
    let root_table = build_root_table(log_height);
    let mut out = data.to_vec();
    cpu_evals_idft(&mut out, &root_table, width);
    out
}

// ── Verification against leanVM's DFT ────────────────────────────────────

/// Evaluate the multilinear polynomial (given by evals on {0,1}^n) at the
/// point derived from ω^i: (ω^i, ω^(2i), ω^(4i), ..., ω^(2^(n-1)·i)).
///
/// This is what the DFT output[i] should equal.
pub fn eval_multilinear_at_power(evals: &[u32], i: usize, log_n: usize) -> u32 {
    let omega = KoalaBear::two_adic_generator(log_n);
    let omega_i = omega.exp_u64(i as u64);

    // Build point: (ω^i, ω^(2i), ω^(4i), ...)
    let mut coords = Vec::with_capacity(log_n);
    let mut w = omega_i;
    for _ in 0..log_n {
        coords.push(w);
        w = w * w;
    }

    // Evaluate multilinear: sum over all binary assignments.
    // The DFT convention: bit k of index j corresponds to coords[log_n - 1 - k]
    // (MSB of j → first coordinate, LSB of j → last coordinate).
    let n = 1usize << log_n;
    let mut result = KoalaBear::ZERO;
    for j in 0..n {
        let mut term = kb(evals[j]);
        for bit in 0..log_n {
            let coord = coords[log_n - 1 - bit];
            if (j >> bit) & 1 == 1 {
                term = term * coord;
            } else {
                term = term * (KoalaBear::ONE - coord);
            }
        }
        result += term;
    }
    kb_u32(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    #[test]
    fn test_cpu_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        for log_n in 1..=12 {
            let n = 1usize << log_n;
            let data: Vec<u32> = (0..n).map(|_| kb_u32(rng.random::<KoalaBear>())).collect();
            let dft = cpu_dft(&data, log_n, 1);
            let roundtrip = cpu_idft(&dft, log_n, 1);
            assert_eq!(data, roundtrip, "CPU round-trip failed for log_n={log_n}");
        }
    }

    #[test]
    fn test_cpu_dft_matches_naive_eval() {
        let mut rng = StdRng::seed_from_u64(7);
        for log_n in 1..=10 {
            let n = 1usize << log_n;
            let data: Vec<u32> = (0..n).map(|_| kb_u32(rng.random::<KoalaBear>())).collect();
            let dft = cpu_dft(&data, log_n, 1);

            // Check a few random output positions against naive multilinear evaluation.
            for _ in 0..std::cmp::min(10, n) {
                let i = rng.random_range(0..n);
                let expected = eval_multilinear_at_power(&data, i, log_n);
                assert_eq!(
                    dft[i], expected,
                    "CPU DFT mismatch at i={i} for log_n={log_n}"
                );
            }
        }
    }
}
