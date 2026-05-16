//! GPU multilinear polynomial folding.
//!
//! Core sumcheck primitive: fold one variable of a multilinear polynomial
//! at a challenge point r, halving the number of evaluations.
//!
//! Three type variants:
//! - base→base: data and r are base field
//! - base→ext: data is base, r is quintic extension (first sumcheck fold)
//! - ext→ext: both data and r are quintic extension (subsequent folds)
//!
//! Three addressing modes:
//! - LSB (bit=0): pairs are (2j, 2j+1)
//! - Half: pairs are (j, j+n/2) (fold_multilinear convention)
//! - Arbitrary bit: interleaved pairs at stride 1<<bit

use std::ffi::CString;
use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use koala_bear::{KoalaBear, extension::QuinticExtensionField};

type EF = QuinticExtensionField<KoalaBear>;

fn kb_as_u32(v: &KoalaBear) -> u32 {
    unsafe { std::mem::transmute::<KoalaBear, u32>(*v) }
}

fn ef_as_u32(v: &EF) -> [u32; 5] {
    unsafe { std::mem::transmute::<EF, [u32; 5]>(*v) }
}

/// Addressing mode for fold pairs.
#[derive(Clone, Copy, Debug)]
pub enum FoldMode {
    /// Pairs (2j, 2j+1) — contiguous.
    Lsb,
    /// Pairs (j, j+n/2) — halves.
    Half,
    /// Pairs with stride 1<<bit.
    AtBit(u32),
}

/// GPU polynomial folder. Owns the loaded CUDA module.
pub struct GpuPolyFold {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    // base → base
    fn_base_lsb: cuda_sys::CUfunction,
    fn_base_half: cuda_sys::CUfunction,
    fn_base_at_bit: cuda_sys::CUfunction,
    // base → ext
    fn_b2e_lsb: cuda_sys::CUfunction,
    fn_b2e_half: cuda_sys::CUfunction,
    fn_b2e_at_bit: cuda_sys::CUfunction,
    // ext → ext
    fn_ext_lsb: cuda_sys::CUfunction,
    fn_ext_half: cuda_sys::CUfunction,
    fn_ext_at_bit: cuda_sys::CUfunction,
    fn_evals_to_coeffs_ext_first_layer: cuda_sys::CUfunction,
    fn_evals_to_coeffs_ext_layer: cuda_sys::CUfunction,
    fn_bit_reverse_ext: cuda_sys::CUfunction,
}

unsafe impl Send for GpuPolyFold {}
unsafe impl Sync for GpuPolyFold {}

impl Drop for GpuPolyFold {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_result::module::unload(self.cu_module);
        }
    }
}

impl GpuPolyFold {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let cubin = include_bytes!(concat!(env!("OUT_DIR"), "/poly_fold.cubin"));
        let cu_module = unsafe { cuda_result::module::load_data(cubin.as_ptr().cast()) }
            .expect("failed to load poly_fold cubin");

        let load = |name: &str| {
            let c = CString::new(name).unwrap();
            unsafe { cuda_result::module::get_function(cu_module, c) }
                .unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };

        Self {
            stream,
            cu_module,
            fn_base_lsb: load("fold_base_lsb_kernel"),
            fn_base_half: load("fold_base_half_kernel"),
            fn_base_at_bit: load("fold_base_at_bit_kernel"),
            fn_b2e_lsb: load("fold_base_to_ext_lsb_kernel"),
            fn_b2e_half: load("fold_base_to_ext_half_kernel"),
            fn_b2e_at_bit: load("fold_base_to_ext_at_bit_kernel"),
            fn_ext_lsb: load("fold_ext_lsb_kernel"),
            fn_ext_half: load("fold_ext_half_kernel"),
            fn_ext_at_bit: load("fold_ext_at_bit_kernel"),
            fn_evals_to_coeffs_ext_first_layer: load("evals_to_coeffs_ext_first_layer_kernel"),
            fn_evals_to_coeffs_ext_layer: load("evals_to_coeffs_ext_layer_kernel"),
            fn_bit_reverse_ext: load("bit_reverse_ext_kernel"),
        }
    }

    fn launch(
        &self,
        func: cuda_sys::CUfunction,
        args: &mut Vec<*mut std::ffi::c_void>,
        n_pairs: u32,
    ) {
        let threads = 256u32;
        let blocks = (n_pairs + threads - 1) / threads;
        unsafe {
            cuda_result::launch_kernel(
                func,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                args,
            )
            .expect("fold kernel launch failed");
        }
    }

    /// Fold base→base: input is 2*n_pairs base field elements, output is n_pairs.
    pub fn fold_base(&self, data: &[u32], r: u32, mode: FoldMode) -> Vec<u32> {
        let n_pairs = match mode {
            FoldMode::Lsb | FoldMode::Half => (data.len() / 2) as u32,
            FoldMode::AtBit(_) => (data.len() / 2) as u32,
        };

        let d_data = self.stream.memcpy_stod(data).unwrap();
        let mut d_out = self.stream.alloc_zeros::<u32>(n_pairs as usize).unwrap();

        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);

            match mode {
                FoldMode::Lsb => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                    ];
                    self.launch(self.fn_base_lsb, &mut args, n_pairs);
                }
                FoldMode::Half => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                    ];
                    self.launch(self.fn_base_half, &mut args, n_pairs);
                }
                FoldMode::AtBit(bit) => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                        &bit as *const _ as *mut _,
                    ];
                    self.launch(self.fn_base_at_bit, &mut args, n_pairs);
                }
            }
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_out).unwrap()
    }

    /// Fold base→ext: input is 2*n_pairs base field elements,
    /// output is n_pairs * 5 (quintic ext).
    pub fn fold_base_to_ext(&self, data: &[u32], r_ext: &[u32; 5], mode: FoldMode) -> Vec<u32> {
        let n_pairs = (data.len() / 2) as u32;

        let d_data = self.stream.memcpy_stod(data).unwrap();
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();

        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);

            match mode {
                FoldMode::Lsb => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r_ptr as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                    ];
                    self.launch(self.fn_b2e_lsb, &mut args, n_pairs);
                }
                FoldMode::Half => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r_ptr as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                    ];
                    self.launch(self.fn_b2e_half, &mut args, n_pairs);
                }
                FoldMode::AtBit(bit) => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r_ptr as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                        &bit as *const _ as *mut _,
                    ];
                    self.launch(self.fn_b2e_at_bit, &mut args, n_pairs);
                }
            }
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_out).unwrap()
    }

    /// Fold ext→ext: input is 2*n_pairs*5 ext elements, output is n_pairs*5.
    pub fn fold_ext(&self, data: &[u32], r_ext: &[u32; 5], mode: FoldMode) -> Vec<u32> {
        let n_pairs = (data.len() / 10) as u32; // 2 * n_pairs * 5 elements

        let d_data = self.stream.memcpy_stod(data).unwrap();
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();

        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);

            match mode {
                FoldMode::Lsb => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r_ptr as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                    ];
                    self.launch(self.fn_ext_lsb, &mut args, n_pairs);
                }
                FoldMode::Half => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r_ptr as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                    ];
                    self.launch(self.fn_ext_half, &mut args, n_pairs);
                }
                FoldMode::AtBit(bit) => {
                    let mut args: Vec<*mut std::ffi::c_void> = vec![
                        &data_ptr as *const _ as *mut _,
                        &out_ptr as *const _ as *mut _,
                        &r_ptr as *const _ as *mut _,
                        &n_pairs as *const _ as *mut _,
                        &bit as *const _ as *mut _,
                    ];
                    self.launch(self.fn_ext_at_bit, &mut args, n_pairs);
                }
            }
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_out).unwrap()
    }

    // ── Device-resident fold APIs ──────────────────────────────────────────

    /// Fold base→ext on device buffers. Returns new CudaSlice on device.
    pub fn fold_base_to_ext_device(
        &self,
        d_data: &CudaSlice<u32>,
        n_pairs: u32,
        r_ext: &[u32; 5],
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            self.launch(self.fn_b2e_half, &mut args, n_pairs);
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Fold ext→ext on device buffers. Returns new CudaSlice on device.
    pub fn fold_ext_device(
        &self,
        d_data: &CudaSlice<u32>,
        n_pairs: u32,
        r_ext: &[u32; 5],
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            self.launch(self.fn_ext_half, &mut args, n_pairs);
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Fold ext→ext on device buffers with LSB mode (pairs at 2j, 2j+1).
    pub fn fold_ext_lsb_device(
        &self,
        d_data: &CudaSlice<u32>,
        n_pairs: u32,
        r_ext: &[u32; 5],
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            self.launch(self.fn_ext_lsb, &mut args, n_pairs);
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Fold ext→ext on device buffers with AtBit mode (pairs at stride 2^bit).
    pub fn fold_ext_at_bit_device(
        &self,
        d_data: &CudaSlice<u32>,
        n_pairs: u32,
        r_ext: &[u32; 5],
        bit: u32,
    ) -> CudaSlice<u32> {
        let d_r = self.stream.memcpy_stod(r_ext.as_slice()).unwrap();
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
                &bit as *const _ as *mut _,
            ];
            self.launch(self.fn_ext_at_bit, &mut args, n_pairs);
        }
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Fold base→ext on device buffers using a device-resident challenge.
    pub fn fold_base_to_ext_device_with_challenge<D, R>(
        &self,
        d_data: &D,
        n_pairs: u32,
        d_r: &R,
    ) -> CudaSlice<u32>
    where
        D: DevicePtr<u32>,
        R: DevicePtr<u32>,
    {
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        self.fold_base_to_ext_device_with_challenge_into_async(d_data, n_pairs, d_r, &mut d_out);
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Fold base→ext on device buffers using a device-resident challenge into
    /// a caller-provided output buffer. Safe to use during graph capture.
    pub fn fold_base_to_ext_device_with_challenge_into_async<D, R>(
        &self,
        d_data: &D,
        n_pairs: u32,
        d_r: &R,
        d_out: &mut CudaSlice<u32>,
    ) where
        D: DevicePtr<u32>,
        R: DevicePtr<u32>,
    {
        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            self.launch(self.fn_b2e_half, &mut args, n_pairs);
        }
    }

    /// Fold ext→ext on device buffers using a device-resident challenge.
    pub fn fold_ext_device_with_challenge<D, R>(
        &self,
        d_data: &D,
        n_pairs: u32,
        d_r: &R,
    ) -> CudaSlice<u32>
    where
        D: DevicePtr<u32>,
        R: DevicePtr<u32>,
    {
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        self.fold_ext_device_with_challenge_into_async(d_data, n_pairs, d_r, &mut d_out);
        self.stream.synchronize().unwrap();
        d_out
    }

    /// Fold ext→ext on device buffers using a device-resident challenge into
    /// a caller-provided output buffer. Safe to use during graph capture.
    pub fn fold_ext_device_with_challenge_into_async<D, R>(
        &self,
        d_data: &D,
        n_pairs: u32,
        d_r: &R,
        d_out: &mut CudaSlice<u32>,
    ) where
        D: DevicePtr<u32>,
        R: DevicePtr<u32>,
    {
        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            self.launch(self.fn_ext_half, &mut args, n_pairs);
        }
    }

    /// In-place multilinear evals -> coeffs transform for extension-field data.
    ///
    /// `d_data` stores `n_elements` quintic extension values as contiguous 5-word elements.
    /// The transform matches `backend::poly::evals_to_coeffs`.
    pub fn evals_to_coeffs_ext_in_place(&self, d_data: &mut CudaSlice<u32>, n_elements: u32) {
        self.evals_to_coeffs_ext_in_place_async(d_data, n_elements);
        self.stream.synchronize().unwrap();
    }

    /// Async in-place multilinear evals -> coeffs transform for extension-field data.
    pub fn evals_to_coeffs_ext_in_place_async(&self, d_data: &mut CudaSlice<u32>, n_elements: u32) {
        assert!(n_elements.is_power_of_two());
        let threads = 256u32;
        let n_pairs = n_elements >> 1;

        let mut half = 1u32;
        while half < n_elements {
            let blocks = (n_pairs + threads - 1) / threads;
            let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &half as *const _ as *mut _,
                &n_elements as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_evals_to_coeffs_ext_layer,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("evals_to_coeffs ext layer kernel failed");
            }
            half <<= 1;
        }

        if n_elements == 1 {
            return;
        }

        let log_n = n_elements.trailing_zeros();
        let blocks = (n_elements + threads - 1) / threads;
        let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &data_ptr as *const _ as *mut _,
            &log_n as *const _ as *mut _,
            &n_elements as *const _ as *mut _,
        ];
        unsafe {
            cuda_result::launch_kernel(
                self.fn_bit_reverse_ext,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("bit_reverse ext kernel failed");
        }
    }

    /// Async out-of-place multilinear evals -> coeffs transform for extension-field data.
    pub fn evals_to_coeffs_ext_device_async<D>(&self, d_data: &D, n_elements: u32) -> CudaSlice<u32>
    where
        D: DevicePtr<u32>,
    {
        assert!(n_elements.is_power_of_two());
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_elements as usize) * 5)
            .unwrap();
        let threads = 256u32;
        let first_blocks = ((n_elements >> 1).max(1) + threads - 1) / threads;
        {
            let (input_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_out.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &input_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &n_elements as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_evals_to_coeffs_ext_first_layer,
                    (first_blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("evals_to_coeffs ext first-layer kernel failed");
            }
        }

        if n_elements > 2 {
            let n_pairs = n_elements >> 1;
            let mut half = 2u32;
            while half < n_elements {
                let blocks = (n_pairs + threads - 1) / threads;
                let (data_ptr, _g1) = d_out.device_ptr_mut(&self.stream);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &data_ptr as *const _ as *mut _,
                    &half as *const _ as *mut _,
                    &n_elements as *const _ as *mut _,
                ];
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_evals_to_coeffs_ext_layer,
                        (blocks, 1, 1),
                        (threads, 1, 1),
                        0,
                        self.stream.cu_stream(),
                        &mut args,
                    )
                    .expect("evals_to_coeffs ext layer kernel failed");
                }
                half <<= 1;
            }
        }

        if n_elements > 1 {
            let log_n = n_elements.trailing_zeros();
            let blocks = (n_elements + threads - 1) / threads;
            let (data_ptr, _g1) = d_out.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &log_n as *const _ as *mut _,
                &n_elements as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_bit_reverse_ext,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("bit_reverse ext kernel failed");
            }
        }

        d_out
    }

    pub fn evals_to_coeffs_ext_device<D>(&self, d_data: &D, n_elements: u32) -> CudaSlice<u32>
    where
        D: DevicePtr<u32>,
    {
        let d_out = self.evals_to_coeffs_ext_device_async(d_data, n_elements);
        self.stream.synchronize().unwrap();
        d_out
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

// ── CPU reference implementations ────────────────────────────────────────

/// CPU fold base→base (LSB).
pub fn cpu_fold_base_lsb(data: &[u32], r: u32) -> Vec<u32> {
    let r_kb = unsafe { std::mem::transmute::<u32, KoalaBear>(r) };
    data.chunks_exact(2)
        .map(|c| {
            let lo = unsafe { std::mem::transmute::<u32, KoalaBear>(c[0]) };
            let hi = unsafe { std::mem::transmute::<u32, KoalaBear>(c[1]) };
            let res = lo + (hi - lo) * r_kb;
            unsafe { std::mem::transmute::<KoalaBear, u32>(res) }
        })
        .collect()
}

/// CPU fold base→base (half).
pub fn cpu_fold_base_half(data: &[u32], r: u32) -> Vec<u32> {
    let n = data.len() / 2;
    let r_kb = unsafe { std::mem::transmute::<u32, KoalaBear>(r) };
    (0..n)
        .map(|j| {
            let lo = unsafe { std::mem::transmute::<u32, KoalaBear>(data[j]) };
            let hi = unsafe { std::mem::transmute::<u32, KoalaBear>(data[j + n]) };
            let res = lo + (hi - lo) * r_kb;
            unsafe { std::mem::transmute::<KoalaBear, u32>(res) }
        })
        .collect()
}

/// CPU fold base→base (arbitrary bit).
pub fn cpu_fold_base_at_bit(data: &[u32], r: u32, bit: u32) -> Vec<u32> {
    let n = data.len() / 2;
    let stride = 1usize << bit;
    let lo_mask = stride - 1;
    let r_kb = unsafe { std::mem::transmute::<u32, KoalaBear>(r) };
    (0..n)
        .map(|j| {
            let i0 = ((j >> bit) << (bit as usize + 1)) | (j & lo_mask);
            let i1 = i0 | stride;
            let lo = unsafe { std::mem::transmute::<u32, KoalaBear>(data[i0]) };
            let hi = unsafe { std::mem::transmute::<u32, KoalaBear>(data[i1]) };
            let res = lo + (hi - lo) * r_kb;
            unsafe { std::mem::transmute::<KoalaBear, u32>(res) }
        })
        .collect()
}

/// CPU fold base→ext (LSB).
pub fn cpu_fold_base_to_ext_lsb(data: &[u32], r_ext: &[u32; 5]) -> Vec<u32> {
    let r = unsafe { std::mem::transmute::<[u32; 5], EF>(*r_ext) };
    data.chunks_exact(2)
        .flat_map(|c| {
            let lo = unsafe { std::mem::transmute::<u32, KoalaBear>(c[0]) };
            let hi = unsafe { std::mem::transmute::<u32, KoalaBear>(c[1]) };
            let diff: EF = (hi - lo).into();
            let res: EF = EF::from(lo) + diff * r;
            let arr: [u32; 5] = unsafe { std::mem::transmute(res) };
            arr
        })
        .collect()
}

/// CPU fold base→ext (half).
pub fn cpu_fold_base_to_ext_half(data: &[u32], r_ext: &[u32; 5]) -> Vec<u32> {
    let n = data.len() / 2;
    let r = unsafe { std::mem::transmute::<[u32; 5], EF>(*r_ext) };
    (0..n)
        .flat_map(|j| {
            let lo = unsafe { std::mem::transmute::<u32, KoalaBear>(data[j]) };
            let hi = unsafe { std::mem::transmute::<u32, KoalaBear>(data[j + n]) };
            let diff: EF = (hi - lo).into();
            let res: EF = EF::from(lo) + diff * r;
            let arr: [u32; 5] = unsafe { std::mem::transmute(res) };
            arr
        })
        .collect()
}

/// CPU fold base→ext (arbitrary bit).
pub fn cpu_fold_base_to_ext_at_bit(data: &[u32], r_ext: &[u32; 5], bit: u32) -> Vec<u32> {
    let n = data.len() / 2;
    let stride = 1usize << bit;
    let lo_mask = stride - 1;
    let r = unsafe { std::mem::transmute::<[u32; 5], EF>(*r_ext) };
    (0..n)
        .flat_map(|j| {
            let i0 = ((j >> bit) << (bit as usize + 1)) | (j & lo_mask);
            let i1 = i0 | stride;
            let lo = unsafe { std::mem::transmute::<u32, KoalaBear>(data[i0]) };
            let hi = unsafe { std::mem::transmute::<u32, KoalaBear>(data[i1]) };
            let diff: EF = (hi - lo).into();
            let res: EF = EF::from(lo) + diff * r;
            let arr: [u32; 5] = unsafe { std::mem::transmute(res) };
            arr
        })
        .collect()
}

/// CPU fold ext→ext (LSB).
pub fn cpu_fold_ext_lsb(data: &[u32], r_ext: &[u32; 5]) -> Vec<u32> {
    let r = unsafe { std::mem::transmute::<[u32; 5], EF>(*r_ext) };
    data.chunks_exact(10) // 2 ext elements = 10 u32s
        .flat_map(|c| {
            let lo = unsafe { std::mem::transmute::<[u32; 5], EF>(c[..5].try_into().unwrap()) };
            let hi = unsafe { std::mem::transmute::<[u32; 5], EF>(c[5..].try_into().unwrap()) };
            let res: EF = lo + (hi - lo) * r;
            let arr: [u32; 5] = unsafe { std::mem::transmute(res) };
            arr
        })
        .collect()
}

/// CPU fold ext→ext (half).
pub fn cpu_fold_ext_half(data: &[u32], r_ext: &[u32; 5]) -> Vec<u32> {
    let n_pairs = data.len() / 10;
    let r = unsafe { std::mem::transmute::<[u32; 5], EF>(*r_ext) };
    (0..n_pairs)
        .flat_map(|j| {
            let lo = unsafe {
                std::mem::transmute::<[u32; 5], EF>(data[j * 5..(j + 1) * 5].try_into().unwrap())
            };
            let hi = unsafe {
                std::mem::transmute::<[u32; 5], EF>(
                    data[(j + n_pairs) * 5..(j + n_pairs + 1) * 5]
                        .try_into()
                        .unwrap(),
                )
            };
            let res: EF = lo + (hi - lo) * r;
            let arr: [u32; 5] = unsafe { std::mem::transmute(res) };
            arr
        })
        .collect()
}

/// CPU fold ext→ext (arbitrary bit).
pub fn cpu_fold_ext_at_bit(data: &[u32], r_ext: &[u32; 5], bit: u32) -> Vec<u32> {
    let n_pairs = data.len() / 10;
    let stride = 1usize << bit;
    let lo_mask = stride - 1;
    let r = unsafe { std::mem::transmute::<[u32; 5], EF>(*r_ext) };
    (0..n_pairs)
        .flat_map(|j| {
            let i0 = ((j >> bit) << (bit as usize + 1)) | (j & lo_mask);
            let i1 = i0 | stride;
            let lo = unsafe {
                std::mem::transmute::<[u32; 5], EF>(data[i0 * 5..(i0 + 1) * 5].try_into().unwrap())
            };
            let hi = unsafe {
                std::mem::transmute::<[u32; 5], EF>(data[i1 * 5..(i1 + 1) * 5].try_into().unwrap())
            };
            let res: EF = lo + (hi - lo) * r;
            let arr: [u32; 5] = unsafe { std::mem::transmute(res) };
            arr
        })
        .collect()
}
