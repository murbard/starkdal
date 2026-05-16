//! GPU Logup fingerprint computation and endianness reorder.

use std::ffi::CString;
use std::sync::Arc;

use koala_bear::{KoalaBear, extension::QuinticExtensionField};
use cudarc::driver::safe::{CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};

type EF = QuinticExtensionField<KoalaBear>;

fn kb(v: u32) -> KoalaBear { unsafe { std::mem::transmute(v) } }
fn kb_u32(v: KoalaBear) -> u32 { unsafe { std::mem::transmute(v) } }

pub struct GpuLogup {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    fn_fingerprint: cuda_sys::CUfunction,
    fn_reorder: cuda_sys::CUfunction,
    fn_reorder_ext: cuda_sys::CUfunction,
}

unsafe impl Send for GpuLogup {}
unsafe impl Sync for GpuLogup {}

impl Drop for GpuLogup {
    fn drop(&mut self) {
        unsafe { let _ = cuda_result::module::unload(self.cu_module); }
    }
}

impl GpuLogup {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let cubin = include_bytes!(concat!(env!("OUT_DIR"), "/logup.cubin"));
        let cu_module = unsafe { cuda_result::module::load_data(cubin.as_ptr().cast()) }
            .expect("failed to load logup cubin");

        let load = |name: &str| {
            let c = CString::new(name).unwrap();
            unsafe { cuda_result::module::get_function(cu_module, c) }
                .unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };

        Self {
            stream,
            cu_module,
            fn_fingerprint: load("fingerprint_kernel"),
            fn_reorder: load("endianness_reorder_kernel"),
            fn_reorder_ext: load("endianness_reorder_ext_kernel"),
        }
    }

    fn launch(&self, func: cuda_sys::CUfunction, args: &mut Vec<*mut std::ffi::c_void>, n: u32) {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        unsafe {
            cuda_result::launch_kernel(
                func, (blocks, 1, 1), (threads, 1, 1), 0,
                self.stream.cu_stream(), args,
            ).expect("kernel launch failed");
        }
    }

    /// Compute fingerprints: denom[i] = c - Σ_j columns[j][i] * alphas[j].
    ///
    /// `columns_flat`: col-major, n_cols * n_rows base field elements.
    /// `alphas`: n_cols ext field challenges (n_cols * 5 u32s).
    /// `c`: ext field challenge (5 u32s).
    ///
    /// Returns n_rows ext field elements (n_rows * 5 u32s).
    pub fn fingerprint(
        &self,
        columns_flat: &[u32],
        alphas: &[u32],
        c: &[u32; 5],
        n_rows: u32,
        n_cols: u32,
    ) -> Vec<u32> {
        assert_eq!(columns_flat.len(), (n_rows as usize) * (n_cols as usize));
        assert_eq!(alphas.len(), (n_cols as usize) * 5);

        let d_cols = self.stream.memcpy_stod(columns_flat).unwrap();
        let d_alphas = self.stream.memcpy_stod(alphas).unwrap();
        let d_c = self.stream.memcpy_stod(c.as_slice()).unwrap();
        let mut d_out = self.stream.alloc_zeros::<u32>((n_rows as usize) * 5).unwrap();

        {
            let (cols_ptr, _g1) = d_cols.device_ptr(&self.stream);
            let (alphas_ptr, _g2) = d_alphas.device_ptr(&self.stream);
            let (c_ptr, _g3) = d_c.device_ptr(&self.stream);
            let (out_ptr, _g4) = d_out.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &cols_ptr as *const _ as *mut _,
                &alphas_ptr as *const _ as *mut _,
                &c_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &n_rows as *const _ as *mut _,
                &n_cols as *const _ as *mut _,
            ];
            self.launch(self.fn_fingerprint, &mut args, n_rows);
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_out).unwrap()
    }

    /// Endianness reorder (bit-reversal within chunks) for base field.
    pub fn endianness_reorder(&self, data: &[u32], chunk_log: u32) -> Vec<u32> {
        let n = data.len() as u32;
        let d_src = self.stream.memcpy_stod(data).unwrap();
        let mut d_dst = self.stream.alloc_zeros::<u32>(data.len()).unwrap();
        {
            let (src_ptr, _g1) = d_src.device_ptr(&self.stream);
            let (dst_ptr, _g2) = d_dst.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &src_ptr as *const _ as *mut _,
                &dst_ptr as *const _ as *mut _,
                &n as *const _ as *mut _,
                &chunk_log as *const _ as *mut _,
            ];
            self.launch(self.fn_reorder, &mut args, n);
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_dst).unwrap()
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
}

// ── CPU references ───────────────────────────────────────────────────────

pub fn cpu_fingerprint(
    columns_flat: &[u32],
    alphas: &[u32],
    c: &[u32; 5],
    n_rows: usize,
    n_cols: usize,
) -> Vec<u32> {
    let c_ef: EF = unsafe { std::mem::transmute(*c) };
    let mut out = Vec::with_capacity(n_rows * 5);
    for row in 0..n_rows {
        let mut fp = EF::default();
        for col in 0..n_cols {
            let val = kb(columns_flat[col * n_rows + row]);
            let alpha: EF = unsafe {
                std::mem::transmute::<[u32; 5], EF>(alphas[col * 5..(col + 1) * 5].try_into().unwrap())
            };
            fp = fp + alpha * val;
        }
        let denom = c_ef - fp;
        let arr: [u32; 5] = unsafe { std::mem::transmute(denom) };
        out.extend_from_slice(&arr);
    }
    out
}

pub fn cpu_endianness_reorder(data: &[u32], chunk_log: u32) -> Vec<u32> {
    let n = data.len();
    let mask = (1usize << chunk_log) - 1;
    let shift = usize::BITS as usize - chunk_log as usize;
    let mut out = vec![0u32; n];
    for idx in 0..n {
        let lo = idx & mask;
        let hi = idx & !mask;
        let rev = lo.reverse_bits() >> shift;
        let src_idx = hi | rev;
        out[idx] = data[src_idx];
    }
    out
}
