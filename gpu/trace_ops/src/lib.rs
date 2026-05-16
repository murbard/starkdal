//! GPU trace table operations (glue kernels).
//!
//! These kernels are computationally trivial but necessary to keep data on GPU
//! between heavy compute phases, avoiding PCIe round-trips.

use std::ffi::CString;
use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use field::PrimeField32;
use koala_bear::{KoalaBear, extension::QuinticExtensionField};

type EF = QuinticExtensionField<KoalaBear>;

fn kb(v: u32) -> KoalaBear {
    unsafe { std::mem::transmute(v) }
}
fn ef_u32(v: EF) -> [u32; 5] {
    unsafe { std::mem::transmute(v) }
}

pub struct GpuTraceOps {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    fn_access_count: cuda_sys::CUfunction,
    fn_access_count_simple: cuda_sys::CUfunction,
    fn_canonical_to_monty: cuda_sys::CUfunction,
    fn_fill_base: cuda_sys::CUfunction,
    fn_negate_base: cuda_sys::CUfunction,
    fn_negate_ext: cuda_sys::CUfunction,
    fn_fill_ext: cuda_sys::CUfunction,
    fn_fill_ext_offset: cuda_sys::CUfunction,
    fn_base_to_ext: cuda_sys::CUfunction,
    fn_copy_column: cuda_sys::CUfunction,
    fn_shift_down: cuda_sys::CUfunction,
    fn_shift_down_to_offset: cuda_sys::CUfunction,
    fn_bit_reverse: cuda_sys::CUfunction,
    fn_bit_reverse_ext: cuda_sys::CUfunction,
    fn_mle_fold_b2e: cuda_sys::CUfunction,
    fn_mle_fold_ext: cuda_sys::CUfunction,
}

unsafe impl Send for GpuTraceOps {}
unsafe impl Sync for GpuTraceOps {}

impl Drop for GpuTraceOps {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_result::module::unload(self.cu_module);
        }
    }
}

impl GpuTraceOps {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let cubin = include_bytes!(concat!(env!("OUT_DIR"), "/trace_ops.cubin"));
        let cu_module = unsafe { cuda_result::module::load_data(cubin.as_ptr().cast()) }
            .expect("failed to load trace_ops cubin");

        let load = |name: &str| {
            let c = CString::new(name).unwrap();
            unsafe { cuda_result::module::get_function(cu_module, c) }
                .unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };

        Self {
            stream,
            cu_module,
            fn_access_count: load("access_count_kernel"),
            fn_access_count_simple: load("access_count_simple_kernel"),
            fn_canonical_to_monty: load("canonical_to_monty_kernel"),
            fn_fill_base: load("fill_base_kernel"),
            fn_negate_base: load("negate_base_kernel"),
            fn_negate_ext: load("negate_ext_kernel"),
            fn_fill_ext: load("fill_ext_kernel"),
            fn_fill_ext_offset: load("fill_ext_offset_kernel"),
            fn_base_to_ext: load("base_to_ext_kernel"),
            fn_copy_column: load("copy_column_kernel"),
            fn_shift_down: load("shift_down_kernel"),
            fn_shift_down_to_offset: load("shift_down_to_offset_kernel"),
            fn_bit_reverse: load("bit_reverse_kernel"),
            fn_bit_reverse_ext: load("bit_reverse_ext_within_chunks_kernel"),
            fn_mle_fold_b2e: load("mle_fold_base_to_ext_kernel"),
            fn_mle_fold_ext: load("mle_fold_ext_kernel"),
        }
    }

    fn launch(&self, func: cuda_sys::CUfunction, args: &mut Vec<*mut std::ffi::c_void>, n: u32) {
        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;
        unsafe {
            cuda_result::launch_kernel(
                func,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                args,
            )
            .expect("kernel launch failed");
        }
    }

    /// Copy `n` elements from a device slice into `dst[dst_offset..]`.
    ///
    /// This keeps witness assembly on GPU when the source data is already
    /// resident on device.
    pub fn copy_to_offset_device<S, D>(&self, src: &S, dst: &mut D, n: u32, dst_offset: u32)
    where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(src.len() >= n as usize);
        assert!(dst.len() >= dst_offset as usize + n as usize);
        let (src_ptr, _g1) = src.device_ptr(&self.stream);
        let (dst_ptr, _g2) = dst.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &dst_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
            &dst_offset as *const _ as *mut _,
        ];
        self.launch(self.fn_copy_column, &mut args, n);
    }

    /// Histogram: for each column[i], increment acc[canonical(column[i]) + j]
    /// for `j in 0..n_values`. Both buffers stay on device.
    pub fn access_count_into_device<S, D>(&self, column: &S, acc: &mut D, n: u32, n_values: u32)
    where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(column.len() >= n as usize);
        let (col_ptr, _g1) = column.device_ptr(&self.stream);
        let (acc_ptr, _g2) = acc.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &col_ptr as *const _ as *mut _,
            &acc_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
            &n_values as *const _ as *mut _,
        ];
        self.launch(self.fn_access_count, &mut args, n);
    }

    /// Single-value histogram variant with device-resident input/output.
    pub fn access_count_simple_into_device<S, D>(&self, column: &S, acc: &mut D, n: u32)
    where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(column.len() >= n as usize);
        let (col_ptr, _g1) = column.device_ptr(&self.stream);
        let (acc_ptr, _g2) = acc.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &col_ptr as *const _ as *mut _,
            &acc_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];
        self.launch(self.fn_access_count_simple, &mut args, n);
    }

    /// Convert canonical u32 counts to KoalaBear Montgomery form on device.
    pub fn canonical_to_monty_device<S>(&self, src: &S, n: u32) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        assert!(src.len() >= n as usize);
        let mut d_out = self.stream.alloc_zeros::<u32>(n as usize).unwrap();
        let src_ptr = {
            let (p, _) = src.device_ptr(&self.stream);
            p
        };
        let out_ptr = {
            let (p, _) = d_out.device_ptr_mut(&self.stream);
            p
        };
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];
        self.launch(self.fn_canonical_to_monty, &mut args, n);
        d_out
    }

    pub fn fill_base(&self, value: u32, n: u32) -> CudaSlice<u32> {
        let mut d_out = self.stream.alloc_zeros::<u32>(n as usize).unwrap();
        let out_ptr = {
            let (p, _) = d_out.device_ptr_mut(&self.stream);
            p
        };
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &out_ptr as *const _ as *mut _,
            &value as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];
        self.launch(self.fn_fill_base, &mut args, n);
        d_out
    }

    pub fn negate_base<S>(&self, src: &S, n: u32) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        let mut d_out = self.stream.alloc_zeros::<u32>(n as usize).unwrap();
        self.negate_base_into(src, &mut d_out, n);
        d_out
    }

    pub fn negate_base_into<S, D>(&self, src: &S, dst: &mut D, n: u32)
    where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(src.len() >= n as usize);
        assert!(dst.len() >= n as usize);
        let (src_ptr, _) = src.device_ptr(&self.stream);
        let (out_ptr, _) = dst.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];
        self.launch(self.fn_negate_base, &mut args, n);
    }

    pub fn negate_ext<S>(&self, src: &S, n: u32) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        let mut d_out = self.stream.alloc_zeros::<u32>((n as usize) * 5).unwrap();
        self.negate_ext_into(src, &mut d_out, n);
        d_out
    }

    pub fn negate_ext_into<S, D>(&self, src: &S, dst: &mut D, n: u32)
    where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(src.len() >= (n as usize) * 5);
        assert!(dst.len() >= (n as usize) * 5);
        let (src_ptr, _) = src.device_ptr(&self.stream);
        let (out_ptr, _) = dst.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];
        self.launch(self.fn_negate_ext, &mut args, n);
    }

    pub fn fill_ext(&self, value: EF, n: u32) -> CudaSlice<u32> {
        let mut d_out = self.stream.alloc_zeros::<u32>((n as usize) * 5).unwrap();
        let value_words = ef_u32(value);
        let out_ptr = {
            let (p, _) = d_out.device_ptr_mut(&self.stream);
            p
        };
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &out_ptr as *const _ as *mut _,
            &value_words[0] as *const _ as *mut _,
            &value_words[1] as *const _ as *mut _,
            &value_words[2] as *const _ as *mut _,
            &value_words[3] as *const _ as *mut _,
            &value_words[4] as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];
        self.launch(self.fn_fill_ext, &mut args, n);
        d_out
    }

    pub fn fill_ext_at_offset<D>(&self, dst: &mut D, value: EF, n: u32, dst_offset: u32)
    where
        D: DevicePtrMut<u32>,
    {
        if n == 0 {
            return;
        }
        assert!(dst.len() >= ((dst_offset + n) as usize) * 5);
        let value_words = ef_u32(value);
        let dst_ptr = {
            let (p, _) = dst.device_ptr_mut(&self.stream);
            p
        };
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &dst_ptr as *const _ as *mut _,
            &value_words[0] as *const _ as *mut _,
            &value_words[1] as *const _ as *mut _,
            &value_words[2] as *const _ as *mut _,
            &value_words[3] as *const _ as *mut _,
            &value_words[4] as *const _ as *mut _,
            &n as *const _ as *mut _,
            &dst_offset as *const _ as *mut _,
        ];
        self.launch(self.fn_fill_ext_offset, &mut args, n);
    }

    pub fn base_to_ext<S>(&self, src: &S, n: u32) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        let mut d_out = self.stream.alloc_zeros::<u32>((n as usize) * 5).unwrap();
        self.base_to_ext_into(src, n, &mut d_out);
        d_out
    }

    pub fn base_to_ext_into<S, D>(&self, src: &S, n: u32, d_out: &mut D)
    where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(src.len() >= n as usize);
        assert!(d_out.len() >= (n as usize) * 5);
        let src_ptr = {
            let (p, _) = src.device_ptr(&self.stream);
            p
        };
        let out_ptr = {
            let (p, _) = d_out.device_ptr_mut(&self.stream);
            p
        };
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];
        self.launch(self.fn_base_to_ext, &mut args, n);
    }

    pub fn bit_reverse_ext_within_chunks<S>(
        &self,
        src: &S,
        n_elems: u32,
        chunk_log: u32,
    ) -> CudaSlice<u32>
    where
        S: DevicePtr<u32>,
    {
        let mut d_out = self
            .stream
            .alloc_zeros::<u32>((n_elems as usize) * 5)
            .unwrap();
        self.bit_reverse_ext_within_chunks_into(src, &mut d_out, n_elems, chunk_log);
        d_out
    }

    pub fn bit_reverse_ext_within_chunks_into<S, D>(
        &self,
        src: &S,
        dst: &mut D,
        n_elems: u32,
        chunk_log: u32,
    ) where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        assert!(src.len() >= (n_elems as usize) * 5);
        assert!(dst.len() >= (n_elems as usize) * 5);
        let (src_ptr, _) = src.device_ptr(&self.stream);
        let (out_ptr, _) = dst.device_ptr_mut(&self.stream);
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &out_ptr as *const _ as *mut _,
            &n_elems as *const _ as *mut _,
            &chunk_log as *const _ as *mut _,
        ];
        self.launch(self.fn_bit_reverse_ext, &mut args, n_elems);
    }

    /// Shift a device-resident column into `dst[dst_offset..]`.
    ///
    /// This matches AIR "down" semantics: `dst[i] = src[i + 1]` and the final
    /// row repeats `src[n - 1]`.
    pub fn shift_down_to_offset_device<S, D>(&self, src: &S, dst: &mut D, n: u32, dst_offset: u32)
    where
        S: DevicePtr<u32>,
        D: DevicePtrMut<u32>,
    {
        if n == 0 {
            return;
        }
        assert!(src.len() >= n as usize);
        assert!(dst.len() >= dst_offset as usize + n as usize);
        let src_ptr = {
            let (p, _) = src.device_ptr(&self.stream);
            p
        };
        let dst_ptr = {
            let (p, _) = dst.device_ptr_mut(&self.stream);
            p
        };
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &src_ptr as *const _ as *mut _,
            &dst_ptr as *const _ as *mut _,
            &n as *const _ as *mut _,
            &dst_offset as *const _ as *mut _,
        ];
        self.launch(self.fn_shift_down_to_offset, &mut args, n);
    }

    /// Histogram: for each column[i], increment acc[canonical(column[i])].
    /// Returns plain u32 counts (NOT Montgomery form).
    pub fn access_count_simple(&self, column: &[u32], acc_size: usize) -> Vec<u32> {
        let n = column.len() as u32;
        let d_col = self.stream.memcpy_stod(column).unwrap();
        let mut d_acc = self.stream.alloc_zeros::<u32>(acc_size).unwrap();
        {
            let (col_ptr, _g1) = d_col.device_ptr(&self.stream);
            let (acc_ptr, _g2) = d_acc.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &col_ptr as *const _ as *mut _,
                &acc_ptr as *const _ as *mut _,
                &n as *const _ as *mut _,
            ];
            self.launch(self.fn_access_count_simple, &mut args, n);
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_acc).unwrap()
    }

    /// Shift down: dst[i] = src[i+1], dst[n-1] = src[n-1].
    pub fn shift_down(&self, data: &[u32]) -> Vec<u32> {
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
            ];
            self.launch(self.fn_shift_down, &mut args, n);
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_dst).unwrap()
    }

    /// In-place bit-reversal permutation.
    pub fn bit_reverse(&self, data: &mut [u32]) {
        let n = data.len();
        assert!(n.is_power_of_two());
        let log_n = n.trailing_zeros();
        let n_u32 = n as u32;

        let mut d_data = self.stream.memcpy_stod(data).unwrap();
        {
            let (data_ptr, _g1) = d_data.device_ptr_mut(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &log_n as *const _ as *mut _,
                &n_u32 as *const _ as *mut _,
            ];
            self.launch(self.fn_bit_reverse, &mut args, n_u32);
        }
        self.stream.synchronize().unwrap();
        let result = self.stream.memcpy_dtov(&d_data).unwrap();
        data.copy_from_slice(&result);
    }

    /// MLE evaluation: evaluate multilinear polynomial at a point.
    /// `data` is base field evals (2^log_n elements), `point` is ext field (log_n × 5 u32s).
    /// Returns a single extension field element (5 u32s).
    pub fn mle_eval(&self, data: &[u32], point: &[[u32; 5]]) -> [u32; 5] {
        let log_n = point.len();
        assert_eq!(data.len(), 1 << log_n);

        if log_n == 0 {
            // Constant polynomial: embed base element into ext.
            return [data[0], 0, 0, 0, 0];
        }

        // First fold: base → ext with point[0].
        let n_pairs = (data.len() / 2) as u32;
        let d_data = self.stream.memcpy_stod(data).unwrap();
        let d_r = self.stream.memcpy_stod(&point[0]).unwrap();
        let mut d_ext = self
            .stream
            .alloc_zeros::<u32>((n_pairs as usize) * 5)
            .unwrap();
        {
            let (data_ptr, _g1) = d_data.device_ptr(&self.stream);
            let (out_ptr, _g2) = d_ext.device_ptr_mut(&self.stream);
            let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &data_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &r_ptr as *const _ as *mut _,
                &n_pairs as *const _ as *mut _,
            ];
            self.launch(self.fn_mle_fold_b2e, &mut args, n_pairs);
        }
        self.stream.synchronize().unwrap();

        // Subsequent folds: ext → ext.
        let mut current = d_ext;
        let mut cur_n = n_pairs as usize;
        for round in 1..log_n {
            let np = (cur_n / 2) as u32;
            let d_r = self.stream.memcpy_stod(&point[round]).unwrap();
            let mut d_next = self.stream.alloc_zeros::<u32>((np as usize) * 5).unwrap();
            {
                let (data_ptr, _g1) = current.device_ptr(&self.stream);
                let (out_ptr, _g2) = d_next.device_ptr_mut(&self.stream);
                let (r_ptr, _g3) = d_r.device_ptr(&self.stream);
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &data_ptr as *const _ as *mut _,
                    &out_ptr as *const _ as *mut _,
                    &r_ptr as *const _ as *mut _,
                    &np as *const _ as *mut _,
                ];
                self.launch(self.fn_mle_fold_ext, &mut args, np);
            }
            self.stream.synchronize().unwrap();
            current = d_next;
            cur_n = np as usize;
        }

        let result = self.stream.memcpy_dtov(&current).unwrap();
        result[..5].try_into().unwrap()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

// ── CPU references ───────────────────────────────────────────────────────

/// Returns plain u32 counts (NOT Montgomery form), matching the GPU kernel.
pub fn cpu_access_count_simple(column: &[u32], acc_size: usize) -> Vec<u32> {
    let mut acc = vec![0u32; acc_size];
    for &v in column {
        let addr = kb(v).as_canonical_u32() as usize;
        acc[addr] += 1;
    }
    acc
}

pub fn cpu_shift_down(data: &[u32]) -> Vec<u32> {
    let n = data.len();
    let mut out = vec![0u32; n];
    for i in 0..n - 1 {
        out[i] = data[i + 1];
    }
    out[n - 1] = data[n - 1];
    out
}

pub fn cpu_bit_reverse(data: &mut [u32]) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;
    let shift = usize::BITS as usize - log_n;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            data.swap(i, j);
        }
    }
}

pub fn cpu_mle_eval(data: &[u32], point: &[[u32; 5]]) -> [u32; 5] {
    let log_n = point.len();
    assert_eq!(data.len(), 1 << log_n);

    if log_n == 0 {
        return [data[0], 0, 0, 0, 0];
    }

    // First fold: base → ext with point[0].
    let r0: EF = unsafe { std::mem::transmute(point[0]) };
    let mut current: Vec<[u32; 5]> = data
        .chunks_exact(2)
        .map(|c| {
            let lo = kb(c[0]);
            let hi = kb(c[1]);
            let diff: EF = (hi - lo).into();
            let res: EF = EF::from(lo) + diff * r0;
            unsafe { std::mem::transmute(res) }
        })
        .collect();

    // Subsequent folds: ext → ext.
    for round in 1..log_n {
        let r: EF = unsafe { std::mem::transmute(point[round]) };
        current = current
            .chunks_exact(2)
            .map(|c| {
                let lo: EF = unsafe { std::mem::transmute(c[0]) };
                let hi: EF = unsafe { std::mem::transmute(c[1]) };
                let res: EF = lo + (hi - lo) * r;
                unsafe { std::mem::transmute(res) }
            })
            .collect();
    }

    current[0]
}
