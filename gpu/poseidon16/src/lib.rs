//! GPU-accelerated Poseidon16 on KoalaBear.
//!
//! Provides a CUDA kernel for batch `poseidon16_compress` and property-based
//! testing against the CPU reference from the `backend` crate.

use std::ffi::CString;
use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use field::PrimeField32;
use koala_bear::symmetric::Permutation;
use koala_bear::{
    KoalaBear, POSEIDON1_WIDTH, default_koalabear_poseidon1_16, poseidon1_round_constants,
    poseidon1_sparse_first_round_constants, poseidon1_sparse_first_row, poseidon1_sparse_m_i,
    poseidon1_sparse_scalar_round_constants, poseidon1_sparse_v,
};

const WIDTH: usize = POSEIDON1_WIDTH;

fn kb_slice_as_u32(slice: &[KoalaBear]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u32>(), slice.len()) }
}

/// Holds a loaded CUDA module with Poseidon16 kernels and pre-uploaded constants.
///
/// Uses raw CUDA driver API for the module so we can write to __constant__ memory
/// and load functions from the same module instance.
pub struct GpuPoseidon16 {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    compress_fn: cuda_sys::CUfunction,
    permute_fn: cuda_sys::CUfunction,
}

unsafe impl Send for GpuPoseidon16 {}
unsafe impl Sync for GpuPoseidon16 {}

impl Drop for GpuPoseidon16 {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_result::module::unload(self.cu_module);
        }
    }
}

impl GpuPoseidon16 {
    /// Load the PTX module and upload all Poseidon16 constants to device constant memory.
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let cubin = include_bytes!(concat!(env!("OUT_DIR"), "/poseidon16.cubin"));

        let cu_module = unsafe { cuda_result::module::load_data(cubin.as_ptr().cast()) }
            .expect("failed to load poseidon16 cubin");

        let compress_fn = {
            let name = CString::new("poseidon16_compress_kernel").unwrap();
            unsafe { cuda_result::module::get_function(cu_module, name) }
                .expect("compress kernel not found")
        };
        let permute_fn = {
            let name = CString::new("poseidon16_permute_kernel").unwrap();
            unsafe { cuda_result::module::get_function(cu_module, name) }
                .expect("permute kernel not found")
        };

        let mut this = Self {
            stream,
            cu_module,
            compress_fn,
            permute_fn,
        };
        this.upload_constants();
        this
    }

    fn upload_constants(&mut self) {
        let rc = poseidon1_round_constants();
        let rc_flat: Vec<u32> = rc
            .iter()
            .flat_map(|row| kb_slice_as_u32(row))
            .copied()
            .collect();
        self.copy_to_symbol("d_rc", &rc_flat);

        let mds_col_kb =
            KoalaBear::new_array([1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1]);
        let mds_col = kb_slice_as_u32(&mds_col_kb);
        let mut mds_flat = vec![0u32; WIDTH * WIDTH];
        for i in 0..WIDTH {
            for j in 0..WIDTH {
                mds_flat[i * WIDTH + j] = mds_col[(WIDTH + i - j) % WIDTH];
            }
        }
        self.copy_to_symbol("d_mds", &mds_flat);

        let first_rc = kb_slice_as_u32(poseidon1_sparse_first_round_constants());
        self.copy_to_symbol("d_sparse_first_rc", first_rc);

        let m_i = poseidon1_sparse_m_i();
        let m_i_flat: Vec<u32> = m_i
            .iter()
            .flat_map(|row| kb_slice_as_u32(row))
            .copied()
            .collect();
        self.copy_to_symbol("d_sparse_m_i", &m_i_flat);

        let first_rows = poseidon1_sparse_first_row();
        let fr_flat: Vec<u32> = first_rows
            .iter()
            .flat_map(|row| kb_slice_as_u32(row))
            .copied()
            .collect();
        self.copy_to_symbol("d_sparse_first_row", &fr_flat);

        let v_vecs = poseidon1_sparse_v();
        let v_flat: Vec<u32> = v_vecs
            .iter()
            .flat_map(|row| kb_slice_as_u32(row))
            .copied()
            .collect();
        self.copy_to_symbol("d_sparse_v", &v_flat);

        let scalar_rc = poseidon1_sparse_scalar_round_constants();
        let src_flat: Vec<u32> = scalar_rc
            .iter()
            .map(|c| kb_slice_as_u32(std::slice::from_ref(c))[0])
            .collect();
        self.copy_to_symbol("d_sparse_scalar_rc", &src_flat);
    }

    fn copy_to_symbol(&self, name: &str, data: &[u32]) {
        let c_name = CString::new(name).unwrap();
        unsafe {
            let mut dptr: cuda_sys::CUdeviceptr = 0;
            let mut size: usize = 0;
            cuda_sys::cuModuleGetGlobal_v2(&mut dptr, &mut size, self.cu_module, c_name.as_ptr())
                .result()
                .unwrap_or_else(|e| panic!("cuModuleGetGlobal({name}): {e:?}"));

            let data_bytes = data.len() * std::mem::size_of::<u32>();
            assert!(
                data_bytes <= size,
                "constant {name}: data {data_bytes} bytes > symbol {size} bytes"
            );
            cuda_result::memcpy_htod_sync(dptr, data)
                .unwrap_or_else(|e| panic!("memcpy_htod({name}): {e:?}"));
        }
    }

    /// Batch compress: (perm(state) + state)[0..8] for each of n states.
    pub fn compress_batch(&self, input: &CudaSlice<u32>, n: u32) -> CudaSlice<u32> {
        let mut output = self.stream.alloc_zeros::<u32>((n as usize) * 8).unwrap();

        let threads_per_block = 256u32;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        self.launch_kernel(
            self.compress_fn,
            blocks,
            threads_per_block,
            input,
            &mut output,
            n,
        );
        output
    }

    /// Batch permute: full 16-element permutation for each of n states.
    pub fn permute_batch(&self, input: &CudaSlice<u32>, n: u32) -> CudaSlice<u32> {
        let mut output = self.stream.alloc_zeros::<u32>((n as usize) * 16).unwrap();

        let threads_per_block = 256u32;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        self.launch_kernel(
            self.permute_fn,
            blocks,
            threads_per_block,
            input,
            &mut output,
            n,
        );
        output
    }

    fn launch_kernel(
        &self,
        func: cuda_sys::CUfunction,
        blocks: u32,
        threads: u32,
        input: &CudaSlice<u32>,
        output: &mut CudaSlice<u32>,
        n: u32,
    ) {
        // Get raw device pointers via the DevicePtr/DevicePtrMut traits.
        let (in_dptr, _in_guard) = input.device_ptr(&self.stream);
        let (out_dptr, _out_guard) = output.device_ptr_mut(&self.stream);

        // Kernel arguments: pointers to device pointers + scalar n.
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &in_dptr as *const cuda_sys::CUdeviceptr as *mut std::ffi::c_void,
            &out_dptr as *const cuda_sys::CUdeviceptr as *mut std::ffi::c_void,
            &n as *const u32 as *mut std::ffi::c_void,
        ];

        unsafe {
            cuda_result::launch_kernel(
                func,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("kernel launch failed");
        }
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

// ── CPU reference ────────────────────────────────────────────────────────

pub fn cpu_permute(state: &mut [u32; 16]) {
    let p = default_koalabear_poseidon1_16();
    let kb_state: &mut [KoalaBear; 16] =
        unsafe { &mut *(state as *mut [u32; 16] as *mut [KoalaBear; 16]) };
    p.permute_mut(kb_state);
}

pub fn cpu_compress(input: &[u32; 16]) -> [u32; 8] {
    let p = default_koalabear_poseidon1_16();
    let mut kb_state: [KoalaBear; 16] = unsafe { std::mem::transmute(*input) };
    p.compress_in_place(&mut kb_state);
    let out: [u32; 16] = unsafe { std::mem::transmute(kb_state) };
    out[..8].try_into().unwrap()
}

pub fn from_monty(x: u32) -> u32 {
    let kb = unsafe { std::mem::transmute::<u32, KoalaBear>(x) };
    kb.as_canonical_u32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_reference_known_vector() {
        let p = default_koalabear_poseidon1_16();
        let mut input: [KoalaBear; 16] =
            KoalaBear::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        p.permute_mut(&mut input);
        let vals: Vec<u32> = input.iter().map(|x| x.as_canonical_u32()).collect();
        assert_eq!(
            vals,
            vec![
                610090613, 935319874, 1893335292, 796792199, 356405232, 552237741, 55134556,
                1215104204, 1823723405, 1133298033, 1780633798, 1453946561, 710069176, 1128629550,
                1917333254, 1175481618,
            ]
        );
    }

    #[test]
    fn test_cpu_compress_via_u32() {
        let input_kb = KoalaBear::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let input_u32: [u32; 16] = unsafe { std::mem::transmute(input_kb) };

        let p = default_koalabear_poseidon1_16();
        let mut state = input_kb;
        p.compress_in_place(&mut state);
        let expected: [u32; 16] = unsafe { std::mem::transmute(state) };

        let got = cpu_compress(&input_u32);
        assert_eq!(got, expected[..8]);
    }
}
