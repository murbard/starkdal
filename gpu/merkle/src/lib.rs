//! GPU Merkle tree construction for WHIR.
//!
//! Two phases:
//! 1. Leaf hashing: Poseidon16 sponge (RTL) over each row → 8-element digest
//! 2. Binary reduction: compress pairs of sibling digests up to root

use std::ffi::CString;
use std::sync::Arc;

use koala_bear::{
    KoalaBear, POSEIDON1_WIDTH, default_koalabear_poseidon1_16,
    poseidon1_round_constants,
    poseidon1_sparse_first_round_constants, poseidon1_sparse_first_row, poseidon1_sparse_m_i,
    poseidon1_sparse_scalar_round_constants, poseidon1_sparse_v,
};
use koala_bear::symmetric::Permutation;
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::{result as cuda_result, sys as cuda_sys};

pub const DIGEST_ELEMS: usize = 8;
const WIDTH: usize = POSEIDON1_WIDTH;

fn kb_slice_as_u32(slice: &[KoalaBear]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u32>(), slice.len()) }
}

pub struct GpuMerkle {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    fn_leaf_hash: cuda_sys::CUfunction,
    fn_reduce: cuda_sys::CUfunction,
}

unsafe impl Send for GpuMerkle {}
unsafe impl Sync for GpuMerkle {}

impl Drop for GpuMerkle {
    fn drop(&mut self) {
        unsafe { let _ = cuda_result::module::unload(self.cu_module); }
    }
}

impl GpuMerkle {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/merkle.ptx"));
        let c_src = CString::new(ptx_src).unwrap();
        let cu_module = unsafe { cuda_result::module::load_data(c_src.as_ptr().cast()) }
            .expect("failed to load merkle PTX");

        let load = |name: &str| {
            let c = CString::new(name).unwrap();
            unsafe { cuda_result::module::get_function(cu_module, c) }
                .unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };

        let mut this = Self {
            stream,
            cu_module,
            fn_leaf_hash: load("merkle_leaf_hash_kernel"),
            fn_reduce: load("merkle_reduce_kernel"),
        };
        this.upload_poseidon_constants();
        this
    }

    fn upload_poseidon_constants(&mut self) {
        let rc = poseidon1_round_constants();
        let rc_flat: Vec<u32> = rc.iter().flat_map(|r| kb_slice_as_u32(r)).copied().collect();
        self.copy_sym("d_rc", &rc_flat);

        let mds_col_kb = KoalaBear::new_array([1,3,13,22,67,2,15,63,101,1,2,17,11,1,51,1]);
        let mds_col = kb_slice_as_u32(&mds_col_kb);
        let mut mds_flat = vec![0u32; WIDTH * WIDTH];
        for i in 0..WIDTH { for j in 0..WIDTH { mds_flat[i*WIDTH+j] = mds_col[(WIDTH+i-j)%WIDTH]; } }
        self.copy_sym("d_mds", &mds_flat);

        self.copy_sym("d_sparse_first_rc", kb_slice_as_u32(poseidon1_sparse_first_round_constants()));
        let m_i: Vec<u32> = poseidon1_sparse_m_i().iter().flat_map(|r| kb_slice_as_u32(r)).copied().collect();
        self.copy_sym("d_sparse_m_i", &m_i);
        let fr: Vec<u32> = poseidon1_sparse_first_row().iter().flat_map(|r| kb_slice_as_u32(r)).copied().collect();
        self.copy_sym("d_sparse_first_row", &fr);
        let v: Vec<u32> = poseidon1_sparse_v().iter().flat_map(|r| kb_slice_as_u32(r)).copied().collect();
        self.copy_sym("d_sparse_v", &v);
        let src: Vec<u32> = poseidon1_sparse_scalar_round_constants().iter()
            .map(|c| kb_slice_as_u32(std::slice::from_ref(c))[0]).collect();
        self.copy_sym("d_sparse_scalar_rc", &src);
    }

    fn copy_sym(&self, name: &str, data: &[u32]) {
        let c_name = CString::new(name).unwrap();
        unsafe {
            let mut dptr: cuda_sys::CUdeviceptr = 0;
            let mut size: usize = 0;
            cuda_sys::cuModuleGetGlobal_v2(&mut dptr, &mut size, self.cu_module, c_name.as_ptr())
                .result().unwrap_or_else(|e| panic!("cuModuleGetGlobal({name}): {e:?}"));
            assert!(data.len() * 4 <= size);
            cuda_result::memcpy_htod_sync(dptr, data).unwrap();
        }
    }

    /// Build a full Merkle tree from a row-major matrix on the host.
    ///
    /// Returns `(root, all_digests)` where `all_digests[layer][node]` is the
    /// 8-element digest. Layer 0 = leaves, last layer = root.
    pub fn build_tree(
        &self,
        matrix: &[u32],
        height: u32,
        row_width: u32,
        row_stride: u32,
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        let d_matrix = self.stream.memcpy_stod(matrix).unwrap();

        // Phase 1: leaf hashing.
        let mut d_digests = self.stream.alloc_zeros::<u32>((height as usize) * 8).unwrap();
        {
            let (mat_ptr, _g1) = d_matrix.device_ptr(&self.stream);
            let (dig_ptr, _g2) = d_digests.device_ptr_mut(&self.stream);
            let threads = 256u32;
            let blocks = (height + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mat_ptr as *const _ as *mut _,
                &dig_ptr as *const _ as *mut _,
                &height as *const _ as *mut _,
                &row_width as *const _ as *mut _,
                &row_stride as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_leaf_hash, (blocks, 1, 1), (threads, 1, 1),
                    0, self.stream.cu_stream(), &mut args,
                ).expect("leaf hash kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        let mut layers = Vec::new();
        let leaf_digests = self.stream.memcpy_dtov(&d_digests).unwrap();
        layers.push(leaf_digests);

        // Phase 2: binary reduction.
        let mut current = d_digests;
        let mut n = height;
        while n > 1 {
            let n_pairs = n / 2;
            let mut d_parents = self.stream.alloc_zeros::<u32>((n_pairs as usize) * 8).unwrap();
            {
                let (child_ptr, _g1) = current.device_ptr(&self.stream);
                let (parent_ptr, _g2) = d_parents.device_ptr_mut(&self.stream);
                let threads = 256u32;
                let blocks = (n_pairs + threads - 1) / threads;
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &child_ptr as *const _ as *mut _,
                    &parent_ptr as *const _ as *mut _,
                    &n_pairs as *const _ as *mut _,
                ];
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_reduce, (blocks, 1, 1), (threads, 1, 1),
                        0, self.stream.cu_stream(), &mut args,
                    ).expect("reduce kernel failed");
                }
            }
            self.stream.synchronize().unwrap();

            let layer_digests = self.stream.memcpy_dtov(&d_parents).unwrap();
            layers.push(layer_digests);
            current = d_parents;
            n = n_pairs;
        }

        let root = layers.last().unwrap().clone();
        (root, layers)
    }

    /// Build Merkle tree from device-resident data (no htod copy needed).
    pub fn build_tree_from_device(
        &self,
        d_matrix: &CudaSlice<u32>,
        height: u32,
        row_width: u32,
        row_stride: u32,
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        let mut d_digests = self
            .stream
            .alloc_zeros::<u32>((height as usize) * 8)
            .unwrap();

        {
            let (mat_ptr, _g1) = d_matrix.device_ptr(&self.stream);
            let (dig_ptr, _g2) = d_digests.device_ptr_mut(&self.stream);
            let threads = 256u32;
            let blocks = (height + threads - 1) / threads;
            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &mat_ptr as *const _ as *mut _,
                &dig_ptr as *const _ as *mut _,
                &height as *const _ as *mut _,
                &row_width as *const _ as *mut _,
                &row_stride as *const _ as *mut _,
            ];
            unsafe {
                cuda_result::launch_kernel(
                    self.fn_leaf_hash,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("leaf hash kernel failed");
            }
        }
        self.stream.synchronize().unwrap();

        let mut layers = Vec::new();
        let leaf_digests = self.stream.memcpy_dtov(&d_digests).unwrap();
        layers.push(leaf_digests);

        let mut current = d_digests;
        let mut n = height;
        while n > 1 {
            let n_pairs = n / 2;
            let mut d_parents = self
                .stream
                .alloc_zeros::<u32>((n_pairs as usize) * 8)
                .unwrap();
            {
                let (child_ptr, _g1) = current.device_ptr(&self.stream);
                let (parent_ptr, _g2) = d_parents.device_ptr_mut(&self.stream);
                let threads = 256u32;
                let blocks = (n_pairs + threads - 1) / threads;
                let mut args: Vec<*mut std::ffi::c_void> = vec![
                    &child_ptr as *const _ as *mut _,
                    &parent_ptr as *const _ as *mut _,
                    &n_pairs as *const _ as *mut _,
                ];
                unsafe {
                    cuda_result::launch_kernel(
                        self.fn_reduce,
                        (blocks, 1, 1),
                        (threads, 1, 1),
                        0,
                        self.stream.cu_stream(),
                        &mut args,
                    )
                    .expect("reduce kernel failed");
                }
            }
            self.stream.synchronize().unwrap();

            let layer_digests = self.stream.memcpy_dtov(&d_parents).unwrap();
            layers.push(layer_digests);
            current = d_parents;
            n = n_pairs;
        }

        let root = layers.last().unwrap().clone();
        (root, layers)
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
}

// ── CPU reference ────────────────────────────────────────────────────────

/// CPU Poseidon16 compress_in_place: state = perm(state) + state.
fn cpu_compress_in_place(state: &mut [u32; 16]) {
    let p = default_koalabear_poseidon1_16();
    let kb_state: &mut [KoalaBear; 16] = unsafe { &mut *(state as *mut _ as *mut [KoalaBear; 16]) };
    p.compress_in_place(kb_state);
}

/// CPU leaf hash: Poseidon16 sponge (RTL) over row data.
/// The row is processed in REVERSE order (right-to-left), matching the WHIR convention.
pub fn cpu_leaf_hash(row: &[u32]) -> [u32; 8] {
    let mut state = [0u32; 16];

    // Reverse the row to match RTL convention.
    let reversed: Vec<u32> = row.iter().copied().rev().collect();
    let mut pos = 0usize;

    // First chunk: fill state[15..0] RTL.
    for s in (0..16).rev() {
        if pos < reversed.len() {
            state[s] = reversed[pos];
            pos += 1;
        }
    }
    cpu_compress_in_place(&mut state);

    // Subsequent chunks: fill state[15..8] RTL.
    while pos < reversed.len() {
        for s in (8..16).rev() {
            if pos < reversed.len() {
                state[s] = reversed[pos];
                pos += 1;
            }
        }
        cpu_compress_in_place(&mut state);
    }

    state[..8].try_into().unwrap()
}

/// CPU internal node: compress two 8-element digests.
pub fn cpu_compress_pair(left: &[u32; 8], right: &[u32; 8]) -> [u32; 8] {
    let mut state = [0u32; 16];
    state[..8].copy_from_slice(left);
    state[8..].copy_from_slice(right);
    cpu_compress_in_place(&mut state);
    state[..8].try_into().unwrap()
}

/// CPU Merkle tree: full construction.
pub fn cpu_build_tree(
    matrix: &[u32],
    height: usize,
    row_width: usize,
    row_stride: usize,
) -> (Vec<u32>, Vec<Vec<u32>>) {
    // Leaf hashing (row_width elements per row, zero-padded beyond row_stride).
    let mut leaf_digests: Vec<u32> = Vec::with_capacity(height * 8);
    for r in 0..height {
        let mut row = vec![0u32; row_width];
        let actual = row_width.min(row_stride);
        row[..actual].copy_from_slice(&matrix[r * row_stride..r * row_stride + actual]);
        let digest = cpu_leaf_hash(&row);
        leaf_digests.extend_from_slice(&digest);
    }
    let mut layers = vec![leaf_digests];

    // Binary reduction.
    let mut n = height;
    while n > 1 {
        let prev = layers.last().unwrap();
        let n_pairs = n / 2;
        let mut next = Vec::with_capacity(n_pairs * 8);
        for i in 0..n_pairs {
            let left: [u32; 8] = prev[i * 16..i * 16 + 8].try_into().unwrap();
            let right: [u32; 8] = prev[i * 16 + 8..i * 16 + 16].try_into().unwrap();
            let parent = cpu_compress_pair(&left, &right);
            next.extend_from_slice(&parent);
        }
        layers.push(next);
        n = n_pairs;
    }

    let root = layers.last().unwrap().clone();
    (root, layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_leaf_hash_basic() {
        // Hash a 16-element row (single sponge chunk).
        let row = [0u32; 16];
        let digest = cpu_leaf_hash(&row);
        // Non-trivial: Poseidon of zeros is not zeros (due to permutation).
        assert_ne!(digest, [0u32; 8]);
    }

    #[test]
    fn test_cpu_compress_pair_basic() {
        let left = [0u32; 8];
        let right = [0u32; 8];
        let parent = cpu_compress_pair(&left, &right);
        // Should match: Poseidon16_compress([0;16])[0..8]
        let digest2 = cpu_leaf_hash(&[0u32; 16]);
        assert_eq!(parent, digest2, "compress_pair([0;8],[0;8]) should equal leaf_hash([0;16])");
    }
}
