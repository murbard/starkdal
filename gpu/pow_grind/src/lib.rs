//! GPU proof-of-work grinding + field arithmetic testing.
//!
//! Provides:
//! - `GpuPowGrinder`: finds PoW nonces using GPU Poseidon16
//! - `GpuFieldTester`: property-tests KoalaBear base + quintic extension on GPU

use std::ffi::CString;
use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

const KB_P: u64 = 0x7F000001;
fn kb_to_monty(v: u32) -> u32 {
    (((v as u64) << 32) % KB_P) as u32
}
use cudarc::driver::{result as cuda_result, sys as cuda_sys};
use field::PrimeField32;
use koala_bear::{
    KoalaBear, POSEIDON1_WIDTH, default_koalabear_poseidon1_16, poseidon1_round_constants,
    poseidon1_sparse_first_round_constants, poseidon1_sparse_first_row, poseidon1_sparse_m_i,
    poseidon1_sparse_scalar_round_constants, poseidon1_sparse_v,
};

const WIDTH: usize = POSEIDON1_WIDTH;

fn kb_slice_as_u32(slice: &[KoalaBear]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u32>(), slice.len()) }
}

/// GPU proof-of-work grinder + field tester.
pub struct GpuPowGrinder {
    stream: Arc<CudaStream>,
    cu_module: cuda_sys::CUmodule,
    pow_fn: cuda_sys::CUfunction,
    field_test_fn: cuda_sys::CUfunction,
    qe_test_fn: cuda_sys::CUfunction,
}

unsafe impl Send for GpuPowGrinder {}
unsafe impl Sync for GpuPowGrinder {}

impl Drop for GpuPowGrinder {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_result::module::unload(self.cu_module);
        }
    }
}

impl GpuPowGrinder {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let cubin = include_bytes!(concat!(env!("OUT_DIR"), "/pow_grind.cubin"));
        let cu_module = unsafe { cuda_result::module::load_data(cubin.as_ptr().cast()) }
            .expect("failed to load pow_grind cubin");

        let load_fn = |name: &str| {
            let c = CString::new(name).unwrap();
            unsafe { cuda_result::module::get_function(cu_module, c) }
                .unwrap_or_else(|e| panic!("kernel {name} not found: {e:?}"))
        };

        let pow_fn = load_fn("pow_grind_kernel");
        let field_test_fn = load_fn("field_test_kernel");
        let qe_test_fn = load_fn("qe_test_kernel");

        let mut this = Self {
            stream,
            cu_module,
            pow_fn,
            field_test_fn,
            qe_test_fn,
        };
        this.upload_poseidon_constants();
        this
    }

    fn upload_poseidon_constants(&mut self) {
        let rc = poseidon1_round_constants();
        let rc_flat: Vec<u32> = rc
            .iter()
            .flat_map(|r| kb_slice_as_u32(r))
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

        self.copy_to_symbol(
            "d_sparse_first_rc",
            kb_slice_as_u32(poseidon1_sparse_first_round_constants()),
        );

        let m_i = poseidon1_sparse_m_i();
        let m_i_flat: Vec<u32> = m_i
            .iter()
            .flat_map(|r| kb_slice_as_u32(r))
            .copied()
            .collect();
        self.copy_to_symbol("d_sparse_m_i", &m_i_flat);

        let fr: Vec<u32> = poseidon1_sparse_first_row()
            .iter()
            .flat_map(|r| kb_slice_as_u32(r))
            .copied()
            .collect();
        self.copy_to_symbol("d_sparse_first_row", &fr);

        let v: Vec<u32> = poseidon1_sparse_v()
            .iter()
            .flat_map(|r| kb_slice_as_u32(r))
            .copied()
            .collect();
        self.copy_to_symbol("d_sparse_v", &v);

        let src: Vec<u32> = poseidon1_sparse_scalar_round_constants()
            .iter()
            .map(|c| kb_slice_as_u32(std::slice::from_ref(c))[0])
            .collect();
        self.copy_to_symbol("d_sparse_scalar_rc", &src);
    }

    fn copy_to_symbol(&self, name: &str, data: &[u32]) {
        let c_name = CString::new(name).unwrap();
        unsafe {
            let mut dptr: cuda_sys::CUdeviceptr = 0;
            let mut size: usize = 0;
            cuda_sys::cuModuleGetGlobal_v2(&mut dptr, &mut size, self.cu_module, c_name.as_ptr())
                .result()
                .unwrap_or_else(|e| panic!("cuModuleGetGlobal({name}): {e:?}"));
            assert!(data.len() * 4 <= size, "constant {name}: data too large");
            cuda_result::memcpy_htod_sync(dptr, data)
                .unwrap_or_else(|e| panic!("memcpy_htod({name}): {e:?}"));
        }
    }

    /// Find a nonce such that Poseidon16_compress(state_with_nonce)[0] has
    /// `target_bits` trailing zero bits in canonical form.
    ///
    /// `challenger_state`: 16 u32s in Montgomery form (the Fiat-Shamir state).
    /// `nonce_slot`: which of the 16 state elements to replace with the nonce.
    /// `target_bits`: number of trailing zero bits required.
    ///
    /// Returns the winning nonce in Montgomery form, or None if not found
    /// within `max_nonces` attempts.
    pub fn grind(
        &self,
        challenger_state: &[u32; 16],
        nonce_slot: u32,
        target_bits: u32,
        max_nonces: u64,
    ) -> Option<u32> {
        let d_state = self
            .stream
            .memcpy_stod(challenger_state.as_slice())
            .unwrap();
        self.grind_from_device_state(&d_state, 16, nonce_slot, target_bits, max_nonces)
    }

    /// Find a nonce using a challenger state that is already resident on this
    /// GPU context. Entries past `state_len` are treated as zero.
    pub fn grind_from_device_state(
        &self,
        d_challenger_state: &CudaSlice<u32>,
        state_len: u32,
        nonce_slot: u32,
        target_bits: u32,
        max_nonces: u64,
    ) -> Option<u32> {
        let d_nonce = self.grind_from_device_state_device(
            d_challenger_state,
            state_len,
            nonce_slot,
            target_bits,
            max_nonces,
        )?;
        let nonce_host = self.stream.memcpy_dtov(&d_nonce).unwrap();
        Some(nonce_host[0])
    }

    /// Find a nonce using a challenger state already resident on this GPU
    /// context and return the winning nonce as a device buffer.
    pub fn grind_from_device_state_device(
        &self,
        d_challenger_state: &CudaSlice<u32>,
        state_len: u32,
        nonce_slot: u32,
        target_bits: u32,
        max_nonces: u64,
    ) -> Option<CudaSlice<u32>> {
        assert!(state_len <= WIDTH as u32);
        if max_nonces == 0 {
            return None;
        }
        let mut d_nonce = self.stream.alloc_zeros::<u32>(1).unwrap();
        let mut d_flag = self.stream.alloc_zeros::<u32>(1).unwrap();
        self.grind_from_device_state_device_async(
            d_challenger_state,
            state_len,
            nonce_slot,
            target_bits,
            max_nonces,
            &mut d_nonce,
            &mut d_flag,
        );
        self.stream.synchronize().unwrap();

        let flag_host = self.stream.memcpy_dtov(&d_flag).unwrap();
        if flag_host[0] != 0 {
            Some(d_nonce)
        } else {
            None
        }
    }

    /// Launch a single device-side PoW search into preallocated output buffers.
    ///
    /// The caller is responsible for synchronizing the stream and checking `d_flag`.
    #[allow(clippy::too_many_arguments)]
    pub fn grind_from_device_state_device_async(
        &self,
        d_challenger_state: &CudaSlice<u32>,
        state_len: u32,
        nonce_slot: u32,
        target_bits: u32,
        max_nonces: u64,
        d_nonce: &mut CudaSlice<u32>,
        d_flag: &mut CudaSlice<u32>,
    ) {
        assert!(state_len <= WIDTH as u32);
        if max_nonces == 0 {
            return;
        }

        let threads = 256u32;
        let max_blocks = 65_535u32;
        let blocks = ((max_nonces + threads as u64 - 1) / threads as u64)
            .min(max_blocks as u64)
            .max(1) as u32;

        let (state_ptr, _g1) = d_challenger_state.device_ptr(&self.stream);
        let (nonce_ptr, _g2) = d_nonce.device_ptr_mut(&self.stream);
        let (flag_ptr, _g3) = d_flag.device_ptr_mut(&self.stream);

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &state_ptr as *const _ as *mut _,
            &nonce_ptr as *const _ as *mut _,
            &flag_ptr as *const _ as *mut _,
            &state_len as *const _ as *mut _,
            &target_bits as *const _ as *mut _,
            &nonce_slot as *const _ as *mut _,
            &max_nonces as *const _ as *mut _,
        ];

        unsafe {
            cuda_result::launch_kernel(
                self.pow_fn,
                (blocks, 1, 1),
                (threads, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut args,
            )
            .expect("pow_grind kernel launch failed");
        }
    }

    // ── Field testing ────────────────────────────────────────────────────

    /// Run base field tests on GPU: returns [add, sub, mul, cube] for each pair.
    pub fn field_test(&self, a_vals: &[u32], b_vals: &[u32]) -> Vec<u32> {
        let n = a_vals.len() as u32;
        assert_eq!(a_vals.len(), b_vals.len());

        let d_a = self.stream.memcpy_stod(a_vals).unwrap();
        let d_b = self.stream.memcpy_stod(b_vals).unwrap();
        let mut d_out = self.stream.alloc_zeros::<u32>((n as usize) * 4).unwrap();

        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;

        {
            let (a_ptr, _g1) = d_a.device_ptr(&self.stream);
            let (b_ptr, _g2) = d_b.device_ptr(&self.stream);
            let (out_ptr, _g3) = d_out.device_ptr_mut(&self.stream);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &n as *const _ as *mut _,
            ];

            unsafe {
                cuda_result::launch_kernel(
                    self.field_test_fn,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("field_test kernel launch failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_out).unwrap()
    }

    /// Run quintic extension tests: returns [add(5), mul(5), square(5)] per pair.
    pub fn qe_test(&self, a_vals: &[u32], b_vals: &[u32]) -> Vec<u32> {
        let n = (a_vals.len() / 5) as u32;
        assert_eq!(a_vals.len(), b_vals.len());
        assert_eq!(a_vals.len() % 5, 0);

        let d_a = self.stream.memcpy_stod(a_vals).unwrap();
        let d_b = self.stream.memcpy_stod(b_vals).unwrap();
        let mut d_out = self.stream.alloc_zeros::<u32>((n as usize) * 15).unwrap();

        let threads = 256u32;
        let blocks = (n + threads - 1) / threads;

        {
            let (a_ptr, _g1) = d_a.device_ptr(&self.stream);
            let (b_ptr, _g2) = d_b.device_ptr(&self.stream);
            let (out_ptr, _g3) = d_out.device_ptr_mut(&self.stream);

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &out_ptr as *const _ as *mut _,
                &n as *const _ as *mut _,
            ];

            unsafe {
                cuda_result::launch_kernel(
                    self.qe_test_fn,
                    (blocks, 1, 1),
                    (threads, 1, 1),
                    0,
                    self.stream.cu_stream(),
                    &mut args,
                )
                .expect("qe_test kernel launch failed");
            }
        }
        self.stream.synchronize().unwrap();
        self.stream.memcpy_dtov(&d_out).unwrap()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Static convenience: create a temporary GPU context, grind, return nonce.
    /// Returns None if GPU not available.
    pub fn grind_static(
        challenger_state: &[u32; 16],
        nonce_slot: u32,
        target_bits: u32,
        max_nonces: u64,
    ) -> Option<u32> {
        use std::sync::OnceLock;
        static INSTANCE: OnceLock<Option<GpuPowGrinder>> = OnceLock::new();
        let inst = INSTANCE.get_or_init(|| {
            let ctx = cudarc::driver::safe::CudaContext::new(0).ok()?;
            let stream = ctx.default_stream();
            Some(GpuPowGrinder::new(stream))
        });
        inst.as_ref()?
            .grind(challenger_state, nonce_slot, target_bits, max_nonces)
    }
}

// ── CPU reference for PoW ────────────────────────────────────────────────

/// CPU PoW grind: brute-force search for a nonce.
pub fn cpu_pow_grind(challenger_state: &[u32; 16], nonce_slot: usize, target_bits: u32) -> u32 {
    let p = default_koalabear_poseidon1_16();
    let mask = (1u32 << target_bits) - 1;

    for nonce in 0u32.. {
        let mut state: [KoalaBear; 16] = unsafe { std::mem::transmute(*challenger_state) };
        state[nonce_slot] = KoalaBear::new(nonce);
        p.compress_in_place(&mut state);
        let out0 = state[0].as_canonical_u32();
        if out0 & mask == 0 {
            return unsafe { std::mem::transmute::<KoalaBear, u32>(KoalaBear::new(nonce)) };
        }
    }
    unreachable!()
}

// ── CPU reference for field ops ──────────────────────────────────────────

pub fn cpu_field_ops(a: u32, b: u32) -> [u32; 4] {
    let ka = unsafe { std::mem::transmute::<u32, KoalaBear>(a) };
    let kb = unsafe { std::mem::transmute::<u32, KoalaBear>(b) };
    let add: u32 = unsafe { std::mem::transmute(ka + kb) };
    let sub: u32 = unsafe { std::mem::transmute(ka - kb) };
    let mul: u32 = unsafe { std::mem::transmute(ka * kb) };
    let cube: u32 = unsafe { std::mem::transmute(ka * ka * ka) };
    [add, sub, mul, cube]
}

pub use koala_bear::extension::QuinticExtensionField;

pub fn cpu_qe_ops(a: &[u32; 5], b: &[u32; 5]) -> [u32; 15] {
    type QE = QuinticExtensionField<KoalaBear>;
    let qa: QE = unsafe { std::mem::transmute(*a) };
    let qb: QE = unsafe { std::mem::transmute(*b) };
    let add: [u32; 5] = unsafe { std::mem::transmute(qa + qb) };
    let mul: [u32; 5] = unsafe { std::mem::transmute(qa * qb) };
    let sq: [u32; 5] = unsafe { std::mem::transmute(qa * qa) };
    let mut out = [0u32; 15];
    out[0..5].copy_from_slice(&add);
    out[5..10].copy_from_slice(&mul);
    out[10..15].copy_from_slice(&sq);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_field_ops_basic() {
        let one = unsafe { std::mem::transmute::<KoalaBear, u32>(KoalaBear::new(1)) };
        let two = unsafe { std::mem::transmute::<KoalaBear, u32>(KoalaBear::new(2)) };
        let [add, sub, mul, cube] = cpu_field_ops(one, two);
        let three = unsafe { std::mem::transmute::<KoalaBear, u32>(KoalaBear::new(3)) };
        assert_eq!(add, three); // 1 + 2 = 3
        let neg_one =
            unsafe { std::mem::transmute::<KoalaBear, u32>(KoalaBear::new(0x7F000001 - 1)) };
        assert_eq!(sub, neg_one); // 1 - 2 = -1
        assert_eq!(mul, two); // 1 * 2 = 2
        assert_eq!(cube, one); // 1^3 = 1
    }
}
