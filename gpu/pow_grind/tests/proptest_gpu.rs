//! Property-based tests for GPU field arithmetic and PoW grinding.

use std::sync::Arc;

use cudarc::driver::safe::CudaContext;
use gpu_pow_grind::{GpuPowGrinder, cpu_field_ops, cpu_qe_ops, cpu_pow_grind};
use proptest::prelude::*;

const P: u32 = 0x7F000001;

fn gpu() -> (Arc<cudarc::driver::safe::CudaStream>, GpuPowGrinder) {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    let g = GpuPowGrinder::new(stream.clone());
    (stream, g)
}

// ── Base field tests ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_base_field_ops(a in 0..P, b in 0..P) {
        let (_stream, gpu) = gpu();
        let results = gpu.field_test(&[a], &[b]);
        let expected = cpu_field_ops(a, b);
        prop_assert_eq!(&results[..4], &expected[..]);
    }
}

#[test]
fn test_base_field_batch() {
    let (_stream, gpu) = gpu();
    let n = 1000;
    let a_vals: Vec<u32> = (0..n).map(|i| (i * 7 + 3) % P).collect();
    let b_vals: Vec<u32> = (0..n).map(|i| (i * 13 + 11) % P).collect();

    let results = gpu.field_test(&a_vals, &b_vals);
    for i in 0..n as usize {
        let expected = cpu_field_ops(a_vals[i], b_vals[i]);
        assert_eq!(
            &results[i * 4..(i + 1) * 4],
            &expected[..],
            "base field mismatch at index {i}"
        );
    }
}

// ── Quintic extension tests ──────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_quintic_ext_ops(
        a0 in 0..P, a1 in 0..P, a2 in 0..P, a3 in 0..P, a4 in 0..P,
        b0 in 0..P, b1 in 0..P, b2 in 0..P, b3 in 0..P, b4 in 0..P,
    ) {
        let (_stream, gpu) = gpu();
        let a = [a0, a1, a2, a3, a4];
        let b = [b0, b1, b2, b3, b4];
        let a_flat: Vec<u32> = a.to_vec();
        let b_flat: Vec<u32> = b.to_vec();

        let results = gpu.qe_test(&a_flat, &b_flat);
        let expected = cpu_qe_ops(&a, &b);
        prop_assert_eq!(&results[..15], &expected[..]);
    }
}

#[test]
fn test_quintic_ext_batch() {
    let (_stream, gpu) = gpu();
    let n = 200;
    let a_flat: Vec<u32> = (0..n * 5).map(|i| ((i as u64 * 31 + 7) % P as u64) as u32).collect();
    let b_flat: Vec<u32> = (0..n * 5).map(|i| ((i as u64 * 47 + 13) % P as u64) as u32).collect();

    let results = gpu.qe_test(&a_flat, &b_flat);
    for i in 0..n {
        let a: [u32; 5] = a_flat[i * 5..(i + 1) * 5].try_into().unwrap();
        let b: [u32; 5] = b_flat[i * 5..(i + 1) * 5].try_into().unwrap();
        let expected = cpu_qe_ops(&a, &b);
        assert_eq!(
            &results[i * 15..(i + 1) * 15],
            &expected[..],
            "quintic ext mismatch at index {i}"
        );
    }
}

// ── PoW grinding tests ──────────────────────────────────────────────────

#[test]
fn test_pow_grind_8bit() {
    let (_stream, gpu) = gpu();

    // Fixed challenger state (all zeros → some will have trailing zeros).
    let state = [0u32; 16];
    let nonce_slot = 8u32;
    let target_bits = 8u32;

    let gpu_nonce = gpu.grind(&state, nonce_slot, target_bits, 1 << 20)
        .expect("GPU should find 8-bit PoW within 1M attempts");

    let cpu_nonce = cpu_pow_grind(&state, nonce_slot as usize, target_bits);

    // Both should be valid (not necessarily the same nonce — any valid one works).
    // Verify the GPU nonce is actually valid.
    let p = koala_bear::default_koalabear_poseidon1_16();
    let mut check_state: [koala_bear::KoalaBear; 16] = unsafe { std::mem::transmute(state) };
    check_state[nonce_slot as usize] = unsafe { std::mem::transmute(gpu_nonce) };
    use koala_bear::symmetric::Permutation;
    let initial = check_state;
    p.permute_mut(&mut check_state);
    for i in 0..16 { check_state[i] += initial[i]; }
    let out0 = field::PrimeField32::as_canonical_u32(&check_state[0]);
    let mask = (1u32 << target_bits) - 1;
    assert_eq!(out0 & mask, 0, "GPU nonce {gpu_nonce:#010x} does not satisfy {target_bits}-bit PoW");
}

#[test]
fn test_pow_grind_16bit() {
    let (_stream, gpu) = gpu();

    let state = [0u32; 16];
    let nonce_slot = 8u32;
    let target_bits = 16u32;

    let gpu_nonce = gpu.grind(&state, nonce_slot, target_bits, 1 << 22)
        .expect("GPU should find 16-bit PoW within 4M attempts");

    // Verify.
    let p = koala_bear::default_koalabear_poseidon1_16();
    let mut check_state: [koala_bear::KoalaBear; 16] = unsafe { std::mem::transmute(state) };
    check_state[nonce_slot as usize] = unsafe { std::mem::transmute(gpu_nonce) };
    use koala_bear::symmetric::Permutation;
    let initial = check_state;
    p.permute_mut(&mut check_state);
    for i in 0..16 { check_state[i] += initial[i]; }
    let out0 = field::PrimeField32::as_canonical_u32(&check_state[0]);
    let mask = (1u32 << target_bits) - 1;
    assert_eq!(out0 & mask, 0, "GPU nonce does not satisfy {target_bits}-bit PoW");
}
