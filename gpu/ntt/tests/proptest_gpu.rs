//! Property-based tests: GPU NTT matches CPU reference.

use std::sync::Arc;

use cudarc::driver::safe::CudaContext;
use field::{PrimeCharacteristicRing, PrimeField32};
use gpu_ntt::*;
use koala_bear::KoalaBear;
use proptest::prelude::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

const P: u32 = 0x7F000001;

fn gpu() -> GpuNtt {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    GpuNtt::new(stream)
}

fn random_poly(log_n: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = 1usize << log_n;
    (0..n)
        .map(|_| unsafe { std::mem::transmute::<KoalaBear, u32>(rng.random()) })
        .collect()
}

// ── GPU vs CPU DFT ──────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_dft_width1_small(seed in 0u64..10000) {
        let log_n = 8; // 256 elements
        let data = random_poly(log_n, seed);
        let g = gpu();
        let gpu_out = g.dft(&data, log_n, 1);
        let cpu_out = cpu_dft(&data, log_n, 1);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_dft_width1_medium(seed in 0u64..10000) {
        let log_n = 12; // 4096 elements
        let data = random_poly(log_n, seed);
        let g = gpu();
        let gpu_out = g.dft(&data, log_n, 1);
        let cpu_out = cpu_dft(&data, log_n, 1);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_dft_width4(seed in 0u64..10000) {
        let log_n = 8;
        let width = 4;
        let n = (1usize << log_n) * width;
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<u32> = (0..n)
            .map(|_| unsafe { std::mem::transmute::<KoalaBear, u32>(rng.random()) })
            .collect();
        let g = gpu();
        let gpu_out = g.dft(&data, log_n, width);
        let cpu_out = cpu_dft(&data, log_n, width);
        prop_assert_eq!(gpu_out, cpu_out);
    }
}

// ── Round-trip: DFT then IDFT should recover original ────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_roundtrip_width1(seed in 0u64..10000) {
        let log_n = 10; // 1024 elements
        let data = random_poly(log_n, seed);
        let g = gpu();
        let dft = g.dft(&data, log_n, 1);
        let roundtrip = g.idft(&dft, log_n, 1);
        prop_assert_eq!(data, roundtrip);
    }

    #[test]
    fn prop_roundtrip_width4(seed in 0u64..10000) {
        let log_n = 8;
        let width = 4;
        let n = (1usize << log_n) * width;
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<u32> = (0..n)
            .map(|_| unsafe { std::mem::transmute::<KoalaBear, u32>(rng.random()) })
            .collect();
        let g = gpu();
        let dft = g.dft(&data, log_n, width);
        let roundtrip = g.idft(&dft, log_n, width);
        prop_assert_eq!(data, roundtrip);
    }
}

// ── GPU DFT matches naive multilinear evaluation ─────────────────────────

#[test]
fn test_gpu_dft_matches_naive_eval() {
    let g = gpu();
    let mut rng = StdRng::seed_from_u64(99);

    for log_n in 1..=12 {
        let n = 1usize << log_n;
        let data: Vec<u32> = (0..n)
            .map(|_| unsafe { std::mem::transmute::<KoalaBear, u32>(rng.random()) })
            .collect();
        let gpu_dft = g.dft(&data, log_n, 1);

        // Check 10 random positions against naive multilinear evaluation.
        for _ in 0..std::cmp::min(10, n) {
            let i = rng.random_range(0..n);
            let expected = eval_multilinear_at_power(&data, i, log_n);
            assert_eq!(
                gpu_dft[i], expected,
                "GPU DFT mismatch at i={i} for log_n={log_n}"
            );
        }
    }
}

// ── Deterministic edge cases ─────────────────────────────────────────────

#[test]
fn test_dft_size_2() {
    let g = gpu();
    // Smallest DFT: 2 elements.
    let data = vec![100u32 % P, 200u32 % P];
    let gpu_out = g.dft(&data, 1, 1);
    let cpu_out = cpu_dft(&data, 1, 1);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn test_dft_size_4() {
    let g = gpu();
    let data = vec![10u32, 20, 30, 40];
    let gpu_out = g.dft(&data, 2, 1);
    let cpu_out = cpu_dft(&data, 2, 1);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn test_dft_all_zeros() {
    let g = gpu();
    let log_n = 8;
    let data = vec![0u32; 1 << log_n];
    let gpu_out = g.dft(&data, log_n, 1);
    // DFT of all zeros should be all zeros.
    assert!(
        gpu_out.iter().all(|&v| v == 0),
        "DFT of zeros should be zeros"
    );
}

#[test]
fn test_dft_single_one() {
    let g = gpu();
    let log_n = 8;
    let n = 1usize << log_n;
    // data[0] = 1 (monty), rest = 0. This is the "constant 1" multilinear
    // (only the all-zeros evaluation is nonzero).
    // ... actually it's just one nonzero eval point, not a constant polynomial.
    // Just verify GPU == CPU.
    let one_monty = 0x01FFFFFEu32;
    let mut data = vec![0u32; n];
    data[0] = one_monty;
    let gpu_out = g.dft(&data, log_n, 1);
    let cpu_out = cpu_dft(&data, log_n, 1);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn test_roundtrip_large() {
    let g = gpu();
    let log_n = 16; // 65536 elements
    let data = random_poly(log_n, 123);
    let dft = g.dft(&data, log_n, 1);
    let roundtrip = g.idft(&dft, log_n, 1);
    assert_eq!(data, roundtrip, "Round-trip failed for log_n=16");
}

#[test]
fn test_dft_width2_roundtrip() {
    let g = gpu();
    let log_n = 10;
    let width = 2;
    let n = (1usize << log_n) * width;
    let mut rng = StdRng::seed_from_u64(777);
    let data: Vec<u32> = (0..n)
        .map(|_| unsafe { std::mem::transmute::<KoalaBear, u32>(rng.random()) })
        .collect();
    let dft = g.dft(&data, log_n, width);
    let roundtrip = g.idft(&dft, log_n, width);
    assert_eq!(data, roundtrip);
}
