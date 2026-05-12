//! Property-based tests: GPU sumcheck matches CPU reference.

use cudarc::driver::safe::CudaContext;
use gpu_sumcheck::*;
use proptest::prelude::*;

const P: u32 = 0x7F000001;

fn gpu() -> GpuSumcheck {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    GpuSumcheck::new(stream)
}

fn random_base(n: usize, seed: u64) -> Vec<u32> {
    (0..n).map(|i| (((i as u64 + seed) * 997 + 7) % P as u64) as u32).collect()
}

fn random_ext(n: usize, seed: u64) -> Vec<u32> {
    (0..n * 5).map(|i| (((i as u64 + seed) * 1337 + 13) % P as u64) as u32).collect()
}

// ── Product sumcheck: base × ext ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_product_base_ext_small(seed in 0u64..10000) {
        let g = gpu();
        let n = 256; // 128 pairs
        let pol_a = random_base(n, seed);
        let pol_b = random_ext(n, seed + 1000);
        let (gpu_c0, gpu_c2) = g.product_sumcheck_base_ext(&pol_a, &pol_b);
        let (cpu_c0, cpu_c2) = cpu_product_sumcheck_base_ext(&pol_a, &pol_b);
        prop_assert_eq!(gpu_c0, cpu_c0);
        prop_assert_eq!(gpu_c2, cpu_c2);
    }

    #[test]
    fn prop_product_base_ext_medium(seed in 0u64..10000) {
        let g = gpu();
        let n = 4096;
        let pol_a = random_base(n, seed);
        let pol_b = random_ext(n, seed + 2000);
        let (gpu_c0, gpu_c2) = g.product_sumcheck_base_ext(&pol_a, &pol_b);
        let (cpu_c0, cpu_c2) = cpu_product_sumcheck_base_ext(&pol_a, &pol_b);
        prop_assert_eq!(gpu_c0, cpu_c0);
        prop_assert_eq!(gpu_c2, cpu_c2);
    }

    #[test]
    fn prop_product_base_ext_large(seed in 0u64..10000) {
        let g = gpu();
        let n = 1 << 16;
        let pol_a = random_base(n, seed);
        let pol_b = random_ext(n, seed + 3000);
        let (gpu_c0, gpu_c2) = g.product_sumcheck_base_ext(&pol_a, &pol_b);
        let (cpu_c0, cpu_c2) = cpu_product_sumcheck_base_ext(&pol_a, &pol_b);
        prop_assert_eq!(gpu_c0, cpu_c0);
        prop_assert_eq!(gpu_c2, cpu_c2);
    }
}

// ── Product sumcheck: ext × ext ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_product_ext_ext(seed in 0u64..10000) {
        let g = gpu();
        let n = 1024;
        let pol_a = random_ext(n, seed);
        let pol_b = random_ext(n, seed + 5000);
        let (gpu_c0, gpu_c2) = g.product_sumcheck_ext_ext(&pol_a, &pol_b);
        let (cpu_c0, cpu_c2) = cpu_product_sumcheck_ext_ext(&pol_a, &pol_b);
        prop_assert_eq!(gpu_c0, cpu_c0);
        prop_assert_eq!(gpu_c2, cpu_c2);
    }

    #[test]
    fn prop_product_ext_ext_large(seed in 0u64..10000) {
        let g = gpu();
        let n = 1 << 14;
        let pol_a = random_ext(n, seed);
        let pol_b = random_ext(n, seed + 6000);
        let (gpu_c0, gpu_c2) = g.product_sumcheck_ext_ext(&pol_a, &pol_b);
        let (cpu_c0, cpu_c2) = cpu_product_sumcheck_ext_ext(&pol_a, &pol_b);
        prop_assert_eq!(gpu_c0, cpu_c0);
        prop_assert_eq!(gpu_c2, cpu_c2);
    }
}

// ── Eq fold ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_eq_fold(seed in 0u64..10000) {
        let g = gpu();
        let n_pairs = 512;
        let eq_data = random_ext(n_pairs * 2, seed);
        let r: [u32; 5] = std::array::from_fn(|i| (((i as u64 + seed) * 47 + 3) % P as u64) as u32);
        let gpu_out = g.eq_fold(&eq_data, &r);
        let cpu_out = cpu_eq_fold(&eq_data, &r);
        prop_assert_eq!(gpu_out, cpu_out);
    }
}

// ── Deterministic tests ──────────────────────────────────────────────────

#[test]
fn test_product_base_ext_zeros() {
    let g = gpu();
    let n = 64;
    let pol_a = vec![0u32; n];
    let pol_b = vec![0u32; n * 5];
    let (c0, c2) = g.product_sumcheck_base_ext(&pol_a, &pol_b);
    assert_eq!(c0, [0; 5]);
    assert_eq!(c2, [0; 5]);
}

#[test]
fn test_product_base_ext_identity() {
    let g = gpu();
    // pol_a = [1, 1, ...], pol_b = [1, 0, 0, 0, 0, ...] (base field 1 embedded in ext)
    let n = 64;
    let monty_one = 0x01FFFFFEu32;
    let pol_a = vec![monty_one; n];
    let mut pol_b = vec![0u32; n * 5];
    for i in 0..n { pol_b[i * 5] = monty_one; }

    let (gpu_c0, gpu_c2) = g.product_sumcheck_base_ext(&pol_a, &pol_b);
    let (cpu_c0, cpu_c2) = cpu_product_sumcheck_base_ext(&pol_a, &pol_b);
    assert_eq!(gpu_c0, cpu_c0);
    assert_eq!(gpu_c2, cpu_c2);
}

#[test]
fn test_eq_fold_basic() {
    let g = gpu();
    // 2 ext elements: fold with r.
    let eq_data = random_ext(2, 42);
    let r: [u32; 5] = [100, 200, 300, 400, 500];
    let gpu_out = g.eq_fold(&eq_data, &r);
    let cpu_out = cpu_eq_fold(&eq_data, &r);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn test_product_large_exact() {
    let g = gpu();
    let n = 1 << 18;
    let pol_a = random_base(n, 999);
    let pol_b = random_ext(n, 1999);
    let (gpu_c0, gpu_c2) = g.product_sumcheck_base_ext(&pol_a, &pol_b);
    let (cpu_c0, cpu_c2) = cpu_product_sumcheck_base_ext(&pol_a, &pol_b);
    assert_eq!(gpu_c0, cpu_c0, "c0 mismatch at n=2^18");
    assert_eq!(gpu_c2, cpu_c2, "c2 mismatch at n=2^18");
}
