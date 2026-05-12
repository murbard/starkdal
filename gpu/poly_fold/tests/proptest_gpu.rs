//! Property-based tests: GPU polynomial folding matches CPU reference.

use std::sync::Arc;

use cudarc::driver::safe::CudaContext;
use gpu_poly_fold::*;
use proptest::prelude::*;

const P: u32 = 0x7F000001;

fn gpu() -> GpuPolyFold {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    GpuPolyFold::new(stream)
}

// Random base field polynomial of size 2^log_n.
fn arb_base_poly(log_n: usize) -> impl Strategy<Value = Vec<u32>> {
    let n = 1 << log_n;
    prop::collection::vec(0..P, n)
}

// Random extension field challenge.
fn arb_ext_challenge() -> impl Strategy<Value = [u32; 5]> {
    prop::array::uniform5(0..P)
}

// Random extension field polynomial of size 2^log_n (each elem = 5 u32s).
fn arb_ext_poly(log_n: usize) -> impl Strategy<Value = Vec<u32>> {
    let n = (1 << log_n) * 5;
    prop::collection::vec(0..P, n)
}

// ── Base → Base tests ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_fold_base_lsb(data in arb_base_poly(10), r in 0..P) {
        let g = gpu();
        let gpu_out = g.fold_base(&data, r, FoldMode::Lsb);
        let cpu_out = cpu_fold_base_lsb(&data, r);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_fold_base_half(data in arb_base_poly(10), r in 0..P) {
        let g = gpu();
        let gpu_out = g.fold_base(&data, r, FoldMode::Half);
        let cpu_out = cpu_fold_base_half(&data, r);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_fold_base_at_bit_1(data in arb_base_poly(10), r in 0..P) {
        let g = gpu();
        let gpu_out = g.fold_base(&data, r, FoldMode::AtBit(1));
        let cpu_out = cpu_fold_base_at_bit(&data, r, 1);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_fold_base_at_bit_3(data in arb_base_poly(10), r in 0..P) {
        let g = gpu();
        let gpu_out = g.fold_base(&data, r, FoldMode::AtBit(3));
        let cpu_out = cpu_fold_base_at_bit(&data, r, 3);
        prop_assert_eq!(gpu_out, cpu_out);
    }
}

// ── Base → Ext tests ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_fold_b2e_lsb(data in arb_base_poly(10), r in arb_ext_challenge()) {
        let g = gpu();
        let gpu_out = g.fold_base_to_ext(&data, &r, FoldMode::Lsb);
        let cpu_out = cpu_fold_base_to_ext_lsb(&data, &r);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_fold_b2e_half(data in arb_base_poly(10), r in arb_ext_challenge()) {
        let g = gpu();
        let gpu_out = g.fold_base_to_ext(&data, &r, FoldMode::Half);
        let cpu_out = cpu_fold_base_to_ext_half(&data, &r);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_fold_b2e_at_bit_2(data in arb_base_poly(10), r in arb_ext_challenge()) {
        let g = gpu();
        let gpu_out = g.fold_base_to_ext(&data, &r, FoldMode::AtBit(2));
        let cpu_out = cpu_fold_base_to_ext_at_bit(&data, &r, 2);
        prop_assert_eq!(gpu_out, cpu_out);
    }
}

// ── Ext → Ext tests ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_fold_ext_lsb(data in arb_ext_poly(10), r in arb_ext_challenge()) {
        let g = gpu();
        let gpu_out = g.fold_ext(&data, &r, FoldMode::Lsb);
        let cpu_out = cpu_fold_ext_lsb(&data, &r);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_fold_ext_half(data in arb_ext_poly(10), r in arb_ext_challenge()) {
        let g = gpu();
        let gpu_out = g.fold_ext(&data, &r, FoldMode::Half);
        let cpu_out = cpu_fold_ext_half(&data, &r);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_fold_ext_at_bit_4(data in arb_ext_poly(10), r in arb_ext_challenge()) {
        let g = gpu();
        let gpu_out = g.fold_ext(&data, &r, FoldMode::AtBit(4));
        let cpu_out = cpu_fold_ext_at_bit(&data, &r, 4);
        prop_assert_eq!(gpu_out, cpu_out);
    }
}

// ── Edge cases ──────────────────────────────────────────────────────────

#[test]
fn test_fold_base_minimal() {
    let g = gpu();
    // 2 elements → 1 element. out = lo + r*(hi - lo)
    let data = vec![0u32, 0u32]; // both zero
    let r = 0u32;
    let out = g.fold_base(&data, r, FoldMode::Lsb);
    assert_eq!(out, vec![0u32]);
}

#[test]
fn test_fold_base_r_zero() {
    let g = gpu();
    // r=0 → out[j] = data[2j] (just take the "lo" element)
    let data: Vec<u32> = (0..16).map(|i| (i * 37 + 5) % P).collect();
    let out = g.fold_base(&data, 0, FoldMode::Lsb);
    let cpu = cpu_fold_base_lsb(&data, 0);
    assert_eq!(out, cpu);
    // When r=0, result should equal the even-indexed elements
    for (j, &v) in out.iter().enumerate() {
        assert_eq!(v, data[2 * j], "r=0 should select lo at index {j}");
    }
}

#[test]
fn test_fold_base_r_one() {
    let g = gpu();
    // r=1 (in monty form) → out[j] = data[2j+1] (just take the "hi" element)
    let one_monty = 0x01FFFFFEu32; // KB_MONTY_ONE = (1 << 32) % P
    let data: Vec<u32> = (0..16).map(|i| (i * 37 + 5) % P).collect();
    let out = g.fold_base(&data, one_monty, FoldMode::Lsb);
    let cpu = cpu_fold_base_lsb(&data, one_monty);
    assert_eq!(out, cpu);
    // When r=1, result should equal the odd-indexed elements
    for (j, &v) in out.iter().enumerate() {
        assert_eq!(v, data[2 * j + 1], "r=1 should select hi at index {j}");
    }
}

#[test]
fn test_fold_large_base_lsb() {
    let g = gpu();
    let n = 1 << 16; // 64K elements
    let data: Vec<u32> = (0..n).map(|i| ((i as u64 * 1337 + 42) % P as u64) as u32).collect();
    let r = 12345u32 % P;
    let gpu_out = g.fold_base(&data, r, FoldMode::Lsb);
    let cpu_out = cpu_fold_base_lsb(&data, r);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn test_fold_large_ext_half() {
    let g = gpu();
    let n = 1 << 12; // 4K ext elements = 20K u32s per half
    let data: Vec<u32> = (0..(n * 2 * 5)).map(|i| ((i as u64 * 997 + 7) % P as u64) as u32).collect();
    let r: [u32; 5] = [100, 200, 300, 400, 500].map(|v| v % P);
    let gpu_out = g.fold_ext(&data, &r, FoldMode::Half);
    let cpu_out = cpu_fold_ext_half(&data, &r);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn test_fold_chained() {
    // Simulate a sumcheck: fold base→ext, then ext→ext, then ext→ext.
    let g = gpu();
    let data: Vec<u32> = (0..256).map(|i| ((i as u64 * 7919 + 13) % P as u64) as u32).collect();
    let r0: [u32; 5] = [111, 222, 333, 444, 555].map(|v| v % P);
    let r1: [u32; 5] = [666, 777, 888, 999, 1010].map(|v| v % P);
    let r2: [u32; 5] = [1111, 2222, 3333, 4444, 5555].map(|v| v % P);

    // GPU chain
    let g1 = g.fold_base_to_ext(&data, &r0, FoldMode::Lsb);
    let g2 = g.fold_ext(&g1, &r1, FoldMode::Lsb);
    let g3 = g.fold_ext(&g2, &r2, FoldMode::Lsb);

    // CPU chain
    let c1 = cpu_fold_base_to_ext_lsb(&data, &r0);
    let c2 = cpu_fold_ext_lsb(&c1, &r1);
    let c3 = cpu_fold_ext_lsb(&c2, &r2);

    assert_eq!(g3, c3, "chained fold GPU != CPU");
}
