//! Property-based tests: GPU trace ops match CPU reference.

use cudarc::driver::safe::CudaContext;
use gpu_trace_ops::*;
use proptest::prelude::*;

const P: u32 = 0x7F000001;
// Montgomery form of small canonical values (for use as addresses).
fn to_monty(v: u32) -> u32 { (((v as u64) << 32) % P as u64) as u32 }

fn gpu() -> GpuTraceOps {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    GpuTraceOps::new(stream)
}

// ── Access count tests ───────────────────────────────────────────────────

#[test]
fn test_access_count_basic() {
    let g = gpu();
    // Column with addresses 0, 1, 2, 0, 1, 0 (in Montgomery form).
    let column: Vec<u32> = vec![
        to_monty(0), to_monty(1), to_monty(2),
        to_monty(0), to_monty(1), to_monty(0),
    ];
    let gpu_acc = g.access_count_simple(&column, 4);
    let cpu_acc = cpu_access_count_simple(&column, 4);
    assert_eq!(gpu_acc, cpu_acc);
    // Plain counts: addr 0: 3, addr 1: 2, addr 2: 1, addr 3: 0
    assert_eq!(gpu_acc, vec![3, 2, 1, 0]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_access_count(
        n in 100..2000usize,
        max_addr in 4..64u32,
    ) {
        let g = gpu();
        // Generate random addresses in [0, max_addr).
        let column: Vec<u32> = (0..n).map(|i| {
            to_monty(((i * 37 + 13) as u32) % max_addr)
        }).collect();
        let acc_size = max_addr as usize;
        let gpu_acc = g.access_count_simple(&column, acc_size);
        let cpu_acc = cpu_access_count_simple(&column, acc_size);
        prop_assert_eq!(gpu_acc, cpu_acc);
    }
}

// ── Shift down tests ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_shift_down(data in prop::collection::vec(0..P, 64..1024)) {
        let g = gpu();
        let gpu_out = g.shift_down(&data);
        let cpu_out = cpu_shift_down(&data);
        prop_assert_eq!(gpu_out, cpu_out);
    }
}

#[test]
fn test_shift_down_basic() {
    let g = gpu();
    let data = vec![10, 20, 30, 40, 50];
    let gpu_out = g.shift_down(&data);
    assert_eq!(gpu_out, vec![20, 30, 40, 50, 50]); // last element repeated
}

// ── Bit-reversal tests ───────────────────────────────────────────────────

#[test]
fn test_bit_reverse_8() {
    let g = gpu();
    let mut data: Vec<u32> = (0..8).collect();
    g.bit_reverse(&mut data);
    assert_eq!(data, vec![0, 4, 2, 6, 1, 5, 3, 7]);
}

#[test]
fn test_bit_reverse_roundtrip() {
    let g = gpu();
    let original: Vec<u32> = (0..256).map(|i| i * 7 % P).collect();
    let mut data = original.clone();
    g.bit_reverse(&mut data);
    g.bit_reverse(&mut data); // double reversal = identity
    assert_eq!(data, original);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_bit_reverse(log_n in 3..14u32) {
        let g = gpu();
        let n = 1usize << log_n;
        let mut gpu_data: Vec<u32> = (0..n as u32).collect();
        let mut cpu_data = gpu_data.clone();
        g.bit_reverse(&mut gpu_data);
        cpu_bit_reverse(&mut cpu_data);
        prop_assert_eq!(gpu_data, cpu_data);
    }
}

// ── MLE eval tests ───────────────────────────────────────────────────────

#[test]
fn test_mle_eval_constant() {
    let g = gpu();
    // 1-variable polynomial: [a, b]. Eval at point r: a + r*(b-a)
    let data = vec![100u32 % P, 200u32 % P];
    let point = [[500u32 % P, 0, 0, 0, 0]]; // base-field challenge embedded in ext
    let gpu_result = g.mle_eval(&data, &point);
    let cpu_result = cpu_mle_eval(&data, &point);
    assert_eq!(gpu_result, cpu_result);
}

#[test]
fn test_mle_eval_2var() {
    let g = gpu();
    let data = vec![10, 20, 30, 40]; // 2 variables
    let point = [
        [111 % P, 0, 0, 0, 0],
        [222 % P, 0, 0, 0, 0],
    ];
    let gpu_result = g.mle_eval(&data, &point);
    let cpu_result = cpu_mle_eval(&data, &point);
    assert_eq!(gpu_result, cpu_result);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_mle_eval_base_point(log_n in 1..12u32) {
        let g = gpu();
        let n = 1usize << log_n;
        let data: Vec<u32> = (0..n).map(|i| ((i as u64 * 997 + 7) % P as u64) as u32).collect();
        // Base-field point (embedded in ext: only component 0 nonzero).
        let point: Vec<[u32; 5]> = (0..log_n as usize)
            .map(|i| [((i as u64 * 1337 + 42) % P as u64) as u32, 0, 0, 0, 0])
            .collect();
        let gpu_result = g.mle_eval(&data, &point);
        let cpu_result = cpu_mle_eval(&data, &point);
        prop_assert_eq!(gpu_result, cpu_result);
    }

    #[test]
    fn prop_mle_eval_ext_point(log_n in 1..10u32) {
        let g = gpu();
        let n = 1usize << log_n;
        let data: Vec<u32> = (0..n).map(|i| ((i as u64 * 31 + 5) % P as u64) as u32).collect();
        // Full extension field point.
        let point: Vec<[u32; 5]> = (0..log_n as usize)
            .map(|i| {
                let base = (i as u64 * 47 + 13) % P as u64;
                [
                    (base % P as u64) as u32,
                    ((base * 3 + 1) % P as u64) as u32,
                    ((base * 7 + 2) % P as u64) as u32,
                    ((base * 11 + 3) % P as u64) as u32,
                    ((base * 13 + 5) % P as u64) as u32,
                ]
            })
            .collect();
        let gpu_result = g.mle_eval(&data, &point);
        let cpu_result = cpu_mle_eval(&data, &point);
        prop_assert_eq!(gpu_result, cpu_result);
    }
}

#[test]
fn test_mle_eval_large() {
    let g = gpu();
    let log_n = 14;
    let n = 1usize << log_n;
    let data: Vec<u32> = (0..n).map(|i| ((i as u64 * 997 + 7) % P as u64) as u32).collect();
    let point: Vec<[u32; 5]> = (0..log_n)
        .map(|i| [((i as u64 * 1337 + 42) % P as u64) as u32, 0, 0, 0, 0])
        .collect();
    let gpu_result = g.mle_eval(&data, &point);
    let cpu_result = cpu_mle_eval(&data, &point);
    assert_eq!(gpu_result, cpu_result);
}
