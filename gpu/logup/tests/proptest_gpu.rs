//! Property-based tests: GPU logup operations match CPU reference.

use cudarc::driver::safe::CudaContext;
use gpu_logup::*;
use proptest::prelude::*;

const P: u32 = 0x7F000001;

fn gpu() -> GpuLogup {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    GpuLogup::new(stream)
}

// ── Fingerprint tests ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_fingerprint_small(seed in 0u64..10000) {
        let g = gpu();
        let n_rows = 64u32;
        let n_cols = 4u32;
        let columns: Vec<u32> = (0..(n_rows * n_cols) as usize)
            .map(|i| (((i as u64 + seed) * 997 + 7) % P as u64) as u32)
            .collect();
        let alphas: Vec<u32> = (0..(n_cols * 5) as usize)
            .map(|i| (((i as u64 + seed) * 1337 + 13) % P as u64) as u32)
            .collect();
        let c: [u32; 5] = std::array::from_fn(|i| (((i as u64 + seed) * 47 + 3) % P as u64) as u32);

        let gpu_out = g.fingerprint(&columns, &alphas, &c, n_rows, n_cols);
        let cpu_out = cpu_fingerprint(&columns, &alphas, &c, n_rows as usize, n_cols as usize);
        prop_assert_eq!(gpu_out, cpu_out);
    }

    #[test]
    fn prop_fingerprint_medium(seed in 0u64..10000) {
        let g = gpu();
        let n_rows = 1024u32;
        let n_cols = 8u32;
        let columns: Vec<u32> = (0..(n_rows * n_cols) as usize)
            .map(|i| (((i as u64 + seed) * 31 + 5) % P as u64) as u32)
            .collect();
        let alphas: Vec<u32> = (0..(n_cols * 5) as usize)
            .map(|i| (((i as u64 + seed) * 73 + 11) % P as u64) as u32)
            .collect();
        let c: [u32; 5] = std::array::from_fn(|i| (((i as u64 + seed) * 101) % P as u64) as u32);

        let gpu_out = g.fingerprint(&columns, &alphas, &c, n_rows, n_cols);
        let cpu_out = cpu_fingerprint(&columns, &alphas, &c, n_rows as usize, n_cols as usize);
        prop_assert_eq!(gpu_out, cpu_out);
    }
}

#[test]
fn test_fingerprint_single_col() {
    let g = gpu();
    let n_rows = 4u32;
    let n_cols = 1u32;
    let columns = vec![10u32, 20, 30, 40];
    let alphas = vec![100u32, 0, 0, 0, 0]; // base-field alpha
    let c: [u32; 5] = [500, 0, 0, 0, 0];

    let gpu_out = g.fingerprint(&columns, &alphas, &c, n_rows, n_cols);
    let cpu_out = cpu_fingerprint(&columns, &alphas, &c, n_rows as usize, n_cols as usize);
    assert_eq!(gpu_out, cpu_out);
}

#[test]
fn test_fingerprint_all_zeros() {
    let g = gpu();
    let n_rows = 8u32;
    let n_cols = 3u32;
    let columns = vec![0u32; (n_rows * n_cols) as usize];
    let alphas = vec![0u32; (n_cols * 5) as usize];
    let c: [u32; 5] = [1000, 2000, 3000, 4000, 5000];

    let gpu_out = g.fingerprint(&columns, &alphas, &c, n_rows, n_cols);
    let cpu_out = cpu_fingerprint(&columns, &alphas, &c, n_rows as usize, n_cols as usize);
    assert_eq!(gpu_out, cpu_out);
    // With zero columns and zero alphas, fingerprint = 0, so denom = c for all rows.
    for i in 0..n_rows as usize {
        assert_eq!(&gpu_out[i * 5..(i + 1) * 5], &c);
    }
}

// ── Endianness reorder tests ─────────────────────────────────────────────

#[test]
fn test_reorder_chunk2() {
    let g = gpu();
    // chunk_log=1: reverse 1 bit within pairs. Pair (0,1) → (0,1), (2,3) → (2,3). No change for 1-bit reversal.
    // Actually bit_reverse of 0 in 1 bit = 0, bit_reverse of 1 in 1 bit = 1. So identity.
    let data: Vec<u32> = (0..8).collect();
    let out = g.endianness_reorder(&data, 1);
    assert_eq!(out, data); // 1-bit reversal is identity
}

#[test]
fn test_reorder_chunk4() {
    let g = gpu();
    // chunk_log=2: reverse 2 bits within groups of 4.
    // 0b00→0b00, 0b01→0b10, 0b10→0b01, 0b11→0b11
    // So within [0,1,2,3]: reorder to [0,2,1,3]
    let data: Vec<u32> = (0..8).collect();
    let out = g.endianness_reorder(&data, 2);
    assert_eq!(out, vec![0, 2, 1, 3, 4, 6, 5, 7]);
}

#[test]
fn test_reorder_chunk8() {
    let g = gpu();
    // chunk_log=3: reverse 3 bits within groups of 8.
    let data: Vec<u32> = (0..8).collect();
    let out = g.endianness_reorder(&data, 3);
    let cpu_out = cpu_endianness_reorder(&data, 3);
    assert_eq!(out, cpu_out);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_reorder(log_n in 4..14u32, chunk_log in 1..4u32) {
        let g = gpu();
        let n = 1usize << log_n;
        let data: Vec<u32> = (0..n as u32).collect();
        let gpu_out = g.endianness_reorder(&data, chunk_log);
        let cpu_out = cpu_endianness_reorder(&data, chunk_log);
        prop_assert_eq!(gpu_out, cpu_out);
    }
}

#[test]
fn test_reorder_roundtrip() {
    let g = gpu();
    // Applying the same reorder twice is identity.
    let data: Vec<u32> = (0..256).collect();
    let once = g.endianness_reorder(&data, 3);
    let twice = g.endianness_reorder(&once, 3);
    assert_eq!(twice, data);
}
