//! Test that GPU prepare_evals_for_fft matches the CPU version.

use cudarc::driver::safe::CudaContext;
use gpu_ntt::GpuNtt;
use rand::{RngExt, SeedableRng, rngs::StdRng};

const P: u32 = 0x7F000001;

fn cpu_prepare_evals_for_fft(evals: &[u32], folding_factor: usize, log_inv_rate: usize, dft_n_cols: usize) -> Vec<u32> {
    let n_blocks = 1usize << folding_factor;
    let full_len = evals.len() << log_inv_rate;
    let block_size = full_len / n_blocks;
    let log_block_size = block_size.trailing_zeros() as usize;
    let out_len = block_size * dft_n_cols;

    (0..out_len)
        .map(|i| {
            let block_index = i % dft_n_cols;
            let offset_in_block = i / dft_n_cols;
            let src_index = ((block_index << log_block_size) + offset_in_block) >> log_inv_rate;
            if src_index < evals.len() { evals[src_index] } else { 0 }
        })
        .collect()
}

#[test]
fn test_prepare_evals_for_fft_matches_cpu() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    let ntt = GpuNtt::new(stream.clone());
    let mut rng = StdRng::seed_from_u64(42);

    for log_n in [10, 14, 18] {
        for folding_factor in [3, 5, 7] {
            for log_inv_rate in [1, 2] {
                let n = 1usize << log_n;
                let n_cols = (1usize << folding_factor).min(n);
                if n < n_cols { continue; }

                let evals: Vec<u32> = (0..n).map(|_| rng.random_range(0..P)).collect();

                let cpu_result = cpu_prepare_evals_for_fft(&evals, folding_factor, log_inv_rate, n_cols);

                let d_evals = stream.memcpy_stod(&evals).unwrap();
                let d_result = ntt.prepare_evals_for_fft_device(
                    &d_evals,
                    n as u32,
                    n_cols as u32,
                    log_inv_rate as u32,
                );
                let gpu_result = stream.memcpy_dtov(&d_result).unwrap();

                assert_eq!(
                    gpu_result.len(),
                    cpu_result.len(),
                    "Length mismatch: log_n={log_n}, ff={folding_factor}, lr={log_inv_rate}"
                );

                let mismatches: Vec<usize> = (0..cpu_result.len())
                    .filter(|&i| gpu_result[i] != cpu_result[i])
                    .collect();

                assert!(
                    mismatches.is_empty(),
                    "Mismatches at {} positions (first 5: {:?}) for log_n={log_n}, ff={folding_factor}, lr={log_inv_rate}",
                    mismatches.len(),
                    &mismatches[..mismatches.len().min(5)],
                );
            }
        }
    }
}

/// Test full reorder+DFT+Merkle pipeline against CPU Merkle on the same polynomial.
#[test]
fn test_reorder_dft_gpu_matches_cpu() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let stream = ctx.default_stream();
    let ntt = GpuNtt::new(stream.clone());
    let mut rng = StdRng::seed_from_u64(99);

    // Small test case: 2^10 evals, folding_factor=3, log_inv_rate=1.
    let log_n = 10;
    let n = 1usize << log_n;
    let folding_factor = 3;
    let log_inv_rate = 1;
    let n_cols = 1usize << folding_factor; // 8

    let evals: Vec<u32> = (0..n).map(|_| rng.random_range(0..P)).collect();

    // CPU: prepare + DFT
    let cpu_prepared = cpu_prepare_evals_for_fft(&evals, folding_factor, log_inv_rate, n_cols);
    let full_len = n << log_inv_rate;
    let block_size = full_len / n_cols;
    let log_height = block_size.trailing_zeros() as usize;
    let cpu_dft = gpu_ntt::cpu_dft(&cpu_prepared, log_height, n_cols);

    // GPU: reorder_and_dft_device
    let d_evals = stream.memcpy_stod(&evals).unwrap();
    let d_dft = ntt.reorder_and_dft_device(&d_evals, n as u32, folding_factor, log_inv_rate);
    let gpu_dft = stream.memcpy_dtov(&d_dft).unwrap();

    assert_eq!(gpu_dft.len(), cpu_dft.len(), "DFT output length mismatch");

    let mismatches: Vec<usize> = (0..cpu_dft.len())
        .filter(|&i| gpu_dft[i] != cpu_dft[i])
        .collect();

    assert!(
        mismatches.is_empty(),
        "DFT mismatches at {} positions (first 5: {:?})",
        mismatches.len(),
        &mismatches[..mismatches.len().min(5)],
    );
}
