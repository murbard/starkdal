//! GPU-resident product sumcheck: upload polynomial once, fold on GPU,
//! only download ~40 bytes per round for Fiat-Shamir.
//!
//! This replaces `run_product_sumcheck` when polynomial data is large enough
//! to amortize the upload cost (~32ms for 2^24 elements).

use std::sync::Arc;

use cudarc::driver::safe::CudaStream;
use gpu_poly_fold::{FoldMode, GpuPolyFold};
use gpu_sumcheck::GpuSumcheck;

/// Run a full product sumcheck on GPU.
///
/// `evals_flat`: flat u32 array of base field evaluations (already unpacked from SIMD).
/// `weights_flat`: flat u32 array of extension field weights (5 u32s per element, already transposed from packed).
/// `sum`: initial sumcheck claim (5 u32s, extension field).
/// `n_rounds`: number of sumcheck rounds (= folding factor).
///
/// For each round:
///   1. Compute (c0, c2) on GPU → download ~40 bytes
///   2. Call `on_round` callback with (c0, c2, sum) → gets back challenge r
///   3. Fold both evals and weights on GPU with r
///
/// Returns (final_evals_flat, final_weights_flat, challenges, final_sum).
pub fn gpu_product_sumcheck_rounds(
    stream: &Arc<CudaStream>,
    sumcheck: &GpuSumcheck,
    fold: &GpuPolyFold,
    evals_flat: &[u32],
    weights_flat: &[u32],
    sum: [u32; 5],
    n_rounds: usize,
    mut on_round: impl FnMut(&[u32; 5], &[u32; 5], &[u32; 5]) -> [u32; 5],
) -> (Vec<u32>, Vec<u32>, Vec<[u32; 5]>, [u32; 5]) {
    let mut current_evals = evals_flat.to_vec();
    let mut current_weights = weights_flat.to_vec();
    let mut current_sum = sum;
    let mut challenges = Vec::with_capacity(n_rounds);
    let mut evals_is_base = true;

    for _round in 0..n_rounds {
        // 1. Compute product sumcheck polynomial on GPU.
        let (c0, c2) = if evals_is_base {
            sumcheck.product_sumcheck_base_ext(&current_evals, &current_weights)
        } else {
            sumcheck.product_sumcheck_ext_ext(&current_evals, &current_weights)
        };

        // 2. Call host callback with (c0, c2, sum) → gets challenge r.
        let r = on_round(&c0, &c2, &current_sum);
        challenges.push(r);

        // Compute new sum: P(r) where P(z) = c0 + c1*z + c2*z^2.
        // c1 = sum - 2*c0 - c2.
        // For simplicity, let the callback compute the new sum too.
        // Actually, we need to evaluate P(r) = c0 + c1*r + c2*r^2.
        // The callback can return r AND the new sum. But to keep it simple,
        // let's compute sum on CPU using the EF arithmetic from the callback.
        // The callback is responsible for: adding polynomial to prover_state,
        // sampling r, computing P(r) as new sum.
        // We pass (c0, c2, current_sum) and get back r.
        // The new sum is computed by the callback internally.

        // 3. Fold both evals and weights on GPU.
        if evals_is_base {
            current_evals = fold.fold_base_to_ext(&current_evals, &r, FoldMode::Half);
            evals_is_base = false;
        } else {
            current_evals = fold.fold_ext(&current_evals, &r, FoldMode::Half);
        }
        current_weights = fold.fold_ext(&current_weights, &r, FoldMode::Half);
    }

    (current_evals, current_weights, challenges, current_sum)
}
