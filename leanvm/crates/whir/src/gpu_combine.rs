//! GPU-accelerated combine_statement.
//!
//! Builds equality polynomial weights directly on GPU.
//! Eliminates the 640MB upload that was bottlenecking the GPU product sumcheck.

use field::{ExtensionField, PrimeCharacteristicRing};
use poly::*;
use cudarc::driver::safe::CudaSlice;

use crate::SparseStatement;
use crate::gpu_backend;
use poly::matrix_next_mle_folded;

/// Build combined weights on GPU.
///
/// Reimplements combine_statement but keeps the result on GPU as a CudaSlice.
/// For each sparse statement, builds eq(point, x) on GPU and accumulates.
///
/// Returns (d_weights_flat, sum) where d_weights_flat has n_total * 5 u32s
/// (flat extension field, NOT packed).
///
/// Returns None if GPU not available or statements are too complex.
pub(crate) fn gpu_combine_statement<EF>(
    statements: &[SparseStatement<EF>],
    gamma: EF,
    num_variables: usize,
) -> Option<(CudaSlice<u32>, EF)>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION; // 5
    let n_total = 1usize << num_variables;

    // Allocate zero weights on GPU.
    let mut d_weights = g.stream.memcpy_stod(
        &vec![0u32; n_total * dim],
    ).ok()?;

    let mut combined_sum = EF::ZERO;
    let mut gamma_pow = EF::ONE;

    for smt in statements {
        let inner_n_vars = smt.inner_num_variables();
        let inner_n = 1usize << inner_n_vars;

        // Build the polynomial on GPU (eq or next_mle depending on is_next).
        let d_poly = if smt.is_next {
            // next_mle: compute on CPU, upload to GPU (small: ~few KB per statement).
            let next_values = matrix_next_mle_folded::<EF>(&smt.point.0);
            let next_u32: &[u32] = unsafe {
                std::slice::from_raw_parts(next_values.as_ptr().cast::<u32>(), next_values.len() * dim)
            };
            g.stream.memcpy_stod(next_u32).ok()?
        } else {
            // eq polynomial: build on GPU.
            let point_ef_u32: Vec<[u32; 5]> = smt.point.0.iter().map(|p| {
                let mut out = [0u32; 5];
                let coeffs = p.as_basis_coefficients_slice();
                for (j, c) in coeffs.iter().enumerate() {
                    out[j] = unsafe { *(c as *const PF<EF> as *const u32) };
                }
                out
            }).collect();
            g.sumcheck.eq_polynomial_device(&point_ef_u32)
        };

        for evaluation in &smt.values {

            // Scale by gamma_pow and accumulate into weights at the selector offset.
            let scalar_u32 = {
                let mut out = [0u32; 5];
                let coeffs = gamma_pow.as_basis_coefficients_slice();
                for (j, c) in coeffs.iter().enumerate() {
                    out[j] = unsafe { *(c as *const PF<EF> as *const u32) };
                }
                out
            };

            let selector = evaluation.selector;
            let inner_n = 1usize << inner_n_vars;

            if selector == 0 && inner_n == n_total {
                // Dense case: polynomial covers the full domain. Accumulate directly.
                g.sumcheck.eq_accumulate_device(
                    &mut d_weights,
                    &d_poly,
                    &scalar_u32,
                    n_total as u32,
                );
            } else {
                // Sparse case: eq covers a sub-range at selector offset.
                // Use offset accumulator: weights[selector * inner_n + j] += scalar * eq[j]
                let offset = (selector * inner_n) as u32;
                g.sumcheck.eq_accumulate_offset_device(
                    &mut d_weights,
                    &d_poly,
                    &scalar_u32,
                    offset,
                    inner_n as u32,
                );
            }

            combined_sum += evaluation.value * gamma_pow;
            gamma_pow *= gamma;
        }
    }

    Some((d_weights, combined_sum))
}
