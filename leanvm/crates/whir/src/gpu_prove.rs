//! GPU-resident initial sumcheck rounds.
//!
//! Uploads unpacked polynomial and packed weights to GPU once.
//! All fold + product sumcheck rounds operate on device memory.
//! Only ~200 bytes per round crosses PCIe for Fiat-Shamir.

use fiat_shamir::FSProver;
use field::{ExtensionField, Field, PrimeCharacteristicRing};
use poly::*;
use tracing::instrument;

use crate::SumcheckSingle;
use crate::gpu_backend;

/// GPU-accelerated initial sumcheck rounds.
///
/// Takes unpacked base-field evals and packed ext-field weights.
/// Uploads both to GPU, transposes weights, runs product sumcheck + fold
/// for `folding_factor` rounds, downloads result.
///
/// Returns None if GPU not available.
#[instrument(name = "GPU initial sumcheck", skip_all)]
pub(crate) fn gpu_initial_sumcheck_rounds<EF>(
    base_evals: &[PF<EF>],
    weights_packed: &[EFPacking<EF>],
    sum: EF,
    prover_state: &mut impl FSProver<EF>,
    folding_factor: usize,
    pow_bits: usize,
) -> Option<(SumcheckSingle<EF>, MultilinearPoint<EF>)>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION; // 5
    let n_evals = base_evals.len();
    let pf_width = packing_width::<EF>(); // SIMD packing width for extension field

    // Flat u32 view of base-field evals.
    let evals_u32: &[u32] = unsafe { std::slice::from_raw_parts(base_evals.as_ptr().cast::<u32>(), n_evals) };

    // Transpose packed weights to flat ext layout on GPU.
    let n_packed = weights_packed.len();
    let u32_per_packed = dim * pf_width;
    let weights_packed_u32: &[u32] =
        unsafe { std::slice::from_raw_parts(weights_packed.as_ptr().cast::<u32>(), n_packed * u32_per_packed) };
    let n_weight_scalars = n_packed * pf_width;

    assert_eq!(
        n_evals, n_weight_scalars,
        "GPU initial sumcheck: evals ({n_evals}) != weights ({n_weight_scalars})"
    );

    let weights_flat = if pf_width == 1 {
        weights_packed_u32.to_vec()
    } else {
        g.sumcheck
            .transpose_packed_ext(weights_packed_u32, n_packed as u32, dim as u32, pf_width as u32)
    };

    tracing::info!(
        "GPU initial sumcheck: n={n_evals}, folding_factor={folding_factor}, upload={:.1}MB",
        (n_evals * 4 + n_weight_scalars * dim * 4) as f64 / 1e6,
    );

    // Upload data to GPU ONCE.
    let mut d_evals = g.stream.memcpy_stod(evals_u32).ok()?;
    let mut d_weights = g.stream.memcpy_stod(&weights_flat).ok()?;

    let mut current_sum = sum;
    let mut challenges = Vec::with_capacity(folding_factor);
    let mut evals_is_base = true;
    let mut n_elements = n_evals;

    for _round in 0..folding_factor {
        let half = (n_elements / 2) as u32;

        // 1. Compute (c0, c2) on GPU — data stays on device, only 40 bytes downloaded.
        let (c0_u32, c2_u32) = if evals_is_base {
            g.sumcheck.product_sumcheck_base_ext_device(&d_evals, &d_weights, half)
        } else {
            g.sumcheck.product_sumcheck_ext_ext_device(&d_evals, &d_weights, half)
        };

        // 2. Convert c0, c2 to EF, compute polynomial (CPU, ~200 bytes).
        let c0 = ef_from_u32::<EF>(&c0_u32);
        let c2 = ef_from_u32::<EF>(&c2_u32);
        let c1 = current_sum - c0.double() - c2;
        let poly = DensePolynomial::new(vec![c0, c1, c2]);

        // 3. Fiat-Shamir (CPU, ~200 bytes total).
        prover_state.add_sumcheck_polynomial(&poly.coeffs, None);
        prover_state.pow_grinding(pow_bits);
        let r: EF = prover_state.sample();
        current_sum = poly.evaluate(r);
        let r_u32 = ef_to_u32::<EF>(&r);
        challenges.push(r);

        // 4. Fold evals and weights ON GPU — data stays on device.
        if evals_is_base {
            d_evals = g.fold.fold_base_to_ext_device(&d_evals, half, &r_u32);
            evals_is_base = false;
        } else {
            d_evals = g.fold.fold_ext_device(&d_evals, half, &r_u32);
        }
        d_weights = g.fold.fold_ext_device(&d_weights, half, &r_u32);
        n_elements /= 2;
    }

    // Download final folded data from GPU (small: ~few MB).
    let current_evals = g.stream.memcpy_dtov(&d_evals).ok()?;
    let current_weights = g.stream.memcpy_dtov(&d_weights).ok()?;

    // Reconstruct as ExtensionPacked for downstream compatibility.
    let n_final_ext = current_evals.len() / dim;
    let n_packed_final = n_final_ext / pf_width;
    assert_eq!(n_packed_final * pf_width, n_final_ext);

    // Repack evals: flat → packed layout.
    let evals_packed_out = repack_to_extension_packed::<EF>(&current_evals, n_packed_final, dim, pf_width);
    let weights_packed_out = repack_to_extension_packed::<EF>(&current_weights, n_packed_final, dim, pf_width);

    let sumcheck = SumcheckSingle {
        evals: MleOwned::ExtensionPacked(evals_packed_out),
        weights: MleOwned::ExtensionPacked(weights_packed_out),
        sum: current_sum,
    };

    Some((sumcheck, MultilinearPoint(challenges)))
}

/// GPU initial sumcheck with DEVICE-RESIDENT weights (from gpu_combine_statement).
/// Only evals need to be uploaded (~128MB). Weights are already on GPU (~640MB saved!).
#[instrument(name = "GPU initial sumcheck (device weights)", skip_all)]
pub(crate) fn gpu_initial_sumcheck_with_device_weights<EF>(
    base_evals: &[PF<EF>],
    d_weights: cudarc::driver::safe::CudaSlice<u32>,
    sum: EF,
    prover_state: &mut impl FSProver<EF>,
    folding_factor: usize,
    pow_bits: usize,
    num_variables: usize,
) -> Option<(SumcheckSingle<EF>, MultilinearPoint<EF>)>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION;
    let n_evals = base_evals.len();
    let pf_width = packing_width::<EF>();

    let evals_u32: &[u32] = unsafe { std::slice::from_raw_parts(base_evals.as_ptr().cast::<u32>(), n_evals) };

    tracing::info!(
        "GPU initial sumcheck (device weights): n={n_evals}, upload_evals={:.1}MB, weights already on GPU",
        (n_evals * 4) as f64 / 1e6,
    );

    // Upload ONLY evals (weights are already on GPU!).
    let mut d_evals = g.stream.memcpy_stod(evals_u32).ok()?;
    let mut d_weights = d_weights; // already on device

    let mut current_sum = sum;
    let mut challenges = Vec::with_capacity(folding_factor);
    let mut evals_is_base = true;
    let mut n_elements = n_evals;

    for _round in 0..folding_factor {
        let half = (n_elements / 2) as u32;

        let (c0_u32, c2_u32) = if evals_is_base {
            g.sumcheck.product_sumcheck_base_ext_device(&d_evals, &d_weights, half)
        } else {
            g.sumcheck.product_sumcheck_ext_ext_device(&d_evals, &d_weights, half)
        };

        let c0 = ef_from_u32::<EF>(&c0_u32);
        let c2 = ef_from_u32::<EF>(&c2_u32);
        let c1 = current_sum - c0.double() - c2;
        let poly = DensePolynomial::new(vec![c0, c1, c2]);

        prover_state.add_sumcheck_polynomial(&poly.coeffs, None);
        prover_state.pow_grinding(pow_bits);
        let r: EF = prover_state.sample();
        current_sum = poly.evaluate(r);
        let r_u32 = ef_to_u32::<EF>(&r);
        challenges.push(r);

        if evals_is_base {
            d_evals = g.fold.fold_base_to_ext_device(&d_evals, half, &r_u32);
            evals_is_base = false;
        } else {
            d_evals = g.fold.fold_ext_device(&d_evals, half, &r_u32);
        }
        d_weights = g.fold.fold_ext_device(&d_weights, half, &r_u32);
        n_elements /= 2;
    }

    let current_evals = g.stream.memcpy_dtov(&d_evals).ok()?;
    let current_weights = g.stream.memcpy_dtov(&d_weights).ok()?;

    let n_final_ext = current_evals.len() / dim;
    let n_packed_final = n_final_ext / pf_width;
    assert_eq!(n_packed_final * pf_width, n_final_ext);

    let evals_packed_out = repack_to_extension_packed::<EF>(&current_evals, n_packed_final, dim, pf_width);
    let weights_packed_out = repack_to_extension_packed::<EF>(&current_weights, n_packed_final, dim, pf_width);

    let sumcheck = SumcheckSingle {
        evals: MleOwned::ExtensionPacked(evals_packed_out),
        weights: MleOwned::ExtensionPacked(weights_packed_out),
        sum: current_sum,
    };

    Some((sumcheck, MultilinearPoint(challenges)))
}

/// Repack flat u32 array to Vec<EFPacking<EF>>.
fn repack_to_extension_packed<EF: ExtensionField<PF<EF>>>(
    flat: &[u32],
    n_packed: usize,
    dim: usize,
    pf_width: usize,
) -> Vec<EFPacking<EF>> {
    let u32_per_packed = dim * pf_width;
    let mut packed_u32 = vec![0u32; n_packed * u32_per_packed];
    for pi in 0..n_packed {
        for comp in 0..dim {
            for lane in 0..pf_width {
                let ext_idx = pi * pf_width + lane;
                packed_u32[pi * u32_per_packed + comp * pf_width + lane] = flat[ext_idx * dim + comp];
            }
        }
    }
    unsafe {
        let mut out = std::mem::ManuallyDrop::new(packed_u32);
        Vec::from_raw_parts(out.as_mut_ptr().cast::<EFPacking<EF>>(), n_packed, n_packed)
    }
}

pub(crate) fn ef_from_u32<EF: ExtensionField<PF<EF>>>(v: &[u32; 5]) -> EF {
    EF::from_basis_coefficients_fn(|j| unsafe { *(&v[j] as *const u32 as *const PF<EF>) })
}

pub(crate) fn ef_to_u32<EF: ExtensionField<PF<EF>>>(v: &EF) -> [u32; 5] {
    let mut out = [0u32; 5];
    let coeffs = v.as_basis_coefficients_slice();
    for (j, c) in coeffs.iter().enumerate() {
        out[j] = unsafe { *(c as *const PF<EF> as *const u32) };
    }
    out
}
