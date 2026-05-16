//! GPU-accelerated GKR layer proving.
//!
//! Matches the CPU's Fiat-Shamir sequence exactly by using the same
//! bit-reversed fold ordering. The GPU kernel takes a `fold_bit` parameter
//! that specifies which bit to fold on each round, eliminating the need for
//! SIMD packing while producing identical round polynomials.

use super::layers::LayerStorage;
use backend::*;
use cudarc::driver::safe::CudaSlice;

/// GPU-accelerated GKR layer proving.
///
/// Operates on the layer data in its native layout (bit-reversed or natural).
/// Uses `fold_bit` parameter to match the CPU's fold schedule exactly.
///
/// IMPORTANT: No prover_state interaction before the alpha sample.
/// The alpha sample is the FIRST F-S interaction, matching the CPU.
pub(super) fn gpu_prove_gkr_layer<EF: ExtensionField<PF<EF>>>(
    prover_state: &mut impl FSProver<EF>,
    layer: &LayerStorage<'_, EF>,
    claim_point: &MultilinearPoint<EF>,
    claim_num: EF,
    claim_den: EF,
) -> Option<(MultilinearPoint<EF>, EF, EF)> {
    // Only handle Natural layers (which have been converted already by the GKR tree).
    // For Initial/PackedBr layers, fall back to CPU (no F-S interaction yet).
    let (nums, dens) = match layer {
        LayerStorage::Natural { nums, dens } => (nums.as_ref(), dens.as_ref()),
        _ => return None,
    };

    let n = nums.len();
    if n < 64 {
        return None;
    }

    // Get GPU backend (no F-S interaction).
    use std::sync::OnceLock;
    static GPU_SC: OnceLock<Option<(gpu_sumcheck::GpuSumcheck, gpu_poly_fold::GpuPolyFold)>> = OnceLock::new();
    let (gpu, fold_gpu) = GPU_SC
        .get_or_init(|| {
            let ctx = cudarc::driver::safe::CudaContext::new(0).ok()?;
            let stream = ctx.default_stream();
            let sc = gpu_sumcheck::GpuSumcheck::new(stream.clone());
            let fo = gpu_poly_fold::GpuPolyFold::new(stream);
            Some((sc, fo))
        })
        .as_ref()?;

    let dim = EF::DIMENSION;
    let n_rounds = claim_point.0.len();

    // ── COMMIT to GPU path. Sample alpha. ──
    let alpha: EF = prover_state.sample();
    let expected_sum = claim_num + alpha * claim_den;

    // Even/odd split: left = evens, right = odds.
    let (num_l, num_r) = super::sumcheck_utils::even_odd_split(nums);
    let (den_l, den_r) = super::sumcheck_utils::even_odd_split(dens);

    // Upload to GPU.
    let flat = |v: &[EF]| -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u32>(), v.len() * dim) }.to_vec()
    };
    let mut d_nl = gpu.stream().memcpy_stod(&flat(&num_l)).unwrap();
    let mut d_nr = gpu.stream().memcpy_stod(&flat(&num_r)).unwrap();
    let mut d_dl = gpu.stream().memcpy_stod(&flat(&den_l)).unwrap();
    let mut d_dr = gpu.stream().memcpy_stod(&flat(&den_r)).unwrap();

    // The eq point is the claim_point reversed (matching CPU's eq_alphas_rev).
    let eq_point: Vec<EF> = claim_point.0.iter().rev().copied().collect();
    let mut sum = expected_sum;
    let mut mmf = EF::ONE;
    let mut challenges = Vec::with_capacity(n_rounds);

    let ef_to_u32 = |v: &EF| -> [u32; 5] { unsafe { std::mem::transmute_copy(v) } };
    let ef_from_u32 = |v: &[u32; 5]| -> EF { unsafe { std::mem::transmute_copy(v) } };

    // Track actual (non-padded) element counts for padding correction.
    let mut active_l = num_l.len();
    let mut active_r = num_r.len();

    for round in 0..n_rounds {
        let eq_alpha = eq_point[n_rounds - 1 - round];
        let eq_prefix = &eq_point[..n_rounds - 1 - round];

        let active_pairs = active_l.div_ceil(2);
        let eq_table = eval_eq::<EF>(eq_prefix);

        // Build eq table on GPU.
        let eq_u32 = flat(&eq_table);
        let d_eq = gpu.stream().memcpy_stod(&eq_u32).unwrap();

        // GPU quotient sumcheck (fold_bit = 0, adjacent pairs).
        let (c0n_u32, c2n_u32, c0d_u32, c2d_u32) =
            gpu.gkr_quotient_sumcheck_device(&d_nl, &d_nr, &d_dl, &d_dr, &d_eq, active_pairs as u32, 0);
        let c0_num = ef_from_u32(&c0n_u32);
        let c2_num = ef_from_u32(&c2n_u32);
        let c0_den = ef_from_u32(&c0d_u32);
        let c2_den = ef_from_u32(&c2d_u32);

        // Padding correction: alpha * mle_of_zeros_then_ones(active_pairs, eq_prefix).
        let padding_sum = alpha * mle_of_zeros_then_ones(active_pairs, eq_prefix);

        let c0_raw = c0_num + alpha * c0_den + padding_sum;
        let c2_raw = c2_num + alpha * c2_den;
        let c0_mmf = c0_raw * mmf;
        let c2_mmf = c2_raw * mmf;
        let h1_mmf = (sum - (EF::ONE - eq_alpha) * c0_mmf) / eq_alpha;
        let c1_mmf = h1_mmf - c0_mmf - c2_mmf;
        let bare = DensePolynomial::new(vec![c0_mmf, c1_mmf, c2_mmf]);

        prover_state.add_sumcheck_polynomial(&bare.coeffs, Some(eq_alpha));
        let r: EF = prover_state.sample();
        let eq_eval = (EF::ONE - eq_alpha) * (EF::ONE - r) + eq_alpha * r;
        sum = eq_eval * bare.evaluate(r);
        mmf *= eq_eval;
        challenges.push(r);

        // Fold all 4 arrays (LSB mode, adjacent pairs).
        let r_u32 = ef_to_u32(&r);
        d_nl = fold_gpu.fold_ext_lsb_device(&d_nl, active_pairs as u32, &r_u32);
        d_nr = fold_gpu.fold_ext_lsb_device(&d_nr, active_pairs as u32, &r_u32);
        d_dl = fold_gpu.fold_ext_lsb_device(&d_dl, active_pairs as u32, &r_u32);
        d_dr = fold_gpu.fold_ext_lsb_device(&d_dr, active_pairs as u32, &r_u32);

        active_l = active_l.div_ceil(2);
        active_r = active_r.div_ceil(2);
    }

    // Download final 4 values.
    let get_final = |d: &CudaSlice<u32>| -> EF {
        let v = gpu.stream().memcpy_dtov(d).unwrap();
        ef_from_u32(&v[..5].try_into().unwrap())
    };
    let nl_val = get_final(&d_nl);
    let nr_val = get_final(&d_nr);
    let dl_val = get_final(&d_dl);
    let dr_val = get_final(&d_dr);

    let inner_evals = [nl_val, nr_val, dl_val, dr_val];
    prover_state.add_extension_scalars(&inner_evals);
    let beta: EF = prover_state.sample();
    let next_num = (EF::ONE - beta) * nl_val + beta * nr_val;
    let next_den = (EF::ONE - beta) * dl_val + beta * dr_val;

    let mut q_natural: Vec<EF> = challenges.into_iter().rev().collect();
    q_natural.push(beta);

    Some((MultilinearPoint(q_natural), next_num, next_den))
}
