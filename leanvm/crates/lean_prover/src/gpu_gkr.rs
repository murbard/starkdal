//! GPU GKR quotient protocol — full implementation.
//!
//! Layer building: GPU `gkr_sum_quotients_at_bit_device`.
//! Layer proving: GPU quotient sumcheck + fold. Four separate arrays
//! (num_l, num_r, den_l, den_r) kept throughout, matching CPU's `run_phase2_sumcheck`.

use super::gpu_prove_execution::{Gpu, GpuFsPhase, TranscriptChunk};
use backend::*;
use cudarc::driver::{safe::CudaSlice, sys};
use sub_protocols::N_VARS_TO_SEND_GKR_COEFFS;

type F = lean_vm::F;
type EF = lean_vm::EF;

pub(crate) struct GpuGkrQuotientOutput {
    pub(crate) quotient: Option<EF>,
    pub(crate) point: Option<MultilinearPoint<EF>>,
    pub(crate) d_point_words: CudaSlice<u32>,
}

fn ef_to_u32(v: &EF) -> [u32; 5] {
    unsafe { std::mem::transmute_copy(v) }
}
#[cfg(test)]
fn kb_u32(v: F) -> u32 {
    unsafe { std::mem::transmute(v) }
}
fn kb_from_u32(v: u32) -> F {
    unsafe { std::mem::transmute(v) }
}
fn ef_from_u32(v: &[u32; 5]) -> EF {
    unsafe { std::mem::transmute_copy(v) }
}
fn flat_ef(v: &[EF]) -> Vec<u32> {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u32>(), v.len() * 5) }.to_vec()
}

struct GkrRoundCaptureState {
    d_eq: CudaSlice<u32>,
    d_eq_alpha: CudaSlice<u32>,
    d_eq_prefix: CudaSlice<u32>,
    eq_prefix_len: u32,
    active_pairs: u32,
    input_buf_len: u32,
}

struct GkrLayerCaptureState {
    d_alpha: CudaSlice<u32>,
    d_sum: CudaSlice<u32>,
    d_mmf: CudaSlice<u32>,
    rounds: Vec<GkrRoundCaptureState>,
    d_nl_states: Vec<CudaSlice<u32>>,
    d_nr_states: Vec<CudaSlice<u32>>,
    d_dl_states: Vec<CudaSlice<u32>>,
    d_dr_states: Vec<CudaSlice<u32>>,
    d_round_challenges: CudaSlice<u32>,
    d_round_rs: Vec<CudaSlice<u32>>,
    d_round_tails: Vec<CudaSlice<u32>>,
    d_c0n_partials: CudaSlice<u32>,
    d_c2n_partials: CudaSlice<u32>,
    d_c0d_partials: CudaSlice<u32>,
    d_c2d_partials: CudaSlice<u32>,
    d_c0n_out: CudaSlice<u32>,
    d_c2n_out: CudaSlice<u32>,
    d_c0d_out: CudaSlice<u32>,
    d_c2d_out: CudaSlice<u32>,
    d_beta: CudaSlice<u32>,
    d_next_claim_num: CudaSlice<u32>,
    d_next_claim_den: CudaSlice<u32>,
    d_inner_words: CudaSlice<u32>,
    d_next_point_words: CudaSlice<u32>,
}

fn prepare_gkr_layer_capture_state(
    g: &Gpu,
    d_layer_nums: CudaSlice<u32>,
    d_layer_dens: CudaSlice<u32>,
    layer_active_len: usize,
    layer_buf_len: usize,
    n_rounds: usize,
) -> GkrLayerCaptureState {
    let zero = ef_to_u32(&EF::ZERO);
    let one = ef_to_u32(&EF::ONE);
    let left_actual = layer_active_len.div_ceil(2);
    let initial_buf_len = left_actual.next_power_of_two().max(1);

    debug_assert_eq!(g.d_ext_zero_one.len(), zero.len() + one.len());
    let mut d_nl_raw = g.stream.alloc_zeros::<u32>(left_actual * 5).unwrap();
    let mut d_nr_raw = g.stream.alloc_zeros::<u32>(left_actual * 5).unwrap();
    let mut d_dl_raw = g.stream.alloc_zeros::<u32>(left_actual * 5).unwrap();
    let mut d_dr_raw = g.stream.alloc_zeros::<u32>(left_actual * 5).unwrap();
    g.sumcheck.fold_multi_col_ext_at_bit_into_async(
        &d_layer_nums,
        &mut d_nl_raw,
        layer_buf_len as u32,
        left_actual as u32,
        1,
        &g.d_ext_zero_one,
        0,
        0,
    );
    g.sumcheck.fold_multi_col_ext_at_bit_into_async(
        &d_layer_nums,
        &mut d_nr_raw,
        layer_buf_len as u32,
        left_actual as u32,
        1,
        &g.d_ext_zero_one,
        5,
        0,
    );
    g.sumcheck.fold_multi_col_ext_at_bit_into_async(
        &d_layer_dens,
        &mut d_dl_raw,
        layer_buf_len as u32,
        left_actual as u32,
        1,
        &g.d_ext_zero_one,
        0,
        0,
    );
    g.sumcheck.fold_multi_col_ext_at_bit_into_async(
        &d_layer_dens,
        &mut d_dr_raw,
        layer_buf_len as u32,
        left_actual as u32,
        1,
        &g.d_ext_zero_one,
        5,
        0,
    );

    let mut active_l = left_actual;
    let mut working_buf_len = initial_buf_len;
    let mut rounds = Vec::with_capacity(n_rounds);
    let mut output_buf_lens = Vec::with_capacity(n_rounds);
    let mut max_blocks = 1usize;
    for round in 0..n_rounds {
        let eq_prefix_len = n_rounds - 1 - round;
        let active_pairs = active_l.div_ceil(2);
        let eq_len = (1usize << eq_prefix_len).max(1);
        max_blocks = max_blocks.max(active_pairs.div_ceil(256).max(1));
        output_buf_lens.push(active_pairs.next_power_of_two().max(1));
        rounds.push(GkrRoundCaptureState {
            d_eq: g.sumcheck.stream().alloc_zeros::<u32>(eq_len * 5).unwrap(),
            d_eq_alpha: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
            d_eq_prefix: g
                .sumcheck
                .stream()
                .alloc_zeros::<u32>((eq_prefix_len * 5).max(1))
                .unwrap(),
            eq_prefix_len: eq_prefix_len as u32,
            active_pairs: active_pairs as u32,
            input_buf_len: working_buf_len as u32,
        });
        active_l = active_pairs;
        working_buf_len = output_buf_lens[round];
    }

    let mut d_nl_states = Vec::with_capacity(n_rounds + 1);
    let mut d_nr_states = Vec::with_capacity(n_rounds + 1);
    let mut d_dl_states = Vec::with_capacity(n_rounds + 1);
    let mut d_dr_states = Vec::with_capacity(n_rounds + 1);
    let mut d_nl0 = g.trace_ops.fill_ext(EF::ZERO, initial_buf_len as u32);
    let mut d_nr0 = g.trace_ops.fill_ext(EF::ZERO, initial_buf_len as u32);
    let mut d_dl0 = g.trace_ops.fill_ext(EF::ONE, initial_buf_len as u32);
    let mut d_dr0 = g.trace_ops.fill_ext(EF::ONE, initial_buf_len as u32);
    g.trace_ops
        .copy_to_offset_device(&d_nl_raw, &mut d_nl0, (left_actual * 5) as u32, 0);
    g.trace_ops
        .copy_to_offset_device(&d_nr_raw, &mut d_nr0, (left_actual * 5) as u32, 0);
    g.trace_ops
        .copy_to_offset_device(&d_dl_raw, &mut d_dl0, (left_actual * 5) as u32, 0);
    g.trace_ops
        .copy_to_offset_device(&d_dr_raw, &mut d_dr0, (left_actual * 5) as u32, 0);
    d_nl_states.push(d_nl0);
    d_nr_states.push(d_nr0);
    d_dl_states.push(d_dl0);
    d_dr_states.push(d_dr0);
    for &buf_len in &output_buf_lens {
        d_nl_states.push(g.trace_ops.fill_ext(EF::ZERO, buf_len as u32));
        d_nr_states.push(g.trace_ops.fill_ext(EF::ZERO, buf_len as u32));
        d_dl_states.push(g.trace_ops.fill_ext(EF::ONE, buf_len as u32));
        d_dr_states.push(g.trace_ops.fill_ext(EF::ONE, buf_len as u32));
    }

    let partial_words = max_blocks * 5;
    GkrLayerCaptureState {
        d_alpha: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_sum: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_mmf: g.trace_ops.fill_ext(EF::ONE, 1),
        rounds,
        d_nl_states,
        d_nr_states,
        d_dl_states,
        d_dr_states,
        d_round_challenges: g.sumcheck.stream().alloc_zeros::<u32>((n_rounds + 1) * 5).unwrap(),
        d_round_rs: (0..n_rounds)
            .map(|_| g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap())
            .collect(),
        d_round_tails: (0..n_rounds)
            .map(|_| g.sumcheck.stream().alloc_zeros::<u32>(10).unwrap())
            .collect(),
        d_c0n_partials: g.sumcheck.stream().alloc_zeros::<u32>(partial_words).unwrap(),
        d_c2n_partials: g.sumcheck.stream().alloc_zeros::<u32>(partial_words).unwrap(),
        d_c0d_partials: g.sumcheck.stream().alloc_zeros::<u32>(partial_words).unwrap(),
        d_c2d_partials: g.sumcheck.stream().alloc_zeros::<u32>(partial_words).unwrap(),
        d_c0n_out: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_c2n_out: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_c0d_out: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_c2d_out: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_beta: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_next_claim_num: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_next_claim_den: g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap(),
        d_inner_words: g.sumcheck.stream().alloc_zeros::<u32>(20).unwrap(),
        d_next_point_words: g.sumcheck.stream().alloc_zeros::<u32>((n_rounds + 1) * 5).unwrap(),
    }
}

fn run_gkr_layers_captured(
    g: &Gpu,
    mut capture_layers: Vec<GkrLayerCaptureState>,
    d_point_words: CudaSlice<u32>,
    d_claim_num: CudaSlice<u32>,
    d_claim_den: CudaSlice<u32>,
    d_challenger_state: &mut CudaSlice<u32>,
    d_empty_observe: &CudaSlice<u32>,
    d_p16_rc: &CudaSlice<u32>,
    d_p16_mds: &CudaSlice<u32>,
    d_p16_sparse: &CudaSlice<u32>,
) -> (Vec<TranscriptChunk>, CudaSlice<u32>) {
    let stream = g.sumcheck.stream().clone();
    stream
        .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
        .unwrap();

    for layer_idx in 0..capture_layers.len() {
        let (completed_layers, remaining_layers) = capture_layers.split_at_mut(layer_idx);
        let layer = &mut remaining_layers[0];
        let current_claim_num = if layer_idx == 0 {
            &d_claim_num
        } else {
            &completed_layers[layer_idx - 1].d_next_claim_num
        };
        let current_claim_den = if layer_idx == 0 {
            &d_claim_den
        } else {
            &completed_layers[layer_idx - 1].d_next_claim_den
        };
        let current_point_words = if layer_idx == 0 {
            &d_point_words
        } else {
            &completed_layers[layer_idx - 1].d_next_point_words
        };

        g.sumcheck.challenger_sample_exts_device_into_async(
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            1,
            d_empty_observe,
            &mut layer.d_alpha,
        );
        g.sumcheck.ext_affine_combine_into_async(
            current_claim_num,
            current_claim_den,
            &layer.d_alpha,
            &mut layer.d_sum,
        );

        for round_idx in 0..layer.rounds.len() {
            let round = &mut layer.rounds[round_idx];
            g.sumcheck.eq_polynomial_device_from_flat_point_words_into_async(
                current_point_words,
                round.eq_prefix_len,
                &mut round.d_eq,
            );
            if round.eq_prefix_len > 0 {
                g.sumcheck.memcpy_d2d_async(
                    current_point_words,
                    0,
                    &mut round.d_eq_prefix,
                    0,
                    round.eq_prefix_len as usize * 5,
                );
            }
            g.sumcheck.memcpy_d2d_async(
                current_point_words,
                round.eq_prefix_len as usize * 5,
                &mut round.d_eq_alpha,
                0,
                5,
            );

            let (nl_done, nl_rest) = layer.d_nl_states.split_at_mut(round_idx + 1);
            let (nr_done, nr_rest) = layer.d_nr_states.split_at_mut(round_idx + 1);
            let (dl_done, dl_rest) = layer.d_dl_states.split_at_mut(round_idx + 1);
            let (dr_done, dr_rest) = layer.d_dr_states.split_at_mut(round_idx + 1);
            let d_nl = &nl_done[round_idx];
            let d_nr = &nr_done[round_idx];
            let d_dl = &dl_done[round_idx];
            let d_dr = &dr_done[round_idx];
            let d_nl_next = &mut nl_rest[0];
            let d_nr_next = &mut nr_rest[0];
            let d_dl_next = &mut dl_rest[0];
            let d_dr_next = &mut dr_rest[0];

            g.sumcheck.gkr_quotient_sumcheck_device_resident_into_async(
                d_nl,
                d_nr,
                d_dl,
                d_dr,
                &round.d_eq,
                round.active_pairs,
                0,
                &mut layer.d_c0n_partials,
                &mut layer.d_c2n_partials,
                &mut layer.d_c0d_partials,
                &mut layer.d_c2d_partials,
                &mut layer.d_c0n_out,
                &mut layer.d_c2n_out,
                &mut layer.d_c0d_out,
                &mut layer.d_c2d_out,
            );
            g.sumcheck.gkr_round_protocol_step_precomputed_async(
                &layer.d_c0n_out,
                &layer.d_c2n_out,
                &layer.d_c0d_out,
                &layer.d_c2d_out,
                &layer.d_alpha,
                &round.d_eq_alpha,
                &round.d_eq_prefix,
                round.eq_prefix_len,
                round.active_pairs,
                &mut layer.d_sum,
                &mut layer.d_mmf,
                &mut layer.d_round_rs[round_idx],
                &mut layer.d_round_tails[round_idx],
                d_challenger_state,
                d_p16_rc,
                d_p16_mds,
                d_p16_sparse,
            );
            g.sumcheck.memcpy_d2d_async(
                &layer.d_round_rs[round_idx],
                0,
                &mut layer.d_round_challenges,
                round_idx * 5,
                5,
            );
            g.sumcheck.fold_multi_col_ext_at_bit_into_async(
                d_nl,
                d_nl_next,
                round.input_buf_len,
                round.active_pairs,
                1,
                &layer.d_round_rs[round_idx],
                0,
                0,
            );
            g.sumcheck.fold_multi_col_ext_at_bit_into_async(
                d_nr,
                d_nr_next,
                round.input_buf_len,
                round.active_pairs,
                1,
                &layer.d_round_rs[round_idx],
                0,
                0,
            );
            g.sumcheck.fold_multi_col_ext_at_bit_into_async(
                d_dl,
                d_dl_next,
                round.input_buf_len,
                round.active_pairs,
                1,
                &layer.d_round_rs[round_idx],
                0,
                0,
            );
            g.sumcheck.fold_multi_col_ext_at_bit_into_async(
                d_dr,
                d_dr_next,
                round.input_buf_len,
                round.active_pairs,
                1,
                &layer.d_round_rs[round_idx],
                0,
                0,
            );
        }

        let d_final_nl = layer.d_nl_states.last().unwrap();
        let d_final_nr = layer.d_nr_states.last().unwrap();
        let d_final_dl = layer.d_dl_states.last().unwrap();
        let d_final_dr = layer.d_dr_states.last().unwrap();
        g.sumcheck.gkr_finalize_layer_async(
            d_final_nl,
            d_final_nr,
            d_final_dl,
            d_final_dr,
            &mut layer.d_beta,
            &mut layer.d_next_claim_num,
            &mut layer.d_next_claim_den,
            &mut layer.d_inner_words,
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
        );
        g.sumcheck.memcpy_d2d_async(
            &layer.d_beta,
            0,
            &mut layer.d_round_challenges,
            layer.rounds.len() * 5,
            5,
        );
        g.sumcheck.reverse_gkr_challenges_into_point_async(
            &layer.d_round_challenges,
            layer.rounds.len() as u32,
            &mut layer.d_next_point_words,
        );
    }

    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = stream
        .end_capture(graph_flags)
        .unwrap()
        .expect("stream capture produced no graph");
    graph.launch().unwrap();

    let mut transcript_chunks = Vec::new();
    let last_layer_idx = capture_layers.len() - 1;
    let mut final_point_words = None;
    for (layer_idx, layer) in capture_layers.into_iter().enumerate() {
        for d_words in layer.d_round_tails {
            transcript_chunks.push(TranscriptChunk {
                d_words: d_words.into(),
                n_words: 10,
            });
        }
        transcript_chunks.push(TranscriptChunk {
            d_words: layer.d_inner_words.into(),
            n_words: 20,
        });
        if layer_idx == last_layer_idx {
            final_point_words = Some(layer.d_next_point_words);
        }
    }

    (transcript_chunks, final_point_words.expect("missing final GKR point"))
}

fn gpu_mle_eval_ext_device(
    g: &Gpu,
    d_evals: &CudaSlice<u32>,
    n_elements: usize,
    d_point_words: &CudaSlice<u32>,
    n_coords: usize,
) -> CudaSlice<u32> {
    assert_eq!(n_elements, 1usize << n_coords);

    if n_coords == 0 {
        return g.stream.clone_dtod(&d_evals.slice(0..5)).unwrap();
    }

    let d_first_coord = d_point_words.slice(0..5);
    let mut current_len = n_elements;
    let mut current = g
        .fold
        .fold_ext_device_with_challenge(d_evals, (current_len / 2) as u32, &d_first_coord);
    current_len /= 2;

    for coord_idx in 1..n_coords {
        let start = coord_idx * 5;
        let end = start + 5;
        let d_coord = d_point_words.slice(start..end);
        current = g
            .fold
            .fold_ext_device_with_challenge(&current, (current_len / 2) as u32, &d_coord);
        current_len /= 2;
    }

    current
}

fn gpu_mle_eval_ext_device_into(
    g: &Gpu,
    d_evals: &CudaSlice<u32>,
    n_elements: usize,
    d_point_words: &CudaSlice<u32>,
    n_coords: usize,
    d_scratch_a: &mut CudaSlice<u32>,
    d_scratch_b: &mut CudaSlice<u32>,
    d_out: &mut CudaSlice<u32>,
) {
    assert_eq!(n_elements, 1usize << n_coords);
    assert!(d_out.len() >= 5);

    if n_coords == 0 {
        g.trace_ops.copy_to_offset_device(d_evals, d_out, 5, 0);
        return;
    }

    let first_coord = d_point_words.slice(0..5);
    if n_coords == 1 {
        g.fold
            .fold_ext_device_with_challenge_into_async(d_evals, 1, &first_coord, d_out);
        return;
    }

    assert!(d_scratch_a.len() >= (n_elements / 2) * 5);
    assert!(d_scratch_b.len() >= (n_elements / 2) * 5);

    let mut current_len = n_elements;
    g.fold
        .fold_ext_device_with_challenge_into_async(d_evals, (current_len / 2) as u32, &first_coord, d_scratch_a);
    current_len /= 2;
    let mut current_in_a = true;

    for coord_idx in 1..n_coords {
        let start = coord_idx * 5;
        let end = start + 5;
        let d_coord = d_point_words.slice(start..end);
        let n_pairs = current_len / 2;
        if n_pairs == 1 {
            if current_in_a {
                g.fold
                    .fold_ext_device_with_challenge_into_async(d_scratch_a, 1, &d_coord, d_out);
            } else {
                g.fold
                    .fold_ext_device_with_challenge_into_async(d_scratch_b, 1, &d_coord, d_out);
            }
        } else if current_in_a {
            g.fold
                .fold_ext_device_with_challenge_into_async(d_scratch_a, n_pairs as u32, &d_coord, d_scratch_b);
            current_in_a = false;
        } else {
            g.fold
                .fold_ext_device_with_challenge_into_async(d_scratch_b, n_pairs as u32, &d_coord, d_scratch_a);
            current_in_a = true;
        }
        current_len = n_pairs;
    }
}

pub(crate) fn p16c(g: &Gpu) -> (&CudaSlice<u32>, &CudaSlice<u32>, &CudaSlice<u32>) {
    g.p16.as_slices()
}

pub fn gpu_prove_gkr_quotient(
    g: &Gpu,
    prover_state: &mut impl FSProver<EF>,
    numerators_br: &[F],
    denominators_packed: &[EFPacking<EF>],
    pivot: usize,
) -> (EF, MultilinearPoint<EF>) {
    tracing::info!(
        "GKR quotient: n_vars={} (standalone GPU path)",
        log2_ceil_usize(numerators_br.len())
    );
    let active_len = numerators_br.len();
    let nums: Vec<EF> = numerators_br.iter().map(|&x| EF::from(x)).collect();
    let dens: Vec<EF> = unpack_extension::<EF>(denominators_packed);
    let d_n = g.stream.memcpy_stod(&flat_ef(&nums)).unwrap();
    let d_d = g.stream.memcpy_stod(&flat_ef(&dens)).unwrap();
    let mut fs = GpuFsPhase::new_for_standalone_host_boundary(g, prover_state);
    let out = gpu_prove_gkr_quotient_from_device_ext(g, &mut fs, d_n, d_d, active_len, pivot, true, None, None, None);
    fs.finish(g, prover_state);
    (
        out.quotient.expect("standalone GKR quotient should be downloaded"),
        out.point.expect("standalone GKR point should be downloaded"),
    )
}

pub fn gpu_prove_gkr_quotient_from_device(
    g: &Gpu,
    prover_state: &mut impl FSProver<EF>,
    d_numerators_br: &CudaSlice<u32>,
    d_denominators_br: &CudaSlice<u32>,
    active_len: usize,
    pivot: usize,
) -> (EF, MultilinearPoint<EF>) {
    let full_n = active_len.next_power_of_two();
    let d_nums_ext = g.trace_ops.base_to_ext(d_numerators_br, active_len as u32);
    let mut d_nums_padded = g.stream.alloc_zeros::<u32>(full_n * 5).unwrap();
    let mut d_dens_padded = g.trace_ops.fill_ext(EF::ONE, full_n as u32);
    g.trace_ops
        .copy_to_offset_device(&d_nums_ext, &mut d_nums_padded, (active_len * 5) as u32, 0);
    g.trace_ops
        .copy_to_offset_device(d_denominators_br, &mut d_dens_padded, (active_len * 5) as u32, 0);
    let mut fs = GpuFsPhase::new_for_standalone_host_boundary(g, prover_state);
    let out = gpu_prove_gkr_quotient_from_device_ext(
        g,
        &mut fs,
        d_nums_padded,
        d_dens_padded,
        active_len,
        pivot,
        true,
        None,
        None,
        None,
    );
    fs.finish(g, prover_state);
    (
        out.quotient.expect("standalone GKR quotient should be downloaded"),
        out.point.expect("standalone GKR point should be downloaded"),
    )
}

pub(crate) fn gpu_prove_gkr_quotient_from_device_with_fs(
    g: &Gpu,
    fs: &mut GpuFsPhase,
    d_numerators_br: &CudaSlice<u32>,
    d_denominators_br: &CudaSlice<u32>,
    active_len: usize,
    pivot: usize,
    d_initial_point_words: CudaSlice<u32>,
    mut d_nums_padded: CudaSlice<u32>,
    mut d_dens_padded: CudaSlice<u32>,
    d_top_transcript: CudaSlice<u32>,
    d_claim_num: CudaSlice<u32>,
    d_claim_den: CudaSlice<u32>,
    d_mle_scratch_a: CudaSlice<u32>,
    d_mle_scratch_b: CudaSlice<u32>,
) -> GpuGkrQuotientOutput {
    let full_n = active_len.next_power_of_two();
    assert_eq!(d_nums_padded.len(), full_n * 5);
    assert_eq!(d_dens_padded.len(), full_n * 5);
    g.stream
        .memset_zeros(&mut d_nums_padded)
        .expect("zero padded GKR numerator workspace");
    g.trace_ops
        .base_to_ext_into(d_numerators_br, active_len as u32, &mut d_nums_padded);
    g.trace_ops
        .fill_ext_at_offset(&mut d_dens_padded, EF::ONE, full_n as u32, 0);
    g.trace_ops
        .copy_to_offset_device(d_denominators_br, &mut d_dens_padded, (active_len * 5) as u32, 0);
    gpu_prove_gkr_quotient_from_device_ext(
        g,
        fs,
        d_nums_padded,
        d_dens_padded,
        active_len,
        pivot,
        false,
        Some(d_initial_point_words),
        Some(d_top_transcript),
        Some((d_claim_num, d_claim_den, d_mle_scratch_a, d_mle_scratch_b)),
    )
}

fn gpu_prove_gkr_quotient_from_device_ext(
    g: &Gpu,
    fs: &mut GpuFsPhase,
    d_n: CudaSlice<u32>,
    d_d: CudaSlice<u32>,
    active_len: usize,
    pivot: usize,
    download_quotient: bool,
    initial_point_workspace: Option<CudaSlice<u32>>,
    top_transcript_workspace: Option<CudaSlice<u32>>,
    top_claim_workspaces: Option<(CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u32>)>,
) -> GpuGkrQuotientOutput {
    let total_n_vars = log2_ceil_usize(active_len);
    assert!(total_n_vars > N_VARS_TO_SEND_GKR_COEFFS);
    let full_n = active_len.next_power_of_two();
    let mut transcript_chunks = Vec::<TranscriptChunk>::new();

    // ── Build layers ──
    struct DeviceLayer {
        d_nums: CudaSlice<u32>,
        d_dens: CudaSlice<u32>,
        active_len: usize,
        buf_len: usize,
        chunk_log: usize,
    }
    let mut layers: Vec<DeviceLayer> = Vec::new();
    let mut current = DeviceLayer {
        d_nums: d_n,
        d_dens: d_d,
        active_len,
        buf_len: full_n,
        chunk_log: pivot,
    };

    let mut cur_n_vars = log2_strict_usize(full_n);
    let mut cl = pivot;
    while cur_n_vars > N_VARS_TO_SEND_GKR_COEFFS {
        let n_pairs = current.buf_len / 2;
        let new_active_len = current.active_len.div_ceil(2);
        let fb = if cl > 0 { (cl - 1) as u32 } else { 0 };
        let (d_nn, d_nd) =
            g.sumcheck
                .gkr_sum_quotients_at_bit_device(&current.d_nums, &current.d_dens, n_pairs as u32, fb);
        layers.push(current);
        current = DeviceLayer {
            d_nums: d_nn,
            d_dens: d_nd,
            active_len: new_active_len,
            buf_len: n_pairs,
            chunk_log: cl.saturating_sub(1),
        };
        cur_n_vars -= 1;
        if cl > 0 {
            cl -= 1;
        }
    }

    let top_active_len = current.active_len;
    let top_buf_len = current.buf_len;
    let top_chunk_log = current.chunk_log;
    let (d_top_nums, d_top_dens) = if top_chunk_log > 0 {
        let d_nums_nat_active =
            g.trace_ops
                .bit_reverse_ext_within_chunks(&current.d_nums, top_active_len as u32, top_chunk_log as u32);
        let d_dens_nat_active =
            g.trace_ops
                .bit_reverse_ext_within_chunks(&current.d_dens, top_active_len as u32, top_chunk_log as u32);
        let mut d_nums_nat = g.trace_ops.fill_ext(EF::ZERO, top_buf_len as u32);
        let mut d_dens_nat = g.trace_ops.fill_ext(EF::ONE, top_buf_len as u32);
        g.trace_ops
            .copy_to_offset_device(&d_nums_nat_active, &mut d_nums_nat, (top_active_len * 5) as u32, 0);
        g.trace_ops
            .copy_to_offset_device(&d_dens_nat_active, &mut d_dens_nat, (top_active_len * 5) as u32, 0);
        (d_nums_nat, d_dens_nat)
    } else {
        (current.d_nums, current.d_dens)
    };

    // ── Send top layer ──
    let n_top_words = top_buf_len * 5;
    let mut d_top_transcript = top_transcript_workspace.unwrap_or_else(|| {
        g.stream
            .alloc_zeros::<u32>(n_top_words * 2)
            .expect("alloc GKR top transcript")
    });
    assert_eq!(d_top_transcript.len(), n_top_words * 2);
    let quotient = download_quotient.then(|| {
        ef_from_u32(
            &g.sumcheck
                .sum_quotients_ext_device(&d_top_nums, &d_top_dens, top_buf_len as u32),
        )
    });
    let point_len = N_VARS_TO_SEND_GKR_COEFFS;
    let mut d_point_words = initial_point_workspace.unwrap_or_else(|| {
        g.stream
            .alloc_zeros::<u32>(point_len * 5)
            .expect("alloc GKR initial point")
    });
    let (d_claim_num, d_claim_den, _top_claim_scratch_guards) = if let Some((
        mut d_claim_num,
        mut d_claim_den,
        mut d_mle_scratch_a,
        mut d_mle_scratch_b,
    )) = top_claim_workspaces
    {
        let stream = g.sumcheck.stream().clone();
        stream
            .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin GKR top claim graph capture");
        g.sumcheck
            .memcpy_d2d_async(&d_top_nums, 0, &mut d_top_transcript, 0, n_top_words);
        g.sumcheck
            .memcpy_d2d_async(&d_top_dens, 0, &mut d_top_transcript, n_top_words, n_top_words);
        fs.observe_base_scalars_device(g, d_top_transcript, n_top_words * 2);
        fs.sample_ext_vec_device_into(g, point_len, &mut d_point_words);
        gpu_mle_eval_ext_device_into(
            g,
            &d_top_nums,
            top_buf_len,
            &d_point_words,
            point_len,
            &mut d_mle_scratch_a,
            &mut d_mle_scratch_b,
            &mut d_claim_num,
        );
        gpu_mle_eval_ext_device_into(
            g,
            &d_top_dens,
            top_buf_len,
            &d_point_words,
            point_len,
            &mut d_mle_scratch_a,
            &mut d_mle_scratch_b,
            &mut d_claim_den,
        );
        let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph = stream
            .end_capture(graph_flags)
            .expect("end GKR top claim graph capture")
            .expect("GKR top claim capture produced no graph");
        graph.launch().expect("launch GKR top claim graph");
        (d_claim_num, d_claim_den, Some((d_mle_scratch_a, d_mle_scratch_b)))
    } else {
        g.sumcheck
            .memcpy_d2d_async(&d_top_nums, 0, &mut d_top_transcript, 0, n_top_words);
        g.sumcheck
            .memcpy_d2d_async(&d_top_dens, 0, &mut d_top_transcript, n_top_words, n_top_words);
        fs.observe_base_scalars_device(g, d_top_transcript, n_top_words * 2);
        fs.sample_ext_vec_device_into(g, point_len, &mut d_point_words);
        (
            gpu_mle_eval_ext_device(g, &d_top_nums, top_buf_len, &d_point_words, point_len),
            gpu_mle_eval_ext_device(g, &d_top_dens, top_buf_len, &d_point_words, point_len),
            None,
        )
    };

    // ── Prove all remaining layers in one captured graph ──
    let mut capture_layers = Vec::with_capacity(layers.len());
    let mut capture_point_len = point_len;
    for layer in layers.into_iter().rev() {
        let layer_active_len = layer.active_len;
        let layer_buf_len = layer.buf_len;
        let layer_chunk_log = layer.chunk_log;
        let (d_layer_nums, d_layer_dens) = if layer_chunk_log > 0 {
            let d_nums_nat_active = g.trace_ops.bit_reverse_ext_within_chunks(
                &layer.d_nums,
                layer_active_len as u32,
                layer_chunk_log as u32,
            );
            let d_dens_nat_active = g.trace_ops.bit_reverse_ext_within_chunks(
                &layer.d_dens,
                layer_active_len as u32,
                layer_chunk_log as u32,
            );
            let mut d_nums_nat = g.trace_ops.fill_ext(EF::ZERO, layer_buf_len as u32);
            let mut d_dens_nat = g.trace_ops.fill_ext(EF::ONE, layer_buf_len as u32);
            g.trace_ops
                .copy_to_offset_device(&d_nums_nat_active, &mut d_nums_nat, (layer_active_len * 5) as u32, 0);
            g.trace_ops
                .copy_to_offset_device(&d_dens_nat_active, &mut d_dens_nat, (layer_active_len * 5) as u32, 0);
            (d_nums_nat, d_dens_nat)
        } else {
            (layer.d_nums, layer.d_dens)
        };
        capture_layers.push(prepare_gkr_layer_capture_state(
            g,
            d_layer_nums,
            d_layer_dens,
            layer_active_len,
            layer_buf_len,
            capture_point_len,
        ));
        capture_point_len += 1;
    }
    let (mut layer_chunks, d_point_words) = run_gkr_layers_captured(
        g,
        capture_layers,
        d_point_words,
        d_claim_num,
        d_claim_den,
        &mut fs.d_challenger_state,
        &fs.d_empty_observe,
        &g.p16.d_rc,
        &g.p16.d_mds,
        &g.p16.d_sparse,
    );
    transcript_chunks.append(&mut layer_chunks);

    fs.transcript_chunks.append(&mut transcript_chunks);

    let point = download_quotient.then(|| {
        let point_words = g.stream.memcpy_dtov(&d_point_words).unwrap();
        MultilinearPoint(
            point_words
                .chunks_exact(5)
                .map(|chunk| ef_from_u32(chunk.try_into().unwrap()))
                .collect(),
        )
    });

    GpuGkrQuotientOutput {
        quotient,
        point,
        d_point_words,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::safe::CudaContext;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use sub_protocols::ENDIANNESS_PIVOT_GKR;
    use sub_protocols::{prove_gkr_quotient, verify_gkr_quotient};
    use utils::{build_prover_state, get_poseidon16};

    fn br_chunks<T: Copy>(v: &[T], chunk_log: usize) -> Vec<T> {
        let chunk_size = 1usize << chunk_log;
        assert_eq!(v.len() % chunk_size, 0);
        let shift = usize::BITS as usize - chunk_log;
        let mut out = v.to_vec();
        for chunk in out.chunks_exact_mut(chunk_size) {
            let src = chunk.to_vec();
            for (p, slot) in chunk.iter_mut().enumerate() {
                *slot = src[p.reverse_bits() >> shift];
            }
        }
        out
    }

    fn make_test_gpu() -> super::super::gpu_prove_execution::Gpu {
        let ctx = CudaContext::new(0).expect("CUDA required");
        unsafe {
            ctx.disable_event_tracking();
        }
        let s = ctx.new_stream().expect("failed to create CUDA stream");
        super::super::gpu_prove_execution::Gpu {
            sumcheck: gpu_sumcheck::GpuSumcheck::new(s.clone()),
            fold: gpu_poly_fold::GpuPolyFold::new(s.clone()),
            ntt: gpu_ntt::GpuNtt::new(s.clone()),
            merkle: gpu_merkle::GpuMerkle::new(s.clone()),
            pow: gpu_pow_grind::GpuPowGrinder::new(s.clone()),
            trace_ops: gpu_trace_ops::GpuTraceOps::new(s.clone()),
            p16: super::super::gpu_prove_execution::GpuPoseidon16Constants::new(&s),
            d_ext_one: s.memcpy_stod(&[kb_u32(F::ONE), 0, 0, 0, 0]).unwrap(),
            d_ext_zero_one: s.memcpy_stod(&[0, 0, 0, 0, 0, kb_u32(F::ONE), 0, 0, 0, 0]).unwrap(),
            stream: s,
        }
    }

    #[test]
    fn test_gpu_challenger_observe_20_scalars_matches_cpu() {
        let g = make_test_gpu();
        let mut rng = StdRng::seed_from_u64(7);
        let scalars: Vec<F> = (0..20).map(|_| rng.random()).collect();

        let mut cpu_ps = build_prover_state();
        cpu_ps.observe_scalars(&scalars);
        let cpu_sample = cpu_ps.sample();
        let cpu_state = cpu_ps.gpu_challenger_state();

        let (d_p16_rc, d_p16_mds, d_p16_sparse) = p16c(&g);
        let state_words = build_prover_state().gpu_challenger_state().map(kb_u32);
        let mut d_state = g.stream.memcpy_stod(&state_words).unwrap();
        let scalar_words: Vec<u32> = scalars.iter().copied().map(kb_u32).collect();
        let d_scalars = g.stream.memcpy_stod(&scalar_words).unwrap();
        g.sumcheck.challenger_observe_device_scalars(
            &mut d_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            &d_scalars,
            scalars.len() as u32,
        );
        let d_sample = g
            .sumcheck
            .challenger_sample_exts_device(&mut d_state, d_p16_rc, d_p16_mds, d_p16_sparse, 1);
        let gpu_sample_words = g.stream.memcpy_dtov(&d_sample).unwrap();
        let gpu_state_words = g.stream.memcpy_dtov(&d_state).unwrap();
        let gpu_sample = ef_from_u32(&gpu_sample_words[..5].try_into().unwrap());
        let gpu_state = [
            kb_from_u32(gpu_state_words[0]),
            kb_from_u32(gpu_state_words[1]),
            kb_from_u32(gpu_state_words[2]),
            kb_from_u32(gpu_state_words[3]),
            kb_from_u32(gpu_state_words[4]),
            kb_from_u32(gpu_state_words[5]),
            kb_from_u32(gpu_state_words[6]),
            kb_from_u32(gpu_state_words[7]),
        ];

        assert_eq!(
            gpu_sample, cpu_sample,
            "challenger sample mismatch after 20-scalar observe"
        );
        assert_eq!(
            gpu_state, cpu_state,
            "challenger state mismatch after 20-scalar observe"
        );
    }

    #[test]
    fn test_gpu_sample_from_specific_state_matches_cpu() {
        let g = make_test_gpu();
        // This is the exact state from the logup output during test_prove_fibonacci
        let specific_state: [u32; 8] = [
            1244653872, 1739405030, 277879316, 1585366579, 1686547249, 1649058240, 1102484181, 1426005795,
        ];
        let state_kb: [F; 8] = specific_state.map(kb_from_u32);

        // CPU sample
        let mut cpu_ps = build_prover_state();
        // Inject the specific state
        cpu_ps.inject_gpu_transcript_state(&[], state_kb);
        let cpu_sample: EF = cpu_ps.sample();
        eprintln!("CPU sample from specific state: {:?}", ef_to_u32(&cpu_sample));

        // GPU sample
        let (d_p16_rc, d_p16_mds, d_p16_sparse) = p16c(&g);
        let mut d_state = g.stream.memcpy_stod(&specific_state).unwrap();
        let d_samples =
            g.sumcheck
                .challenger_observe_and_sample_exts(&mut d_state, d_p16_rc, d_p16_mds, d_p16_sparse, &[], 1);
        let gpu_sample = ef_from_u32(&d_samples[0]);
        eprintln!("GPU sample from specific state: {:?}", ef_to_u32(&gpu_sample));

        assert_eq!(
            gpu_sample,
            cpu_sample,
            "sample from specific state mismatch: gpu={:?} cpu={:?}",
            ef_to_u32(&gpu_sample),
            ef_to_u32(&cpu_sample),
        );
    }

    #[test]
    fn test_gpu_lagrange_and_expand_match_cpu() {
        let g = make_test_gpu();
        let mut rng = StdRng::seed_from_u64(99);

        for n in [3usize, 5, 12] {
            let evals: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let d_evals = g.stream.memcpy_stod(&flat_ef(&evals)).unwrap();
            let mut d_coeffs = g.stream.alloc_zeros::<u32>(n * 5).unwrap();
            g.sumcheck.test_lagrange_interp(&d_evals, n as u32, &mut d_coeffs);
            let gpu_coeff_words = g.stream.memcpy_dtov(&d_coeffs).unwrap();
            let gpu_coeffs: Vec<EF> = gpu_coeff_words
                .chunks_exact(5)
                .map(|chunk| ef_from_u32(chunk.try_into().unwrap()))
                .collect();

            let points: Vec<(F, EF)> = evals
                .iter()
                .enumerate()
                .map(|(i, &v)| (F::from_u32(i as u32), v))
                .collect();
            let cpu_coeffs = DensePolynomial::lagrange_interpolation(&points).unwrap().coeffs;
            assert_eq!(gpu_coeffs, cpu_coeffs, "lagrange mismatch for n={n}");

            let alpha: EF = rng.random();
            let gpu_full_words =
                g.sumcheck
                    .test_expand_bare_to_full(&flat_ef(&cpu_coeffs), n as u32, &ef_to_u32(&alpha));
            let gpu_full: Vec<EF> = gpu_full_words
                .chunks_exact(5)
                .map(|chunk| ef_from_u32(chunk.try_into().unwrap()))
                .collect();
            let cpu_full = expand_bare_to_full(&cpu_coeffs, alpha);
            assert_eq!(gpu_full, cpu_full, "expand mismatch for n={n}");
        }
    }

    #[test]
    fn test_gpu_challenger_observe_60_scalars_then_sample() {
        // Reproduce the AIR sumcheck scenario: observe 60 scalars (cc polynomial),
        // then sample a challenge. Start from a non-trivial state.
        let g = make_test_gpu();
        let mut rng = StdRng::seed_from_u64(42);

        // Non-trivial initial state (simulate post-logup state)
        let init_state: [F; 8] = std::array::from_fn(|_| rng.random());
        let init_words: [u32; 8] = init_state.map(kb_u32);

        // 60 scalars to observe (simulating cc polynomial with mfd=11)
        let observe_scalars: Vec<F> = (0..60).map(|_| rng.random()).collect();

        // CPU path
        let mut cpu_ps = build_prover_state();
        cpu_ps.inject_gpu_transcript_state(&[], init_state);
        cpu_ps.observe_scalars(&observe_scalars);
        let cpu_sample: EF = cpu_ps.sample();
        let cpu_post_state = cpu_ps.gpu_challenger_state().map(kb_u32);

        // GPU path
        let (d_p16_rc, d_p16_mds, d_p16_sparse) = p16c(&g);
        let mut d_state = g.stream.memcpy_stod(&init_words).unwrap();
        let scalar_words: Vec<u32> = observe_scalars.iter().copied().map(kb_u32).collect();
        let d_scalars = g.stream.memcpy_stod(&scalar_words).unwrap();
        g.sumcheck
            .challenger_observe_device_scalars(&mut d_state, d_p16_rc, d_p16_mds, d_p16_sparse, &d_scalars, 60);
        let d_sample =
            g.sumcheck
                .challenger_observe_and_sample_exts(&mut d_state, d_p16_rc, d_p16_mds, d_p16_sparse, &[], 1);
        let gpu_sample = ef_from_u32(&d_sample[0]);
        let gpu_post_state = g.stream.memcpy_dtov(&d_state).unwrap();

        eprintln!("CPU sample after 60 observe: {:?}", ef_to_u32(&cpu_sample));
        eprintln!("GPU sample after 60 observe: {:?}", ef_to_u32(&gpu_sample));
        eprintln!("CPU post-state: {:?}", cpu_post_state);
        eprintln!("GPU post-state: {:?}", &gpu_post_state[..8]);

        assert_eq!(gpu_sample, cpu_sample, "sample after 60 observe mismatch");
        assert_eq!(
            &gpu_post_state[..8],
            &cpu_post_state[..],
            "post-state after 60 observe mismatch"
        );
    }

    #[test]
    fn test_gpu_gkr_round_protocol_step_matches_cpu() {
        let g = make_test_gpu();

        let mut rng = StdRng::seed_from_u64(11);
        let c0_num: EF = rng.random();
        let c2_num: EF = rng.random();
        let c0_den: EF = rng.random();
        let c2_den: EF = rng.random();
        let alpha: EF = rng.random();
        let eq_alpha: EF = rng.random();
        let eq_prefix: Vec<EF> = (0..3).map(|_| rng.random()).collect();
        let active_pairs = 7usize;
        let init_sum: EF = rng.random();
        let init_mmf: EF = rng.random();

        let padding_sum = alpha * mle_of_zeros_then_ones(active_pairs, &eq_prefix);
        let c0_raw = c0_num + alpha * c0_den + padding_sum;
        let c2_raw = c2_num + alpha * c2_den;
        let c0_mmf = c0_raw * init_mmf;
        let c2_mmf = c2_raw * init_mmf;
        let h1_mmf = (init_sum - (EF::ONE - eq_alpha) * c0_mmf) / eq_alpha;
        let c1_mmf = h1_mmf - c0_mmf - c2_mmf;
        let bare = vec![c0_mmf, c1_mmf, c2_mmf];

        let mut cpu_ps = build_prover_state();
        cpu_ps.add_sumcheck_polynomial(&bare, Some(eq_alpha));
        let cpu_r = cpu_ps.sample();
        let cpu_eq_eval = (EF::ONE - eq_alpha) * (EF::ONE - cpu_r) + eq_alpha * cpu_r;
        let cpu_sum = cpu_eq_eval * DensePolynomial::new(bare.clone()).evaluate(cpu_r);
        let cpu_mmf = init_mmf * cpu_eq_eval;
        let cpu_state = cpu_ps.gpu_challenger_state();

        let (d_p16_rc, d_p16_mds, d_p16_sparse) = p16c(&g);
        let init_state_words = build_prover_state().gpu_challenger_state().map(kb_u32);
        let mut d_state = g.stream.memcpy_stod(&init_state_words).unwrap();
        let mut d_sum = g.stream.memcpy_stod(&ef_to_u32(&init_sum)).unwrap();
        let mut d_mmf = g.stream.memcpy_stod(&ef_to_u32(&init_mmf)).unwrap();

        let d_c0n = g.stream.memcpy_stod(&ef_to_u32(&c0_num)).unwrap();
        let d_c2n = g.stream.memcpy_stod(&ef_to_u32(&c2_num)).unwrap();
        let d_c0d = g.stream.memcpy_stod(&ef_to_u32(&c0_den)).unwrap();
        let d_c2d = g.stream.memcpy_stod(&ef_to_u32(&c2_den)).unwrap();
        let d_alpha = g.stream.memcpy_stod(&ef_to_u32(&alpha)).unwrap();
        let eq_prefix_u32: Vec<[u32; 5]> = eq_prefix.iter().map(ef_to_u32).collect();

        let (d_r, d_tail) = g.sumcheck.gkr_round_protocol_step(
            &d_c0n,
            &d_c2n,
            &d_c0d,
            &d_c2d,
            &d_alpha,
            &ef_to_u32(&eq_alpha),
            &eq_prefix_u32,
            active_pairs as u32,
            &mut d_sum,
            &mut d_mmf,
            &mut d_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
        );

        let gpu_r_words = g.stream.memcpy_dtov(&d_r).unwrap();
        let gpu_tail_words = g.stream.memcpy_dtov(&d_tail).unwrap();
        let gpu_sum_words = g.stream.memcpy_dtov(&d_sum).unwrap();
        let gpu_mmf_words = g.stream.memcpy_dtov(&d_mmf).unwrap();
        let gpu_state_words = g.stream.memcpy_dtov(&d_state).unwrap();

        let gpu_r = ef_from_u32(&gpu_r_words[..5].try_into().unwrap());
        let gpu_tail: Vec<EF> = gpu_tail_words
            .chunks_exact(5)
            .map(|chunk| ef_from_u32(&chunk.try_into().unwrap()))
            .collect();
        let gpu_sum = ef_from_u32(&gpu_sum_words[..5].try_into().unwrap());
        let gpu_mmf = ef_from_u32(&gpu_mmf_words[..5].try_into().unwrap());
        let gpu_state = [
            kb_from_u32(gpu_state_words[0]),
            kb_from_u32(gpu_state_words[1]),
            kb_from_u32(gpu_state_words[2]),
            kb_from_u32(gpu_state_words[3]),
            kb_from_u32(gpu_state_words[4]),
            kb_from_u32(gpu_state_words[5]),
            kb_from_u32(gpu_state_words[6]),
            kb_from_u32(gpu_state_words[7]),
        ];

        assert_eq!(gpu_tail, vec![c1_mmf, c2_mmf], "round tail mismatch");
        assert_eq!(gpu_r, cpu_r, "round challenge mismatch");
        assert_eq!(gpu_sum, cpu_sum, "updated sum mismatch");
        assert_eq!(gpu_mmf, cpu_mmf, "updated mmf mismatch");
        assert_eq!(gpu_state, cpu_state, "challenger state mismatch after round step");
    }

    #[test]
    fn test_gpu_gkr_quotient_matches_cpu() {
        const LOG_N: usize = 15;
        const PIVOT: usize = ENDIANNESS_PIVOT_GKR;
        let n = 1usize << LOG_N;
        let w = packing_log_width::<EF>();
        let active_packed = n >> w;

        let mut rng = StdRng::seed_from_u64(42);
        let nums_raw: Vec<F> = (0..n).map(|_| rng.random()).collect();
        let dens_raw: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let nums_br_flat: Vec<F> = br_chunks(&nums_raw, PIVOT);
        let dens_br_flat: Vec<EF> = br_chunks(&dens_raw, PIVOT);
        let nums_br_packed: Vec<PFPacking<EF>> = {
            let mut v: Vec<PFPacking<EF>> = unsafe { uninitialized_vec(active_packed) };
            PFPacking::<EF>::unpack_slice_mut(&mut v).copy_from_slice(&nums_br_flat);
            v
        };
        let dens_br_packed: Vec<EFPacking<EF>> = pack_extension(&dens_br_flat);

        let mut cpu_ps = build_prover_state();
        let (cpu_q, cpu_pt) = prove_gkr_quotient::<EF>(&mut cpu_ps, &nums_br_packed, &dens_br_packed, PIVOT);

        let g = make_test_gpu();
        let mut gpu_ps = build_prover_state();
        let (gpu_q, gpu_pt) = gpu_prove_gkr_quotient(&g, &mut gpu_ps, &nums_br_flat, &dens_br_packed, PIVOT);

        println!("CPU quotient: {:?}", ef_to_u32(&cpu_q));
        println!("GPU quotient: {:?}", ef_to_u32(&gpu_q));
        assert_eq!(cpu_q, gpu_q, "quotient mismatch");
        println!("CPU point len={}, GPU point len={}", cpu_pt.0.len(), gpu_pt.0.len());
        for (i, (c, g_val)) in cpu_pt.0.iter().zip(&gpu_pt.0).enumerate() {
            if c != g_val {
                println!("FIRST DIVERGENCE at point[{i}]");
                break;
            }
        }
        // Compare final challenger states
        let cpu_ch = cpu_ps.gpu_challenger_state();
        let gpu_ch = gpu_ps.gpu_challenger_state();
        println!("CPU final challenger: {:?}", cpu_ch.map(|f| ef_to_u32(&EF::from(f))[0]));
        println!("GPU final challenger: {:?}", gpu_ch.map(|f| ef_to_u32(&EF::from(f))[0]));
        if cpu_ch != gpu_ch {
            println!("CHALLENGER STATE MISMATCH");
        }
        let cpu_proof = cpu_ps.into_proof();
        let gpu_proof = gpu_ps.into_proof();
        println!(
            "CPU transcript len={}, GPU transcript len={}",
            cpu_proof.transcript().len(),
            gpu_proof.transcript().len()
        );
        let cpu_top: Vec<u32> = cpu_proof
            .transcript()
            .chunks_exact(5)
            .take(8)
            .map(|chunk| ef_to_u32(&EF::from(chunk[0]))[0])
            .collect();
        let gpu_top: Vec<u32> = gpu_proof
            .transcript()
            .chunks_exact(5)
            .take(8)
            .map(|chunk| ef_to_u32(&EF::from(chunk[0]))[0])
            .collect();
        println!("CPU top first-word scalars: {:?}", cpu_top);
        println!("GPU top first-word scalars: {:?}", gpu_top);
        for (i, (c, g_val)) in cpu_proof.transcript().iter().zip(gpu_proof.transcript()).enumerate() {
            if c != g_val {
                println!("FIRST DIVERGENCE at transcript[{i}]");
                println!(
                    "CPU transcript[{i}]={:?} GPU transcript[{i}]={:?}",
                    ef_to_u32(&EF::from(*c))[0],
                    ef_to_u32(&EF::from(*g_val))[0]
                );
                break;
            }
        }
        // Verify both
        let cpu_ok = verify_gkr_quotient::<EF>(
            &mut VerifierState::new(cpu_proof, get_poseidon16().clone()).unwrap(),
            LOG_N,
        );
        println!("CPU verify: {:?}", cpu_ok.is_ok());
        let gpu_ok = verify_gkr_quotient::<EF>(
            &mut VerifierState::new(gpu_proof, get_poseidon16().clone()).unwrap(),
            LOG_N,
        );
        println!("GPU verify: {:?}", gpu_ok.is_ok());
        assert!(cpu_ok.is_ok(), "CPU GKR verify failed");
        assert!(gpu_ok.is_ok(), "GPU GKR verify failed");
    }
}
