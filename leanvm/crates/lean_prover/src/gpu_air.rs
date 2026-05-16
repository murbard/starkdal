//! GPU AIR sumcheck captured as a single CUDA graph launch.

use std::collections::BTreeMap;

use backend::*;
use cudarc::driver::safe::{CudaSlice, DevicePtr};
use cudarc::driver::sys;
use lean_vm::*;
use utils::*;

use super::gpu_logup::GpuLogupBusEvals;
use super::gpu_prove_execution::{Gpu, GpuFsPhase, GpuUploadedExecutionTrace, TranscriptChunk};

type F = lean_vm::F;
type EF = lean_vm::EF;

const ENDIANNESS_PIVOT_AIR: usize = 12;
const PACKING_LOG_WIDTH: usize = {
    use backend::*;
    packing_log_width::<EF>()
};

const TS_MMF: usize = 5;
const TS_NZ: usize = 15;
const TS_ETAK: usize = 16;
const TS_PAD: usize = 21;
const TS_JOIN: usize = 26;
const TS_STRIDE: usize = 32;
const AIR_NO_PADDING_SENTINEL: u32 = u32::MAX;

#[derive(Clone, Copy)]
pub(crate) struct RoundSchedule {
    n_rows: u32,
    total_pairs: u32,
    active_pairs: u32,
    blocks: u32,
    fold_bit: u32,
    ze_offset_words: usize,
    is_base: bool,
}

pub(crate) struct GpuAirTableStaticPlan {
    table: Table,
    log_n_rows: usize,
    n_rows: usize,
    n_up_cols: usize,
    n_down_cols: usize,
    degree: usize,
    join_round: usize,
    max_ext_pairs: usize,
    max_partial_words: usize,
    schedules: Vec<RoundSchedule>,
    d_ext_up_a: Option<CudaSlice<u32>>,
    d_ext_up_b: Option<CudaSlice<u32>>,
    d_ext_down_a: Option<CudaSlice<u32>>,
    d_ext_down_b: Option<CudaSlice<u32>>,
    d_partial_sums: Option<CudaSlice<u32>>,
    d_alphas: Option<CudaSlice<u32>>,
    d_eq_point_scratch: Option<CudaSlice<u32>>,
    d_eq_rounds: Option<Vec<CudaSlice<u32>>>,
    d_eval_values: Option<CudaSlice<u32>>,
    d_eval_point: Option<CudaSlice<u32>>,
}

pub(crate) struct GpuAirStaticMetadata {
    tables_sorted: Vec<(Table, VarCount)>,
    table_plans: Vec<GpuAirTableStaticPlan>,
    n_tables: usize,
    n_rounds: usize,
    max_full_degree: usize,
    ze_words_max: usize,
    d_round_meta: Vec<CudaSlice<u32>>,
    d_round_meta_work: Option<Vec<CudaSlice<u32>>>,
    d_round_extra: Vec<CudaSlice<u32>>,
    d_ze_offsets: Vec<CudaSlice<u32>>,
    d_table_state: CudaSlice<u32>,
    d_table_state_work: Option<CudaSlice<u32>>,
    d_logup_bus_numerators: Option<CudaSlice<u32>>,
    d_logup_bus_denominators: Option<CudaSlice<u32>>,
    d_logup_bus_contrib: Option<CudaSlice<u32>>,
    d_logup_bus_alphas: Option<CudaSlice<u32>>,
    d_air_alpha_powers: Option<CudaSlice<u32>>,
    d_table_pad_evals: Option<CudaSlice<u32>>,
    d_eta_powers: Option<CudaSlice<u32>>,
    d_ze_all: Option<CudaSlice<u32>>,
    d_cc_scratch: Option<CudaSlice<u32>>,
    d_bare_scratch: Option<CudaSlice<u32>>,
    d_transcript: Option<CudaSlice<u32>>,
    d_transcript_len: Option<CudaSlice<u32>>,
    d_challenges: Option<CudaSlice<u32>>,
    d_eval_transcript: Option<CudaSlice<u32>>,
    logup_bus_constant_inputs: gpu_sumcheck::LogupConstantDeviceInputs,
}

struct TablePlan<'a> {
    table: Table,
    log_n_rows: usize,
    n_up_cols: u32,
    n_down_cols: u32,
    degree: usize,
    join_round: usize,
    d_base_cols: &'a CudaSlice<u32>,
    d_base_down_cols: &'a CudaSlice<u32>,
    d_ext_up_a: CudaSlice<u32>,
    d_ext_up_b: CudaSlice<u32>,
    d_ext_down_a: Option<CudaSlice<u32>>,
    d_ext_down_b: Option<CudaSlice<u32>>,
    d_partial_sums: CudaSlice<u32>,
    d_alphas: CudaSlice<u32>,
    schedules: Vec<RoundSchedule>,
    _d_eq_point_scratch: CudaSlice<u32>,
    d_eq_rounds: Vec<CudaSlice<u32>>,
    d_eval_values: CudaSlice<u32>,
    d_eval_point: CudaSlice<u32>,
}

pub struct GpuAirWhirStatementTargets<'a> {
    pub table_targets: Vec<GpuAirWhirTableStatementTargets<'a>>,
    pub public_memory_target: GpuAirPublicMemoryStatementTarget<'a>,
}

pub struct GpuAirWhirTableStatementTargets<'a> {
    pub table: Table,
    pub d_eq_point: &'a mut CudaSlice<u32>,
    pub eq_values: Vec<GpuAirWhirValueStatementTarget<'a>>,
    pub d_next_point: Option<&'a mut CudaSlice<u32>>,
    pub next_values: Vec<GpuAirWhirValueStatementTarget<'a>>,
}

pub struct GpuAirWhirValueStatementTarget<'a> {
    pub value_offset_words: usize,
    pub d_value: &'a mut CudaSlice<u32>,
}

pub struct GpuAirPublicMemoryStatementTarget<'a> {
    pub n_vars: usize,
    pub d_point: &'a mut CudaSlice<u32>,
    pub d_value: &'a mut CudaSlice<u32>,
    pub d_scratch_a: CudaSlice<u32>,
    pub d_scratch_b: CudaSlice<u32>,
}

fn ef_u32(v: EF) -> [u32; 5] {
    unsafe { std::mem::transmute(v) }
}
fn concat_table_bus_ext_scalars_into(
    g: &Gpu,
    tables_sorted: &[(Table, VarCount)],
    values: &[GpuLogupBusEvals],
    select: fn(&GpuLogupBusEvals) -> &CudaSlice<u32>,
    d_out: &mut CudaSlice<u32>,
) {
    debug_assert_eq!(tables_sorted.len(), values.len());
    debug_assert!(d_out.len() >= tables_sorted.len() * 5);
    for (idx, ((table, _), value)) in tables_sorted.iter().zip(values.iter()).enumerate() {
        debug_assert_eq!(*table, value.table);
        g.sumcheck.memcpy_d2d_async(select(value), 0, d_out, idx * 5, 5);
    }
}

fn copy_air_eval_into_whir_target(
    g: &Gpu,
    d_eval_values: &CudaSlice<u32>,
    target: &mut GpuAirWhirValueStatementTarget<'_>,
) {
    assert!(target.value_offset_words + 5 <= d_eval_values.len());
    assert!(target.d_value.len() >= 5);
    g.sumcheck
        .memcpy_d2d_async(d_eval_values, target.value_offset_words, target.d_value, 0, 5);
}

fn fill_air_whir_statement_targets(
    g: &Gpu,
    d_eval_point: &CudaSlice<u32>,
    d_eval_values: &CudaSlice<u32>,
    table_target: &mut GpuAirWhirTableStatementTargets<'_>,
) {
    assert_eq!(table_target.d_eq_point.len(), d_eval_point.len());
    g.sumcheck
        .memcpy_d2d_async(d_eval_point, 0, table_target.d_eq_point, 0, d_eval_point.len());
    for target in &mut table_target.eq_values {
        copy_air_eval_into_whir_target(g, d_eval_values, target);
    }
    if let Some(d_next_point) = table_target.d_next_point.as_mut() {
        let d_next_point = &mut **d_next_point;
        assert_eq!(d_next_point.len(), d_eval_point.len());
        g.sumcheck
            .memcpy_d2d_async(d_eval_point, 0, d_next_point, 0, d_eval_point.len());
    }
    for target in &mut table_target.next_values {
        copy_air_eval_into_whir_target(g, d_eval_values, target);
    }
}

fn gpu_mle_eval_base_device_into<D>(
    g: &Gpu,
    d_evals: &D,
    n_elements: usize,
    d_point_words: &CudaSlice<u32>,
    n_vars: usize,
    d_scratch_a: &mut CudaSlice<u32>,
    d_scratch_b: &mut CudaSlice<u32>,
    d_out: &mut CudaSlice<u32>,
) where
    D: DevicePtr<u32>,
{
    const EXT_DIM: usize = 5;
    assert_eq!(
        n_elements,
        1usize << n_vars,
        "gpu_mle_eval_base_device: n_elements={n_elements} n_vars={n_vars} d_evals.len={}",
        d_evals.len()
    );
    assert!(d_out.len() >= EXT_DIM);

    if n_vars == 0 {
        g.stream.memset_zeros(d_out).expect("zero constant ext mle output");
        g.trace_ops.copy_to_offset_device(d_evals, d_out, 1, 0);
        return;
    }

    let first_coord = d_point_words.slice(0..EXT_DIM);
    if n_vars == 1 {
        g.fold
            .fold_base_to_ext_device_with_challenge_into_async(d_evals, 1, &first_coord, d_out);
        return;
    }

    let mut current_len = n_elements;
    g.fold.fold_base_to_ext_device_with_challenge_into_async(
        d_evals,
        (current_len / 2) as u32,
        &first_coord,
        d_scratch_a,
    );
    current_len /= 2;
    let mut current_in_a = true;

    for coord_idx in 1..n_vars {
        let start = coord_idx * EXT_DIM;
        let end = start + EXT_DIM;
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

fn fill_public_memory_whir_statement_target(
    g: &Gpu,
    fs: &mut GpuFsPhase,
    d_memory: &CudaSlice<u32>,
    target: &mut GpuAirPublicMemoryStatementTarget<'_>,
) {
    fs.sample_ext_vec_device_into(g, target.n_vars, target.d_point);
    gpu_mle_eval_base_device_into(
        g,
        d_memory,
        1usize << target.n_vars,
        target.d_point,
        target.n_vars,
        &mut target.d_scratch_a,
        &mut target.d_scratch_b,
        target.d_value,
    );
}

fn p16c(g: &Gpu) -> (&CudaSlice<u32>, &CudaSlice<u32>, &CudaSlice<u32>) {
    g.p16.as_slices()
}

fn take_air_workspace<T>(slot: &mut Option<T>, name: &str) -> T {
    slot.take()
        .unwrap_or_else(|| panic!("AIR workspace already consumed: {name}"))
}

fn pivot(n_vars: usize) -> usize {
    ENDIANNESS_PIVOT_AIR.min(n_vars)
}

fn in_phase_1(rounds_done: usize, n_vars: usize) -> bool {
    let p = pivot(n_vars);
    rounds_done + PACKING_LOG_WIDTH < p && rounds_done + PACKING_LOG_WIDTH + 1 < n_vars
}

fn folding_bit(rounds_done: usize, n_vars: usize) -> usize {
    let p = pivot(n_vars);
    if rounds_done < p { p - 1 - rounds_done } else { 0 }
}

fn active_pairs(current_unpadded_len: usize, n_vars: usize, rounds_done: usize, total_pairs: u32) -> u32 {
    if in_phase_1(rounds_done, n_vars) {
        (((current_unpadded_len / 2) >> PACKING_LOG_WIDTH) << PACKING_LOG_WIDTH) as u32
    } else {
        (current_unpadded_len.div_ceil(2) as u32).min(total_pairs)
    }
}

fn partial_threads(table: Table, is_base: bool) -> u32 {
    match (table, is_base) {
        (Table::Poseidon16(_), true) => 128,
        (Table::Poseidon16(_), false) => 64,
        (Table::Execution(_), true) | (Table::ExtensionOp(_), true) => 256,
        (Table::Execution(_), false) | (Table::ExtensionOp(_), false) => 128,
    }
}

fn fill_air_eq_round_from_gkr_point(
    g: &Gpu,
    d_gkr_point_words: &CudaSlice<u32>,
    total_coords: usize,
    suffix_len: usize,
    len: usize,
    pivot: usize,
    d_point_scratch: &mut CudaSlice<u32>,
    d_out: &mut CudaSlice<u32>,
) {
    if len == 0 {
        debug_assert!(d_out.len() >= 5);
        return;
    }
    g.sumcheck
        .eq_polynomial_device_from_permuted_suffix_point_words_into_async(
            d_gkr_point_words,
            total_coords,
            suffix_len,
            len,
            pivot,
            d_point_scratch,
            d_out,
        );
}

pub(crate) fn build_air_static_metadata(g: &Gpu, traces: &BTreeMap<Table, TableTrace>) -> GpuAirStaticMetadata {
    let tables_log_heights: BTreeMap<Table, VarCount> =
        traces.iter().map(|(table, trace)| (*table, trace.log_n_rows)).collect();
    let tables_sorted = sort_tables_by_height(&tables_log_heights);
    let n_tables = tables_sorted.len();
    let n_rounds = tables_sorted.iter().map(|(_, log)| *log).max().unwrap_or(0);
    let max_full_degree = tables_sorted
        .iter()
        .map(|(table, _)| table.degree_air() + 1)
        .max()
        .unwrap_or(0);

    let mut table_state_host = vec![0u32; n_tables * TS_STRIDE];
    let mut round_meta_host = vec![vec![0u32; n_tables * TS_STRIDE]; n_rounds];
    let mut round_extra_host = vec![vec![0u32; n_tables * 4]; n_rounds];
    let mut ze_offsets_by_round = vec![vec![0u32; n_tables]; n_rounds];
    let mut table_plans = Vec::with_capacity(n_tables);
    let mut total_eval_transcript_words = 0usize;

    for (idx, (table, log_n_rows)) in tables_sorted.iter().enumerate() {
        let trace = &traces[table];
        let n_rows = 1usize << log_n_rows;
        let n_up_cols = table.n_columns();
        let n_down_cols = table.down_column_indexes().len();
        total_eval_transcript_words += (n_up_cols + n_down_cols) * 5;
        let pv = pivot(*log_n_rows);
        let state = &mut table_state_host[idx * TS_STRIDE..(idx + 1) * TS_STRIDE];
        for (slot, word) in state[TS_MMF..TS_MMF + 5].iter_mut().zip(ef_u32(EF::ONE)) {
            *slot = word;
        }

        let join_round = n_rounds - log_n_rows;
        let mut schedules = Vec::with_capacity(*log_n_rows);
        let mut current_unpadded_len = trace.non_padded_n_rows.next_multiple_of(1usize << pv).min(n_rows);
        let mut current_n_rows = n_rows as u32;
        let mut max_partial_words = 1usize;

        for local_round in 0..*log_n_rows {
            let remaining = log_n_rows - local_round;
            let total_pairs = (1u32 << remaining) / 2;
            let act_pairs = active_pairs(current_unpadded_len, *log_n_rows, local_round, total_pairs);
            let fb = folding_bit(local_round, *log_n_rows) as u32;
            let is_base = local_round == 0;
            let blocks = act_pairs.div_ceil(partial_threads(*table, is_base));
            let degree = table.degree_air();
            max_partial_words = max_partial_words.max(degree * blocks as usize * 5);

            let meta = &mut round_meta_host[join_round + local_round][idx * TS_STRIDE..(idx + 1) * TS_STRIDE];
            meta[TS_JOIN] = 1;
            meta[TS_NZ] = degree as u32;
            let extra = &mut round_extra_host[join_round + local_round][idx * 4..(idx + 1) * 4];
            extra[0] = *log_n_rows as u32;
            extra[1] = local_round as u32;
            extra[2] = if act_pairs < total_pairs {
                current_unpadded_len as u32
            } else {
                AIR_NO_PADDING_SENTINEL
            };
            extra[3] = pv as u32;

            schedules.push(RoundSchedule {
                n_rows: current_n_rows,
                total_pairs,
                active_pairs: act_pairs,
                blocks,
                fold_bit: fb,
                ze_offset_words: 0,
                is_base,
            });

            current_unpadded_len = current_unpadded_len.div_ceil(2);
            current_n_rows = total_pairs;
        }

        let max_ext_pairs = schedules.first().map(|s| s.total_pairs as usize).unwrap_or(1).max(1);
        let n_alpha_words = (table.n_constraints() + 1) * 5;
        let d_ext_up_a = g
            .sumcheck
            .stream()
            .alloc_zeros::<u32>(n_up_cols * max_ext_pairs * 5)
            .unwrap();
        let d_ext_up_b = g
            .sumcheck
            .stream()
            .alloc_zeros::<u32>(n_up_cols * max_ext_pairs * 5)
            .unwrap();
        let d_ext_down_a = if n_down_cols > 0 {
            Some(
                g.sumcheck
                    .stream()
                    .alloc_zeros::<u32>(n_down_cols * max_ext_pairs * 5)
                    .unwrap(),
            )
        } else {
            None
        };
        let d_ext_down_b = if n_down_cols > 0 {
            Some(
                g.sumcheck
                    .stream()
                    .alloc_zeros::<u32>(n_down_cols * max_ext_pairs * 5)
                    .unwrap(),
            )
        } else {
            None
        };
        let d_partial_sums = g.sumcheck.stream().alloc_zeros::<u32>(max_partial_words).unwrap();
        let d_alphas = g.sumcheck.stream().alloc_zeros::<u32>(n_alpha_words).unwrap();
        let d_eq_point_scratch = g
            .sumcheck
            .stream()
            .alloc_zeros::<u32>(log_n_rows.saturating_sub(1).max(1) * 5)
            .unwrap();
        let d_eq_rounds = (0..*log_n_rows)
            .map(|local_round| {
                let remaining = *log_n_rows - local_round;
                let len = remaining - 1;
                if len == 0 {
                    g.d_ext_one.clone()
                } else {
                    g.sumcheck.stream().alloc_zeros::<u32>((1usize << len) * 5).unwrap()
                }
            })
            .collect();
        let d_eval_values = g
            .sumcheck
            .stream()
            .alloc_zeros::<u32>((n_up_cols + n_down_cols) * 5)
            .unwrap();
        let d_eval_point = g.sumcheck.stream().alloc_zeros::<u32>(*log_n_rows * 5).unwrap();
        table_plans.push(GpuAirTableStaticPlan {
            table: *table,
            log_n_rows: *log_n_rows,
            n_rows,
            n_up_cols,
            n_down_cols,
            degree: table.degree_air(),
            join_round,
            max_ext_pairs,
            max_partial_words,
            schedules,
            d_ext_up_a: Some(d_ext_up_a),
            d_ext_up_b: Some(d_ext_up_b),
            d_ext_down_a,
            d_ext_down_b,
            d_partial_sums: Some(d_partial_sums),
            d_alphas: Some(d_alphas),
            d_eq_point_scratch: Some(d_eq_point_scratch),
            d_eq_rounds: Some(d_eq_rounds),
            d_eval_values: Some(d_eval_values),
            d_eval_point: Some(d_eval_point),
        });
    }

    let mut ze_words_max = 1usize;
    for round in 0..n_rounds {
        let mut off = 0u32;
        for (idx, plan) in table_plans.iter_mut().enumerate() {
            ze_offsets_by_round[round][idx] = off;
            if round >= plan.join_round {
                let local_round = round - plan.join_round;
                plan.schedules[local_round].ze_offset_words = off as usize * 5;
                off += plan.degree as u32;
            }
        }
        ze_words_max = ze_words_max.max(off as usize * 5);
    }

    let d_round_meta = round_meta_host
        .iter()
        .map(|meta| g.sumcheck.stream().memcpy_stod(meta).unwrap())
        .collect::<Vec<_>>();
    let d_round_meta_work = round_meta_host
        .iter()
        .map(|meta| g.sumcheck.stream().alloc_zeros::<u32>(meta.len()).unwrap())
        .collect::<Vec<_>>();
    let d_round_extra = round_extra_host
        .iter()
        .map(|extra| g.sumcheck.stream().memcpy_stod(extra).unwrap())
        .collect();
    let d_ze_offsets = ze_offsets_by_round
        .iter()
        .map(|offsets| g.sumcheck.stream().memcpy_stod(offsets).unwrap())
        .collect();
    let d_table_state = g.sumcheck.stream().memcpy_stod(&table_state_host).unwrap();
    let d_table_state_work = g.sumcheck.stream().alloc_zeros::<u32>(table_state_host.len()).unwrap();
    let d_logup_bus_numerators = g.sumcheck.stream().alloc_zeros::<u32>(n_tables * 5).unwrap();
    let d_logup_bus_denominators = g.sumcheck.stream().alloc_zeros::<u32>(n_tables * 5).unwrap();
    let d_air_alpha_powers = g
        .sumcheck
        .stream()
        .alloc_zeros::<u32>((max_air_constraints() + 1) * 5)
        .unwrap();
    let d_table_pad_evals = g.sumcheck.stream().alloc_zeros::<u32>(n_tables * 5).unwrap();
    let d_eta_powers = g.sumcheck.stream().alloc_zeros::<u32>(n_tables * 5).unwrap();
    let d_ze_all = g.sumcheck.stream().alloc_zeros::<u32>(ze_words_max).unwrap();
    let d_cc_scratch = g
        .sumcheck
        .stream()
        .alloc_zeros::<u32>((max_full_degree + 1) * 5)
        .unwrap();
    let d_bare_scratch = g
        .sumcheck
        .stream()
        .alloc_zeros::<u32>((n_tables * max_full_degree * 5).max(1))
        .unwrap();
    let transcript_stride_words = (max_full_degree + 1) * 5 - 5;
    let d_transcript = g
        .sumcheck
        .stream()
        .alloc_zeros::<u32>(transcript_stride_words.max(1) * n_rounds.max(1))
        .unwrap();
    let d_transcript_len = g.sumcheck.stream().alloc_zeros::<u32>(1).unwrap();
    let d_challenges = g.sumcheck.stream().alloc_zeros::<u32>(n_rounds.max(1) * 5).unwrap();
    let d_eval_transcript = g
        .sumcheck
        .stream()
        .alloc_zeros::<u32>(total_eval_transcript_words)
        .unwrap();
    let logup_alpha_len = log2_ceil_usize(max_bus_width_including_domainsep());
    let alpha_last_idx = ((1usize << logup_alpha_len) - 1) as u32;
    let logup_bus_alpha_indices = [0, 1, 2, 3, alpha_last_idx];
    let logup_bus_alpha_signs = [0, 0, 0, 0, 0];
    let d_logup_bus_contrib = g.sumcheck.stream().alloc_zeros::<u32>(5).unwrap();
    let d_logup_bus_alphas = g
        .sumcheck
        .stream()
        .alloc_zeros::<u32>(logup_bus_alpha_indices.len() * 5)
        .unwrap();
    let logup_bus_constant_inputs =
        g.sumcheck
            .upload_logup_constant_inputs(&logup_bus_alpha_indices, &logup_bus_alpha_signs, &[], &[]);

    GpuAirStaticMetadata {
        tables_sorted,
        table_plans,
        n_tables,
        n_rounds,
        max_full_degree,
        ze_words_max,
        d_round_meta,
        d_round_meta_work: Some(d_round_meta_work),
        d_round_extra,
        d_ze_offsets,
        d_table_state,
        d_table_state_work: Some(d_table_state_work),
        d_logup_bus_numerators: Some(d_logup_bus_numerators),
        d_logup_bus_denominators: Some(d_logup_bus_denominators),
        d_logup_bus_contrib: Some(d_logup_bus_contrib),
        d_logup_bus_alphas: Some(d_logup_bus_alphas),
        d_air_alpha_powers: Some(d_air_alpha_powers),
        d_table_pad_evals: Some(d_table_pad_evals),
        d_eta_powers: Some(d_eta_powers),
        d_ze_all: Some(d_ze_all),
        d_cc_scratch: Some(d_cc_scratch),
        d_bare_scratch: Some(d_bare_scratch),
        d_transcript: Some(d_transcript),
        d_transcript_len: Some(d_transcript_len),
        d_challenges: Some(d_challenges),
        d_eval_transcript: Some(d_eval_transcript),
        logup_bus_constant_inputs,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_prove_air_sumcheck(
    g: &Gpu,
    fs: &mut GpuFsPhase,
    d_logup_c: &CudaSlice<u32>,
    d_logup_alphas_eq_poly: &CudaSlice<u32>,
    uploaded_trace: &mut GpuUploadedExecutionTrace,
    d_gkr_point_words: &CudaSlice<u32>,
    d_logup_bus_evals: &[GpuLogupBusEvals],
    mut d_bus_beta: CudaSlice<u32>,
    mut d_air_alpha: CudaSlice<u32>,
    mut d_air_eta: CudaSlice<u32>,
    whir_statement_targets: GpuAirWhirStatementTargets<'_>,
) {
    let tables = &uploaded_trace.tables;
    let d_memory = &uploaded_trace.d_memory;
    let d_air_bus_directions = &uploaded_trace.d_air_bus_directions;
    let air_static = &mut uploaded_trace.air_static;
    let tables_sorted = &air_static.tables_sorted;
    let n_tables = air_static.n_tables;
    let n_rounds = air_static.n_rounds;
    let max_full_degree = air_static.max_full_degree;
    let GpuAirWhirStatementTargets {
        table_targets,
        mut public_memory_target,
    } = whir_statement_targets;
    let mut whir_table_targets = table_targets.into_iter();

    let (d_p16_rc, d_p16_mds, d_p16_sparse) = p16c(g);
    let stream = g.sumcheck.stream().clone();
    stream
        .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
        .unwrap();
    fs.sample_ext_vec_device_into(g, 1, &mut d_bus_beta);
    fs.sample_ext_vec_device_into(g, 1, &mut d_air_alpha);
    fs.sample_ext_vec_device_into(g, 1, &mut d_air_eta);

    let mut d_logup_bus_contrib = take_air_workspace(&mut air_static.d_logup_bus_contrib, "logup bus contrib");
    let mut d_logup_alphas = take_air_workspace(&mut air_static.d_logup_bus_alphas, "logup bus alphas");
    g.sumcheck.logup_prepare_constants_from_device_inputs_into_async(
        d_logup_alphas_eq_poly,
        &air_static.logup_bus_constant_inputs,
        &mut d_logup_bus_contrib,
        &mut d_logup_alphas,
    );
    let mut d_air_alpha_powers = take_air_workspace(&mut air_static.d_air_alpha_powers, "AIR alpha powers");
    g.sumcheck.extension_powers_device_into_async(
        &d_air_alpha,
        (max_air_constraints() + 1) as u32,
        &mut d_air_alpha_powers,
    );
    let mut d_bus_numerators = take_air_workspace(&mut air_static.d_logup_bus_numerators, "logup bus numerators");
    concat_table_bus_ext_scalars_into(
        g,
        tables_sorted,
        d_logup_bus_evals,
        |evals| &evals.d_numerator,
        &mut d_bus_numerators,
    );
    let mut d_bus_denominators = take_air_workspace(&mut air_static.d_logup_bus_denominators, "logup bus denominators");
    concat_table_bus_ext_scalars_into(
        g,
        tables_sorted,
        d_logup_bus_evals,
        |evals| &evals.d_denominator,
        &mut d_bus_denominators,
    );
    debug_assert_eq!(d_air_bus_directions.len(), n_tables);

    let mut plans = Vec::with_capacity(n_tables);

    for static_plan in &mut air_static.table_plans {
        let table = static_plan.table;
        let log_n_rows = static_plan.log_n_rows;
        let n_up_cols = static_plan.n_up_cols;
        let n_down_cols = static_plan.n_down_cols;
        let pv = pivot(log_n_rows);
        let uploaded = tables.get(&table).expect("GPU AIR requires staged device table trace");
        let (d_base_cols, d_base_down_cols) = (&uploaded.d_air_base_cols, &uploaded.d_air_base_down_cols);

        let mut d_alphas = take_air_workspace(&mut static_plan.d_alphas, "table AIR alphas");
        let n_alpha_words = d_alphas.len();
        g.sumcheck
            .memcpy_d2d_async(&d_air_alpha_powers, 0, &mut d_alphas, 0, n_alpha_words);
        let mut d_eq_point_scratch = take_air_workspace(&mut static_plan.d_eq_point_scratch, "AIR eq point scratch");
        let mut d_eq_rounds = take_air_workspace(&mut static_plan.d_eq_rounds, "AIR eq rounds");
        for local_round in 0..static_plan.schedules.len() {
            let remaining = log_n_rows - local_round;
            fill_air_eq_round_from_gkr_point(
                g,
                d_gkr_point_words,
                d_gkr_point_words.len() / 5,
                log_n_rows,
                remaining - 1,
                pv,
                &mut d_eq_point_scratch,
                &mut d_eq_rounds[local_round],
            );
        }

        plans.push(TablePlan {
            table,
            log_n_rows,
            n_up_cols: n_up_cols as u32,
            n_down_cols: n_down_cols as u32,
            degree: static_plan.degree,
            join_round: static_plan.join_round,
            d_base_cols,
            d_base_down_cols,
            d_ext_up_a: take_air_workspace(&mut static_plan.d_ext_up_a, "AIR ext up A"),
            d_ext_up_b: take_air_workspace(&mut static_plan.d_ext_up_b, "AIR ext up B"),
            d_ext_down_a: if n_down_cols > 0 {
                Some(take_air_workspace(&mut static_plan.d_ext_down_a, "AIR ext down A"))
            } else {
                None
            },
            d_ext_down_b: if n_down_cols > 0 {
                Some(take_air_workspace(&mut static_plan.d_ext_down_b, "AIR ext down B"))
            } else {
                None
            },
            d_partial_sums: take_air_workspace(&mut static_plan.d_partial_sums, "AIR partial sums"),
            d_alphas,
            schedules: static_plan.schedules.clone(),
            _d_eq_point_scratch: d_eq_point_scratch,
            d_eq_rounds,
            d_eval_values: take_air_workspace(&mut static_plan.d_eval_values, "AIR eval values"),
            d_eval_point: take_air_workspace(&mut static_plan.d_eval_point, "AIR eval point"),
        });
    }

    let mut d_table_pad_evals = take_air_workspace(&mut air_static.d_table_pad_evals, "table pad evals");
    for (idx, static_plan) in air_static.table_plans.iter().enumerate() {
        let table = static_plan.table;
        g.sumcheck.air_table_pad_eval_device_into_async(
            &tables[&table].d_all_cols,
            &plans[idx].d_alphas,
            &d_logup_alphas,
            &d_bus_beta,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            table.index() as u32,
            static_plan.n_rows as u32,
            &mut d_table_pad_evals,
            idx * 5,
        );
    }

    let mut d_round_meta = take_air_workspace(&mut air_static.d_round_meta_work, "round meta work");
    for (src, dst) in air_static.d_round_meta.iter().zip(d_round_meta.iter_mut()) {
        let n_words = src.len();
        g.sumcheck.memcpy_d2d_async(src, 0, dst, 0, n_words);
    }
    for round in 0..n_rounds {
        g.sumcheck.patch_air_round_pad_evals_async(
            &d_table_pad_evals,
            &air_static.d_round_extra[round],
            &mut d_round_meta[round],
            n_tables as u32,
        );
    }

    let mut d_table_state = take_air_workspace(&mut air_static.d_table_state_work, "table state work");
    let table_state_words = air_static.d_table_state.len();
    g.sumcheck
        .memcpy_d2d_async(&air_static.d_table_state, 0, &mut d_table_state, 0, table_state_words);
    let mut d_eta_powers = take_air_workspace(&mut air_static.d_eta_powers, "eta powers");
    g.sumcheck
        .extension_powers_device_into_async(&d_air_eta, n_tables as u32, &mut d_eta_powers);
    for idx in 0..n_tables {
        g.trace_ops.copy_to_offset_device(
            &d_eta_powers.slice(idx * 5..idx * 5 + 5),
            &mut d_table_state,
            5,
            (idx * TS_STRIDE + TS_ETAK) as u32,
        );
    }
    g.sumcheck.init_air_table_sums_from_logup_device_async(
        &d_bus_numerators,
        &d_bus_denominators,
        d_logup_c,
        &d_bus_beta,
        d_air_bus_directions,
        &mut d_table_state,
        n_tables as u32,
    );
    let mut d_ze_all = take_air_workspace(&mut air_static.d_ze_all, "zero-eval workspace");
    let mut d_cc_scratch = take_air_workspace(&mut air_static.d_cc_scratch, "combined coeff scratch");
    let mut d_bare_scratch = take_air_workspace(&mut air_static.d_bare_scratch, "bare coeff scratch");
    let transcript_stride_words = (max_full_degree + 1) * 5 - 5;
    let mut d_transcript = take_air_workspace(&mut air_static.d_transcript, "AIR transcript");
    let mut d_transcript_len = take_air_workspace(&mut air_static.d_transcript_len, "AIR transcript length");
    let mut d_challenges = take_air_workspace(&mut air_static.d_challenges, "AIR challenges");
    let mut d_eval_transcript = take_air_workspace(&mut air_static.d_eval_transcript, "AIR eval transcript");
    stream.memset_zeros(&mut d_ze_all).expect("zero AIR ze workspace");

    let mut current_ext_is_a = vec![true; plans.len()];
    for round in 0..n_rounds {
        g.sumcheck.patch_air_table_state_from_gkr_async(
            &d_round_meta[round],
            &air_static.d_round_extra[round],
            d_gkr_point_words,
            (d_gkr_point_words.len() / 5) as u32,
            &mut d_table_state,
            n_tables as u32,
        );

        for (idx, plan) in plans.iter_mut().enumerate() {
            if round < plan.join_round {
                continue;
            }

            let local_round = round - plan.join_round;
            let schedule = plan.schedules[local_round];

            // Skip kernel launch when active_pairs=0 (all padding, protocol step handles it)
            if schedule.active_pairs == 0 {
                continue;
            }

            match (plan.table, schedule.is_base) {
                (Table::Execution(_), true) => {
                    g.sumcheck.launch_air_execution_multi_z_fb_into(
                        &plan.d_base_cols,
                        &plan.d_base_down_cols,
                        &plan.d_eq_rounds[local_round],
                        &plan.d_alphas,
                        &d_logup_alphas,
                        &d_bus_beta,
                        &mut plan.d_partial_sums,
                        schedule.n_rows,
                        schedule.active_pairs,
                        schedule.fold_bit,
                    );
                }
                (Table::ExtensionOp(_), true) => {
                    g.sumcheck.launch_air_ext_op_multi_z_fb_into(
                        &plan.d_base_cols,
                        &plan.d_base_down_cols,
                        &plan.d_eq_rounds[local_round],
                        &plan.d_alphas,
                        &d_logup_alphas,
                        &d_bus_beta,
                        &mut plan.d_partial_sums,
                        schedule.n_rows,
                        schedule.active_pairs,
                        schedule.fold_bit,
                    );
                }
                (Table::Poseidon16(_), true) => {
                    g.sumcheck.launch_air_poseidon16_multi_z_fb_into(
                        &plan.d_base_cols,
                        &plan.d_eq_rounds[local_round],
                        &plan.d_alphas,
                        d_p16_rc,
                        d_p16_mds,
                        d_p16_sparse,
                        &d_logup_alphas,
                        &d_bus_beta,
                        &mut plan.d_partial_sums,
                        schedule.n_rows,
                        schedule.active_pairs,
                        schedule.fold_bit,
                    );
                }
                (Table::Execution(_), false) => {
                    let d_in = if current_ext_is_a[idx] {
                        &plan.d_ext_up_a
                    } else {
                        &plan.d_ext_up_b
                    };
                    let d_down = if current_ext_is_a[idx] {
                        plan.d_ext_down_a.as_ref()
                    } else {
                        plan.d_ext_down_b.as_ref()
                    }
                    .expect("execution down cols");
                    g.sumcheck.launch_air_execution_multi_z_ext_fb_into(
                        d_in,
                        d_down,
                        &plan.d_eq_rounds[local_round],
                        &plan.d_alphas,
                        &d_logup_alphas,
                        &d_bus_beta,
                        &mut plan.d_partial_sums,
                        schedule.n_rows,
                        schedule.active_pairs,
                        schedule.fold_bit,
                    );
                }
                (Table::ExtensionOp(_), false) => {
                    let d_in = if current_ext_is_a[idx] {
                        &plan.d_ext_up_a
                    } else {
                        &plan.d_ext_up_b
                    };
                    let d_down = if current_ext_is_a[idx] {
                        plan.d_ext_down_a.as_ref()
                    } else {
                        plan.d_ext_down_b.as_ref()
                    }
                    .expect("extension-op down cols");
                    g.sumcheck.launch_air_ext_op_multi_z_ext_fb_into(
                        d_in,
                        d_down,
                        &plan.d_eq_rounds[local_round],
                        &plan.d_alphas,
                        &d_logup_alphas,
                        &d_bus_beta,
                        &mut plan.d_partial_sums,
                        schedule.n_rows,
                        schedule.active_pairs,
                        schedule.fold_bit,
                    );
                }
                (Table::Poseidon16(_), false) => {
                    let d_in = if current_ext_is_a[idx] {
                        &plan.d_ext_up_a
                    } else {
                        &plan.d_ext_up_b
                    };
                    g.sumcheck.launch_air_poseidon16_multi_z_ext_fb_into(
                        d_in,
                        &plan.d_eq_rounds[local_round],
                        &plan.d_alphas,
                        d_p16_rc,
                        d_p16_mds,
                        d_p16_sparse,
                        &d_logup_alphas,
                        &d_bus_beta,
                        &mut plan.d_partial_sums,
                        schedule.n_rows,
                        schedule.active_pairs,
                        schedule.fold_bit,
                    );
                }
            }

            for z in 0..plan.degree {
                let src_offset_words = z * schedule.blocks as usize * 5;
                let ze_offset_words = schedule.ze_offset_words + z * 5;
                g.sumcheck.launch_reduce_ext_into_async(
                    &plan.d_partial_sums,
                    src_offset_words,
                    &mut d_ze_all,
                    ze_offset_words,
                    schedule.blocks,
                );
            }
        }
        g.sumcheck.build_air_bare_coeffs_async(
            &d_ze_all,
            &air_static.d_ze_offsets[round],
            &d_table_state,
            n_tables as u32,
            max_full_degree as u32,
            &mut d_bare_scratch,
        );
        g.sumcheck.protocol_step_async(
            &d_ze_all,
            &air_static.d_ze_offsets[round],
            &mut d_table_state,
            n_tables as u32,
            max_full_degree as u32,
            &mut d_cc_scratch,
            &mut d_bare_scratch,
            &mut d_challenges,
            round * 5,
            &mut d_transcript,
            round * transcript_stride_words,
            &mut d_transcript_len,
            &mut fs.d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
        );

        for (idx, plan) in plans.iter_mut().enumerate() {
            if round < plan.join_round {
                continue;
            }
            let local_round = round - plan.join_round;
            let schedule = plan.schedules[local_round];

            if schedule.is_base {
                g.sumcheck.fold_multi_col_b2e_at_bit_into_async(
                    &plan.d_base_cols,
                    &mut plan.d_ext_up_a,
                    schedule.n_rows,
                    schedule.total_pairs,
                    plan.n_up_cols,
                    &d_challenges,
                    round * 5,
                    schedule.fold_bit,
                );
                if plan.n_down_cols > 0 {
                    g.sumcheck.fold_multi_col_b2e_at_bit_into_async(
                        &plan.d_base_down_cols,
                        plan.d_ext_down_a.as_mut().unwrap(),
                        schedule.n_rows,
                        schedule.total_pairs,
                        plan.n_down_cols,
                        &d_challenges,
                        round * 5,
                        schedule.fold_bit,
                    );
                }
                current_ext_is_a[idx] = true;
            } else if current_ext_is_a[idx] {
                g.sumcheck.fold_multi_col_ext_at_bit_into_async(
                    &plan.d_ext_up_a,
                    &mut plan.d_ext_up_b,
                    schedule.n_rows,
                    schedule.total_pairs,
                    plan.n_up_cols,
                    &d_challenges,
                    round * 5,
                    schedule.fold_bit,
                );
                if plan.n_down_cols > 0 {
                    g.sumcheck.fold_multi_col_ext_at_bit_into_async(
                        plan.d_ext_down_a.as_ref().unwrap(),
                        plan.d_ext_down_b.as_mut().unwrap(),
                        schedule.n_rows,
                        schedule.total_pairs,
                        plan.n_down_cols,
                        &d_challenges,
                        round * 5,
                        schedule.fold_bit,
                    );
                }
                current_ext_is_a[idx] = false;
            } else {
                g.sumcheck.fold_multi_col_ext_at_bit_into_async(
                    &plan.d_ext_up_b,
                    &mut plan.d_ext_up_a,
                    schedule.n_rows,
                    schedule.total_pairs,
                    plan.n_up_cols,
                    &d_challenges,
                    round * 5,
                    schedule.fold_bit,
                );
                if plan.n_down_cols > 0 {
                    g.sumcheck.fold_multi_col_ext_at_bit_into_async(
                        plan.d_ext_down_b.as_ref().unwrap(),
                        plan.d_ext_down_a.as_mut().unwrap(),
                        schedule.n_rows,
                        schedule.total_pairs,
                        plan.n_down_cols,
                        &d_challenges,
                        round * 5,
                        schedule.fold_bit,
                    );
                }
                current_ext_is_a[idx] = true;
            }
        }
    }

    let mut eval_transcript_offset = 0usize;
    for (idx, plan) in plans.iter_mut().enumerate() {
        let mut whir_table_target = whir_table_targets
            .next()
            .unwrap_or_else(|| panic!("missing AIR WHIR statement targets for {}", plan.table.name()));
        debug_assert_eq!(whir_table_target.table, plan.table);
        let n_up_words = plan.n_up_cols as usize * 5;
        let n_down_words = plan.n_down_cols as usize * 5;
        if current_ext_is_a[idx] {
            g.trace_ops.copy_to_offset_device(
                &plan.d_ext_up_a.slice(..n_up_words),
                &mut plan.d_eval_values,
                n_up_words as u32,
                0,
            );
            if plan.n_down_cols > 0 {
                g.trace_ops.copy_to_offset_device(
                    &plan.d_ext_down_a.as_ref().unwrap().slice(..n_down_words),
                    &mut plan.d_eval_values,
                    n_down_words as u32,
                    n_up_words as u32,
                );
            }
        } else {
            g.trace_ops.copy_to_offset_device(
                &plan.d_ext_up_b.slice(..n_up_words),
                &mut plan.d_eval_values,
                n_up_words as u32,
                0,
            );
            if plan.n_down_cols > 0 {
                g.trace_ops.copy_to_offset_device(
                    &plan.d_ext_down_b.as_ref().unwrap().slice(..n_down_words),
                    &mut plan.d_eval_values,
                    n_down_words as u32,
                    n_up_words as u32,
                );
            }
        }
        g.sumcheck.reversed_suffix_point_words_into_async(
            &d_challenges,
            n_rounds,
            plan.log_n_rows,
            &mut plan.d_eval_point,
        );
        fill_air_whir_statement_targets(g, &plan.d_eval_point, &plan.d_eval_values, &mut whir_table_target);
        let n_eval_words = n_up_words + n_down_words;
        g.sumcheck.memcpy_d2d_async(
            &plan.d_eval_values,
            0,
            &mut d_eval_transcript,
            eval_transcript_offset,
            n_eval_words,
        );
        eval_transcript_offset += n_eval_words;
        g.sumcheck.challenger_observe_device_scalars_async(
            &mut fs.d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            &plan.d_eval_values,
            n_eval_words as u32,
        );
    }
    debug_assert!(whir_table_targets.next().is_none());
    fill_public_memory_whir_statement_target(g, fs, d_memory, &mut public_memory_target);

    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = stream
        .end_capture(graph_flags)
        .unwrap()
        .expect("stream capture produced no graph");
    // nothing to do here

    graph.launch().unwrap();

    fs.transcript_chunks.push(TranscriptChunk {
        d_words: d_transcript.into(),
        n_words: n_rounds * transcript_stride_words,
    });
    fs.transcript_chunks.push(TranscriptChunk {
        d_words: d_eval_transcript.into(),
        n_words: eval_transcript_offset,
    });

    drop(plans);
}
