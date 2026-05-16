//! GPU logup: GPU fingerprint building + GPU GKR quotient protocol.
//!
//! The logup numerator/denominator witness is assembled on device from the
//! staged trace, staged bytecode columns, and device access-count buffers.
//! The GKR quotient protocol then consumes that device witness directly.
//!
//! The F-S sequence is identical to `prove_generic_logup`.

use std::collections::{BTreeMap, BTreeSet};

use backend::*;
use cudarc::driver::{
    safe::{CudaSlice, DevicePtr},
    sys,
};
use lean_vm::*;
use sub_protocols::*;

use super::gpu_gkr::gpu_prove_gkr_quotient_from_device_with_fs;
use super::gpu_prove_execution::{Gpu, GpuFsPhase, GpuUploadedExecutionTrace, GpuUploadedTableTrace};

const ALPHA_ZERO_INDEX: u32 = u32::MAX;

pub struct GpuLogupOutput {
    pub d_c: CudaSlice<u32>,
    pub d_alphas_eq_poly: CudaSlice<u32>,
    pub d_gkr_point_words: CudaSlice<u32>,
    pub bus_evals: Vec<GpuLogupBusEvals>,
    pub(crate) _d_alpha_challenges: CudaSlice<u32>,
    pub(crate) _dot_accumulation_guards: Vec<LogupExtDotAccumulationGuard>,
}

pub struct GpuLogupWhirStatementTargets<'a> {
    pub d_memory_and_acc_point: &'a mut CudaSlice<u32>,
    pub d_value_memory: &'a mut CudaSlice<u32>,
    pub d_value_memory_acc: &'a mut CudaSlice<u32>,
    pub d_bytecode_and_acc_point: &'a mut CudaSlice<u32>,
    pub d_value_bytecode_acc: &'a mut CudaSlice<u32>,
    pub table_targets: Vec<GpuLogupWhirTableStatementTargets<'a>>,
}

pub struct GpuLogupWhirTableStatementTargets<'a> {
    pub table: Table,
    pub d_point: &'a mut CudaSlice<u32>,
    pub values: Vec<GpuLogupWhirColumnStatementTarget<'a>>,
}

pub struct GpuLogupWhirColumnStatementTarget<'a> {
    pub col_index: ColIndex,
    pub d_value: &'a mut CudaSlice<u32>,
}

pub struct GpuLogupBusEvals {
    pub table: Table,
    pub d_numerator: CudaSlice<u32>,
    pub d_denominator: CudaSlice<u32>,
}

pub(crate) struct LogupExtDotAccumulationGuard {
    pub d_out: CudaSlice<u32>,
    _d_base_sum: CudaSlice<u32>,
    _d_values: Option<CudaSlice<u32>>,
    _prepared_scalars: gpu_sumcheck::LogupPreparedConstants,
}

fn kb_u32(v: F) -> u32 {
    unsafe { std::mem::transmute(v) }
}

pub(crate) struct GpuLogupStaticMetadata {
    alpha_last_idx: u32,
    memory_len: usize,
    log_memory: usize,
    bytecode_n_rows: usize,
    log_bytecode: usize,
    total_active_len: usize,
    total_gkr_n_vars: usize,
    pivot: usize,
    bytecode_offset: usize,
    max_table_height: usize,
    table_plans: Vec<GpuLogupTablePlan>,
    memory_den_inputs: gpu_sumcheck::LogupConstantDeviceInputs,
    bytecode_den_inputs: gpu_sumcheck::LogupConstantDeviceInputs,
    execution_den_inputs: gpu_sumcheck::LogupConstantDeviceInputs,
    bus_den_inputs: BTreeMap<Table, gpu_sumcheck::LogupConstantDeviceInputs>,
    lookup_den_inputs: BTreeMap<Table, Vec<gpu_sumcheck::LogupConstantDeviceInputs>>,
    bus_data_scalar_inputs: BTreeMap<Table, gpu_sumcheck::LogupConstantDeviceInputs>,
    memory_den_prepared: Option<gpu_sumcheck::LogupPreparedConstants>,
    bytecode_den_prepared: Option<gpu_sumcheck::LogupPreparedConstants>,
    execution_den_prepared: Option<gpu_sumcheck::LogupPreparedConstants>,
    bus_den_prepared: BTreeMap<Table, Option<gpu_sumcheck::LogupPreparedConstants>>,
    lookup_den_prepared: BTreeMap<Table, Vec<Option<gpu_sumcheck::LogupPreparedConstants>>>,
    bus_data_scalar_prepared: BTreeMap<Table, Option<gpu_sumcheck::LogupPreparedConstants>>,
    d_memory_num_br: Option<CudaSlice<u32>>,
    d_memory_num: Option<CudaSlice<u32>>,
    d_memory_den: Option<CudaSlice<u32>>,
    d_memory_den_br: Option<CudaSlice<u32>>,
    d_bytecode_num_br: Option<CudaSlice<u32>>,
    d_bytecode_num: Option<CudaSlice<u32>>,
    d_bytecode_den: Option<CudaSlice<u32>>,
    d_bytecode_den_br: Option<CudaSlice<u32>>,
    d_value_memory_acc_for_transcript: Option<CudaSlice<u32>>,
    d_value_memory_for_transcript: Option<CudaSlice<u32>>,
    d_value_bytecode_acc_for_transcript: Option<CudaSlice<u32>>,
    d_mle_scratch_a: Option<CudaSlice<u32>>,
    d_mle_scratch_b: Option<CudaSlice<u32>>,
    d_mle_eval_temp: Option<CudaSlice<u32>>,
    d_numerators: Option<CudaSlice<u32>>,
    d_denominators: Option<CudaSlice<u32>>,
    d_gkr_initial_point: Option<CudaSlice<u32>>,
    d_gkr_nums_padded: Option<CudaSlice<u32>>,
    d_gkr_dens_padded: Option<CudaSlice<u32>>,
    d_gkr_top_transcript: Option<CudaSlice<u32>>,
    d_gkr_claim_num: Option<CudaSlice<u32>>,
    d_gkr_claim_den: Option<CudaSlice<u32>>,
    d_gkr_mle_scratch_a: Option<CudaSlice<u32>>,
    d_gkr_mle_scratch_b: Option<CudaSlice<u32>>,
    d_base_ones: CudaSlice<u32>,
}

struct GpuLogupTablePlan {
    table: Table,
    log_n_rows: usize,
    is_execution: bool,
    bus_selector: ColIndex,
    bus_direction: GpuLogupBusDirection,
    execution_offset: Option<usize>,
    bus_offset: usize,
    bus_data: Vec<GpuLogupBusDataPlan>,
    bus_data_column_count: usize,
    lookups: Vec<GpuLogupLookupPlan>,
    logup_value_cols: Vec<ColIndex>,
    d_exec_cols: Option<CudaSlice<u32>>,
    d_exec_instr_transcript: Option<CudaSlice<u32>>,
    d_bus_num_br: Option<CudaSlice<u32>>,
    d_bus_num: Option<CudaSlice<u32>>,
    d_bus_cols: Option<CudaSlice<u32>>,
    d_bus_data_values: Option<CudaSlice<u32>>,
    d_lookup_cols: Vec<Option<CudaSlice<u32>>>,
    d_exec_den: Option<CudaSlice<u32>>,
    d_bus_den: Option<CudaSlice<u32>>,
    d_lookup_den: Vec<Option<CudaSlice<u32>>>,
    d_exec_den_br: Option<CudaSlice<u32>>,
    d_bus_den_br: Option<CudaSlice<u32>>,
    d_lookup_den_br: Vec<Option<CudaSlice<u32>>>,
    d_logup_value_eval_transcript: Vec<Option<CudaSlice<u32>>>,
    d_bus_numerator_for_air: Option<CudaSlice<u32>>,
    d_bus_numerator_for_transcript: Option<CudaSlice<u32>>,
    d_bus_denominator_for_air: Option<CudaSlice<u32>>,
    d_bus_denominator_for_transcript: Option<CudaSlice<u32>>,
    d_bus_data_base_sum: Option<CudaSlice<u32>>,
    d_bus_data_dot_out: Option<CudaSlice<u32>>,
}

impl GpuLogupTablePlan {
    fn n_rows(&self) -> usize {
        1usize << self.log_n_rows
    }
}

#[derive(Clone, Copy)]
enum GpuLogupBusDirection {
    Pull,
    Push,
}

impl GpuLogupBusDirection {
    fn is_pull(self) -> bool {
        matches!(self, Self::Pull)
    }
}

enum GpuLogupBusDataPlan {
    Column(ColIndex),
    Constant { d_value: Option<CudaSlice<u32>> },
}

struct GpuLogupLookupPlan {
    index: ColIndex,
    values: Vec<ColIndex>,
    value_offsets: Vec<usize>,
}

fn upload_logup_constant_inputs(
    g: &Gpu,
    alpha_indices: &[u32],
    alpha_negated: &[u32],
    contrib_terms: &[(u32, F)],
) -> gpu_sumcheck::LogupConstantDeviceInputs {
    let contrib_indices: Vec<u32> = contrib_terms.iter().map(|(idx, _)| *idx).collect();
    let contrib_coeffs: Vec<u32> = contrib_terms.iter().map(|(_, coeff)| kb_u32(*coeff)).collect();
    g.sumcheck
        .upload_logup_constant_inputs(alpha_indices, alpha_negated, &contrib_indices, &contrib_coeffs)
}

pub(crate) fn build_logup_static_metadata(
    g: &Gpu,
    traces: &BTreeMap<Table, TableTrace>,
    memory_len: usize,
    bytecode_n_rows: usize,
) -> GpuLogupStaticMetadata {
    let logup_alpha_len = log2_ceil_usize(max_bus_width_including_domainsep());
    let alpha_last_idx = ((1usize << logup_alpha_len) - 1) as u32;
    let log_memory = log2_strict_usize(memory_len);
    let log_bytecode = log2_strict_usize(bytecode_n_rows);

    debug_assert_eq!(LOGUP_MEMORY_DOMAINSEP, 0);
    let memory_den_inputs = upload_logup_constant_inputs(g, &[0, 1], &[0, 0], &[]);
    let memory_den_prepared = Some(g.sumcheck.alloc_logup_prepared_constants(&memory_den_inputs));

    let bytecode_alphas: Vec<u32> = (0..=N_INSTRUCTION_COLUMNS as u32).collect();
    let bytecode_alpha_signs = vec![0u32; bytecode_alphas.len()];
    let bytecode_den_inputs = upload_logup_constant_inputs(
        g,
        &bytecode_alphas,
        &bytecode_alpha_signs,
        &[(alpha_last_idx, F::from_usize(LOGUP_BYTECODE_DOMAINSEP))],
    );
    let bytecode_den_prepared = Some(g.sumcheck.alloc_logup_prepared_constants(&bytecode_den_inputs));

    let mut exec_alphas: Vec<u32> = (0..=N_INSTRUCTION_COLUMNS as u32).collect();
    exec_alphas.push(ALPHA_ZERO_INDEX);
    let exec_alpha_signs = vec![0u32; exec_alphas.len()];
    let execution_den_inputs = upload_logup_constant_inputs(
        g,
        &exec_alphas,
        &exec_alpha_signs,
        &[(alpha_last_idx, F::from_usize(LOGUP_BYTECODE_DOMAINSEP))],
    );
    let execution_den_prepared = Some(g.sumcheck.alloc_logup_prepared_constants(&execution_den_inputs));

    let tables_log_heights: BTreeMap<Table, usize> = traces.iter().map(|(t, tr)| (*t, tr.log_n_rows)).collect();
    let tables_sorted = sort_tables_by_height(&tables_log_heights);
    let mut table_plans = Vec::with_capacity(tables_sorted.len());
    let mut bus_den_inputs = BTreeMap::new();
    let mut lookup_den_inputs = BTreeMap::new();
    let mut bus_data_scalar_inputs = BTreeMap::new();
    let mut bus_den_prepared = BTreeMap::new();
    let mut lookup_den_prepared = BTreeMap::new();
    let mut bus_data_scalar_prepared = BTreeMap::new();

    for (table, log_n_rows) in &tables_sorted {
        let bus = table.bus();
        let lookups = table.lookups();
        let mut bus_contrib_terms = vec![(alpha_last_idx, -F::from_usize(LOGUP_PRECOMPILE_DOMAINSEP))];
        let mut bus_alphas = Vec::<u32>::new();
        let mut bus_alpha_signs = Vec::<u32>::new();
        let mut bus_data_plan = Vec::with_capacity(bus.data.len());
        let mut bus_data_column_count = 0usize;
        for (j, entry) in bus.data.iter().enumerate() {
            match entry {
                BusData::Column(col) => {
                    bus_alphas.push(j as u32);
                    bus_alpha_signs.push(1);
                    bus_data_plan.push(GpuLogupBusDataPlan::Column(*col));
                    bus_data_column_count += 1;
                }
                BusData::Constant(val) => {
                    bus_contrib_terms.push((j as u32, -F::from_usize(*val)));
                    bus_data_plan.push(GpuLogupBusDataPlan::Constant {
                        d_value: Some(g.trace_ops.fill_ext(EF::from(F::from_usize(*val)), 1)),
                    });
                }
            }
        }
        bus_alphas.push(ALPHA_ZERO_INDEX);
        bus_alpha_signs.push(0);
        let bus_den_input = upload_logup_constant_inputs(g, &bus_alphas, &bus_alpha_signs, &bus_contrib_terms);
        let bus_den_prepared_entry = g.sumcheck.alloc_logup_prepared_constants(&bus_den_input);
        bus_den_inputs.insert(*table, bus_den_input);
        bus_den_prepared.insert(*table, Some(bus_den_prepared_entry));

        let bus_data_alpha_indices: Vec<u32> = (0..bus.data.len() as u32).collect();
        let bus_data_alpha_signs = vec![0u32; bus_data_alpha_indices.len()];
        let bus_data_scalar_input = upload_logup_constant_inputs(
            g,
            &bus_data_alpha_indices,
            &bus_data_alpha_signs,
            &[(alpha_last_idx, F::from_usize(LOGUP_PRECOMPILE_DOMAINSEP))],
        );
        let bus_data_scalar_prepared_entry = g.sumcheck.alloc_logup_prepared_constants(&bus_data_scalar_input);
        bus_data_scalar_inputs.insert(*table, bus_data_scalar_input);
        bus_data_scalar_prepared.insert(*table, Some(bus_data_scalar_prepared_entry));

        let mut table_lookup_inputs = Vec::new();
        let mut table_lookup_prepared = Vec::new();
        for lookup in &lookups {
            for i in 0..lookup.values.len() {
                let lookup_input =
                    upload_logup_constant_inputs(g, &[0, 1, ALPHA_ZERO_INDEX], &[0, 0, 0], &[(1, F::from_usize(i))]);
                table_lookup_prepared.push(Some(g.sumcheck.alloc_logup_prepared_constants(&lookup_input)));
                table_lookup_inputs.push(lookup_input);
            }
        }
        lookup_den_inputs.insert(*table, table_lookup_inputs);
        lookup_den_prepared.insert(*table, table_lookup_prepared);

        let mut logup_value_cols = BTreeSet::<ColIndex>::new();
        if table.is_execution_table() {
            logup_value_cols.insert(COL_PC);
            for i in 0..N_INSTRUCTION_COLUMNS {
                logup_value_cols.insert(N_RUNTIME_COLUMNS + i);
            }
        }
        for lookup in &lookups {
            logup_value_cols.insert(lookup.index);
            for col_index in &lookup.values {
                logup_value_cols.insert(*col_index);
            }
        }

        let lookups: Vec<GpuLogupLookupPlan> = lookups
            .into_iter()
            .map(|lookup| GpuLogupLookupPlan {
                index: lookup.index,
                values: lookup.values,
                value_offsets: Vec::new(),
            })
            .collect();
        let n_rows = 1usize << log_n_rows;
        let d_exec_cols = if table.is_execution_table() {
            Some(
                g.stream
                    .alloc_zeros::<u32>((N_INSTRUCTION_COLUMNS + 1) * n_rows)
                    .unwrap(),
            )
        } else {
            None
        };
        let d_exec_den = if table.is_execution_table() {
            Some(g.stream.alloc_zeros::<u32>(n_rows * 5).unwrap())
        } else {
            None
        };
        let d_exec_den_br = if table.is_execution_table() {
            Some(g.stream.alloc_zeros::<u32>(n_rows * 5).unwrap())
        } else {
            None
        };
        let d_exec_instr_transcript = if table.is_execution_table() {
            Some(g.stream.alloc_zeros::<u32>(N_INSTRUCTION_COLUMNS * 5).unwrap())
        } else {
            None
        };
        let d_bus_num_br = Some(g.stream.alloc_zeros::<u32>(n_rows).unwrap());
        let d_bus_num = Some(g.stream.alloc_zeros::<u32>(n_rows).unwrap());
        let d_bus_cols = Some(
            g.stream
                .alloc_zeros::<u32>((bus_data_column_count * n_rows).max(1))
                .unwrap(),
        );
        let d_bus_data_values = Some(g.stream.alloc_zeros::<u32>((bus.data.len() * 5).max(1)).unwrap());
        let d_bus_den = Some(g.stream.alloc_zeros::<u32>(n_rows * 5).unwrap());
        let d_bus_den_br = Some(g.stream.alloc_zeros::<u32>(n_rows * 5).unwrap());
        let d_lookup_cols = lookups
            .iter()
            .flat_map(|lookup: &GpuLogupLookupPlan| lookup.values.iter())
            .map(|_| Some(g.stream.alloc_zeros::<u32>(2 * n_rows).unwrap()))
            .collect();
        let d_lookup_den = lookups
            .iter()
            .flat_map(|lookup: &GpuLogupLookupPlan| lookup.values.iter())
            .map(|_| Some(g.stream.alloc_zeros::<u32>(n_rows * 5).unwrap()))
            .collect();
        let d_lookup_den_br = lookups
            .iter()
            .flat_map(|lookup: &GpuLogupLookupPlan| lookup.values.iter())
            .map(|_| Some(g.stream.alloc_zeros::<u32>(n_rows * 5).unwrap()))
            .collect();
        let d_logup_value_eval_transcript = logup_value_cols
            .iter()
            .map(|_| Some(g.stream.alloc_zeros::<u32>(5).unwrap()))
            .collect();
        let d_bus_numerator_for_air = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
        let d_bus_numerator_for_transcript = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
        let d_bus_denominator_for_air = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
        let d_bus_denominator_for_transcript = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
        let d_bus_data_base_sum = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
        let d_bus_data_dot_out = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
        let bus_direction = match bus.direction {
            BusDirection::Pull => GpuLogupBusDirection::Pull,
            BusDirection::Push => GpuLogupBusDirection::Push,
        };
        table_plans.push(GpuLogupTablePlan {
            table: *table,
            log_n_rows: *log_n_rows,
            is_execution: table.is_execution_table(),
            bus_selector: bus.selector,
            bus_direction,
            execution_offset: None,
            bus_offset: 0,
            bus_data: bus_data_plan,
            bus_data_column_count,
            lookups,
            logup_value_cols: logup_value_cols.into_iter().collect(),
            d_exec_cols,
            d_exec_instr_transcript,
            d_bus_num_br,
            d_bus_num,
            d_bus_cols,
            d_bus_data_values,
            d_lookup_cols,
            d_exec_den,
            d_bus_den,
            d_lookup_den,
            d_exec_den_br,
            d_bus_den_br,
            d_lookup_den_br,
            d_logup_value_eval_transcript,
            d_bus_numerator_for_air,
            d_bus_numerator_for_transcript,
            d_bus_denominator_for_air,
            d_bus_denominator_for_transcript,
            d_bus_data_base_sum,
            d_bus_data_dot_out,
        });
    }

    let max_table_height = table_plans[0].n_rows();
    let min_section_log = log_bytecode.min(table_plans.last().unwrap().log_n_rows);
    let pivot = ENDIANNESS_PIVOT_GKR.min(min_section_log);
    let bytecode_offset = memory_len;
    let mut offset = memory_len + max_table_height.max(bytecode_n_rows);
    for plan in &mut table_plans {
        let n_rows = plan.n_rows();
        if plan.is_execution {
            plan.execution_offset = Some(offset);
            offset += n_rows;
        }
        plan.bus_offset = offset;
        offset += n_rows;
        for lookup in &mut plan.lookups {
            lookup.value_offsets = Vec::with_capacity(lookup.values.len());
            for _ in &lookup.values {
                lookup.value_offsets.push(offset);
                offset += n_rows;
            }
        }
    }
    let total_active_len = offset;
    let total_gkr_n_vars = log2_ceil_usize(total_active_len);
    let d_memory_num_br = Some(g.stream.alloc_zeros::<u32>(memory_len).unwrap());
    let d_memory_num = Some(g.stream.alloc_zeros::<u32>(memory_len).unwrap());
    let d_memory_den = Some(g.stream.alloc_zeros::<u32>(memory_len * 5).unwrap());
    let d_memory_den_br = Some(g.stream.alloc_zeros::<u32>(memory_len * 5).unwrap());
    let d_bytecode_num_br = Some(g.stream.alloc_zeros::<u32>(bytecode_n_rows).unwrap());
    let d_bytecode_num = Some(g.stream.alloc_zeros::<u32>(bytecode_n_rows).unwrap());
    let d_bytecode_den = Some(g.stream.alloc_zeros::<u32>(bytecode_n_rows * 5).unwrap());
    let d_bytecode_den_br = Some(g.stream.alloc_zeros::<u32>(bytecode_n_rows * 5).unwrap());
    let d_value_memory_acc_for_transcript = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
    let d_value_memory_for_transcript = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
    let d_value_bytecode_acc_for_transcript = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
    let max_mle_len = memory_len.max(bytecode_n_rows).max(max_table_height).max(2);
    let d_mle_scratch_a = Some(g.stream.alloc_zeros::<u32>((max_mle_len / 2) * 5).unwrap());
    let d_mle_scratch_b = Some(g.stream.alloc_zeros::<u32>((max_mle_len / 2) * 5).unwrap());
    let d_mle_eval_temp = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
    let d_numerators = Some(g.stream.alloc_zeros::<u32>(total_active_len).unwrap());
    let d_denominators = Some(g.stream.alloc_zeros::<u32>(total_active_len * 5).unwrap());
    let d_gkr_initial_point = Some(g.stream.alloc_zeros::<u32>(N_VARS_TO_SEND_GKR_COEFFS * 5).unwrap());
    let gkr_full_n = total_active_len.next_power_of_two();
    let d_gkr_nums_padded = Some(g.stream.alloc_zeros::<u32>(gkr_full_n * 5).unwrap());
    let d_gkr_dens_padded = Some(g.stream.alloc_zeros::<u32>(gkr_full_n * 5).unwrap());
    let d_gkr_top_transcript = Some(
        g.stream
            .alloc_zeros::<u32>((1usize << N_VARS_TO_SEND_GKR_COEFFS) * 10)
            .unwrap(),
    );
    let top_gkr_len = 1usize << N_VARS_TO_SEND_GKR_COEFFS;
    let d_gkr_claim_num = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
    let d_gkr_claim_den = Some(g.stream.alloc_zeros::<u32>(5).unwrap());
    let d_gkr_mle_scratch_a = Some(g.stream.alloc_zeros::<u32>((top_gkr_len / 2) * 5).unwrap());
    let d_gkr_mle_scratch_b = Some(g.stream.alloc_zeros::<u32>((top_gkr_len / 2) * 5).unwrap());
    let d_base_ones = g.trace_ops.fill_base(kb_u32(F::ONE), max_table_height as u32);

    GpuLogupStaticMetadata {
        alpha_last_idx,
        memory_len,
        log_memory,
        bytecode_n_rows,
        log_bytecode,
        total_active_len,
        total_gkr_n_vars,
        pivot,
        bytecode_offset,
        max_table_height,
        table_plans,
        memory_den_inputs,
        bytecode_den_inputs,
        execution_den_inputs,
        bus_den_inputs,
        lookup_den_inputs,
        bus_data_scalar_inputs,
        memory_den_prepared,
        bytecode_den_prepared,
        execution_den_prepared,
        bus_den_prepared,
        lookup_den_prepared,
        bus_data_scalar_prepared,
        d_memory_num_br,
        d_memory_num,
        d_memory_den,
        d_memory_den_br,
        d_bytecode_num_br,
        d_bytecode_num,
        d_bytecode_den,
        d_bytecode_den_br,
        d_value_memory_acc_for_transcript,
        d_value_memory_for_transcript,
        d_value_bytecode_acc_for_transcript,
        d_mle_scratch_a,
        d_mle_scratch_b,
        d_mle_eval_temp,
        d_numerators,
        d_denominators,
        d_gkr_initial_point,
        d_gkr_nums_padded,
        d_gkr_dens_padded,
        d_gkr_top_transcript,
        d_gkr_claim_num,
        d_gkr_claim_den,
        d_gkr_mle_scratch_a,
        d_gkr_mle_scratch_b,
        d_base_ones,
    }
}

fn add_ext_devices_into(
    g: &Gpu,
    d_a: &CudaSlice<u32>,
    d_b: &CudaSlice<u32>,
    mut d_out: CudaSlice<u32>,
) -> CudaSlice<u32> {
    g.sumcheck.ext_add_into_async(d_a, d_b, &mut d_out);
    d_out
}

fn concat_device_ext_scalars_into(g: &Gpu, d_scalars: &[CudaSlice<u32>], d_out: &mut CudaSlice<u32>) {
    assert!(d_out.len() >= d_scalars.len() * 5);
    for (idx, d_scalar) in d_scalars.iter().enumerate() {
        g.sumcheck.memcpy_d2d_async(d_scalar, 0, d_out, idx * 5, 5);
    }
}

fn observe_ext_scalars_device(
    g: &Gpu,
    fs: &mut GpuFsPhase,
    d_scalars: &[CudaSlice<u32>],
    mut d_transcript: CudaSlice<u32>,
) {
    if d_scalars.is_empty() {
        return;
    }
    concat_device_ext_scalars_into(g, d_scalars, &mut d_transcript);
    fs.observe_ext_scalars_device(g, d_transcript, d_scalars.len());
}

fn observe_ext_scalar_device(g: &Gpu, fs: &mut GpuFsPhase, d_scalar: CudaSlice<u32>) {
    fs.observe_ext_scalars_device(g, d_scalar, 1);
}

fn take_logup_workspace(slot: &mut Option<CudaSlice<u32>>, label: &str) -> CudaSlice<u32> {
    slot.take()
        .unwrap_or_else(|| panic!("logup workspace already consumed: {label}"))
}

fn take_logup_prepared(
    slot: &mut Option<gpu_sumcheck::LogupPreparedConstants>,
    label: &str,
) -> gpu_sumcheck::LogupPreparedConstants {
    slot.take()
        .unwrap_or_else(|| panic!("logup prepared workspace already consumed: {label}"))
}

fn copy_device_words_into(
    g: &Gpu,
    d_src: &CudaSlice<u32>,
    mut d_dst: CudaSlice<u32>,
    n_words: usize,
) -> CudaSlice<u32> {
    assert!(d_src.len() >= n_words);
    assert!(d_dst.len() >= n_words);
    if n_words > 0 {
        g.sumcheck.memcpy_d2d_async(d_src, 0, &mut d_dst, 0, n_words);
    }
    d_dst
}

fn copy_ext_scalar_into(g: &Gpu, d_src: &CudaSlice<u32>, d_dst: CudaSlice<u32>) -> CudaSlice<u32> {
    copy_device_words_into(g, d_src, d_dst, 5)
}

fn copy_ext_scalar_into_ref(g: &Gpu, d_src: &CudaSlice<u32>, d_dst: &mut CudaSlice<u32>) {
    assert!(d_dst.len() >= 5);
    g.sumcheck.memcpy_d2d_async(d_src, 0, d_dst, 0, 5);
}

fn logup_value_idx(planned_cols: &[ColIndex], col_index: ColIndex) -> usize {
    planned_cols
        .iter()
        .position(|planned_col| *planned_col == col_index)
        .expect("observed logup column must be planned")
}

fn store_logup_column_eval_copy_once(
    g: &Gpu,
    written: &mut [bool],
    planned_cols: &[ColIndex],
    targets: &mut [GpuLogupWhirColumnStatementTarget<'_>],
    col_index: ColIndex,
    d_value: &CudaSlice<u32>,
) {
    let value_idx = logup_value_idx(planned_cols, col_index);
    assert_eq!(
        targets.len(),
        planned_cols.len(),
        "planned logup WHIR statement value target mismatch"
    );
    assert!(!written[value_idx], "duplicate logup value column {}", col_index);
    assert_eq!(
        targets[value_idx].col_index, col_index,
        "planned logup WHIR statement column target mismatch"
    );
    copy_ext_scalar_into_ref(g, d_value, targets[value_idx].d_value);
    written[value_idx] = true;
}

fn suffix_point_words_into(g: &Gpu, d_point_words: &CudaSlice<u32>, n_vars: usize, d_out: &mut CudaSlice<u32>) {
    let n_words = n_vars * 5;
    assert!(d_point_words.len() >= n_words);
    assert!(d_out.len() >= n_words);
    if n_words == 0 {
        return;
    }
    let start = d_point_words.len() - n_words;
    g.sumcheck.memcpy_d2d_async(d_point_words, start, d_out, 0, n_words);
}

fn gpu_mle_eval_base_device_point_observed_device_into(
    g: &Gpu,
    fs: &mut GpuFsPhase,
    d_evals: &impl DevicePtr<u32>,
    n_elements: usize,
    d_point_words: &CudaSlice<u32>,
    n_vars: usize,
    d_mle_scratch_a: &mut CudaSlice<u32>,
    d_mle_scratch_b: &mut CudaSlice<u32>,
    mut d_eval_for_transcript: CudaSlice<u32>,
    d_eval_for_statement: &mut CudaSlice<u32>,
) {
    gpu_mle_eval_base_device_point_into(
        g,
        d_evals,
        n_elements,
        d_point_words,
        n_vars,
        d_mle_scratch_a,
        d_mle_scratch_b,
        &mut d_eval_for_transcript,
    );
    copy_ext_scalar_into_ref(g, &d_eval_for_transcript, d_eval_for_statement);
    observe_ext_scalar_device(g, fs, d_eval_for_transcript);
}

#[allow(clippy::too_many_arguments)]
fn gpu_mle_eval_base_device_point_into<D>(
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
    assert_eq!(n_elements, 1usize << n_vars);
    assert!(d_out.len() >= EXT_DIM);
    if n_vars == 0 {
        g.stream.memset_zeros(d_out).expect("zero constant MLE output");
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

fn apply_bus_direction_device_into(
    g: &Gpu,
    d_value: &CudaSlice<u32>,
    direction: GpuLogupBusDirection,
    mut d_out: CudaSlice<u32>,
) -> CudaSlice<u32> {
    if direction.is_pull() {
        g.trace_ops.negate_ext_into(d_value, &mut d_out, 1);
        d_out
    } else {
        copy_ext_scalar_into(g, d_value, d_out)
    }
}

fn ext_dot_accumulate_from_prepared_scalars(
    g: &Gpu,
    d_base_sum: CudaSlice<u32>,
    d_values: CudaSlice<u32>,
    n_terms: u32,
    d_out: CudaSlice<u32>,
    prepared_scalars: gpu_sumcheck::LogupPreparedConstants,
) -> LogupExtDotAccumulationGuard {
    let mut d_out = d_out;
    g.sumcheck
        .ext_dot_accumulate_into_async(&d_base_sum, &d_values, &prepared_scalars.d_alphas, n_terms, &mut d_out);
    LogupExtDotAccumulationGuard {
        d_out,
        _d_base_sum: d_base_sum,
        _d_values: Some(d_values),
        _prepared_scalars: prepared_scalars,
    }
}

#[allow(clippy::too_many_arguments)]
fn gpu_eval_uploaded_column_device_at_point_words_into(
    g: &Gpu,
    tables: &BTreeMap<Table, GpuUploadedTableTrace>,
    table: Table,
    col_index: usize,
    n_rows: usize,
    d_point_words: &CudaSlice<u32>,
    n_vars: usize,
    d_mle_scratch_a: &mut CudaSlice<u32>,
    d_mle_scratch_b: &mut CudaSlice<u32>,
    d_out: &mut CudaSlice<u32>,
) {
    let d_all_cols = &tables[&table].d_all_cols;
    let d_col = d_all_cols.slice(col_index * n_rows..(col_index + 1) * n_rows);
    gpu_mle_eval_base_device_point_into(
        g,
        &d_col,
        n_rows,
        d_point_words,
        n_vars,
        d_mle_scratch_a,
        d_mle_scratch_b,
        d_out,
    );
}

#[allow(clippy::too_many_arguments)]
fn logup_fingerprint_device_from_alpha_eq_into(
    g: &Gpu,
    d_columns: &CudaSlice<u32>,
    d_c: &CudaSlice<u32>,
    d_alphas_eq_poly: &CudaSlice<u32>,
    constant_inputs: &gpu_sumcheck::LogupConstantDeviceInputs,
    prepared: &mut gpu_sumcheck::LogupPreparedConstants,
    d_denoms: &mut CudaSlice<u32>,
    n_rows: u32,
    n_cols: u32,
) {
    g.sumcheck
        .logup_prepare_constants_from_device_inputs_into_prepared_async(d_alphas_eq_poly, constant_inputs, prepared);
    g.sumcheck.logup_fingerprint_device_with_constants_into_async(
        d_columns,
        d_c,
        &prepared.d_contrib,
        &prepared.d_alphas,
        n_rows,
        n_cols,
        d_denoms,
    )
}

fn build_logup_witness_on_device(
    g: &Gpu,
    d_c: &CudaSlice<u32>,
    d_alphas_eq_poly: &CudaSlice<u32>,
    uploaded_trace: &mut GpuUploadedExecutionTrace,
    d_memory_acc: &CudaSlice<u32>,
    d_bytecode_acc: &CudaSlice<u32>,
) -> (usize, usize, CudaSlice<u32>, CudaSlice<u32>) {
    let d_memory = &uploaded_trace.d_memory;
    let d_bytecode_cols = &uploaded_trace.d_bytecode_cols;
    let tables = &uploaded_trace.tables;
    let logup_static = &mut uploaded_trace.logup_static;
    let memory_len = logup_static.memory_len;
    let bytecode_n_rows = logup_static.bytecode_n_rows;
    let total_active_len = logup_static.total_active_len;
    let pivot = logup_static.pivot;
    let max_table_height = logup_static.max_table_height;

    debug_assert_eq!(d_memory.len(), memory_len);
    debug_assert_eq!(d_bytecode_cols.len(), bytecode_n_rows * N_INSTRUCTION_COLUMNS);
    debug_assert_eq!(logup_static.alpha_last_idx, (d_alphas_eq_poly.len() / 5 - 1) as u32);
    debug_assert_eq!(LOGUP_MEMORY_DOMAINSEP, 0);

    let mut d_numerators = take_logup_workspace(&mut logup_static.d_numerators, "logup numerator vector");
    g.stream
        .memset_zeros(&mut d_numerators)
        .expect("zero logup numerator workspace");
    let mut d_denominators = take_logup_workspace(&mut logup_static.d_denominators, "logup denominator vector");
    g.trace_ops
        .fill_ext_at_offset(&mut d_denominators, EF::ONE, total_active_len as u32, 0);
    let mut memory_den_prepared = take_logup_prepared(&mut logup_static.memory_den_prepared, "memory denominator");
    let mut bytecode_den_prepared =
        take_logup_prepared(&mut logup_static.bytecode_den_prepared, "bytecode denominator");
    let mut execution_den_prepared =
        take_logup_prepared(&mut logup_static.execution_den_prepared, "execution denominator");
    let mut d_memory_num_br = take_logup_workspace(&mut logup_static.d_memory_num_br, "memory numerator bit-reverse");
    g.sumcheck.bit_reverse_within_chunks_device_into_async(
        d_memory_acc,
        &mut d_memory_num_br,
        memory_len as u32,
        pivot as u32,
    );
    let mut d_memory_num = take_logup_workspace(&mut logup_static.d_memory_num, "memory numerator");
    g.trace_ops
        .negate_base_into(&d_memory_num_br, &mut d_memory_num, memory_len as u32);
    g.trace_ops
        .copy_to_offset_device(&d_memory_num, &mut d_numerators, memory_len as u32, 0);
    let mut d_memory_den = take_logup_workspace(&mut logup_static.d_memory_den, "memory denominator");
    logup_fingerprint_device_from_alpha_eq_into(
        g,
        d_memory,
        d_c,
        d_alphas_eq_poly,
        &logup_static.memory_den_inputs,
        &mut memory_den_prepared,
        &mut d_memory_den,
        memory_len as u32,
        1,
    );
    let mut d_memory_den_br = take_logup_workspace(&mut logup_static.d_memory_den_br, "memory denominator bit-reverse");
    g.trace_ops.bit_reverse_ext_within_chunks_into(
        &d_memory_den,
        &mut d_memory_den_br,
        memory_len as u32,
        pivot as u32,
    );
    g.trace_ops
        .copy_to_offset_device(&d_memory_den_br, &mut d_denominators, (memory_len * 5) as u32, 0);

    let mut d_bytecode_num_br =
        take_logup_workspace(&mut logup_static.d_bytecode_num_br, "bytecode numerator bit-reverse");
    g.sumcheck.bit_reverse_within_chunks_device_into_async(
        d_bytecode_acc,
        &mut d_bytecode_num_br,
        bytecode_n_rows as u32,
        pivot as u32,
    );
    let mut d_bytecode_num = take_logup_workspace(&mut logup_static.d_bytecode_num, "bytecode numerator");
    g.trace_ops
        .negate_base_into(&d_bytecode_num_br, &mut d_bytecode_num, bytecode_n_rows as u32);
    g.trace_ops.copy_to_offset_device(
        &d_bytecode_num,
        &mut d_numerators,
        bytecode_n_rows as u32,
        logup_static.bytecode_offset as u32,
    );
    let mut d_bytecode_den = take_logup_workspace(&mut logup_static.d_bytecode_den, "bytecode denominator");
    logup_fingerprint_device_from_alpha_eq_into(
        g,
        d_bytecode_cols,
        d_c,
        d_alphas_eq_poly,
        &logup_static.bytecode_den_inputs,
        &mut bytecode_den_prepared,
        &mut d_bytecode_den,
        bytecode_n_rows as u32,
        N_INSTRUCTION_COLUMNS as u32,
    );
    let mut d_bytecode_den_br =
        take_logup_workspace(&mut logup_static.d_bytecode_den_br, "bytecode denominator bit-reverse");
    g.trace_ops.bit_reverse_ext_within_chunks_into(
        &d_bytecode_den,
        &mut d_bytecode_den_br,
        bytecode_n_rows as u32,
        pivot as u32,
    );
    g.trace_ops.copy_to_offset_device(
        &d_bytecode_den_br,
        &mut d_denominators,
        (bytecode_n_rows * 5) as u32,
        (logup_static.bytecode_offset * 5) as u32,
    );
    let first_table_offset = logup_static.bytecode_offset + max_table_height.max(bytecode_n_rows);
    let table_plans = &mut logup_static.table_plans;
    if let Some(execution_offset) = table_plans[0].execution_offset {
        debug_assert_eq!(first_table_offset, execution_offset);
    } else {
        debug_assert_eq!(first_table_offset, table_plans[0].bus_offset);
    }
    let bus_den_inputs = &logup_static.bus_den_inputs;
    let lookup_den_inputs = &logup_static.lookup_den_inputs;
    let bus_den_prepared = &mut logup_static.bus_den_prepared;
    let lookup_den_prepared = &mut logup_static.lookup_den_prepared;
    let d_base_ones = &logup_static.d_base_ones;

    for plan in table_plans {
        let table = plan.table;
        let n_rows = plan.n_rows();
        let d_all_cols = &tables[&table].d_all_cols;

        if let Some(execution_offset) = plan.execution_offset {
            g.trace_ops.copy_to_offset_device(
                &d_base_ones.slice(..n_rows),
                &mut d_numerators,
                n_rows as u32,
                execution_offset as u32,
            );

            let mut d_exec_cols = take_logup_workspace(&mut plan.d_exec_cols, "execution logup columns");
            for i in 0..N_INSTRUCTION_COLUMNS {
                let src = d_all_cols.slice((N_RUNTIME_COLUMNS + i) * n_rows..(N_RUNTIME_COLUMNS + i + 1) * n_rows);
                g.trace_ops
                    .copy_to_offset_device(&src, &mut d_exec_cols, n_rows as u32, (i * n_rows) as u32);
            }
            let d_pc = d_all_cols.slice(COL_PC * n_rows..(COL_PC + 1) * n_rows);
            g.trace_ops.copy_to_offset_device(
                &d_pc,
                &mut d_exec_cols,
                n_rows as u32,
                (N_INSTRUCTION_COLUMNS * n_rows) as u32,
            );
            let mut d_exec_den = take_logup_workspace(&mut plan.d_exec_den, "execution logup denominator");
            logup_fingerprint_device_from_alpha_eq_into(
                g,
                &d_exec_cols,
                d_c,
                d_alphas_eq_poly,
                &logup_static.execution_den_inputs,
                &mut execution_den_prepared,
                &mut d_exec_den,
                n_rows as u32,
                (N_INSTRUCTION_COLUMNS + 1) as u32,
            );
            let mut d_exec_den_br =
                take_logup_workspace(&mut plan.d_exec_den_br, "execution logup bit-reversed denominator");
            g.trace_ops.bit_reverse_ext_within_chunks_into(
                &d_exec_den,
                &mut d_exec_den_br,
                n_rows as u32,
                pivot as u32,
            );
            g.trace_ops.copy_to_offset_device(
                &d_exec_den_br,
                &mut d_denominators,
                (n_rows * 5) as u32,
                (execution_offset * 5) as u32,
            );
        }

        let selector = d_all_cols.slice(plan.bus_selector * n_rows..(plan.bus_selector + 1) * n_rows);
        let mut d_bus_num_br = take_logup_workspace(&mut plan.d_bus_num_br, "bus numerator bit-reverse");
        g.sumcheck.bit_reverse_within_chunks_device_into_async(
            &selector,
            &mut d_bus_num_br,
            n_rows as u32,
            pivot as u32,
        );
        if plan.bus_direction.is_pull() {
            let mut d_bus_num = take_logup_workspace(&mut plan.d_bus_num, "bus numerator");
            g.trace_ops
                .negate_base_into(&d_bus_num_br, &mut d_bus_num, n_rows as u32);
            g.trace_ops
                .copy_to_offset_device(&d_bus_num, &mut d_numerators, n_rows as u32, plan.bus_offset as u32);
        } else {
            g.trace_ops
                .copy_to_offset_device(&d_bus_num_br, &mut d_numerators, n_rows as u32, plan.bus_offset as u32);
        }

        let mut d_bus_cols = take_logup_workspace(&mut plan.d_bus_cols, "bus data columns");
        let mut bus_col_idx = 0usize;
        for entry in &plan.bus_data {
            if let GpuLogupBusDataPlan::Column(col) = entry {
                let src = d_all_cols.slice(*col * n_rows..(*col + 1) * n_rows);
                g.trace_ops
                    .copy_to_offset_device(&src, &mut d_bus_cols, n_rows as u32, (bus_col_idx * n_rows) as u32);
                bus_col_idx += 1;
            }
        }
        debug_assert_eq!(bus_col_idx, plan.bus_data_column_count);
        let mut d_bus_den = take_logup_workspace(&mut plan.d_bus_den, "bus denominator");
        let mut d_bus_den_prepared = take_logup_prepared(
            bus_den_prepared
                .get_mut(&table)
                .expect("planned logup bus denominator workspace"),
            "bus denominator",
        );
        logup_fingerprint_device_from_alpha_eq_into(
            g,
            &d_bus_cols,
            d_c,
            d_alphas_eq_poly,
            &bus_den_inputs[&table],
            &mut d_bus_den_prepared,
            &mut d_bus_den,
            n_rows as u32,
            plan.bus_data_column_count as u32,
        );
        let mut d_bus_den_br = take_logup_workspace(&mut plan.d_bus_den_br, "bus bit-reversed denominator");
        g.trace_ops
            .bit_reverse_ext_within_chunks_into(&d_bus_den, &mut d_bus_den_br, n_rows as u32, pivot as u32);
        g.trace_ops.copy_to_offset_device(
            &d_bus_den_br,
            &mut d_denominators,
            (n_rows * 5) as u32,
            (plan.bus_offset * 5) as u32,
        );

        let lookup_inputs = &lookup_den_inputs[&table];
        let lookup_prepared = lookup_den_prepared
            .get_mut(&table)
            .expect("planned logup lookup denominator workspaces");
        let mut lookup_input_idx = 0usize;
        for lookup in &plan.lookups {
            debug_assert_eq!(lookup.values.len(), lookup.value_offsets.len());
            for (value_col, value_offset) in lookup.values.iter().zip(&lookup.value_offsets) {
                g.trace_ops.copy_to_offset_device(
                    &d_base_ones.slice(..n_rows),
                    &mut d_numerators,
                    n_rows as u32,
                    *value_offset as u32,
                );

                let mut d_lookup_cols =
                    take_logup_workspace(&mut plan.d_lookup_cols[lookup_input_idx], "lookup columns");
                let d_value = d_all_cols.slice(*value_col * n_rows..(*value_col + 1) * n_rows);
                let d_index = d_all_cols.slice(lookup.index * n_rows..(lookup.index + 1) * n_rows);
                g.trace_ops
                    .copy_to_offset_device(&d_value, &mut d_lookup_cols, n_rows as u32, 0);
                g.trace_ops
                    .copy_to_offset_device(&d_index, &mut d_lookup_cols, n_rows as u32, n_rows as u32);

                let mut d_lookup_den =
                    take_logup_workspace(&mut plan.d_lookup_den[lookup_input_idx], "lookup denominator");
                let mut d_lookup_prepared =
                    take_logup_prepared(&mut lookup_prepared[lookup_input_idx], "lookup denominator");
                logup_fingerprint_device_from_alpha_eq_into(
                    g,
                    &d_lookup_cols,
                    d_c,
                    d_alphas_eq_poly,
                    &lookup_inputs[lookup_input_idx],
                    &mut d_lookup_prepared,
                    &mut d_lookup_den,
                    n_rows as u32,
                    2,
                );
                let mut d_lookup_den_br = take_logup_workspace(
                    &mut plan.d_lookup_den_br[lookup_input_idx],
                    "lookup bit-reversed denominator",
                );
                g.trace_ops.bit_reverse_ext_within_chunks_into(
                    &d_lookup_den,
                    &mut d_lookup_den_br,
                    n_rows as u32,
                    pivot as u32,
                );
                g.trace_ops.copy_to_offset_device(
                    &d_lookup_den_br,
                    &mut d_denominators,
                    (n_rows * 5) as u32,
                    (*value_offset * 5) as u32,
                );
                lookup_input_idx += 1;
            }
        }
        debug_assert_eq!(lookup_input_idx, lookup_inputs.len());
    }

    (total_active_len, pivot, d_numerators, d_denominators)
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_prove_logup(
    g: &Gpu,
    fs: &mut GpuFsPhase,
    mut d_c: CudaSlice<u32>,
    mut d_alpha_challenges: CudaSlice<u32>,
    mut d_alphas_eq_poly: CudaSlice<u32>,
    uploaded_trace: &mut GpuUploadedExecutionTrace,
    d_memory_acc: &CudaSlice<u32>,
    d_bytecode_acc: &CudaSlice<u32>,
    whir_statement_targets: GpuLogupWhirStatementTargets<'_>,
) -> GpuLogupOutput {
    let memory_len = uploaded_trace.d_memory.len();
    assert!(memory_len.is_power_of_two());

    let total_active_len;
    let total_gkr_n_vars;
    {
        let logup_static = &uploaded_trace.logup_static;
        assert_eq!(memory_len, d_memory_acc.len());
        assert_eq!(memory_len, logup_static.memory_len);
        assert_eq!(logup_static.bytecode_n_rows, d_bytecode_acc.len());
        assert_eq!(
            uploaded_trace.d_bytecode_cols.len(),
            logup_static.bytecode_n_rows * N_INSTRUCTION_COLUMNS
        );
        total_active_len = logup_static.total_active_len;
        total_gkr_n_vars = logup_static.total_gkr_n_vars;
    }

    assert_eq!(d_c.len(), 5, "logup c workspace must hold one extension scalar");
    assert_eq!(
        d_alpha_challenges.len() % 5,
        0,
        "logup alpha challenge workspace must hold extension scalars"
    );
    let logup_alpha_len = d_alpha_challenges.len() / 5;
    assert_eq!(
        d_alphas_eq_poly.len(),
        (1usize << logup_alpha_len) * 5,
        "logup alpha eq-polynomial workspace has wrong length"
    );
    {
        let stream = g.stream.clone();
        stream
            .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin logup challenge graph capture");
        fs.sample_ext_vec_device_into(g, 1, &mut d_c);
        fs.sample_ext_vec_device_into(g, logup_alpha_len, &mut d_alpha_challenges);
        g.sumcheck.eq_polynomial_device_from_flat_point_words_into_async(
            &d_alpha_challenges,
            logup_alpha_len as u32,
            &mut d_alphas_eq_poly,
        );
        let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph = stream
            .end_capture(graph_flags)
            .expect("end logup challenge graph capture")
            .expect("logup challenge capture produced no graph");
        graph.launch().expect("launch logup challenge graph");
    }

    let (device_active_len, pivot, d_numerators, d_denominators) =
        build_logup_witness_on_device(g, &d_c, &d_alphas_eq_poly, uploaded_trace, d_memory_acc, d_bytecode_acc);
    debug_assert_eq!(device_active_len, total_active_len);

    let d_gkr_initial_point = take_logup_workspace(
        &mut uploaded_trace.logup_static.d_gkr_initial_point,
        "GKR initial challenge point",
    );
    let d_gkr_nums_padded = take_logup_workspace(
        &mut uploaded_trace.logup_static.d_gkr_nums_padded,
        "GKR padded numerator workspace",
    );
    let d_gkr_dens_padded = take_logup_workspace(
        &mut uploaded_trace.logup_static.d_gkr_dens_padded,
        "GKR padded denominator workspace",
    );
    let d_gkr_top_transcript = take_logup_workspace(
        &mut uploaded_trace.logup_static.d_gkr_top_transcript,
        "GKR top transcript workspace",
    );
    let d_gkr_claim_num = take_logup_workspace(
        &mut uploaded_trace.logup_static.d_gkr_claim_num,
        "GKR top claim numerator workspace",
    );
    let d_gkr_claim_den = take_logup_workspace(
        &mut uploaded_trace.logup_static.d_gkr_claim_den,
        "GKR top claim denominator workspace",
    );
    let d_gkr_mle_scratch_a = take_logup_workspace(
        &mut uploaded_trace.logup_static.d_gkr_mle_scratch_a,
        "GKR top MLE scratch A",
    );
    let d_gkr_mle_scratch_b = take_logup_workspace(
        &mut uploaded_trace.logup_static.d_gkr_mle_scratch_b,
        "GKR top MLE scratch B",
    );
    let gkr_output = gpu_prove_gkr_quotient_from_device_with_fs(
        g,
        fs,
        &d_numerators,
        &d_denominators,
        total_active_len,
        pivot,
        d_gkr_initial_point,
        d_gkr_nums_padded,
        d_gkr_dens_padded,
        d_gkr_top_transcript,
        d_gkr_claim_num,
        d_gkr_claim_den,
        d_gkr_mle_scratch_a,
        d_gkr_mle_scratch_b,
    );
    let d_claim_point_gkr_words = gkr_output.d_point_words;
    debug_assert_eq!(d_claim_point_gkr_words.len(), total_gkr_n_vars * 5);

    let tables = &uploaded_trace.tables;
    let d_memory = &uploaded_trace.d_memory;
    let logup_static = &mut uploaded_trace.logup_static;
    let log_memory = logup_static.log_memory;
    let log_bytecode = logup_static.log_bytecode;
    let bytecode_n_rows = logup_static.bytecode_n_rows;
    let alpha_last_idx = logup_static.alpha_last_idx;
    let mut d_mle_scratch_a = take_logup_workspace(&mut logup_static.d_mle_scratch_a, "logup MLE scratch A");
    let mut d_mle_scratch_b = take_logup_workspace(&mut logup_static.d_mle_scratch_b, "logup MLE scratch B");
    let mut d_mle_eval_temp = take_logup_workspace(&mut logup_static.d_mle_eval_temp, "logup MLE eval temp");

    // The post-GKR logup evaluations have fixed control flow from uploaded
    // metadata. Capture them as one graph while keeping the exact FS absorb
    // boundaries used by the CPU protocol.
    let stream = g.stream.clone();
    stream
        .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin logup eval graph capture");

    let GpuLogupWhirStatementTargets {
        d_memory_and_acc_point,
        d_value_memory,
        d_value_memory_acc,
        d_bytecode_and_acc_point,
        d_value_bytecode_acc,
        table_targets,
    } = whir_statement_targets;
    let mut table_targets = table_targets.into_iter();

    suffix_point_words_into(g, &d_claim_point_gkr_words, log_memory, d_memory_and_acc_point);
    gpu_mle_eval_base_device_point_observed_device_into(
        g,
        fs,
        d_memory_acc,
        memory_len,
        d_memory_and_acc_point,
        log_memory,
        &mut d_mle_scratch_a,
        &mut d_mle_scratch_b,
        take_logup_workspace(
            &mut logup_static.d_value_memory_acc_for_transcript,
            "memory accumulator eval for transcript",
        ),
        d_value_memory_acc,
    );
    gpu_mle_eval_base_device_point_observed_device_into(
        g,
        fs,
        d_memory,
        memory_len,
        d_memory_and_acc_point,
        log_memory,
        &mut d_mle_scratch_a,
        &mut d_mle_scratch_b,
        take_logup_workspace(
            &mut logup_static.d_value_memory_for_transcript,
            "memory eval for transcript",
        ),
        d_value_memory,
    );

    suffix_point_words_into(g, &d_claim_point_gkr_words, log_bytecode, d_bytecode_and_acc_point);
    gpu_mle_eval_base_device_point_observed_device_into(
        g,
        fs,
        d_bytecode_acc,
        bytecode_n_rows,
        d_bytecode_and_acc_point,
        log_bytecode,
        &mut d_mle_scratch_a,
        &mut d_mle_scratch_b,
        take_logup_workspace(
            &mut logup_static.d_value_bytecode_acc_for_transcript,
            "bytecode accumulator eval for transcript",
        ),
        d_value_bytecode_acc,
    );

    let table_plans = &mut logup_static.table_plans;
    let bus_data_scalar_prepared = &mut logup_static.bus_data_scalar_prepared;
    let bus_data_scalar_inputs = &logup_static.bus_data_scalar_inputs;
    let mut bus_evals = Vec::<GpuLogupBusEvals>::with_capacity(table_plans.len());
    let mut dot_accumulation_guards = Vec::<LogupExtDotAccumulationGuard>::new();
    debug_assert_eq!(alpha_last_idx, (d_alphas_eq_poly.len() / 5 - 1) as u32);
    for plan in table_plans {
        let table = plan.table;
        let log_n_rows = plan.log_n_rows;
        let n_rows = plan.n_rows();
        let GpuLogupWhirTableStatementTargets {
            table: target_table,
            d_point: d_inner_point,
            values: mut table_value_targets,
        } = table_targets
            .next()
            .unwrap_or_else(|| panic!("missing logup WHIR statement targets for {}", table.name()));
        debug_assert_eq!(target_table, table);
        suffix_point_words_into(g, &d_claim_point_gkr_words, log_n_rows, d_inner_point);
        let mut d_table_values_written = vec![false; plan.logup_value_cols.len()];

        if plan.is_execution {
            let pc_value_idx = logup_value_idx(&plan.logup_value_cols, COL_PC);
            let mut d_eval_on_pc = take_logup_workspace(
                &mut plan.d_logup_value_eval_transcript[pc_value_idx],
                "execution pc transcript value",
            );
            gpu_eval_uploaded_column_device_at_point_words_into(
                g,
                tables,
                table,
                COL_PC,
                n_rows,
                d_inner_point,
                log_n_rows,
                &mut d_mle_scratch_a,
                &mut d_mle_scratch_b,
                &mut d_eval_on_pc,
            );
            store_logup_column_eval_copy_once(
                g,
                &mut d_table_values_written,
                &plan.logup_value_cols,
                &mut table_value_targets,
                COL_PC,
                &d_eval_on_pc,
            );
            observe_ext_scalar_device(g, fs, d_eval_on_pc);
            let mut d_instr_transcript = take_logup_workspace(
                &mut plan.d_exec_instr_transcript,
                "execution instruction transcript workspace",
            );
            for i in 0..N_INSTRUCTION_COLUMNS {
                gpu_eval_uploaded_column_device_at_point_words_into(
                    g,
                    tables,
                    table,
                    N_RUNTIME_COLUMNS + i,
                    n_rows,
                    d_inner_point,
                    log_n_rows,
                    &mut d_mle_scratch_a,
                    &mut d_mle_scratch_b,
                    &mut d_mle_eval_temp,
                );
                store_logup_column_eval_copy_once(
                    g,
                    &mut d_table_values_written,
                    &plan.logup_value_cols,
                    &mut table_value_targets,
                    N_RUNTIME_COLUMNS + i,
                    &d_mle_eval_temp,
                );
                g.sumcheck
                    .memcpy_d2d_async(&d_mle_eval_temp, 0, &mut d_instr_transcript, i * 5, 5);
            }
            fs.observe_ext_scalars_device(g, d_instr_transcript, N_INSTRUCTION_COLUMNS);
        }

        let mut d_eval_on_selector =
            take_logup_workspace(&mut plan.d_bus_numerator_for_transcript, "bus numerator for transcript");
        if plan.bus_direction.is_pull() {
            gpu_eval_uploaded_column_device_at_point_words_into(
                g,
                tables,
                table,
                plan.bus_selector,
                n_rows,
                d_inner_point,
                log_n_rows,
                &mut d_mle_scratch_a,
                &mut d_mle_scratch_b,
                &mut d_mle_eval_temp,
            );
            d_eval_on_selector =
                apply_bus_direction_device_into(g, &d_mle_eval_temp, plan.bus_direction, d_eval_on_selector);
        } else {
            gpu_eval_uploaded_column_device_at_point_words_into(
                g,
                tables,
                table,
                plan.bus_selector,
                n_rows,
                d_inner_point,
                log_n_rows,
                &mut d_mle_scratch_a,
                &mut d_mle_scratch_b,
                &mut d_eval_on_selector,
            );
        }
        let d_eval_on_selector_for_air = copy_ext_scalar_into(
            g,
            &d_eval_on_selector,
            take_logup_workspace(&mut plan.d_bus_numerator_for_air, "bus numerator for AIR"),
        );
        observe_ext_scalar_device(g, fs, d_eval_on_selector);
        let mut d_bus_data_values = take_logup_workspace(&mut plan.d_bus_data_values, "bus-data value workspace");
        for (term_idx, entry) in plan.bus_data.iter_mut().enumerate() {
            match entry {
                GpuLogupBusDataPlan::Column(col) => {
                    gpu_eval_uploaded_column_device_at_point_words_into(
                        g,
                        tables,
                        table,
                        *col,
                        n_rows,
                        d_inner_point,
                        log_n_rows,
                        &mut d_mle_scratch_a,
                        &mut d_mle_scratch_b,
                        &mut d_mle_eval_temp,
                    );
                    g.sumcheck
                        .memcpy_d2d_async(&d_mle_eval_temp, 0, &mut d_bus_data_values, term_idx * 5, 5);
                }
                GpuLogupBusDataPlan::Constant { d_value } => {
                    let d_value = take_logup_workspace(d_value, "uploaded logup bus constant");
                    g.sumcheck
                        .memcpy_d2d_async(&d_value, 0, &mut d_bus_data_values, term_idx * 5, 5);
                }
            }
        }
        let mut prepared_bus_data_scalars = take_logup_prepared(
            bus_data_scalar_prepared
                .get_mut(&table)
                .expect("planned logup bus-data scalar workspace"),
            "bus-data scalar",
        );
        g.sumcheck
            .logup_prepare_constants_from_device_inputs_into_prepared_async(
                &d_alphas_eq_poly,
                &bus_data_scalar_inputs[&table],
                &mut prepared_bus_data_scalars,
            );
        let eval_on_data_guard = ext_dot_accumulate_from_prepared_scalars(
            g,
            add_ext_devices_into(
                g,
                &d_c,
                &prepared_bus_data_scalars.d_contrib,
                take_logup_workspace(&mut plan.d_bus_data_base_sum, "bus-data base sum"),
            ),
            d_bus_data_values,
            plan.bus_data.len() as u32,
            take_logup_workspace(&mut plan.d_bus_data_dot_out, "bus-data dot output"),
            prepared_bus_data_scalars,
        );
        let d_eval_on_data_for_air = copy_ext_scalar_into(
            g,
            &eval_on_data_guard.d_out,
            take_logup_workspace(&mut plan.d_bus_denominator_for_air, "bus denominator for AIR"),
        );
        let d_eval_on_data_for_transcript = copy_ext_scalar_into(
            g,
            &eval_on_data_guard.d_out,
            take_logup_workspace(
                &mut plan.d_bus_denominator_for_transcript,
                "bus denominator for transcript",
            ),
        );
        observe_ext_scalar_device(g, fs, d_eval_on_data_for_transcript);
        dot_accumulation_guards.push(eval_on_data_guard);
        bus_evals.push(GpuLogupBusEvals {
            table,
            d_numerator: d_eval_on_selector_for_air,
            d_denominator: d_eval_on_data_for_air,
        });

        for lookup in &plan.lookups {
            let index_value_idx = logup_value_idx(&plan.logup_value_cols, lookup.index);
            let mut d_index_eval = take_logup_workspace(
                &mut plan.d_logup_value_eval_transcript[index_value_idx],
                "lookup index transcript value",
            );
            gpu_eval_uploaded_column_device_at_point_words_into(
                g,
                tables,
                table,
                lookup.index,
                n_rows,
                d_inner_point,
                log_n_rows,
                &mut d_mle_scratch_a,
                &mut d_mle_scratch_b,
                &mut d_index_eval,
            );
            store_logup_column_eval_copy_once(
                g,
                &mut d_table_values_written,
                &plan.logup_value_cols,
                &mut table_value_targets,
                lookup.index,
                &d_index_eval,
            );
            observe_ext_scalar_device(g, fs, d_index_eval);
            for col_index in &lookup.values {
                let value_idx = logup_value_idx(&plan.logup_value_cols, *col_index);
                let mut d_value_eval = take_logup_workspace(
                    &mut plan.d_logup_value_eval_transcript[value_idx],
                    "lookup value transcript value",
                );
                gpu_eval_uploaded_column_device_at_point_words_into(
                    g,
                    tables,
                    table,
                    *col_index,
                    n_rows,
                    d_inner_point,
                    log_n_rows,
                    &mut d_mle_scratch_a,
                    &mut d_mle_scratch_b,
                    &mut d_value_eval,
                );
                store_logup_column_eval_copy_once(
                    g,
                    &mut d_table_values_written,
                    &plan.logup_value_cols,
                    &mut table_value_targets,
                    *col_index,
                    &d_value_eval,
                );
                observe_ext_scalar_device(g, fs, d_value_eval);
            }
        }
        assert!(
            d_table_values_written.iter().all(|written| *written),
            "not all planned logup WHIR statement values were filled for {}",
            table.name()
        );
    }
    debug_assert!(table_targets.next().is_none());

    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = stream
        .end_capture(graph_flags)
        .expect("end logup eval graph capture")
        .expect("logup eval capture produced no graph");
    graph.launch().expect("launch logup eval graph");

    GpuLogupOutput {
        d_c,
        d_alphas_eq_poly,
        d_gkr_point_words: d_claim_point_gkr_words,
        bus_evals,
        _d_alpha_challenges: d_alpha_challenges,
        _dot_accumulation_guards: dot_accumulation_guards,
    }
}
