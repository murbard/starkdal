//! COMPLETE GPU prove_execution — full protocol reimplementation.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

use backend::{
    GpuDeviceSlice, GpuTranscriptChunk, GpuTranscriptSeed, GpuWhirProverWorkspaces, poseidon1_round_constants,
    poseidon1_sparse_first_round_constants, poseidon1_sparse_first_row, poseidon1_sparse_m_i,
    poseidon1_sparse_scalar_round_constants, poseidon1_sparse_v,
};
use cudarc::driver::{
    safe::{CudaContext, CudaSlice, CudaStream},
    sys,
};

use crate::prove_execution::ExecutionProof;
use crate::*;
use lean_vm::*;
use sub_protocols::*;
use tracing::info_span;
use utils::build_prover_state;

pub(crate) struct Gpu {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) sumcheck: gpu_sumcheck::GpuSumcheck,
    pub(crate) fold: gpu_poly_fold::GpuPolyFold,
    pub(crate) ntt: gpu_ntt::GpuNtt,
    pub(crate) merkle: gpu_merkle::GpuMerkle,
    pub(crate) pow: gpu_pow_grind::GpuPowGrinder,
    pub(crate) trace_ops: gpu_trace_ops::GpuTraceOps,
    pub(crate) p16: GpuPoseidon16Constants,
    pub(crate) d_ext_one: CudaSlice<u32>,
    pub(crate) d_ext_zero_one: CudaSlice<u32>,
}

pub(crate) struct GpuPoseidon16Constants {
    pub(crate) d_rc: CudaSlice<u32>,
    pub(crate) d_mds: CudaSlice<u32>,
    pub(crate) d_sparse: CudaSlice<u32>,
}

impl GpuPoseidon16Constants {
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Self {
        let rc = poseidon1_round_constants();
        let rf: Vec<u32> = rc.iter().flat_map(|r| r.iter().map(|v| kb_u32(*v))).collect();
        let d_rc = stream.memcpy_stod(&rf).unwrap();

        let mds: [u32; 16] = [1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1].map(|v| kb_u32(F::from_u32(v)));
        let d_mds = stream.memcpy_stod(&mds).unwrap();

        let mi = poseidon1_sparse_m_i();
        let fr = poseidon1_sparse_first_row();
        let v = poseidon1_sparse_v();
        let sr = poseidon1_sparse_scalar_round_constants();
        let mut sparse: Vec<u32> = Vec::with_capacity(912);
        for row in mi.iter() {
            for val in row.iter() {
                sparse.push(kb_u32(*val));
            }
        }
        for row in fr.iter() {
            for val in row.iter() {
                sparse.push(kb_u32(*val));
            }
        }
        for row in v.iter() {
            for i in 0..15 {
                sparse.push(kb_u32(row[i]));
            }
        }
        for val in sr.iter() {
            sparse.push(kb_u32(*val));
        }
        while sparse.len() < 896 {
            sparse.push(0);
        }
        for val in poseidon1_sparse_first_round_constants().iter() {
            sparse.push(kb_u32(*val));
        }
        let d_sparse = stream.memcpy_stod(&sparse).unwrap();
        Self { d_rc, d_mds, d_sparse }
    }

    pub(crate) fn as_slices(&self) -> (&CudaSlice<u32>, &CudaSlice<u32>, &CudaSlice<u32>) {
        (&self.d_rc, &self.d_mds, &self.d_sparse)
    }
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

pub(crate) fn gpu() -> Option<&'static Gpu> {
    GPU.get_or_init(|| {
        let ctx = CudaContext::new(0).ok()?;
        // Disable event tracking so CudaSlice::device_ptr() doesn't insert
        // stream.wait(event) calls, which break CUDA graph capture.
        unsafe {
            ctx.disable_event_tracking();
        }
        let s = ctx.new_stream().expect("failed to create CUDA stream");
        tracing::info!("GPU prover initialized (full pipeline)");
        let p16 = GpuPoseidon16Constants::new(&s);
        let d_ext_one = s.memcpy_stod(&[kb_u32(F::ONE), 0, 0, 0, 0]).unwrap();
        let d_ext_zero_one = s.memcpy_stod(&[0, 0, 0, 0, 0, kb_u32(F::ONE), 0, 0, 0, 0]).unwrap();
        Some(Gpu {
            sumcheck: gpu_sumcheck::GpuSumcheck::new(s.clone()),
            fold: gpu_poly_fold::GpuPolyFold::new(s.clone()),
            ntt: gpu_ntt::GpuNtt::new(s.clone()),
            merkle: gpu_merkle::GpuMerkle::new(s.clone()),
            pow: gpu_pow_grind::GpuPowGrinder::new(s.clone()),
            trace_ops: gpu_trace_ops::GpuTraceOps::new(s.clone()),
            p16,
            d_ext_one,
            d_ext_zero_one,
            stream: s,
        })
    })
    .as_ref()
}

const ENDIANNESS_PIVOT_AIR_UPLOAD: usize = 12;
const WHIR_FINAL_MATERIALIZATION_MIN_WORDS: usize = 1 << 20;

pub(crate) struct GpuUploadedTableTrace {
    pub(crate) d_all_cols: CudaSlice<u32>,
    pub(crate) d_air_base_cols: CudaSlice<u32>,
    pub(crate) d_air_base_down_cols: CudaSlice<u32>,
    _d_raw_air_down_cols: Option<CudaSlice<u32>>,
}

pub(crate) struct GpuUploadedExecutionTrace {
    pub(crate) d_memory: CudaSlice<u32>,
    pub(crate) d_bytecode_cols: CudaSlice<u32>,
    pub(crate) tables: BTreeMap<Table, GpuUploadedTableTrace>,
    pub(crate) d_air_bus_directions: CudaSlice<u32>,
    pub(crate) logup_static: crate::gpu_logup::GpuLogupStaticMetadata,
    pub(crate) air_static: crate::gpu_air::GpuAirStaticMetadata,
}

struct GpuUploadedProverPlan {
    uploaded_fs: GpuUploadedFsState,
    uploaded_trace: GpuUploadedExecutionTrace,
    d_memory_acc: CudaSlice<u32>,
    d_bytecode_acc: CudaSlice<u32>,
    memory_len: usize,
    bytecode_acc_len: usize,
    stack_commit_plan: GpuStackCommitPlan,
    whir_statement_plan: GpuWhirStatementPlan,
    active_workspaces: GpuActiveProverWorkspaces,
}

struct GpuStackCommitPlan {
    stacked_n_vars: usize,
    n_base_evals: usize,
    actual_data_len: usize,
    ff0: usize,
    n_blocks: usize,
    n_evals: u32,
    n_cols: u32,
    merkle_height: u32,
    starting_log_inv_rate: usize,
    commitment_ood_samples: usize,
    whir_cfg: WhirConfig<EF>,
    table_column_plans: Vec<GpuStackedTableColumnPlan>,
}

struct GpuStackedTableColumnPlan {
    table: Table,
    n_rows: usize,
    n_columns: usize,
}

struct GpuWhirStatementPlan {
    public_memory_n_vars: usize,
    d_public_memory_mle_scratch_a: Option<CudaSlice<u32>>,
    d_public_memory_mle_scratch_b: Option<CudaSlice<u32>>,
    device_statement_prefix: Vec<GpuSparseStatement>,
    global_slots: GpuWhirGlobalStatementSlots,
    table_slots: Vec<GpuWhirTableStatementSlots>,
}

struct GpuWhirGlobalStatementSlots {
    memory_stmt_idx: usize,
    public_memory_stmt_idx: usize,
    bytecode_stmt_idx: usize,
}

struct GpuWhirTableStatementSlots {
    table: Table,
    logup_stmt_idx: usize,
    logup_value_slots: Vec<GpuWhirColumnValueSlot>,
    air_next_stmt_idx: Option<usize>,
    air_next_value_slots: Vec<GpuWhirAirValueSlot>,
    air_eq_stmt_idx: usize,
    air_eq_value_slots: Vec<GpuWhirAirValueSlot>,
}

struct GpuActiveProverWorkspaces {
    d_stacked_poly: Option<CudaSlice<u32>>,
    initial_ntt_twiddles: Option<gpu_ntt::GpuNttTwiddles>,
    d_initial_dft_output: Option<CudaSlice<u32>>,
    initial_merkle_layers: Option<Vec<CudaSlice<u32>>>,
    d_initial_ood_univariate_points: Option<CudaSlice<u32>>,
    d_initial_ood_challenges: Option<CudaSlice<u32>>,
    d_initial_ood_answer_states: Vec<Option<CudaSlice<u32>>>,
    d_logup_c: Option<CudaSlice<u32>>,
    d_logup_alpha_challenges: Option<CudaSlice<u32>>,
    d_logup_alphas_eq_poly: Option<CudaSlice<u32>>,
    d_bus_beta: Option<CudaSlice<u32>>,
    d_air_alpha: Option<CudaSlice<u32>>,
    d_air_eta: Option<CudaSlice<u32>>,
    whir_workspaces: Option<GpuWhirProverWorkspaces>,
}

struct GpuWhirColumnValueSlot {
    col_index: usize,
    value_idx: usize,
}

struct GpuWhirAirValueSlot {
    value_offset_words: usize,
    value_idx: usize,
}

struct GpuWhirColumnValuePlan {
    col_index: usize,
    selector: usize,
}

struct GpuWhirAirValuePlan {
    selector: usize,
    value_offset_words: usize,
}

fn kb_u32(v: lean_vm::F) -> u32 {
    unsafe { std::mem::transmute(v) }
}

fn kb_from_u32(v: u32) -> lean_vm::F {
    unsafe { std::mem::transmute(v) }
}

fn f_words(slice: &[lean_vm::F]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), slice.len()) }
}

fn take_gpu_workspace<T>(slot: &mut Option<T>, name: &str) -> T {
    slot.take()
        .unwrap_or_else(|| panic!("GPU workspace already consumed: {name}"))
}

pub(crate) struct GpuFsPhase {
    pub(crate) d_challenger_state: CudaSlice<u32>,
    pub(crate) d_empty_observe: CudaSlice<u32>,
    pub(crate) transcript_chunks: Vec<TranscriptChunk>,
}

pub(crate) struct GpuUploadedFsState {
    d_challenger_state: CudaSlice<u32>,
    d_empty_observe: CudaSlice<u32>,
}

pub(crate) struct TranscriptChunk {
    pub(crate) d_words: GpuDeviceSlice,
    pub(crate) n_words: usize,
}

impl GpuFsPhase {
    pub(crate) fn upload_initial(g: &Gpu, prover_state: &impl FSProver<EF>) -> GpuUploadedFsState {
        let state_words = prover_state.gpu_challenger_state().map(kb_u32);
        let d_challenger_state = g.stream.memcpy_stod(&state_words).unwrap();
        let d_empty_observe = g.stream.alloc_zeros::<u32>(1).unwrap();
        GpuUploadedFsState {
            d_challenger_state,
            d_empty_observe,
        }
    }

    pub(crate) fn from_uploaded(uploaded: GpuUploadedFsState) -> Self {
        Self {
            d_challenger_state: uploaded.d_challenger_state,
            d_empty_observe: uploaded.d_empty_observe,
            transcript_chunks: Vec::new(),
        }
    }

    pub(crate) fn new_for_standalone_host_boundary(g: &Gpu, prover_state: &impl FSProver<EF>) -> Self {
        Self::from_uploaded(Self::upload_initial(g, prover_state))
    }

    pub(crate) fn observe_base_scalars_device<D>(&mut self, g: &Gpu, d_scalars: D, n_words: usize)
    where
        D: Into<GpuDeviceSlice>,
    {
        if n_words == 0 {
            return;
        }
        let d_scalars = d_scalars.into();
        let (d_p16_rc, d_p16_mds, d_p16_sparse) = g.p16.as_slices();
        g.sumcheck.challenger_observe_device_scalars_async(
            &mut self.d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            &d_scalars,
            n_words as u32,
        );
        self.transcript_chunks.push(TranscriptChunk {
            d_words: d_scalars,
            n_words,
        });
    }

    pub(crate) fn observe_ext_scalars_device<D>(&mut self, g: &Gpu, d_scalars: D, n_ext_scalars: usize)
    where
        D: Into<GpuDeviceSlice>,
    {
        self.observe_base_scalars_device(g, d_scalars, n_ext_scalars * 5);
    }

    pub(crate) fn sample_ext_vec_device(&mut self, g: &Gpu, len: usize) -> CudaSlice<u32> {
        let mut d_samples = g.stream.alloc_zeros::<u32>(len * 5).unwrap();
        self.sample_ext_vec_device_into(g, len, &mut d_samples);
        d_samples
    }

    pub(crate) fn sample_ext_vec_device_into(&mut self, g: &Gpu, len: usize, d_samples: &mut CudaSlice<u32>) {
        assert_eq!(
            d_samples.len(),
            len * 5,
            "sample_ext_vec_device_into workspace has wrong length"
        );
        let (d_p16_rc, d_p16_mds, d_p16_sparse) = g.p16.as_slices();
        g.sumcheck.challenger_sample_exts_device_into_async(
            &mut self.d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            len as u32,
            &self.d_empty_observe,
            d_samples,
        );
    }

    pub(crate) fn sample_ext_device(&mut self, g: &Gpu) -> CudaSlice<u32> {
        self.sample_ext_vec_device(g, 1)
    }

    pub(crate) fn finish(self, g: &Gpu, prover_state: &mut impl FSProver<EF>) {
        let total_words: usize = self.transcript_chunks.iter().map(|chunk| chunk.n_words).sum();
        let mut d_final = g.stream.alloc_zeros::<u32>(total_words + 8).unwrap();
        let mut offset = 0usize;
        for chunk in &self.transcript_chunks {
            g.sumcheck
                .memcpy_d2d_async(&chunk.d_words, 0, &mut d_final, offset, chunk.n_words);
            offset += chunk.n_words;
        }
        g.sumcheck
            .memcpy_d2d_async(&self.d_challenger_state, 0, &mut d_final, total_words, 8);
        let final_words = g.stream.memcpy_dtov(&d_final).unwrap();
        let transcript_scalars = final_words[..total_words]
            .iter()
            .copied()
            .map(kb_from_u32)
            .collect::<Vec<_>>();
        let challenger_state_words = &final_words[total_words..total_words + 8];
        let final_challenger_state = [
            kb_from_u32(challenger_state_words[0]),
            kb_from_u32(challenger_state_words[1]),
            kb_from_u32(challenger_state_words[2]),
            kb_from_u32(challenger_state_words[3]),
            kb_from_u32(challenger_state_words[4]),
            kb_from_u32(challenger_state_words[5]),
            kb_from_u32(challenger_state_words[6]),
            kb_from_u32(challenger_state_words[7]),
        ];
        prover_state.inject_gpu_transcript_state(&transcript_scalars, final_challenger_state);
    }

    pub(crate) fn into_whir_transcript_seed(self) -> GpuTranscriptSeed {
        GpuTranscriptSeed {
            d_challenger_state: self.d_challenger_state,
            d_empty_observe: self.d_empty_observe,
            transcript_chunks: self
                .transcript_chunks
                .into_iter()
                .map(|chunk| GpuTranscriptChunk {
                    d_words: chunk.d_words,
                    n_words: chunk.n_words,
                })
                .collect(),
        }
    }
}

fn upload_trace_once(
    g: &Gpu,
    memory: &[F],
    bytecode_multilinear: &[F],
    traces: &BTreeMap<Table, TableTrace>,
) -> GpuUploadedExecutionTrace {
    let d_memory = g.stream.memcpy_stod(f_words(memory)).unwrap();
    let bytecode_stride = N_INSTRUCTION_COLUMNS.next_power_of_two();
    let bytecode_n_rows = bytecode_multilinear.len() / bytecode_stride;
    let mut bytecode_cols = Vec::with_capacity(bytecode_n_rows * N_INSTRUCTION_COLUMNS);
    for col in 0..N_INSTRUCTION_COLUMNS {
        for row in 0..bytecode_n_rows {
            bytecode_cols.push(kb_u32(bytecode_multilinear[row * bytecode_stride + col]));
        }
    }
    let d_bytecode_cols = g.stream.memcpy_stod(&bytecode_cols).unwrap();
    let mut tables = BTreeMap::new();

    for (table, trace) in traces {
        let n_rows = 1usize << trace.log_n_rows;
        let mut all_col_data = Vec::with_capacity(trace.columns.len() * n_rows);
        for col in &trace.columns {
            for &value in &col[..n_rows] {
                all_col_data.push(kb_u32(value));
            }
        }
        let d_all_cols = g.stream.memcpy_stod(&all_col_data).unwrap();

        let n_up_cols = table.n_columns();
        let down_idxs = table.down_column_indexes();
        let n_down_cols = down_idxs.len();
        let pv = ENDIANNESS_PIVOT_AIR_UPLOAD.min(trace.log_n_rows);

        let d_air_base_cols = g.sumcheck.bit_reverse_within_chunks_device_async(
            &d_all_cols.slice(0..n_up_cols * n_rows),
            (n_up_cols * n_rows) as u32,
            pv as u32,
        );

        let (d_air_base_down_cols, d_raw_air_down_cols) = if n_down_cols > 0 {
            let mut d_raw_down = g.stream.alloc_zeros::<u32>(n_down_cols * n_rows).unwrap();
            for (down_idx, &ci) in down_idxs.iter().enumerate() {
                let src = d_all_cols.slice(ci * n_rows..(ci + 1) * n_rows);
                g.trace_ops.shift_down_to_offset_device(
                    &src,
                    &mut d_raw_down,
                    n_rows as u32,
                    (down_idx * n_rows) as u32,
                );
            }
            let d_down = g.sumcheck.bit_reverse_within_chunks_device_async(
                &d_raw_down,
                (n_down_cols * n_rows) as u32,
                pv as u32,
            );
            (d_down, Some(d_raw_down))
        } else {
            (g.stream.alloc_zeros::<u32>(1).unwrap(), None)
        };

        tables.insert(
            *table,
            GpuUploadedTableTrace {
                d_all_cols,
                d_air_base_cols,
                d_air_base_down_cols,
                _d_raw_air_down_cols: d_raw_air_down_cols,
            },
        );
    }

    let tables_heights: BTreeMap<Table, usize> =
        traces.iter().map(|(table, trace)| (*table, trace.log_n_rows)).collect();
    let tables_sorted = sort_tables_by_height(&tables_heights);
    let air_bus_directions: Vec<u32> = tables_sorted
        .iter()
        .map(|(table, _)| match table.bus().direction {
            BusDirection::Pull => 0,
            BusDirection::Push => 1,
        })
        .collect();
    let d_air_bus_directions = g.stream.memcpy_stod(&air_bus_directions).unwrap();
    let logup_static = crate::gpu_logup::build_logup_static_metadata(g, traces, memory.len(), bytecode_n_rows);
    let air_static = crate::gpu_air::build_air_static_metadata(g, traces);

    GpuUploadedExecutionTrace {
        d_memory,
        d_bytecode_cols,
        tables,
        d_air_bus_directions,
        logup_static,
        air_static,
    }
}

fn build_access_counts_on_device(
    g: &Gpu,
    uploaded_trace: &GpuUploadedExecutionTrace,
    memory_len: usize,
    bytecode_len: usize,
    traces: &BTreeMap<Table, TableTrace>,
) -> (CudaSlice<u32>, CudaSlice<u32>) {
    let mut d_memory_acc_plain = g.stream.alloc_zeros::<u32>(memory_len).unwrap();
    for (table, trace) in traces {
        let n_rows = 1usize << trace.log_n_rows;
        let d_all_cols = &uploaded_trace.tables[table].d_all_cols;
        for lookup in table.lookups() {
            let d_index = d_all_cols.slice(lookup.index * n_rows..(lookup.index + 1) * n_rows);
            g.trace_ops.access_count_into_device(
                &d_index,
                &mut d_memory_acc_plain,
                n_rows as u32,
                lookup.values.len() as u32,
            );
        }
    }

    let execution_trace = &traces[&Table::execution()];
    let execution_n_rows = 1usize << execution_trace.log_n_rows;
    let mut d_bytecode_acc_plain = g.stream.alloc_zeros::<u32>(bytecode_len).unwrap();
    let d_exec_cols = &uploaded_trace.tables[&Table::execution()].d_all_cols;
    let d_pc = d_exec_cols.slice(COL_PC * execution_n_rows..(COL_PC + 1) * execution_n_rows);
    g.trace_ops
        .access_count_simple_into_device(&d_pc, &mut d_bytecode_acc_plain, execution_n_rows as u32);

    let d_memory_acc = g
        .trace_ops
        .canonical_to_monty_device(&d_memory_acc_plain, memory_len as u32);
    let d_bytecode_acc = g
        .trace_ops
        .canonical_to_monty_device(&d_bytecode_acc_plain, bytecode_len as u32);
    (d_memory_acc, d_bytecode_acc)
}

fn build_whir_statement_plan(
    g: &Gpu,
    stacked_n_vars: usize,
    memory_len: usize,
    public_memory_size: usize,
    bytecode_log_size: usize,
    traces: &BTreeMap<Table, TableTrace>,
) -> GpuWhirStatementPlan {
    let tables_heights: BTreeMap<Table, usize> =
        traces.iter().map(|(table, trace)| (*table, trace.log_n_rows)).collect();
    let tables_sorted = sort_tables_by_height(&tables_heights);
    debug_assert!(tables_sorted[0].0.is_execution_table());

    let memory_n_vars = log2_strict_usize(memory_len);
    let max_table_n_vars = tables_sorted[0].1;
    let mut table_offset = (2 << memory_n_vars) + (1 << bytecode_log_size.max(max_table_n_vars));
    let mut device_statement_prefix = Vec::new();

    let memory_stmt_idx = device_statement_prefix.len();
    device_statement_prefix.push(GpuSparseStatement::new(
        stacked_n_vars,
        memory_n_vars,
        g.stream
            .alloc_zeros::<u32>(memory_n_vars * 5)
            .expect("alloc memory WHIR statement point"),
        vec![
            GpuSparseValue::new(0, g.stream.alloc_zeros::<u32>(5).expect("alloc memory WHIR value")),
            GpuSparseValue::new(1, g.stream.alloc_zeros::<u32>(5).expect("alloc memory-acc WHIR value")),
        ],
    ));

    let public_memory_n_vars = log2_strict_usize(public_memory_size);
    let public_memory_stmt_idx = device_statement_prefix.len();
    device_statement_prefix.push(GpuSparseStatement::new(
        stacked_n_vars,
        public_memory_n_vars,
        g.stream
            .alloc_zeros::<u32>(public_memory_n_vars * 5)
            .expect("alloc public-memory WHIR statement point"),
        vec![GpuSparseValue::new(
            0,
            g.stream
                .alloc_zeros::<u32>(5)
                .expect("alloc public-memory WHIR statement value"),
        )],
    ));

    let bytecode_stmt_idx = device_statement_prefix.len();
    device_statement_prefix.push(GpuSparseStatement::new(
        stacked_n_vars,
        bytecode_log_size,
        g.stream
            .alloc_zeros::<u32>(bytecode_log_size * 5)
            .expect("alloc bytecode WHIR statement point"),
        vec![GpuSparseValue::new(
            (2 * memory_len) >> bytecode_log_size,
            g.stream
                .alloc_zeros::<u32>(5)
                .expect("alloc bytecode-acc WHIR statement value"),
        )],
    ));

    let mut table_slots = Vec::with_capacity(tables_sorted.len());

    for (table, log_n_rows) in tables_sorted {
        let selector_base = table_offset >> log_n_rows;
        let n_up_cols = table.n_columns();
        if table.is_execution_table() {
            let pc_start_selector = table_offset + (COL_PC << log_n_rows);
            let pc_end_selector = table_offset + ((COL_PC + 1) << log_n_rows) - 1;
            device_statement_prefix.push(GpuSparseStatement::new(
                stacked_n_vars,
                0,
                g.stream.alloc_zeros::<u32>(0).expect("alloc empty WHIR point"),
                vec![GpuSparseValue::new(
                    pc_start_selector,
                    g.trace_ops.fill_ext(EF::from_usize(STARTING_PC), 1),
                )],
            ));
            device_statement_prefix.push(GpuSparseStatement::new(
                stacked_n_vars,
                0,
                g.stream.alloc_zeros::<u32>(0).expect("alloc empty WHIR point"),
                vec![GpuSparseValue::new(
                    pc_end_selector,
                    g.trace_ops.fill_ext(EF::from_usize(ENDING_PC), 1),
                )],
            ));
        }

        let mut logup_cols = BTreeSet::new();
        if table.is_execution_table() {
            logup_cols.insert(COL_PC);
            for i in 0..N_INSTRUCTION_COLUMNS {
                logup_cols.insert(N_RUNTIME_COLUMNS + i);
            }
        }
        for lookup in table.lookups() {
            logup_cols.insert(lookup.index);
            for value_col in lookup.values {
                logup_cols.insert(value_col);
            }
        }
        let logup_value_specs = logup_cols
            .into_iter()
            .map(|col_index| GpuWhirColumnValuePlan {
                col_index,
                selector: selector_base + col_index,
            })
            .collect::<Vec<_>>();
        let air_eq_value_specs = (0..n_up_cols)
            .map(|col_index| GpuWhirAirValuePlan {
                selector: selector_base + col_index,
                value_offset_words: col_index * 5,
            })
            .collect::<Vec<_>>();
        let mut air_next_value_specs = table
            .down_column_indexes()
            .into_iter()
            .enumerate()
            .map(|(idx, col_index)| GpuWhirAirValuePlan {
                selector: selector_base + col_index,
                value_offset_words: (n_up_cols + idx) * 5,
            })
            .collect::<Vec<_>>();
        air_next_value_specs.sort_by_key(|plan| plan.selector);

        let logup_stmt_idx = device_statement_prefix.len();
        let logup_values = logup_value_specs
            .iter()
            .map(|plan| {
                GpuSparseValue::new(
                    plan.selector,
                    g.stream
                        .alloc_zeros::<u32>(5)
                        .expect("alloc table logup WHIR statement value"),
                )
            })
            .collect::<Vec<_>>();
        let logup_value_slots = logup_value_specs
            .iter()
            .enumerate()
            .map(|(value_idx, plan)| GpuWhirColumnValueSlot {
                col_index: plan.col_index,
                value_idx,
            })
            .collect::<Vec<_>>();
        device_statement_prefix.push(GpuSparseStatement::new(
            stacked_n_vars,
            log_n_rows,
            g.stream
                .alloc_zeros::<u32>(log_n_rows * 5)
                .expect("alloc table logup WHIR statement point"),
            logup_values,
        ));

        let (air_next_stmt_idx, air_next_value_slots) = if air_next_value_specs.is_empty() {
            (None, Vec::new())
        } else {
            let stmt_idx = device_statement_prefix.len();
            let values = air_next_value_specs
                .iter()
                .map(|plan| {
                    GpuSparseValue::new(
                        plan.selector,
                        g.stream
                            .alloc_zeros::<u32>(5)
                            .expect("alloc table AIR-next WHIR statement value"),
                    )
                })
                .collect::<Vec<_>>();
            let slots = air_next_value_specs
                .iter()
                .enumerate()
                .map(|(value_idx, plan)| GpuWhirAirValueSlot {
                    value_offset_words: plan.value_offset_words,
                    value_idx,
                })
                .collect::<Vec<_>>();
            device_statement_prefix.push(GpuSparseStatement::new_next(
                stacked_n_vars,
                log_n_rows,
                g.stream
                    .alloc_zeros::<u32>(log_n_rows * 5)
                    .expect("alloc table AIR-next WHIR statement point"),
                values,
            ));
            (Some(stmt_idx), slots)
        };

        let air_eq_stmt_idx = device_statement_prefix.len();
        let air_eq_values = air_eq_value_specs
            .iter()
            .map(|plan| {
                GpuSparseValue::new(
                    plan.selector,
                    g.stream
                        .alloc_zeros::<u32>(5)
                        .expect("alloc table AIR-eq WHIR statement value"),
                )
            })
            .collect::<Vec<_>>();
        let air_eq_value_slots = air_eq_value_specs
            .iter()
            .enumerate()
            .map(|(value_idx, plan)| GpuWhirAirValueSlot {
                value_offset_words: plan.value_offset_words,
                value_idx,
            })
            .collect::<Vec<_>>();
        device_statement_prefix.push(GpuSparseStatement::new(
            stacked_n_vars,
            log_n_rows,
            g.stream
                .alloc_zeros::<u32>(log_n_rows * 5)
                .expect("alloc table AIR-eq WHIR statement point"),
            air_eq_values,
        ));

        table_slots.push(GpuWhirTableStatementSlots {
            table,
            logup_stmt_idx,
            logup_value_slots,
            air_next_stmt_idx,
            air_next_value_slots,
            air_eq_stmt_idx,
            air_eq_value_slots,
        });
        table_offset += n_up_cols << log_n_rows;
    }

    let public_memory_mle_scratch_words = if public_memory_n_vars > 1 {
        (public_memory_size / 2) * 5
    } else {
        0
    };

    GpuWhirStatementPlan {
        public_memory_n_vars,
        d_public_memory_mle_scratch_a: Some(
            g.stream
                .alloc_zeros::<u32>(public_memory_mle_scratch_words)
                .expect("alloc public-memory MLE scratch A"),
        ),
        d_public_memory_mle_scratch_b: Some(
            g.stream
                .alloc_zeros::<u32>(public_memory_mle_scratch_words)
                .expect("alloc public-memory MLE scratch B"),
        ),
        device_statement_prefix,
        global_slots: GpuWhirGlobalStatementSlots {
            memory_stmt_idx,
            public_memory_stmt_idx,
            bytecode_stmt_idx,
        },
        table_slots,
    }
}

fn build_stack_commit_plan(
    memory_len: usize,
    bytecode_acc_len: usize,
    whir_config: &WhirConfigBuilder,
    traces: &BTreeMap<Table, TableTrace>,
) -> GpuStackCommitPlan {
    let tables_heights: BTreeMap<Table, usize> =
        traces.iter().map(|(table, trace)| (*table, trace.log_n_rows)).collect();
    let tables_sorted = sort_tables_by_height(&tables_heights);
    let stacked_n_vars = compute_stacked_n_vars(
        log2_strict_usize(memory_len),
        log2_strict_usize(bytecode_acc_len),
        &tables_sorted.iter().cloned().collect(),
    );
    let mut actual_data_len = memory_len * 2;
    let largest_h = 1 << tables_sorted[0].1;
    actual_data_len += largest_h.max(bytecode_acc_len);
    let table_column_plans = tables_sorted
        .iter()
        .map(|(table, log_n_rows)| {
            let n_rows = 1usize << *log_n_rows;
            actual_data_len += table.n_columns() * n_rows;
            GpuStackedTableColumnPlan {
                table: *table,
                n_rows,
                n_columns: table.n_columns(),
            }
        })
        .collect::<Vec<_>>();

    let whir_cfg = WhirConfig::<EF>::new(whir_config, stacked_n_vars);
    let commitment_ood_samples = whir_cfg.commitment_ood_samples;
    let ff0 = whir_config.folding_factor.at_round(0);
    let n_blocks = 1usize << ff0;
    let n_evals = 1u32 << stacked_n_vars;
    let n_cols = n_blocks as u32;
    let full_len = (n_evals as u64) << whir_config.starting_log_inv_rate;
    let merkle_height = (full_len / n_cols as u64) as u32;

    GpuStackCommitPlan {
        stacked_n_vars,
        n_base_evals: 1usize << stacked_n_vars,
        actual_data_len,
        ff0,
        n_blocks,
        n_evals,
        n_cols,
        merkle_height,
        starting_log_inv_rate: whir_config.starting_log_inv_rate,
        commitment_ood_samples,
        whir_cfg,
        table_column_plans,
    }
}

fn build_active_prover_workspaces(
    g: &Gpu,
    stack_commit_plan: &GpuStackCommitPlan,
    whir_statement_plan: &GpuWhirStatementPlan,
) -> GpuActiveProverWorkspaces {
    let logup_alpha_len = log2_ceil_usize(max_bus_width_including_domainsep());
    let initial_ood_univariate_words = stack_commit_plan.commitment_ood_samples * 5;
    let initial_ood_challenge_words = stack_commit_plan.commitment_ood_samples * stack_commit_plan.stacked_n_vars * 5;
    let initial_ood_answer_state_count = stack_commit_plan.stacked_n_vars.max(1);
    let initial_dft_height = ((stack_commit_plan.n_evals as usize) << stack_commit_plan.starting_log_inv_rate)
        / stack_commit_plan.n_cols as usize;
    let whir_final_materialization_word_capacity =
        stack_commit_plan.n_base_evals.max(WHIR_FINAL_MATERIALIZATION_MIN_WORDS);
    let mut initial_ood_answer_len = stack_commit_plan.n_base_evals;
    let d_initial_ood_answer_states = (0..initial_ood_answer_state_count)
        .map(|state_idx| {
            let state_words = if stack_commit_plan.commitment_ood_samples == 0 {
                0
            } else if stack_commit_plan.stacked_n_vars == 0 {
                stack_commit_plan.commitment_ood_samples * 5
            } else {
                initial_ood_answer_len /= 2;
                stack_commit_plan.commitment_ood_samples * initial_ood_answer_len * 5
            };
            Some(
                g.stream
                    .alloc_zeros::<u32>(state_words)
                    .unwrap_or_else(|_| panic!("alloc initial OOD answer state {state_idx}")),
            )
        })
        .collect();
    GpuActiveProverWorkspaces {
        d_stacked_poly: Some(
            g.stream
                .alloc_zeros::<u32>(stack_commit_plan.n_base_evals)
                .expect("alloc stacked polynomial workspace"),
        ),
        initial_ntt_twiddles: Some(gpu_ntt::GpuNttTwiddles::upload(
            &g.stream,
            log2_strict_usize(initial_dft_height),
            stack_commit_plan.n_cols as usize,
        )),
        d_initial_dft_output: Some(
            g.stream
                .alloc_zeros::<u32>((stack_commit_plan.n_evals as usize) << stack_commit_plan.starting_log_inv_rate)
                .expect("alloc initial DFT output workspace"),
        ),
        initial_merkle_layers: Some(gpu_merkle::GpuMerkle::allocate_tree_layers(
            &g.stream,
            stack_commit_plan.merkle_height,
        )),
        d_initial_ood_univariate_points: Some(
            g.stream
                .alloc_zeros::<u32>(initial_ood_univariate_words)
                .expect("alloc initial OOD univariate point workspace"),
        ),
        d_initial_ood_challenges: Some(
            g.stream
                .alloc_zeros::<u32>(initial_ood_challenge_words)
                .expect("alloc initial OOD expanded challenge workspace"),
        ),
        d_initial_ood_answer_states,
        d_logup_c: Some(g.stream.alloc_zeros::<u32>(5).expect("alloc logup c workspace")),
        d_logup_alpha_challenges: Some(
            g.stream
                .alloc_zeros::<u32>(logup_alpha_len * 5)
                .expect("alloc logup alpha challenge workspace"),
        ),
        d_logup_alphas_eq_poly: Some(
            g.stream
                .alloc_zeros::<u32>((1usize << logup_alpha_len) * 5)
                .expect("alloc logup alpha eq polynomial workspace"),
        ),
        d_bus_beta: Some(g.stream.alloc_zeros::<u32>(5).expect("alloc AIR bus beta workspace")),
        d_air_alpha: Some(g.stream.alloc_zeros::<u32>(5).expect("alloc AIR alpha workspace")),
        d_air_eta: Some(g.stream.alloc_zeros::<u32>(5).expect("alloc AIR eta workspace")),
        whir_workspaces: Some(stack_commit_plan.whir_cfg.gpu_allocate_prover_workspaces(
            &g.stream,
            &whir_statement_plan.device_statement_prefix,
            whir_final_materialization_word_capacity,
        )),
    }
}

fn upload_initial_prover_plan(
    g: &Gpu,
    prover_state: &impl FSProver<EF>,
    memory: &[F],
    public_memory_size: usize,
    bytecode: &Bytecode,
    whir_config: &WhirConfigBuilder,
    traces: &BTreeMap<Table, TableTrace>,
) -> GpuUploadedProverPlan {
    assert!(
        backend::initialize_whir_gpu_backend_on_stream(g.stream.clone()),
        "failed to initialize WHIR GPU backend on LeanVM stream during initial plan upload"
    );
    let uploaded_fs = GpuFsPhase::upload_initial(g, prover_state);
    let uploaded_trace = upload_trace_once(g, memory, &bytecode.instructions_multilinear, traces);
    let (d_memory_acc, d_bytecode_acc) =
        build_access_counts_on_device(g, &uploaded_trace, memory.len(), bytecode.padded_size(), traces);
    let stack_commit_plan = build_stack_commit_plan(memory.len(), bytecode.padded_size(), whir_config, traces);
    let whir_statement_plan = build_whir_statement_plan(
        g,
        stack_commit_plan.stacked_n_vars,
        memory.len(),
        public_memory_size,
        bytecode.log_size(),
        traces,
    );
    let active_workspaces = build_active_prover_workspaces(g, &stack_commit_plan, &whir_statement_plan);
    GpuUploadedProverPlan {
        uploaded_fs,
        uploaded_trace,
        d_memory_acc,
        d_bytecode_acc,
        memory_len: memory.len(),
        bytecode_acc_len: bytecode.padded_size(),
        stack_commit_plan,
        whir_statement_plan,
        active_workspaces,
    }
}

fn build_stacked_polynomial_on_device(
    g: &Gpu,
    uploaded_trace: &GpuUploadedExecutionTrace,
    stack_plan: &GpuStackCommitPlan,
    memory_len: usize,
    d_memory_acc: &CudaSlice<u32>,
    bytecode_acc_len: usize,
    d_bytecode_acc: &CudaSlice<u32>,
    mut d_poly: CudaSlice<u32>,
) -> CudaSlice<u32> {
    assert_eq!(d_poly.len(), stack_plan.n_base_evals);

    g.trace_ops
        .copy_to_offset_device(&uploaded_trace.d_memory, &mut d_poly, memory_len as u32, 0);
    let mut offset = memory_len;
    g.trace_ops
        .copy_to_offset_device(d_memory_acc, &mut d_poly, memory_len as u32, offset as u32);
    offset += memory_len;
    g.trace_ops
        .copy_to_offset_device(d_bytecode_acc, &mut d_poly, bytecode_acc_len as u32, offset as u32);
    let largest_table_height = stack_plan
        .table_column_plans
        .first()
        .map(|plan| plan.n_rows)
        .unwrap_or(0);
    offset += largest_table_height.max(bytecode_acc_len);

    for table_plan in &stack_plan.table_column_plans {
        let d_all_cols = &uploaded_trace.tables[&table_plan.table].d_all_cols;
        for col_index in 0..table_plan.n_columns {
            let src = d_all_cols.slice(col_index * table_plan.n_rows..(col_index + 1) * table_plan.n_rows);
            g.trace_ops
                .copy_to_offset_device(&src, &mut d_poly, table_plan.n_rows as u32, offset as u32);
            offset += table_plan.n_rows;
        }
    }

    debug_assert_eq!(offset, stack_plan.actual_data_len);
    d_poly
}

struct GpuMleEvalBaseManyOutput {
    d_values: CudaSlice<u32>,
    _intermediates: Vec<CudaSlice<u32>>,
}

fn gpu_mle_eval_base_many_device(
    g: &Gpu,
    d_evals: &CudaSlice<u32>,
    n_elements: usize,
    d_points_words: &CudaSlice<u32>,
    n_points: usize,
    n_vars: usize,
) -> GpuMleEvalBaseManyOutput {
    const EXT_DIM: usize = 5;
    assert_eq!(
        n_elements,
        1usize << n_vars,
        "gpu_mle_eval_base_many_device: n_elements={n_elements} n_vars={n_vars} d_evals.len()={}",
        d_evals.len()
    );

    if n_points == 0 {
        return GpuMleEvalBaseManyOutput {
            d_values: g.stream.alloc_zeros::<u32>(0).expect("allocate empty base MLE batch"),
            _intermediates: Vec::new(),
        };
    }

    if n_vars == 0 {
        return GpuMleEvalBaseManyOutput {
            d_values: g
                .sumcheck
                .repeat_base_value_as_ext_device_async(d_evals, n_points as u32),
            _intermediates: Vec::new(),
        };
    }

    let point_stride_words = (n_vars * EXT_DIM) as u32;
    let mut current_len = n_elements;
    let mut current = g.sumcheck.fold_single_base_to_many_points_device_async(
        d_evals,
        d_points_words,
        point_stride_words,
        0,
        current_len as u32,
        (current_len / 2) as u32,
        n_points as u32,
    );
    current_len /= 2;

    let mut intermediates = Vec::with_capacity(n_vars.saturating_sub(1));
    for coord_idx in 1..n_vars {
        let previous = current;
        current = g.sumcheck.fold_multi_col_ext_per_point_device_async(
            &previous,
            d_points_words,
            point_stride_words,
            coord_idx as u32,
            current_len as u32,
            (current_len / 2) as u32,
            n_points as u32,
        );
        intermediates.push(previous);
        current_len /= 2;
    }

    GpuMleEvalBaseManyOutput {
        d_values: current,
        _intermediates: intermediates,
    }
}

fn gpu_mle_eval_base_many_device_into(
    g: &Gpu,
    d_evals: &CudaSlice<u32>,
    n_elements: usize,
    d_points_words: &CudaSlice<u32>,
    n_points: usize,
    n_vars: usize,
    d_states: Vec<CudaSlice<u32>>,
) -> GpuMleEvalBaseManyOutput {
    const EXT_DIM: usize = 5;
    assert_eq!(
        n_elements,
        1usize << n_vars,
        "gpu_mle_eval_base_many_device_into: n_elements={n_elements} n_vars={n_vars} d_evals.len()={}",
        d_evals.len()
    );
    assert!(
        d_states.len() >= n_vars.max(1),
        "initial OOD MLE state workspace count mismatch"
    );

    let mut states = d_states.into_iter();
    if n_points == 0 {
        return GpuMleEvalBaseManyOutput {
            d_values: states.next().expect("missing empty MLE output state"),
            _intermediates: Vec::new(),
        };
    }

    if n_vars == 0 {
        let mut d_out = states.next().expect("missing constant MLE output state");
        g.sumcheck
            .repeat_base_value_as_ext_device_into_async(d_evals, n_points as u32, &mut d_out);
        return GpuMleEvalBaseManyOutput {
            d_values: d_out,
            _intermediates: Vec::new(),
        };
    }

    let point_stride_words = (n_vars * EXT_DIM) as u32;
    let mut current_len = n_elements;
    let mut current = states.next().expect("missing first MLE output state");
    g.sumcheck.fold_single_base_to_many_points_device_into_async(
        d_evals,
        d_points_words,
        point_stride_words,
        0,
        current_len as u32,
        (current_len / 2) as u32,
        n_points as u32,
        &mut current,
    );
    current_len /= 2;

    let mut intermediates = Vec::with_capacity(n_vars.saturating_sub(1));
    for coord_idx in 1..n_vars {
        let previous = current;
        let mut next = states
            .next()
            .unwrap_or_else(|| panic!("missing MLE output state for coord {coord_idx}"));
        g.sumcheck.fold_multi_col_ext_per_point_device_into_async(
            &previous,
            d_points_words,
            point_stride_words,
            coord_idx as u32,
            current_len as u32,
            (current_len / 2) as u32,
            n_points as u32,
            &mut next,
        );
        intermediates.push(previous);
        current = next;
        current_len /= 2;
    }

    GpuMleEvalBaseManyOutput {
        d_values: current,
        _intermediates: intermediates,
    }
}

fn owned_device_slice_mut<'a>(slice: &'a mut GpuDeviceSlice, name: &str) -> &'a mut CudaSlice<u32> {
    match slice {
        GpuDeviceSlice::Owned(data) => data,
        GpuDeviceSlice::Shared { .. } => panic!("{name} must be an owned uploaded WHIR statement buffer"),
    }
}

fn public_memory_statement_buffers_mut(
    statement: &mut GpuSparseStatement,
) -> (&mut CudaSlice<u32>, &mut CudaSlice<u32>) {
    let GpuSparseStatement {
        d_point_words, values, ..
    } = statement;
    let d_point = owned_device_slice_mut(d_point_words, "public-memory WHIR statement point");
    let d_value = owned_device_slice_mut(
        &mut values
            .get_mut(0)
            .expect("planned public-memory WHIR statement value")
            .d_value,
        "public-memory WHIR statement value",
    );
    (d_point, d_value)
}

fn memory_statement_buffers_mut(
    statement: &mut GpuSparseStatement,
) -> (&mut CudaSlice<u32>, &mut CudaSlice<u32>, &mut CudaSlice<u32>) {
    let GpuSparseStatement {
        d_point_words, values, ..
    } = statement;
    let d_point = owned_device_slice_mut(d_point_words, "memory WHIR statement point");
    let (first_value, rest_values) = values.split_at_mut(1);
    let d_value_memory = owned_device_slice_mut(&mut first_value[0].d_value, "memory WHIR statement value");
    let d_value_memory_acc = owned_device_slice_mut(
        &mut rest_values
            .get_mut(0)
            .expect("planned memory-acc WHIR statement value")
            .d_value,
        "memory-acc WHIR statement value",
    );
    (d_point, d_value_memory, d_value_memory_acc)
}

fn bytecode_statement_buffers_mut(statement: &mut GpuSparseStatement) -> (&mut CudaSlice<u32>, &mut CudaSlice<u32>) {
    let GpuSparseStatement {
        d_point_words, values, ..
    } = statement;
    let d_point = owned_device_slice_mut(d_point_words, "bytecode WHIR statement point");
    let d_value_bytecode_acc = owned_device_slice_mut(
        &mut values
            .get_mut(0)
            .expect("planned bytecode-acc WHIR statement value")
            .d_value,
        "bytecode-acc WHIR statement value",
    );
    (d_point, d_value_bytecode_acc)
}

fn collect_logup_value_targets_mut<'a>(
    values: &'a mut [GpuSparseValue],
    slots: &[GpuWhirColumnValueSlot],
) -> Vec<crate::gpu_logup::GpuLogupWhirColumnStatementTarget<'a>> {
    let mut targets = Vec::with_capacity(slots.len());
    let mut remaining_values = values;
    let mut base_value_idx = 0usize;

    for slot in slots {
        assert!(
            slot.value_idx >= base_value_idx,
            "table logup WHIR value slots must be strictly increasing"
        );
        let relative_idx = slot.value_idx - base_value_idx;
        let (_, value_and_after) = remaining_values.split_at_mut(relative_idx);
        let (value, after) = value_and_after.split_at_mut(1);
        targets.push(crate::gpu_logup::GpuLogupWhirColumnStatementTarget {
            col_index: slot.col_index,
            d_value: owned_device_slice_mut(&mut value[0].d_value, "table logup WHIR statement value"),
        });
        remaining_values = after;
        base_value_idx = slot.value_idx + 1;
    }

    targets
}

fn logup_table_statement_target_mut<'a>(
    statement: &'a mut GpuSparseStatement,
    table_slots: &GpuWhirTableStatementSlots,
) -> crate::gpu_logup::GpuLogupWhirTableStatementTargets<'a> {
    let GpuSparseStatement {
        d_point_words, values, ..
    } = statement;
    let d_point = owned_device_slice_mut(d_point_words, "table logup WHIR statement point");
    let values = collect_logup_value_targets_mut(values, &table_slots.logup_value_slots);
    crate::gpu_logup::GpuLogupWhirTableStatementTargets {
        table: table_slots.table,
        d_point,
        values,
    }
}

fn collect_logup_table_statement_targets_mut<'a>(
    statements: &'a mut [GpuSparseStatement],
    table_slots: &[GpuWhirTableStatementSlots],
    base_stmt_idx: usize,
) -> Vec<crate::gpu_logup::GpuLogupWhirTableStatementTargets<'a>> {
    let mut targets = Vec::with_capacity(table_slots.len());
    let mut remaining_statements = statements;
    let mut base_idx = base_stmt_idx;

    for slots in table_slots {
        assert!(
            slots.logup_stmt_idx >= base_idx,
            "table logup WHIR statement slots must be strictly increasing"
        );
        let relative_idx = slots.logup_stmt_idx - base_idx;
        let (_, statement_and_after) = remaining_statements.split_at_mut(relative_idx);
        let (statement, after) = statement_and_after.split_at_mut(1);
        targets.push(logup_table_statement_target_mut(&mut statement[0], slots));
        remaining_statements = after;
        base_idx = slots.logup_stmt_idx + 1;
    }

    targets
}

fn logup_whir_statement_targets_mut(
    plan: &mut GpuWhirStatementPlan,
) -> crate::gpu_logup::GpuLogupWhirStatementTargets<'_> {
    let memory_idx = plan.global_slots.memory_stmt_idx;
    let bytecode_idx = plan.global_slots.bytecode_stmt_idx;
    assert_ne!(memory_idx, bytecode_idx);
    let first_logup_idx = plan
        .table_slots
        .first()
        .expect("at least one table WHIR statement plan")
        .logup_stmt_idx;
    assert!(memory_idx < first_logup_idx);
    assert!(bytecode_idx < first_logup_idx);
    let (global_statements, table_and_after) = plan.device_statement_prefix.split_at_mut(first_logup_idx);
    let table_targets = collect_logup_table_statement_targets_mut(table_and_after, &plan.table_slots, first_logup_idx);

    if memory_idx < bytecode_idx {
        let (before_bytecode, bytecode_and_after) = global_statements.split_at_mut(bytecode_idx);
        let memory_statement = &mut before_bytecode[memory_idx];
        let bytecode_statement = &mut bytecode_and_after[0];
        let (d_memory_and_acc_point, d_value_memory, d_value_memory_acc) =
            memory_statement_buffers_mut(memory_statement);
        let (d_bytecode_and_acc_point, d_value_bytecode_acc) = bytecode_statement_buffers_mut(bytecode_statement);
        crate::gpu_logup::GpuLogupWhirStatementTargets {
            d_memory_and_acc_point,
            d_value_memory,
            d_value_memory_acc,
            d_bytecode_and_acc_point,
            d_value_bytecode_acc,
            table_targets,
        }
    } else {
        let (before_memory, memory_and_after) = global_statements.split_at_mut(memory_idx);
        let bytecode_statement = &mut before_memory[bytecode_idx];
        let memory_statement = &mut memory_and_after[0];
        let (d_memory_and_acc_point, d_value_memory, d_value_memory_acc) =
            memory_statement_buffers_mut(memory_statement);
        let (d_bytecode_and_acc_point, d_value_bytecode_acc) = bytecode_statement_buffers_mut(bytecode_statement);
        crate::gpu_logup::GpuLogupWhirStatementTargets {
            d_memory_and_acc_point,
            d_value_memory,
            d_value_memory_acc,
            d_bytecode_and_acc_point,
            d_value_bytecode_acc,
            table_targets,
        }
    }
}

fn collect_air_value_targets_mut<'a>(
    values: &'a mut [GpuSparseValue],
    slots: &[GpuWhirAirValueSlot],
) -> Vec<crate::gpu_air::GpuAirWhirValueStatementTarget<'a>> {
    let mut targets = Vec::with_capacity(slots.len());
    let mut remaining_values = values;
    let mut base_value_idx = 0usize;

    for slot in slots {
        assert!(
            slot.value_idx >= base_value_idx,
            "AIR WHIR value slots must be strictly increasing"
        );
        let relative_idx = slot.value_idx - base_value_idx;
        let (_, value_and_after) = remaining_values.split_at_mut(relative_idx);
        let (value, after) = value_and_after.split_at_mut(1);
        targets.push(crate::gpu_air::GpuAirWhirValueStatementTarget {
            value_offset_words: slot.value_offset_words,
            d_value: owned_device_slice_mut(&mut value[0].d_value, "AIR WHIR statement value"),
        });
        remaining_values = after;
        base_value_idx = slot.value_idx + 1;
    }

    targets
}

fn air_statement_point_and_values_mut<'a>(
    statement: &'a mut GpuSparseStatement,
    slots: &[GpuWhirAirValueSlot],
    point_label: &str,
) -> (
    &'a mut CudaSlice<u32>,
    Vec<crate::gpu_air::GpuAirWhirValueStatementTarget<'a>>,
) {
    let GpuSparseStatement {
        d_point_words, values, ..
    } = statement;
    (
        owned_device_slice_mut(d_point_words, point_label),
        collect_air_value_targets_mut(values, slots),
    )
}

fn collect_air_table_statement_targets_mut<'a>(
    statements: &'a mut [GpuSparseStatement],
    table_slots: &[GpuWhirTableStatementSlots],
    base_stmt_idx: usize,
) -> Vec<crate::gpu_air::GpuAirWhirTableStatementTargets<'a>> {
    let mut targets = Vec::with_capacity(table_slots.len());
    let mut remaining_statements = statements;
    let mut base_idx = base_stmt_idx;

    for slots in table_slots {
        let first_air_idx = slots.air_next_stmt_idx.unwrap_or(slots.air_eq_stmt_idx);
        assert!(
            first_air_idx >= base_idx,
            "AIR WHIR statement slots must be strictly increasing"
        );
        let relative_first_idx = first_air_idx - base_idx;
        let (_, first_and_after) = remaining_statements.split_at_mut(relative_first_idx);

        let (d_next_point, next_values, after_first, next_idx) = if let Some(next_idx) = slots.air_next_stmt_idx {
            assert_eq!(next_idx, first_air_idx);
            let (next_statement, after_next) = first_and_after.split_at_mut(1);
            let (d_next_point, next_values) = air_statement_point_and_values_mut(
                &mut next_statement[0],
                &slots.air_next_value_slots,
                "AIR-next WHIR statement point",
            );
            (Some(d_next_point), next_values, after_next, next_idx)
        } else {
            (None, Vec::new(), first_and_after, first_air_idx.saturating_sub(1))
        };

        let eq_base_idx = next_idx + 1;
        assert!(
            slots.air_eq_stmt_idx >= eq_base_idx,
            "AIR eq WHIR statement must follow AIR-next statement"
        );
        let relative_eq_idx = slots.air_eq_stmt_idx - eq_base_idx;
        let (_, eq_and_after) = after_first.split_at_mut(relative_eq_idx);
        let (eq_statement, after_eq) = eq_and_after.split_at_mut(1);
        let (d_eq_point, eq_values) = air_statement_point_and_values_mut(
            &mut eq_statement[0],
            &slots.air_eq_value_slots,
            "AIR-eq WHIR statement point",
        );

        targets.push(crate::gpu_air::GpuAirWhirTableStatementTargets {
            table: slots.table,
            d_eq_point,
            eq_values,
            d_next_point,
            next_values,
        });
        remaining_statements = after_eq;
        base_idx = slots.air_eq_stmt_idx + 1;
    }

    targets
}

fn air_whir_statement_targets_mut(
    plan: &mut GpuWhirStatementPlan,
    d_public_memory_mle_scratch_a: CudaSlice<u32>,
    d_public_memory_mle_scratch_b: CudaSlice<u32>,
) -> crate::gpu_air::GpuAirWhirStatementTargets<'_> {
    let first_air_idx = plan
        .table_slots
        .first()
        .expect("at least one table WHIR statement plan")
        .air_next_stmt_idx
        .unwrap_or_else(|| plan.table_slots[0].air_eq_stmt_idx);
    let public_memory_n_vars = plan.public_memory_n_vars;
    let public_memory_stmt_idx = plan.global_slots.public_memory_stmt_idx;
    assert!(public_memory_stmt_idx < first_air_idx);
    let (before_air, air_and_after) = plan.device_statement_prefix.split_at_mut(first_air_idx);
    let (d_public_memory_point, d_public_memory_value) =
        public_memory_statement_buffers_mut(&mut before_air[public_memory_stmt_idx]);
    crate::gpu_air::GpuAirWhirStatementTargets {
        table_targets: collect_air_table_statement_targets_mut(air_and_after, &plan.table_slots, first_air_idx),
        public_memory_target: crate::gpu_air::GpuAirPublicMemoryStatementTarget {
            n_vars: public_memory_n_vars,
            d_point: d_public_memory_point,
            d_value: d_public_memory_value,
            d_scratch_a: d_public_memory_mle_scratch_a,
            d_scratch_b: d_public_memory_mle_scratch_b,
        },
    }
}

fn finish_whir_device_statement_prefix(plan: GpuWhirStatementPlan) -> Vec<GpuSparseStatement> {
    plan.device_statement_prefix
}

/// Current post-upload active proving path.
///
/// This is intentionally extracted as one audit surface, but the implementation
/// is still CPU-orchestrated across protocol phases and is not the final
/// device-resident executor required by the goal.
fn prove_uploaded_plan_cpu_orchestrated(
    g: &Gpu,
    prover_state: &mut impl FSProver<EF>,
    uploaded_plan: GpuUploadedProverPlan,
) {
    let GpuUploadedProverPlan {
        uploaded_fs,
        mut uploaded_trace,
        d_memory_acc,
        d_bytecode_acc,
        memory_len,
        bytecode_acc_len,
        stack_commit_plan,
        mut whir_statement_plan,
        mut active_workspaces,
    } = uploaded_plan;
    let mut fs = GpuFsPhase::from_uploaded(uploaded_fs);

    // ═══════════════════════════════════════════════════════════════════
    // STEP 4-5: Polynomial stacking + WHIR commit
    // GPU: DFT→Merkle chained on device. Upload polynomial once.
    // ═══════════════════════════════════════════════════════════════════
    let (stacked_n_vars, d_stacked_poly, initial_gpu_merkle, initial_gpu_ood) = info_span!("GPU stack+commit")
        .in_scope(|| {
            let d_poly = build_stacked_polynomial_on_device(
                g,
                &uploaded_trace,
                &stack_commit_plan,
                memory_len,
                &d_memory_acc,
                bytecode_acc_len,
                &d_bytecode_acc,
                take_gpu_workspace(&mut active_workspaces.d_stacked_poly, "stacked polynomial"),
            );
            tracing::info!(
                "stacked PCS data: {} = 2^{}",
                stack_commit_plan.actual_data_len,
                stack_commit_plan.stacked_n_vars
            );

            // GPU DFT → GPU Merkle.
            let initial_ntt_twiddles = active_workspaces
                .initial_ntt_twiddles
                .as_ref()
                .expect("uploaded initial NTT twiddles");
            let d_dft_output = g.ntt.reorder_and_dft_device_guarded_with_twiddles_into(
                &d_poly,
                stack_commit_plan.n_evals,
                stack_commit_plan.ff0,
                stack_commit_plan.starting_log_inv_rate,
                initial_ntt_twiddles,
                take_gpu_workspace(&mut active_workspaces.d_initial_dft_output, "initial DFT output"),
            );
            let (d_dft, d_dft_guards) = d_dft_output.into_parts();
            let (d_root, merkle_tree) = g.merkle.build_tree_from_device_resident_root_device_into(
                &d_dft,
                stack_commit_plan.merkle_height,
                stack_commit_plan.n_cols,
                stack_commit_plan.n_cols,
                take_gpu_workspace(&mut active_workspaces.initial_merkle_layers, "initial Merkle layers"),
            );
            let initial_merkle = GpuMerkleProverData::new_base_with_guards(
                d_dft,
                merkle_tree,
                stack_commit_plan.merkle_height as usize,
                stack_commit_plan.n_blocks,
                d_dft_guards,
            );

            // Initial commitment OOD answers come from device-resident folds.
            // Inline sample_ood_points (not public from whir crate).
            let initial_ood = {
                let num_samples = stack_commit_plan.commitment_ood_samples;
                let mut d_ood_points = take_gpu_workspace(
                    &mut active_workspaces.d_initial_ood_univariate_points,
                    "initial OOD univariate points",
                );
                let mut d_ood_challenges = take_gpu_workspace(
                    &mut active_workspaces.d_initial_ood_challenges,
                    "initial OOD expanded challenges",
                );
                let d_ood_answer_states = active_workspaces
                    .d_initial_ood_answer_states
                    .iter_mut()
                    .enumerate()
                    .map(|(idx, slot)| take_gpu_workspace(slot, &format!("initial OOD answer state {idx}")))
                    .collect::<Vec<_>>();

                let stream = g.sumcheck.stream().clone();
                stream
                    .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .expect("begin initial OOD graph capture");
                fs.observe_base_scalars_device(g, GpuDeviceSlice::shared(d_root, 0, DIGEST_ELEMS), DIGEST_ELEMS);
                if num_samples > 0 {
                    fs.sample_ext_vec_device_into(g, num_samples, &mut d_ood_points);
                    g.sumcheck.expand_univariate_points_device_into_async(
                        &d_ood_points,
                        num_samples as u32,
                        stack_commit_plan.stacked_n_vars as u32,
                        &mut d_ood_challenges,
                    );
                    let d_ood_answers = gpu_mle_eval_base_many_device_into(
                        g,
                        &d_poly,
                        stack_commit_plan.n_base_evals,
                        &d_ood_challenges,
                        num_samples,
                        stack_commit_plan.stacked_n_vars,
                        d_ood_answer_states,
                    );
                    let d_ood_answer_values = Arc::new(d_ood_answers.d_values);
                    fs.observe_ext_scalars_device(
                        g,
                        GpuDeviceSlice::shared(d_ood_answer_values.clone(), 0, num_samples * 5),
                        num_samples,
                    );
                    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                    let graph = stream
                        .end_capture(graph_flags)
                        .expect("end initial OOD graph capture")
                        .expect("initial OOD capture produced no graph");
                    graph.launch().expect("launch initial OOD graph");
                    GpuInitialOodData {
                        d_points: d_ood_challenges,
                        d_answers: GpuDeviceSlice::shared(d_ood_answer_values, 0, num_samples * 5),
                        n_samples: num_samples,
                        _d_univariate_points: d_ood_points,
                        _answer_intermediates: d_ood_answers._intermediates,
                    }
                } else {
                    let d_ood_answers = gpu_mle_eval_base_many_device_into(
                        g,
                        &d_poly,
                        stack_commit_plan.n_base_evals,
                        &d_ood_challenges,
                        0,
                        stack_commit_plan.stacked_n_vars,
                        d_ood_answer_states,
                    );
                    let d_ood_answer_values = Arc::new(d_ood_answers.d_values);
                    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                    let graph = stream
                        .end_capture(graph_flags)
                        .expect("end initial OOD graph capture")
                        .expect("initial OOD capture produced no graph");
                    graph.launch().expect("launch initial OOD graph");
                    GpuInitialOodData {
                        d_points: d_ood_challenges,
                        d_answers: GpuDeviceSlice::shared(d_ood_answer_values, 0, 0),
                        n_samples: 0,
                        _d_univariate_points: d_ood_points,
                        _answer_intermediates: d_ood_answers._intermediates,
                    }
                }
            };
            (stack_commit_plan.stacked_n_vars, d_poly, initial_merkle, initial_ood)
        });

    // ═══════════════════════════════════════════════════════════════════
    // STEP 6: Logup — device-built fingerprints + GPU GKR quotient
    // ═══════════════════════════════════════════════════════════════════
    let d_logup_c = take_gpu_workspace(&mut active_workspaces.d_logup_c, "logup c");
    let d_logup_alpha_challenges = take_gpu_workspace(
        &mut active_workspaces.d_logup_alpha_challenges,
        "logup alpha challenges",
    );
    let d_logup_alphas_eq_poly = take_gpu_workspace(
        &mut active_workspaces.d_logup_alphas_eq_poly,
        "logup alpha eq polynomial",
    );

    let crate::gpu_logup::GpuLogupOutput {
        d_c: d_logup_c,
        d_alphas_eq_poly: d_logup_alphas_eq_poly,
        d_gkr_point_words,
        bus_evals: logup_bus_evals,
        _d_alpha_challenges,
        _dot_accumulation_guards,
    } = {
        let whir_statement_targets = logup_whir_statement_targets_mut(&mut whir_statement_plan);
        crate::gpu_logup::gpu_prove_logup(
            g,
            &mut fs,
            d_logup_c,
            d_logup_alpha_challenges,
            d_logup_alphas_eq_poly,
            &mut uploaded_trace,
            &d_memory_acc,
            &d_bytecode_acc,
            whir_statement_targets,
        )
    };

    // ═══════════════════════════════════════════════════════════════════
    // STEP 7: AIR sumcheck — GPU constraint evaluation
    // ═══════════════════════════════════════════════════════════════════
    let d_bus_beta = take_gpu_workspace(&mut active_workspaces.d_bus_beta, "AIR bus beta");
    let d_air_alpha = take_gpu_workspace(&mut active_workspaces.d_air_alpha, "AIR alpha");
    let d_air_eta = take_gpu_workspace(&mut active_workspaces.d_air_eta, "AIR eta");

    {
        let d_public_memory_mle_scratch_a = whir_statement_plan
            .d_public_memory_mle_scratch_a
            .take()
            .expect("planned public-memory MLE scratch A");
        let d_public_memory_mle_scratch_b = whir_statement_plan
            .d_public_memory_mle_scratch_b
            .take()
            .expect("planned public-memory MLE scratch B");
        let whir_statement_targets = air_whir_statement_targets_mut(
            &mut whir_statement_plan,
            d_public_memory_mle_scratch_a,
            d_public_memory_mle_scratch_b,
        );
        crate::gpu_air::gpu_prove_air_sumcheck(
            g,
            &mut fs,
            &d_logup_c,
            &d_logup_alphas_eq_poly,
            &mut uploaded_trace,
            &d_gkr_point_words,
            &logup_bus_evals,
            d_bus_beta,
            d_air_alpha,
            d_air_eta,
            whir_statement_targets,
        );
    }
    // Continue the device-resident transcript into WHIR instead of syncing the
    // challenger/proof transcript through the CPU at this phase boundary.
    let whir_transcript_seed = fs.into_whir_transcript_seed();

    // ═══════════════════════════════════════════════════════════════════
    // STEP 8: WHIR prove — GPU product sumcheck + fold + DFT + Merkle
    // Integrated path continues the device-resident transcript into WHIR; WHIR
    // materializes proof data at the final boundary.
    // ═══════════════════════════════════════════════════════════════════

    let device_statement_prefix = finish_whir_device_statement_prefix(whir_statement_plan);

    let whir_cfg = stack_commit_plan.whir_cfg;
    let whir_workspaces = take_gpu_workspace(&mut active_workspaces.whir_workspaces, "WHIR prover workspaces");

    // GPU WHIR prove: product sumcheck + fold + DFT + Merkle all on GPU.
    whir_cfg
        .gpu_prove_from_device_base_with_gpu_commitment_and_transcript(
            prover_state,
            initial_gpu_merkle,
            initial_gpu_ood,
            d_stacked_poly,
            1usize << stacked_n_vars,
            whir_transcript_seed,
            device_statement_prefix,
            whir_workspaces,
        )
        .expect("GPU WHIR prove failed");
    tracing::info!("GPU WHIR prove completed");

    tracing::info!("total pow_grinding time: {} ms", pow_grinding_time().as_millis());
    reset_pow_grinding_time();
}

/// Full GPU prove_execution.
///
/// GPU-backed proving path for generic leanVM executions. Polynomial/column
/// data and integrated-path Fiat-Shamir state stay on GPU as flat u32
/// `CudaSlice`s until final proof materialization, though top-level phase
/// orchestration is still host-visible and remains a blocker for the strict
/// one-launch goal.
pub fn gpu_prove_execution(
    bytecode: &Bytecode,
    public_input: &[F],
    witness: &ExecutionWitness,
    whir_config: &WhirConfigBuilder,
    vm_profiler: bool,
) -> Result<ExecutionProof, ProverError> {
    check_rate(whir_config.starting_log_inv_rate)
        .map_err(|err| panic!("{err}"))
        .unwrap();

    let g = gpu().expect("gpu_prove_execution requires a CUDA-capable GPU");

    // ═══════════════════════════════════════════════════════════════════
    // STEP 1: VM Execution (CPU — cannot be GPU-accelerated)
    // ═══════════════════════════════════════════════════════════════════
    let ExecutionTrace {
        traces,
        public_memory_size,
        mut memory,
        metadata,
    } = info_span!("Witness generation").in_scope(|| -> Result<_, ProverError> {
        let execution_result = info_span!("Executing bytecode")
            .in_scope(|| try_execute_bytecode(bytecode, public_input, witness, vm_profiler))?;
        Ok(info_span!("Building execution trace").in_scope(|| get_execution_trace(bytecode, execution_result)))
    })?;

    let min_memory_size = (1 << MIN_LOG_MEMORY_SIZE).max(1 << bytecode.log_size());
    if memory.len() < min_memory_size {
        memory.resize(min_memory_size, F::ZERO);
    }

    // ═══════════════════════════════════════════════════════════════════
    // STEP 2: ProverState setup (CPU — tiny Fiat-Shamir initialization)
    // ═══════════════════════════════════════════════════════════════════
    let mut prover_state = build_prover_state();
    prover_state.observe_scalars(public_input);
    prover_state.observe_scalars(&poseidon16_compress_pair(&bytecode.hash, &SNARK_DOMAIN_SEP));
    prover_state.add_base_scalars(
        &[
            vec![
                whir_config.starting_log_inv_rate,
                log2_strict_usize(memory.len()),
                public_input.len(),
            ],
            traces.values().map(|t| t.log_n_rows).collect::<Vec<_>>(),
        ]
        .concat()
        .into_iter()
        .map(F::from_usize)
        .collect::<Vec<_>>(),
    );

    for (table, table_trace) in &traces {
        let log_n_rows = table_trace.log_n_rows;
        assert!(log_n_rows >= MIN_LOG_N_ROWS_PER_TABLE, "missing padding");
        let log_limit = max_log_n_rows_per_table(table);
        if log_n_rows > log_limit {
            return Err(TooBigTableError {
                table_name: table.name(),
                log_n_rows,
                log_limit,
            }
            .into());
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // STEP 3: Initial GPU plan upload
    // This is the allowed host-to-device boundary for witness, trace,
    // challenger seed, and static metadata before active proving begins.
    // ═══════════════════════════════════════════════════════════════════
    let uploaded_plan = info_span!("GPU initial prover plan upload").in_scope(|| {
        upload_initial_prover_plan(
            g,
            &prover_state,
            &memory,
            public_memory_size,
            bytecode,
            whir_config,
            &traces,
        )
    });
    prove_uploaded_plan_cpu_orchestrated(g, &mut prover_state, uploaded_plan);

    Ok(ExecutionProof {
        proof: prover_state.into_proof(),
        metadata,
    })
}

#[cfg(test)]
mod gpu_residency_audit {
    const GPU_PROVE_EXECUTION: &str = include_str!("gpu_prove_execution.rs");
    const GPU_AIR: &str = include_str!("gpu_air.rs");
    const GPU_GKR: &str = include_str!("gpu_gkr.rs");
    const GPU_LOGUP: &str = include_str!("gpu_logup.rs");
    const WHIR_GPU_OPEN: &str = include_str!("../../whir/src/gpu_open.rs");
    const WHIR_GPU_COMBINE: &str = include_str!("../../whir/src/gpu_combine.rs");
    const GPU_SUMCHECK: &str = include_str!("../../../../gpu/sumcheck/src/lib.rs");
    const GPU_MERKLE: &str = include_str!("../../../../gpu/merkle/src/lib.rs");
    const GPU_NTT: &str = include_str!("../../../../gpu/ntt/src/lib.rs");

    fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_idx = source
            .find(start)
            .unwrap_or_else(|| panic!("missing audit start marker: {start}"));
        let after_start = &source[start_idx..];
        let end_idx = after_start
            .find(end)
            .unwrap_or_else(|| panic!("missing audit end marker after {start}: {end}"));
        &after_start[..end_idx]
    }

    fn count(haystack: &str, needle: &str) -> usize {
        haystack.match_indices(needle).count()
    }

    #[test]
    fn integrated_active_path_has_no_direct_host_download_or_legacy_finish() {
        let active = section(
            GPU_PROVE_EXECUTION,
            "fn prove_uploaded_plan_cpu_orchestrated",
            "/// Full GPU prove_execution.",
        );
        assert!(
            !active.contains("memcpy_dtov"),
            "post-upload active path must not directly download intermediate GPU data"
        );
        assert!(
            !active.contains(".finish(g, prover_state)"),
            "post-upload active path must not use legacy GPU transcript finalization before WHIR"
        );
        assert!(
            !active.contains("new_for_standalone_host_boundary"),
            "post-upload active path must not create standalone host-boundary Fiat-Shamir state"
        );
    }

    #[test]
    fn integrated_gkr_wrapper_does_not_download_standalone_outputs() {
        let wrapper = section(
            GPU_GKR,
            "pub(crate) fn gpu_prove_gkr_quotient_from_device_with_fs",
            "fn gpu_prove_gkr_quotient_from_device_ext",
        );
        assert!(
            !wrapper.contains("memcpy_dtov"),
            "integrated GKR wrapper must not download quotient or point"
        );
        assert!(
            wrapper.contains("false,"),
            "integrated GKR wrapper must call the shared implementation with download_quotient=false"
        );
    }

    #[test]
    fn integrated_gkr_top_claim_is_graph_captured() {
        let wrapper = section(
            GPU_GKR,
            "pub(crate) fn gpu_prove_gkr_quotient_from_device_with_fs",
            "fn gpu_prove_gkr_quotient_from_device_ext",
        );
        assert!(
            wrapper.contains("Some((d_claim_num, d_claim_den, d_mle_scratch_a, d_mle_scratch_b))"),
            "integrated GKR wrapper must pass uploaded top-claim workspaces"
        );
        let top_claim = section(
            GPU_GKR,
            "begin GKR top claim graph capture",
            "end GKR top claim graph capture",
        );
        assert!(
            top_claim.contains("fs.observe_base_scalars_device")
                && top_claim.contains("fs.sample_ext_vec_device_into(g, point_len, &mut d_point_words)")
                && top_claim.contains("gpu_mle_eval_ext_device_into"),
            "integrated GKR top transcript observe, point sampling, and top-claim MLE must be graph-captured"
        );
        assert!(
            GPU_GKR.contains("launch GKR top claim graph"),
            "captured GKR top-claim graph must be launched before captured GKR layers"
        );
    }

    #[test]
    fn strict_whir_entry_disables_randomness_download() {
        let entry = section(
            WHIR_GPU_OPEN,
            "pub fn gpu_prove_from_device_base_with_gpu_commitment_and_transcript",
            "fn gpu_prove_from_uploaded_base",
        );
        assert!(
            entry.contains("Some(transcript_seed)"),
            "strict WHIR entry must continue from the device transcript seed"
        );
        assert!(
            entry.contains("false,"),
            "strict WHIR entry must pass download_randomness=false"
        );
        assert!(
            entry.contains("Some(whir_workspaces)"),
            "strict WHIR entry must require uploaded WHIR workspaces"
        );
    }

    #[test]
    fn initial_ood_uses_uploaded_answer_workspaces() {
        let active = section(
            GPU_PROVE_EXECUTION,
            "let initial_ood = {",
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 6: Logup",
        );
        assert!(
            active.contains("d_initial_ood_answer_states"),
            "initial OOD MLE must consume answer buffers allocated during initial upload"
        );
        assert!(
            active.contains("gpu_mle_eval_base_many_device_into"),
            "initial OOD MLE must fold into uploaded answer buffers"
        );
        assert!(
            active.contains("begin initial OOD graph capture") && active.contains("launch initial OOD graph"),
            "initial Merkle-root observe, OOD sampling, OOD MLE, and answer observe must be graph-captured"
        );
        let initial_ood_capture = section(
            active,
            "begin initial OOD graph capture",
            "end initial OOD graph capture",
        );
        assert!(
            initial_ood_capture.contains("fs.observe_base_scalars_device")
                && initial_ood_capture.contains("fs.sample_ext_vec_device_into(g, num_samples, &mut d_ood_points)")
                && initial_ood_capture.contains("expand_univariate_points_device_into_async")
                && initial_ood_capture.contains("gpu_mle_eval_base_many_device_into")
                && initial_ood_capture.contains("fs.observe_ext_scalars_device"),
            "initial OOD graph must include root observe, point sampling/expansion, answer MLE, and answer observe"
        );
        assert!(
            !active.contains("gpu_mle_eval_base_many_device("),
            "initial OOD active path must not use the allocating MLE helper"
        );
    }

    #[test]
    fn whir_round_ood_uses_uploaded_answer_workspaces() {
        assert!(
            WHIR_GPU_OPEN.contains("d_ood_answer_states: Vec<Option<CudaSlice<u32>>>"),
            "WHIR round workspaces must include preallocated OOD answer states"
        );
        let round_ood = section(WHIR_GPU_OPEN, "// ── OOD evaluation ──", "// ── STIR queries ──");
        assert!(
            round_ood.contains("d_ood_answer_states"),
            "WHIR round OOD must consume uploaded answer states"
        );
        assert!(
            round_ood.contains("gpu_mle_eval_ext_many_device_into"),
            "WHIR round OOD MLE must fold into uploaded answer states"
        );
        assert!(
            WHIR_GPU_OPEN.contains("begin WHIR round OOD graph capture")
                && WHIR_GPU_OPEN.contains("launch WHIR round OOD graph"),
            "strict WHIR round OOD transcript work must be graph-captured"
        );
        let round_ood_capture = section(
            WHIR_GPU_OPEN,
            "begin WHIR round OOD graph capture",
            "// ── STIR queries ──",
        );
        assert!(
            round_ood_capture.contains("gpu_fs.observe_base_scalars_device")
                && round_ood_capture.contains("gpu_fs.sample_ext_vec_device_into")
                && round_ood_capture.contains("expand_univariate_points_device_into_async")
                && round_ood_capture.contains("gpu_mle_eval_ext_many_device_into")
                && round_ood_capture.contains("gpu_fs.observe_extension_scalars_device"),
            "WHIR round OOD graph must include root observe, point sampling/expansion, answer MLE, and answer observe"
        );
    }

    #[test]
    fn whir_round_stir_queries_use_uploaded_workspaces() {
        assert!(
            WHIR_GPU_OPEN.contains("d_query_pow_witness: Option<CudaSlice<u32>>")
                && WHIR_GPU_OPEN.contains("d_query_pow_flag: Option<CudaSlice<u32>>")
                && WHIR_GPU_OPEN.contains("d_stir_sample_words: Option<CudaSlice<u32>>")
                && WHIR_GPU_OPEN.contains("d_stir_challenges: Option<CudaSlice<u32>>")
                && WHIR_GPU_OPEN.contains("d_stir_indices: Option<CudaSlice<u32>>")
                && WHIR_GPU_OPEN.contains("d_stir_evaluations: Option<CudaSlice<u32>>"),
            "WHIR round workspaces must include uploaded STIR query buffers"
        );
        assert!(
            GPU_SUMCHECK.contains("expand_sampled_base_query_points_device_into_async")
                && GPU_SUMCHECK.contains("challenger_sample_base_scalars_device_into_async")
                && GPU_MERKLE.contains("eval_rows_at_randomness_device_into_async"),
            "strict WHIR STIR query path needs non-allocating GPU helper APIs"
        );
        let stir_queries = section(
            WHIR_GPU_OPEN,
            "// ── STIR queries ──",
            "// ── Add new eq constraints to weights ON GPU ──",
        );
        assert!(
            stir_queries.contains("take_whir_workspace(&mut workspaces.d_query_pow_witness")
                && stir_queries.contains("take_whir_workspace(&mut workspaces.d_query_pow_flag")
                && stir_queries.contains("take_whir_workspace(&mut workspaces.d_stir_sample_words")
                && stir_queries.contains("take_whir_workspace(&mut workspaces.d_stir_challenges")
                && stir_queries.contains("take_whir_workspace(&mut workspaces.d_stir_indices")
                && stir_queries.contains("take_whir_workspace(&mut workspaces.d_stir_evaluations"),
            "strict WHIR STIR query branch must consume uploaded query workspaces"
        );
        assert!(
            WHIR_GPU_OPEN.contains("begin WHIR round STIR query graph capture")
                && WHIR_GPU_OPEN.contains("launch WHIR round STIR query graph"),
            "strict WHIR STIR query work must be graph-captured"
        );
        let stir_capture = section(
            WHIR_GPU_OPEN,
            "begin WHIR round STIR query graph capture",
            "end WHIR round STIR query graph capture",
        );
        assert!(
            stir_capture.contains("pow_grinding_device_into")
                && stir_capture.contains("sample_in_range_device_into")
                && stir_capture.contains("expand_sampled_base_query_points_device_into_async")
                && stir_capture.contains("eval_rows_at_randomness_device_into"),
            "WHIR STIR query graph must include query PoW, query sampling/expansion, and leaf evaluation"
        );
        let strict_pow = section(
            WHIR_GPU_OPEN,
            "fn pow_grinding_device_into",
            "fn finish(self, g: &gpu_backend::GpuBackend, prover_state: &mut impl FSProver<EF>)",
        );
        assert!(
            !strict_pow.contains("memcpy_dtov"),
            "strict query PoW must not read the PoW flag on the host"
        );
    }

    #[test]
    fn whir_round_constraints_use_uploaded_sum_workspaces() {
        assert!(
            WHIR_GPU_OPEN.contains("d_ood_constraint_sum: Option<CudaSlice<u32>>")
                && WHIR_GPU_OPEN.contains("d_stir_constraint_sum: Option<CudaSlice<u32>>"),
            "WHIR round workspaces must include uploaded constraint-sum buffers"
        );
        let constraints = section(
            WHIR_GPU_OPEN,
            "// ── Add new eq constraints to weights ON GPU ──",
            "// ── Product sumcheck rounds ON GPU ──",
        );
        assert!(
            constraints.contains("workspaces.d_ood_constraint_sum")
                && constraints.contains("workspaces.d_stir_constraint_sum"),
            "strict WHIR round constraints must consume uploaded sum workspaces"
        );
        assert!(
            constraints.contains("begin WHIR round constraint graph capture")
                && constraints.contains("launch WHIR round constraint graph"),
            "strict WHIR round constraint accumulation must be graph-captured"
        );
        let constraint_capture = section(
            WHIR_GPU_OPEN,
            "begin WHIR round constraint graph capture",
            "end WHIR round constraint graph capture",
        );
        assert!(
            constraint_capture.contains("gpu_fs.sample_ext_device_into")
                && constraint_capture.contains("extension_powers_device_into_async")
                && constraint_capture.contains("dense_eq_accumulate_from_points_device_async")
                && constraint_capture.contains("ext_dot_accumulate_into_async"),
            "WHIR round constraint graph must include gamma sampling, scalar powers, eq accumulation, and dot accumulation"
        );
    }

    #[test]
    fn whir_initial_constraints_use_uploaded_sum_workspace() {
        assert!(
            WHIR_GPU_OPEN.contains("d_initial_ood_sum: Option<CudaSlice<u32>>"),
            "WHIR initial workspaces must include uploaded initial OOD sum buffer"
        );
        let initial_constraints = section(
            WHIR_GPU_OPEN,
            "InitialCommitmentOod::Device(initial_ood) => {",
            "// Run initial product sumcheck rounds on GPU.",
        );
        assert!(
            initial_constraints.contains("take_whir_workspace(&mut workspaces.d_initial_ood_sum")
                && initial_constraints.contains("ext_dot_accumulate_into_async"),
            "strict WHIR initial constraints must fold OOD answers into the uploaded sum workspace"
        );
        assert!(
            initial_constraints.contains("begin WHIR initial constraint graph capture")
                && initial_constraints.contains("launch WHIR initial constraint graph"),
            "strict WHIR initial constraint accumulation must be graph-captured"
        );
        let initial_capture = section(
            WHIR_GPU_OPEN,
            "begin WHIR initial constraint graph capture",
            "end WHIR initial constraint graph capture",
        );
        assert!(
            initial_capture.contains("gpu_fs.sample_ext_device_into")
                && initial_capture.contains("extension_powers_device_into_async")
                && initial_capture.contains("dense_eq_accumulate_from_points_device_async")
                && initial_capture.contains("ext_dot_accumulate_into_async"),
            "WHIR initial constraint graph must include gamma sampling, scalar powers, eq accumulation, and OOD dot accumulation"
        );
    }

    #[test]
    fn whir_initial_device_statements_use_uploaded_workspaces() {
        assert!(
            WHIR_GPU_OPEN
                .contains("initial_device_statement_accumulation: Option<GpuDeviceStatementAccumulationWorkspaces>"),
            "WHIR workspaces must include uploaded initial device-statement accumulation buffers"
        );
        assert!(
            GPU_PROVE_EXECUTION.contains(".gpu_allocate_prover_workspaces(")
                && GPU_PROVE_EXECUTION.contains("&whir_statement_plan.device_statement_prefix"),
            "initial upload must size WHIR device-statement workspaces from the uploaded statement prefix"
        );
        let initial_device_statements = section(
            WHIR_GPU_OPEN,
            "begin WHIR initial device-statement graph capture",
            "end WHIR initial device-statement graph capture",
        );
        assert!(
            initial_device_statements.contains("gpu_accumulate_device_statements_with_device_gamma_into_async"),
            "strict WHIR initial device-statement accumulation must be graph-captured"
        );
        let helper = section(
            WHIR_GPU_COMBINE,
            "pub(crate) fn gpu_accumulate_device_statements_with_device_gamma_into_async",
            "fn build_statement_poly_device",
        );
        assert!(
            helper.contains("extension_powers_device_into_async")
                && helper.contains("next_mle_device_from_point_words_into_async")
                && helper.contains("eq_polynomial_device_from_flat_point_words_into_async")
                && helper.contains("ext_dot_accumulate_into_async"),
            "device-statement helper must support uploaded gamma powers, polynomials, values, and sum output"
        );
        assert!(
            GPU_SUMCHECK.contains("next_mle_device_from_point_words_into_async"),
            "device-statement next-MLE polynomials need a non-allocating GPU helper"
        );
    }

    #[test]
    fn strict_dfts_use_uploaded_ntt_twiddles() {
        assert!(
            GPU_PROVE_EXECUTION.contains("initial_ntt_twiddles: Option<gpu_ntt::GpuNttTwiddles>")
                && GPU_PROVE_EXECUTION.contains("d_initial_dft_output: Option<CudaSlice<u32>>")
                && GPU_PROVE_EXECUTION.contains("GpuNttTwiddles::upload("),
            "initial stack+commit DFT twiddles and output must be uploaded before active proving"
        );
        let stack_commit = section(
            GPU_PROVE_EXECUTION,
            "// STEP 4-5: Polynomial stacking + WHIR commit",
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 6: Logup",
        );
        assert!(
            stack_commit.contains("reorder_and_dft_device_guarded_with_twiddles_into")
                && stack_commit.contains("d_initial_dft_output")
                && !stack_commit.contains("reorder_and_dft_device_guarded("),
            "active stack+commit DFT must use uploaded NTT twiddles and output workspace"
        );
        assert!(
            WHIR_GPU_OPEN.contains("dft_twiddles: Option<gpu_ntt::GpuNttTwiddles>")
                && WHIR_GPU_OPEN.contains("d_dft_output: Option<CudaSlice<u32>>")
                && WHIR_GPU_OPEN.contains("gpu_ntt::GpuNttTwiddles::upload("),
            "WHIR round DFT twiddles and output buffers must be uploaded with WHIR workspaces"
        );
        let whir_dft = section(
            WHIR_GPU_OPEN,
            "// ── DFT: GPU reorder",
            "// GPU Merkle tree on DFT output.",
        );
        assert!(
            whir_dft.contains("reorder_and_dft_ext_device_guarded_with_twiddles_into")
                && whir_dft.contains("take_whir_workspace(&mut workspaces.d_dft_output")
                && whir_dft.contains("reorder_and_dft_ext_device_guarded("),
            "strict WHIR round DFT must use uploaded NTT twiddles/output while legacy fallback keeps the old path"
        );
        let fused = section(GPU_NTT, "fn dft_in_place_fused_with_twiddles", "/// Per-layer DFT");
        let per_layer = section(
            GPU_NTT,
            "fn dft_in_place_per_layer_with_twiddles",
            "/// In-place inverse evals DFT.",
        );
        let base_into = section(
            GPU_NTT,
            "pub fn reorder_and_dft_device_guarded_with_twiddles_into",
            "/// Extension-field reorder -> DFT pipeline on device.",
        );
        let ext_into = section(
            GPU_NTT,
            "pub fn reorder_and_dft_ext_device_guarded_with_twiddles_into",
            "pub fn stream(&self)",
        );
        assert!(
            !fused.contains("memcpy_stod")
                && !per_layer.contains("memcpy_stod")
                && !base_into.contains("alloc_zeros")
                && !ext_into.contains("alloc_zeros"),
            "preloaded NTT DFT execution must not upload twiddle tables or allocate DFT outputs during active proving"
        );
    }

    #[test]
    fn strict_merkle_uses_uploaded_layer_workspaces() {
        assert!(
            GPU_PROVE_EXECUTION.contains("initial_merkle_layers: Option<Vec<CudaSlice<u32>>>")
                && GPU_PROVE_EXECUTION.contains("GpuMerkle::allocate_tree_layers("),
            "initial stack+commit Merkle layers must be allocated during initial upload"
        );
        let stack_commit = section(
            GPU_PROVE_EXECUTION,
            "// STEP 4-5: Polynomial stacking + WHIR commit",
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 6: Logup",
        );
        assert!(
            stack_commit.contains("build_tree_from_device_resident_root_device_into")
                && stack_commit.contains("initial_merkle_layers"),
            "active initial Merkle commitment must consume uploaded layer workspaces"
        );
        assert!(
            WHIR_GPU_OPEN.contains("merkle_layers: Option<Vec<CudaSlice<u32>>>")
                && WHIR_GPU_OPEN.contains("GpuMerkle::allocate_tree_layers("),
            "WHIR round Merkle layers must be allocated with WHIR workspaces"
        );
        let whir_merkle = section(
            WHIR_GPU_OPEN,
            "// GPU Merkle tree on DFT output.",
            "let new_merkle_data = RoundMerkleProverData::Device",
        );
        assert!(
            whir_merkle.contains("take_whir_workspace(&mut workspaces.merkle_layers")
                && whir_merkle.contains("build_tree_from_device_resident_root_device_into")
                && whir_merkle.contains("build_tree_from_device_resident_root_device("),
            "strict WHIR round Merkle must use uploaded layers while legacy fallback keeps the old path"
        );
        let merkle_into = section(
            GPU_MERKLE,
            "pub fn build_tree_from_device_resident_root_device_into",
            "pub fn eval_rows_at_randomness_device",
        );
        assert!(
            !merkle_into.contains("alloc_zeros"),
            "Merkle _into builder must not allocate digest layers during active proving"
        );
    }

    #[test]
    fn public_memory_whir_statement_is_filled_in_place() {
        let implementation = section(GPU_PROVE_EXECUTION, "struct GpuWhirStatementPlan", "#[cfg(test)]");
        let statement_plan = section(
            GPU_PROVE_EXECUTION,
            "struct GpuWhirStatementPlan",
            "struct GpuWhirGlobalStatementSlots",
        );
        let active_workspaces = section(
            GPU_PROVE_EXECUTION,
            "struct GpuActiveProverWorkspaces",
            "struct GpuWhirColumnValueSlot",
        );
        assert!(
            !active_workspaces.contains("d_public_memory_random_point: Option"),
            "public-memory WHIR point must live in the uploaded statement, not active workspaces"
        );
        assert!(
            !statement_plan.contains("d_public_memory_eval: Option"),
            "public-memory WHIR eval must live in the uploaded statement, not a side buffer"
        );
        assert!(
            !implementation.contains("struct GpuWhirDynamicStatementInputs"),
            "WHIR statement data must not be passed through post-phase dynamic attachment"
        );
        let active = section(
            GPU_PROVE_EXECUTION,
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 7: AIR",
            "let whir_transcript_seed = fs.into_whir_transcript_seed();",
        );
        assert!(
            active.contains("air_whir_statement_targets_mut") && active.contains("d_public_memory_mle_scratch_a"),
            "active path must pass public-memory WHIR target buffers into the AIR graph tail"
        );
        let air_capture = section(
            GPU_AIR,
            "begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)",
            ".end_capture(graph_flags)",
        );
        assert!(
            air_capture.contains("fill_public_memory_whir_statement_target"),
            "public-memory WHIR point/eval fill must run inside the captured AIR graph"
        );
    }

    #[test]
    fn logup_global_whir_statements_are_filled_in_place() {
        let active = section(
            GPU_PROVE_EXECUTION,
            "let crate::gpu_logup::GpuLogupOutput",
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 7: AIR",
        );
        assert!(
            active.contains("logup_whir_statement_targets_mut"),
            "logup must receive uploaded global WHIR statement buffers as write targets"
        );
        assert!(
            !GPU_LOGUP.contains("d_memory_and_acc_point: Option<CudaSlice<u32>>")
                && !GPU_LOGUP.contains("d_bytecode_and_acc_point: Option<CudaSlice<u32>>"),
            "logup must not allocate side point buffers for global WHIR statement attachment"
        );
        assert!(
            !GPU_LOGUP.contains("d_value_memory_for_statement")
                && !GPU_LOGUP.contains("d_value_bytecode_acc_for_statement"),
            "logup must not allocate side value buffers for global WHIR statement attachment"
        );
    }

    #[test]
    fn logup_table_whir_statements_are_filled_in_place() {
        let implementation = section(GPU_PROVE_EXECUTION, "struct GpuWhirStatementPlan", "#[cfg(test)]");
        assert!(
            !implementation.contains("logup_table_evals"),
            "table logup WHIR data must not be passed through post-phase dynamic attachment"
        );
        assert!(
            GPU_PROVE_EXECUTION.contains("fn logup_table_statement_target_mut"),
            "active path must expose uploaded table logup WHIR statement buffers as direct logup write targets"
        );
        assert!(
            !implementation.contains("attach_whir_device_statement_prefix"),
            "post-phase WHIR attachment must not assemble statement point/value messages on CPU"
        );
        assert!(
            GPU_LOGUP.contains("table_targets: Vec<GpuLogupWhirTableStatementTargets<'a>>"),
            "logup must receive table WHIR statement targets"
        );
        assert!(
            !GPU_LOGUP.contains("GpuLogupTableEvals")
                && !GPU_LOGUP.contains("GpuLogupColumnEval")
                && !GPU_LOGUP.contains("d_inner_point: Option<CudaSlice<u32>>")
                && !GPU_LOGUP.contains("d_logup_value_eval_copies"),
            "logup must not allocate or return side buffers for table WHIR statement attachment"
        );
    }

    #[test]
    fn air_table_whir_statements_are_filled_in_place() {
        assert!(
            !GPU_AIR.contains("GpuAirTableEvals") && !GPU_AIR.contains("GpuAirOutput"),
            "AIR must not return table eval side buffers for post-phase WHIR attachment"
        );
        assert!(
            GPU_AIR.contains("GpuAirWhirStatementTargets<'a>") && GPU_AIR.contains("fill_air_whir_statement_targets"),
            "AIR must receive uploaded WHIR statement targets and fill them in-place"
        );
        assert!(
            GPU_PROVE_EXECUTION.contains("fn air_whir_statement_targets_mut"),
            "active path must expose uploaded AIR WHIR statement buffers as direct AIR write targets"
        );
        let active = section(
            GPU_PROVE_EXECUTION,
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 7: AIR",
            "let whir_transcript_seed = fs.into_whir_transcript_seed();",
        );
        assert!(
            active.contains("air_whir_statement_targets_mut") && active.contains("gpu_prove_air_sumcheck("),
            "active AIR path must pass uploaded WHIR statement targets into AIR"
        );
    }

    #[test]
    fn air_challenges_are_sampled_inside_air_graph() {
        let active = section(
            GPU_PROVE_EXECUTION,
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 7: AIR",
            "let whir_transcript_seed = fs.into_whir_transcript_seed();",
        );
        assert!(
            !active.contains("fs.sample_ext_vec_device_into(g, 1"),
            "active Rust scheduler must not launch standalone AIR challenge sampling before AIR"
        );
        let air_capture = section(
            GPU_AIR,
            "begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)",
            ".end_capture(graph_flags)",
        );
        assert!(
            air_capture.contains("fs.sample_ext_vec_device_into(g, 1, &mut d_bus_beta)")
                && air_capture.contains("fs.sample_ext_vec_device_into(g, 1, &mut d_air_alpha)")
                && air_capture.contains("fs.sample_ext_vec_device_into(g, 1, &mut d_air_eta)"),
            "AIR bus beta/alpha/eta sampling must be inside captured AIR graph"
        );
    }

    #[test]
    fn logup_challenges_are_sampled_inside_logup_graph() {
        let active = section(
            GPU_PROVE_EXECUTION,
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 6: Logup",
            "// ═══════════════════════════════════════════════════════════════════\n    // STEP 7: AIR",
        );
        assert!(
            !active.contains("fs.sample_ext_vec_device_into"),
            "active Rust scheduler must not launch standalone logup challenge sampling before logup"
        );
        assert!(
            !active.contains("eq_polynomial_device_from_flat_point_words_into_async"),
            "active Rust scheduler must not build the logup alpha eq polynomial before logup"
        );
        assert!(
            active.contains("d_logup_alpha_challenges") && active.contains("gpu_prove_logup("),
            "active path must pass uploaded logup challenge workspaces into logup"
        );
        let logup_setup = section(
            GPU_LOGUP,
            "begin logup challenge graph capture",
            "end logup challenge graph capture",
        );
        assert!(
            logup_setup.contains("fs.sample_ext_vec_device_into(g, 1, &mut d_c)")
                && logup_setup.contains("fs.sample_ext_vec_device_into(g, logup_alpha_len, &mut d_alpha_challenges)")
                && logup_setup.contains("eq_polynomial_device_from_flat_point_words_into_async"),
            "logup c/alpha sampling and alpha eq-polynomial construction must be captured inside logup"
        );
        assert!(
            GPU_LOGUP.contains("launch logup challenge graph"),
            "captured logup challenge graph must be launched before witness construction consumes it"
        );
    }

    #[test]
    fn logup_post_gkr_evals_are_graph_captured() {
        let begin = GPU_LOGUP
            .find("begin logup eval graph capture")
            .expect("post-GKR logup evaluation staging should begin graph capture");
        let end = GPU_LOGUP
            .find("end logup eval graph capture")
            .expect("post-GKR logup evaluation staging should end graph capture");
        let launch = GPU_LOGUP
            .find("launch logup eval graph")
            .expect("post-GKR logup evaluation graph must be launched after capture");
        assert!(
            begin < end && end < launch,
            "logup eval graph capture markers are out of order"
        );
        assert!(
            !GPU_LOGUP.contains("begin logup witness graph capture"),
            "logup witness capture is known to fail CUDA graph capture on this runtime"
        );
    }

    #[test]
    fn strict_whir_final_queries_use_uploaded_workspaces() {
        assert!(
            WHIR_GPU_OPEN.contains("final_queries: Option<GpuWhirFinalQueryWorkspaces>")
                && WHIR_GPU_OPEN.contains("struct GpuWhirFinalQueryWorkspaces")
                && WHIR_GPU_OPEN.contains("d_query_pow_witness: CudaSlice<u32>")
                && WHIR_GPU_OPEN.contains("d_sample_words: CudaSlice<u32>")
                && WHIR_GPU_OPEN.contains("d_indices: CudaSlice<u32>"),
            "WHIR final query PoW/sample/index buffers must be represented as upload-time workspaces"
        );
        let allocation = section(
            WHIR_GPU_OPEN,
            "final_queries: Some(GpuWhirFinalQueryWorkspaces",
            "final_sumcheck:",
        );
        assert!(
            allocation.contains("alloc WHIR final query PoW witness workspace")
                && allocation.contains("alloc WHIR final query sample workspace")
                && allocation.contains("alloc WHIR final query index workspace"),
            "WHIR final query workspaces must be allocated during initial WHIR workspace upload"
        );
        let final_call = section(
            WHIR_GPU_OPEN,
            "let final_query_workspaces = whir_workspaces",
            "let final_domain_size = domain_size >> self.folding_factor.at_round(round_index);",
        );
        assert!(
            final_call.contains("take_whir_workspace(&mut workspaces.final_queries")
                && final_call.contains("final_query_workspaces"),
            "strict WHIR final round must receive uploaded final-query workspaces"
        );
        let final_round = section(
            WHIR_GPU_OPEN,
            "fn gpu_final_round",
            "// ═══════════════════════════════════════════════════════════════════════\n// Helpers",
        );
        assert!(
            final_round.contains("final_query_workspaces: Option<GpuWhirFinalQueryWorkspaces>")
                && final_round.contains("begin WHIR final query graph capture")
                && final_round.contains("pow_grinding_device_into")
                && final_round.contains("sample_in_range_device_into")
                && final_round.contains("expand_sampled_base_query_points_device_into_async")
                && final_round.contains("launch WHIR final query graph"),
            "strict WHIR final query PoW/sample/index expansion must consume uploaded workspaces inside graph capture"
        );
    }

    #[test]
    fn strict_whir_materialization_has_one_combined_final_download() {
        assert!(
            WHIR_GPU_OPEN.contains("d_final_materialization: Option<CudaSlice<u32>>")
                && WHIR_GPU_OPEN.contains("final_openings: Option<Vec<GpuWhirOpeningWorkspaces>>")
                && WHIR_GPU_OPEN.contains("struct GpuWhirOpeningWorkspaces")
                && WHIR_GPU_OPEN.contains("alloc WHIR final materialization workspace"),
            "strict WHIR materialization/opening buffers must be allocated with WHIR workspaces"
        );
        assert!(
            GPU_MERKLE.contains("gather_rows_device_into_async")
                && GPU_MERKLE.contains("gather_sibling_hashes_device_into_async"),
            "Merkle final-opening staging needs non-allocating row/sibling gather helpers"
        );
        assert!(
            GPU_PROVE_EXECUTION.contains("WHIR_FINAL_MATERIALIZATION_MIN_WORDS")
                && GPU_PROVE_EXECUTION.contains("whir_final_materialization_word_capacity")
                && GPU_PROVE_EXECUTION.contains("whir_final_materialization_word_capacity,"),
            "LeanVM initial upload must provide a final materialization capacity to WHIR workspace allocation"
        );
        let strict_call = section(
            WHIR_GPU_OPEN,
            "let final_materialization_workspace = whir_workspaces",
            "if !download_randomness",
        );
        assert!(
            strict_call.contains("take_whir_workspace(&mut workspaces.d_final_materialization")
                && strict_call.contains("take_whir_workspace(&mut workspaces.final_openings")
                && strict_call.contains("finish_whir_materialization(")
                && strict_call.contains("final_materialization_workspace"),
            "strict WHIR materialization must consume uploaded final materialization/opening workspaces"
        );
        let materialization = section(
            WHIR_GPU_OPEN,
            "fn finish_whir_materialization",
            "/// Materialize all-device Merkle openings with one final packed download.",
        );
        assert!(
            materialization.contains("if require_device_only_path {\n            return None;\n        }"),
            "strict WHIR materialization must fail closed before non-device Merkle fallback"
        );
        assert!(
            materialization.contains("d_final_materialization: Option<CudaSlice<u32>>")
                && materialization.contains("final_opening_workspaces: Option<Vec<GpuWhirOpeningWorkspaces>>")
                && materialization.contains("if d_final.len() < total_words")
                && materialization
                    .contains("if require_device_only_path {\n                return None;\n            }"),
            "strict WHIR materialization must use the uploaded buffer and fail closed if it is undersized"
        );
        let staging = section(
            WHIR_GPU_OPEN,
            "fn stage_device_merkle_opening_plans",
            "fn push_device_merkle_opening_plans",
        );
        assert!(
            staging.contains("require_uploaded_workspaces")
                && staging.contains("gather_rows_device_into_async")
                && staging.contains("gather_sibling_hashes_device_into_async")
                && staging.contains("gather_rows_device_async")
                && staging.contains("gather_sibling_hashes_device_async"),
            "strict opening staging must use uploaded gather workspaces while legacy fallback keeps allocating helpers"
        );
        assert_eq!(
            count(materialization, "memcpy_dtov"),
            1,
            "strict WHIR materialization should have exactly one final combined proof download"
        );
        assert!(
            materialization.contains("download combined WHIR proof materialization"),
            "the allowed WHIR download must be the final combined proof materialization"
        );
    }
}
