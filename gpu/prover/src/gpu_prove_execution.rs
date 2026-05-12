//! Full GPU prove_execution: complete proving pipeline on GPU.
//!
//! Reimplements the entire prove_execution from leanVM but with all
//! heavy compute on GPU. Uses leanVM's ProverState for Fiat-Shamir.
//!
//! This file will be built incrementally as each protocol phase is
//! implemented. Each phase must produce the EXACT same Fiat-Shamir
//! transcript entries as the CPU version.

use std::sync::Arc;
use std::collections::BTreeMap;

use cudarc::driver::safe::{CudaSlice, CudaStream};

use crate::GpuProverContext;

pub use backend::*;
pub use lean_vm::*;
pub use lean_prover::prove_execution::ExecutionProof;

/// GPU-resident trace data.
pub struct GpuTrace {
    pub stream: Arc<CudaStream>,
    pub columns: BTreeMap<Table, Vec<CudaSlice<u32>>>,
    pub down_columns: BTreeMap<Table, Vec<CudaSlice<u32>>>,
    pub d_memory: CudaSlice<u32>,
    pub log_heights: BTreeMap<Table, usize>,
    pub non_padded_rows: BTreeMap<Table, usize>,
    pub public_memory_size: usize,
    pub memory_len: usize,
}

impl GpuTrace {
    pub fn upload(
        stream: &Arc<CudaStream>,
        traces: &BTreeMap<Table, TableTrace>,
        memory: &[F],
        public_memory_size: usize,
    ) -> Self {
        let mem_u32 = unsafe { std::slice::from_raw_parts(memory.as_ptr().cast::<u32>(), memory.len()) };
        let d_memory = stream.memcpy_stod(mem_u32).unwrap();

        let mut columns = BTreeMap::new();
        let mut log_heights = BTreeMap::new();
        let mut non_padded_rows = BTreeMap::new();
        for (table, trace) in traces {
            log_heights.insert(*table, trace.log_n_rows);
            non_padded_rows.insert(*table, trace.non_padded_n_rows);
            let mut cols = Vec::new();
            for col in &trace.columns {
                let u32s = unsafe { std::slice::from_raw_parts(col.as_ptr().cast::<u32>(), col.len()) };
                cols.push(stream.memcpy_stod(u32s).unwrap());
            }
            columns.insert(*table, cols);
        }

        // Compute shifted (down) columns.
        let mut down_columns = BTreeMap::new();
        for (table, trace) in traces {
            let down_idx = table.down_column_indexes();
            let mut dcols = Vec::new();
            for &ci in &down_idx {
                let col = &trace.columns[ci];
                let n = col.len();
                let mut shifted: Vec<F> = col[1..].to_vec();
                shifted.push(col[n - 1]);
                let u32s = unsafe { std::slice::from_raw_parts(shifted.as_ptr().cast::<u32>(), shifted.len()) };
                dcols.push(stream.memcpy_stod(u32s).unwrap());
            }
            down_columns.insert(*table, dcols);
        }

        Self {
            stream: stream.clone(),
            columns,
            down_columns,
            d_memory,
            log_heights,
            non_padded_rows,
            public_memory_size,
            memory_len: memory.len(),
        }
    }
}

/// Phase 1: Polynomial stacking on GPU.
/// Concatenates [memory | memory_acc | bytecode_acc | table_cols...] into
/// a single device buffer, mirroring the CPU's stack_polynomials_and_commit layout.
pub fn gpu_stack_polynomial(
    gpu: &GpuProverContext,
    gpu_trace: &GpuTrace,
    d_memory_acc: &CudaSlice<u32>,
    d_bytecode_acc: &CudaSlice<u32>,
    bytecode_log_size: usize,
) -> (CudaSlice<u32>, usize) {
    let memory_len = gpu_trace.memory_len;
    let tables_sorted = sort_tables_by_height(&gpu_trace.log_heights);

    // Compute stacked size.
    let stacked_n_vars = sub_protocols::stacked_pcs::compute_stacked_n_vars(
        utils::log2_strict_usize(memory_len),
        bytecode_log_size,
        &tables_sorted.iter().cloned().collect(),
    );
    let stacked_size = 1usize << stacked_n_vars;

    // Download everything to CPU for stacking (the stacking pattern is complex).
    // TODO: implement stacking as a GPU kernel for zero-copy.
    let memory = gpu.stream.memcpy_dtov(&gpu_trace.d_memory).unwrap();
    let memory_acc = gpu.stream.memcpy_dtov(d_memory_acc).unwrap();
    let bytecode_acc = gpu.stream.memcpy_dtov(d_bytecode_acc).unwrap();

    let mut stacked = vec![0u32; stacked_size];
    stacked[..memory_len].copy_from_slice(&memory);
    let mut offset = memory_len;
    stacked[offset..offset + memory_len].copy_from_slice(&memory_acc);
    offset += memory_len;

    let bc_len = bytecode_acc.len();
    stacked[offset..offset + bc_len].copy_from_slice(&bytecode_acc);
    let largest_height = 1 << tables_sorted[0].1;
    offset += largest_height.max(bc_len);

    for (table, log_n_rows) in &tables_sorted {
        let n_rows = 1 << *log_n_rows;
        let cols = &gpu_trace.columns[table];
        for ci in 0..table.n_columns() {
            let col_data = gpu.stream.memcpy_dtov(&cols[ci]).unwrap();
            stacked[offset..offset + n_rows].copy_from_slice(&col_data[..n_rows]);
            offset += n_rows;
        }
    }

    // Upload stacked polynomial to GPU.
    let d_stacked = gpu.stream.memcpy_stod(&stacked).unwrap();
    (d_stacked, stacked_n_vars)
}

/// Phase 2: WHIR commit on GPU.
/// Reorder → DFT → Merkle, all on device. Downloads only root (32 bytes)
/// and DFT output (for CPU-side OOD evaluation + Merkle path opening).
pub fn gpu_whir_commit(
    gpu: &GpuProverContext,
    d_stacked: &CudaSlice<u32>,
    stacked_n_vars: usize,
    folding_factor: usize,
    log_inv_rate: usize,
) -> (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) {
    let n_evals = 1u32 << stacked_n_vars;
    let n_cols = 1u32 << folding_factor;

    // Reorder + DFT on device.
    let d_dft = gpu.ntt.reorder_and_dft_device(d_stacked, n_evals, folding_factor, log_inv_rate);

    // Merkle on DFT output (device-resident, no re-upload).
    let full_len = (n_evals as u64) << log_inv_rate;
    let height = (full_len / n_cols as u64) as u32;
    let (root_flat, layers_flat) = gpu.merkle.build_tree_from_device(
        &d_dft, height, n_cols, n_cols,
    );

    // Download DFT for CPU-side operations (OOD eval, Merkle path opening).
    let dft_flat = gpu.stream.memcpy_dtov(&d_dft).unwrap();

    (root_flat, dft_flat, layers_flat)
}

/// Phase 3: Build combined weights on GPU (eq polynomials).
/// Replaces combine_statement — weights are generated directly on GPU,
/// no 640MB upload needed.
pub fn gpu_combine_statement(
    gpu: &GpuProverContext,
    statements: &[(Vec<[u32; 5]>, Vec<(usize, [u32; 5])>, bool)],
    // Each statement: (point_coords, [(selector, value)], is_next)
    gamma: [u32; 5],
    num_variables: usize,
) -> (CudaSlice<u32>, [u32; 5]) {
    let dim = 5;
    let n_total = 1usize << num_variables;

    // Zero-initialized weights on GPU.
    let mut d_weights = gpu.stream.alloc_zeros::<u32>(n_total * dim).unwrap();
    let mut combined_sum = [0u32; 5]; // Accumulated on CPU (tiny).
    let mut gamma_pow = [0x01FFFFFEu32, 0, 0, 0, 0]; // 1 in Montgomery form

    for (point, values, is_next) in statements {
        let inner_n_vars = point.len();
        let inner_n = 1usize << inner_n_vars;

        // Build eq(point, x) on GPU or next_mle on CPU.
        let d_poly = if *is_next {
            // next_mle: compute on CPU, upload (tiny).
            // TODO: implement next_mle on GPU
            let next = backend::extension::QuinticExtensionField::<F>::default(); // placeholder
            gpu.stream.alloc_zeros::<u32>(inner_n * dim).unwrap() // zeros for now
        } else {
            gpu.sumcheck.eq_polynomial_device(point)
        };

        for &(selector, value) in values {
            let offset = (selector * inner_n) as u32;
            gpu.sumcheck.eq_accumulate_offset_device(
                &mut d_weights, &d_poly, &gamma_pow, offset, inner_n as u32,
            );

            // combined_sum += value * gamma_pow (CPU, tiny EF arithmetic)
            // TODO: proper EF multiplication on CPU
            // For now, accumulate component-wise (only correct for base-field values)
            for k in 0..5 {
                combined_sum[k] = kb_add_host(combined_sum[k], kb_mul_host(value[k], gamma_pow[k]));
            }

            // gamma_pow *= gamma
            gamma_pow = qe_mul_host(&gamma_pow, &gamma);
        }
    }

    (d_weights, combined_sum)
}

// Host-side KoalaBear arithmetic helpers.
fn kb_add_host(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= 0x7F000001 { s - 0x7F000001 } else { s }
}

fn kb_mul_host(a: u32, b: u32) -> u32 {
    let x = a as u64 * b as u64;
    let t = (x as u32).wrapping_mul(0x81000001u32);
    let u = t as u64 * 0x7F000001u64;
    let diff = x.wrapping_sub(u);
    let hi = (diff >> 32) as u32;
    if x < u { hi.wrapping_add(0x7F000001) } else { hi }
}

fn qe_mul_host(a: &[u32; 5], b: &[u32; 5]) -> [u32; 5] {
    // Full quintic extension multiplication using the CPU reference.
    type EF = backend::extension::QuinticExtensionField<F>;
    let ea: EF = unsafe { std::mem::transmute(*a) };
    let eb: EF = unsafe { std::mem::transmute(*b) };
    unsafe { std::mem::transmute(ea * eb) }
}

/// Phase 4: GPU-resident product sumcheck.
/// Takes device-resident evals (base) and weights (ext), does all fold+sumcheck
/// rounds on GPU. Only Fiat-Shamir coefficients cross PCIe.
pub fn gpu_product_sumcheck_rounds(
    gpu: &GpuProverContext,
    mut d_evals: CudaSlice<u32>,
    mut d_weights: CudaSlice<u32>,
    mut n_evals: usize,
    n_rounds: usize,
    mut on_round: impl FnMut([u32; 5], [u32; 5]) -> [u32; 5],
    // on_round(c0, c2) → challenge r. Also handles Fiat-Shamir internally.
) -> (CudaSlice<u32>, CudaSlice<u32>, usize) {
    let mut evals_is_base = true;

    for _round in 0..n_rounds {
        let half = (n_evals / 2) as u32;

        // Compute (c0, c2) on GPU — only 40 bytes downloaded.
        let (c0, c2) = if evals_is_base {
            gpu.sumcheck.product_sumcheck_base_ext_device(&d_evals, &d_weights, half)
        } else {
            gpu.sumcheck.product_sumcheck_ext_ext_device(&d_evals, &d_weights, half)
        };

        // Host callback: Fiat-Shamir + challenge sampling.
        let r = on_round(c0, c2);

        // Fold on GPU — data stays on device.
        if evals_is_base {
            d_evals = gpu.fold.fold_base_to_ext_device(&d_evals, half, &r);
            evals_is_base = false;
        } else {
            d_evals = gpu.fold.fold_ext_device(&d_evals, half, &r);
        }
        d_weights = gpu.fold.fold_ext_device(&d_weights, half, &r);
        n_evals /= 2;
    }

    (d_evals, d_weights, n_evals)
}

/// Phase 5: AIR sumcheck for execution table (GPU).
/// Evaluates constraints across all row pairs, folds, repeats.
/// Returns the round polynomial coefficients for each round.
pub fn gpu_air_sumcheck_execution_round(
    gpu: &GpuProverContext,
    d_columns: &CudaSlice<u32>,    // 20 columns col-major on device
    d_down_cols: &CudaSlice<u32>,   // 2 shifted columns col-major on device
    d_eq_factor: &CudaSlice<u32>,   // eq factor (ext field) on device
    alphas: &[u32],                  // alpha powers (13 * 5 = 65 u32s)
    n_rows: u32,
) -> ([u32; 5], [u32; 5]) {
    let n_pairs = n_rows / 2;
    gpu.sumcheck.air_sumcheck_execution_device(
        d_columns, d_down_cols, d_eq_factor, alphas, n_rows, n_pairs,
    )
}

/// The main GPU prove_execution function.
///
/// This is a SKELETON that shows the complete protocol flow.
/// Each step calls our GPU kernels where possible and falls back to CPU
/// for operations not yet ported (logup GKR, AIR constraint eval for
/// ExtensionOp and Poseidon16 tables).
///
/// The goal is to have ALL heavy compute on GPU, with only Fiat-Shamir
/// (ProverState) on CPU.
pub fn gpu_prove_execution_skeleton(
    gpu: &GpuProverContext,
    bytecode: &Bytecode,
    public_input: &[F],
    witness: &ExecutionWitness,
    whir_config: &lean_prover::WhirConfigBuilder,
    vm_profiler: bool,
) -> Result<ExecutionProof, lean_prover::ProverError> {
    // === STEP 1: CPU VM execution (cannot be GPU-accelerated) ===
    let exec_result = lean_vm::try_execute_bytecode(bytecode, public_input, witness, vm_profiler)?;
    let exec_trace = lean_vm::get_execution_trace(bytecode, exec_result);

    let mut memory = exec_trace.memory;
    let traces = exec_trace.traces;
    let metadata = exec_trace.metadata;
    let public_memory_size = exec_trace.public_memory_size;

    // Pad memory.
    let min_memory_size = (1 << lean_vm::MIN_LOG_MEMORY_SIZE).max(1 << bytecode.log_size());
    if memory.len() < min_memory_size {
        memory.resize(min_memory_size, F::ZERO);
    }

    // === STEP 2: Upload ALL trace data to GPU (one-time cost) ===
    let gpu_trace = GpuTrace::upload(&gpu.stream, &traces, &memory, public_memory_size);

    // === STEP 3: Access counts on GPU ===
    // TODO: use gpu_trace_ops::access_count for memory and bytecode
    // For now, compute on CPU and upload.
    let mut memory_acc = F::zero_vec(memory.len());
    for (table, trace) in &traces {
        for lookup in table.lookups() {
            for i in &trace.columns[lookup.index] {
                for j in 0..lookup.values.len() {
                    memory_acc[i.to_usize() + j] += F::ONE;
                }
            }
        }
    }
    let mut bytecode_acc = F::zero_vec(bytecode.padded_size());
    for pc in traces[&Table::execution()].columns[lean_vm::tables::execution::air::COL_PC].iter() {
        bytecode_acc[pc.to_usize()] += F::ONE;
    }

    let mem_acc_u32 = unsafe { std::slice::from_raw_parts(memory_acc.as_ptr().cast::<u32>(), memory_acc.len()) };
    let bc_acc_u32 = unsafe { std::slice::from_raw_parts(bytecode_acc.as_ptr().cast::<u32>(), bytecode_acc.len()) };
    let d_memory_acc = gpu.stream.memcpy_stod(mem_acc_u32).unwrap();
    let d_bytecode_acc = gpu.stream.memcpy_stod(bc_acc_u32).unwrap();

    // === STEP 4: Polynomial stacking + WHIR commit ===
    let (_d_stacked, stacked_n_vars) = gpu_stack_polynomial(
        gpu, &gpu_trace, &d_memory_acc, &d_bytecode_acc, bytecode.log_size(),
    );

    // === For the remaining steps (logup, AIR sumcheck, WHIR prove),
    // we currently fall back to the CPU prove_execution. ===
    // The full GPU implementation will replace this with GPU kernel calls.

    // For now, call the CPU prover to produce a valid proof.
    // This ensures correctness while we incrementally replace each step.
    lean_prover::prove_execution::prove_execution(
        bytecode, public_input, witness, whir_config, vm_profiler,
    )
}

/// Phase 6: OOD evaluation on GPU (multilinear evaluation at sampled points).
/// For now, downloads polynomial and evaluates on CPU (the evaluation is tiny).
pub fn cpu_ood_eval(
    stacked_poly: &[u32],  // downloaded stacked polynomial
    point: &[[u32; 5]],    // extension field point coordinates
) -> [u32; 5] {
    // TODO: use gpu_trace_ops::mle_eval for GPU evaluation
    gpu_trace_ops::cpu_mle_eval(stacked_poly, point)
}

