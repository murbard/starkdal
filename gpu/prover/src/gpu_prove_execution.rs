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
