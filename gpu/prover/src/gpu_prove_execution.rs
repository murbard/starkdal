//! Full GPU prove_execution: complete proving pipeline on GPU.
//!
//! This module provides the building blocks for a complete GPU prover.
//! Each function handles one protocol phase, operating on GPU-resident data.
//! The top-level orchestration ties them together with ProverState on CPU.

use std::sync::Arc;
use std::collections::BTreeMap;

use cudarc::driver::safe::{CudaSlice, CudaStream};

use crate::GpuProverContext;

/// GPU-resident trace data.
pub struct GpuTrace {
    pub stream: Arc<CudaStream>,
    /// All trace columns indexed by (table, col_index).
    pub columns: BTreeMap<lean_vm::Table, Vec<CudaSlice<u32>>>,
    /// Shifted (down) columns indexed by (table, down_col_index).
    pub down_columns: BTreeMap<lean_vm::Table, Vec<CudaSlice<u32>>>,
    /// Memory array.
    pub d_memory: CudaSlice<u32>,
    /// Table metadata.
    pub log_heights: BTreeMap<lean_vm::Table, usize>,
    pub non_padded_rows: BTreeMap<lean_vm::Table, usize>,
    pub public_memory_size: usize,
    pub memory_len: usize,
}

impl GpuTrace {
    /// Upload all trace data to GPU. One-time cost (~100-200MB).
    pub fn upload(
        stream: &Arc<CudaStream>,
        traces: &BTreeMap<lean_vm::Table, lean_vm::TableTrace>,
        memory: &[lean_vm::F],
        public_memory_size: usize,
    ) -> Self {
        let mem_u32 = unsafe {
            std::slice::from_raw_parts(memory.as_ptr().cast::<u32>(), memory.len())
        };
        let d_memory = stream.memcpy_stod(mem_u32).unwrap();

        let mut columns = BTreeMap::new();
        let mut log_heights = BTreeMap::new();
        let mut non_padded_rows = BTreeMap::new();
        for (table, trace) in traces {
            log_heights.insert(*table, trace.log_n_rows);
            non_padded_rows.insert(*table, trace.non_padded_n_rows);
            let mut cols = Vec::new();
            for col in &trace.columns {
                let u32s = unsafe {
                    std::slice::from_raw_parts(col.as_ptr().cast::<u32>(), col.len())
                };
                cols.push(stream.memcpy_stod(u32s).unwrap());
            }
            columns.insert(*table, cols);
        }

        // Shifted columns will be computed on GPU later using shift_down kernel.
        // For now, store empty — they'll be populated when needed.
        let down_columns = BTreeMap::new();

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

/// GPU WHIR commit: reorder → DFT → Merkle, chained on device.
/// Returns (root, dft_output_flat, merkle_layers).
pub fn gpu_whir_commit(
    gpu: &GpuProverContext,
    d_polynomial: &CudaSlice<u32>,
    n_evals: usize,
    folding_factor: usize,
    log_inv_rate: usize,
) -> (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) {
    let n_cols = 1u32 << folding_factor;
    let d_dft = gpu.ntt.reorder_and_dft_device(
        d_polynomial, n_evals as u32, folding_factor, log_inv_rate,
    );
    let full_len = (n_evals as u64) << log_inv_rate;
    let height = (full_len / n_cols as u64) as u32;
    let (root, layers) = gpu.merkle.build_tree_from_device(
        &d_dft, height, n_cols, n_cols,
    );
    let dft_flat = gpu.stream.memcpy_dtov(&d_dft).unwrap();
    (root, dft_flat, layers)
}

/// GPU product sumcheck: all rounds on device.
/// Returns (folded_evals, folded_weights, n_final).
pub fn gpu_product_sumcheck_rounds(
    gpu: &GpuProverContext,
    mut d_evals: CudaSlice<u32>,
    mut d_weights: CudaSlice<u32>,
    mut n_evals: usize,
    n_rounds: usize,
    mut on_round: impl FnMut([u32; 5], [u32; 5]) -> [u32; 5],
) -> (CudaSlice<u32>, CudaSlice<u32>, usize) {
    let mut evals_is_base = true;

    for _round in 0..n_rounds {
        let half = (n_evals / 2) as u32;
        let (c0, c2) = if evals_is_base {
            gpu.sumcheck.product_sumcheck_base_ext_device(&d_evals, &d_weights, half)
        } else {
            gpu.sumcheck.product_sumcheck_ext_ext_device(&d_evals, &d_weights, half)
        };
        let r = on_round(c0, c2);
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

/// GPU eq polynomial generation on device.
pub fn gpu_eq_polynomial(
    gpu: &GpuProverContext,
    point: &[[u32; 5]],
) -> CudaSlice<u32> {
    gpu.sumcheck.eq_polynomial_device(point)
}

/// GPU eq accumulation with offset on device.
pub fn gpu_eq_accumulate(
    gpu: &GpuProverContext,
    d_weights: &mut CudaSlice<u32>,
    d_eq: &CudaSlice<u32>,
    scalar: &[u32; 5],
    offset: u32,
    n: u32,
) {
    gpu.sumcheck.eq_accumulate_offset_device(d_weights, d_eq, scalar, offset, n);
}
