//! Complete GPU prover: reimplements the STARK proving protocol from scratch.
//!
//! All data stays on GPU as flat u32 CudaSlice buffers.
//! Only Fiat-Shamir (ProverState) and VM execution run on CPU.
//! No leanVM type system (MleOwned, EFPacking, etc.).
//!
//! Target: 10-20x speedup over CPU prover.

use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream};

use crate::GpuProverContext;

// Field constants.
const P: u32 = 0x7F000001;
const MONTY_ONE: u32 = 0x01FFFFFE; // Montgomery form of 1
const DIM: usize = 5; // quintic extension dimension
const DIGEST_LEN: usize = 8; // Poseidon16 digest length

/// All GPU-resident state for the prover.
pub struct GpuProverState {
    gpu: GpuProverContext,

    // Trace data (uploaded once).
    /// Execution table columns: n_cols × n_rows base field elements (col-major).
    d_exec_cols: Vec<CudaSlice<u32>>,
    /// Execution table shifted (down) columns.
    d_exec_down: Vec<CudaSlice<u32>>,
    /// ExtensionOp table columns.
    d_extop_cols: Vec<CudaSlice<u32>>,
    d_extop_down: Vec<CudaSlice<u32>>,
    /// Poseidon16 table columns.
    d_pos_cols: Vec<CudaSlice<u32>>,
    /// Memory array.
    d_memory: CudaSlice<u32>,
    /// Memory access count.
    d_memory_acc: CudaSlice<u32>,
    /// Bytecode access count.
    d_bytecode_acc: CudaSlice<u32>,

    // Derived data (built on GPU).
    /// Stacked polynomial.
    d_stacked: Option<CudaSlice<u32>>,

    // Sizes.
    exec_n_rows: usize,
    extop_n_rows: usize,
    pos_n_rows: usize,
    memory_len: usize,
    stacked_n_vars: usize,
}

impl GpuProverState {
    /// Upload all trace data to GPU.
    pub fn upload(
        gpu: GpuProverContext,
        exec_cols: &[Vec<u32>],
        exec_down_indices: &[usize],
        extop_cols: &[Vec<u32>],
        extop_down_indices: &[usize],
        pos_cols: &[Vec<u32>],
        memory: &[u32],
        memory_acc: &[u32],
        bytecode_acc: &[u32],
    ) -> Self {
        let stream = &gpu.stream;

        let upload_cols = |cols: &[Vec<u32>]| -> Vec<CudaSlice<u32>> {
            cols.iter().map(|c| stream.memcpy_stod(c).unwrap()).collect()
        };

        let upload_down = |cols: &[Vec<u32>], indices: &[usize]| -> Vec<CudaSlice<u32>> {
            indices.iter().map(|&i| {
                let col = &cols[i];
                let n = col.len();
                let mut shifted = col[1..].to_vec();
                shifted.push(col[n - 1]);
                stream.memcpy_stod(&shifted).unwrap()
            }).collect()
        };

        let exec_n_rows = if exec_cols.is_empty() { 0 } else { exec_cols[0].len() };
        let extop_n_rows = if extop_cols.is_empty() { 0 } else { extop_cols[0].len() };
        let pos_n_rows = if pos_cols.is_empty() { 0 } else { pos_cols[0].len() };

        Self {
            d_exec_cols: upload_cols(exec_cols),
            d_exec_down: upload_down(exec_cols, exec_down_indices),
            d_extop_cols: upload_cols(extop_cols),
            d_extop_down: upload_down(extop_cols, extop_down_indices),
            d_pos_cols: upload_cols(pos_cols),
            d_memory: stream.memcpy_stod(memory).unwrap(),
            d_memory_acc: stream.memcpy_stod(memory_acc).unwrap(),
            d_bytecode_acc: stream.memcpy_stod(bytecode_acc).unwrap(),
            d_stacked: None,
            exec_n_rows,
            extop_n_rows,
            pos_n_rows,
            memory_len: memory.len(),
            stacked_n_vars: 0,
            gpu,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 3: Polynomial stacking
    // ═══════════════════════════════════════════════════════════════════════

    /// Stack all columns into a single flat buffer on GPU.
    /// Layout: [memory | memory_acc | bytecode_acc(padded) | exec_cols | extop_cols | pos_cols]
    pub fn stack_polynomial(&mut self, bytecode_log_size: usize) {
        // Compute sizes.
        let log_memory = self.memory_len.trailing_zeros() as usize;
        let exec_log = self.exec_n_rows.trailing_zeros() as usize;
        let extop_log = self.extop_n_rows.trailing_zeros() as usize;
        let pos_log = self.pos_n_rows.trailing_zeros() as usize;

        // Stacked layout: memory(2x) + bytecode_acc + exec_cols + extop_cols + pos_cols
        let mut total = 2 * self.memory_len; // memory + memory_acc
        total += self.exec_n_rows.max(1 << bytecode_log_size); // bytecode_acc padded
        let exec_n_cols = self.d_exec_cols.len();
        let extop_n_cols = self.d_extop_cols.len();
        let pos_n_cols = self.d_pos_cols.len();
        total += exec_n_cols * self.exec_n_rows;
        total += extop_n_cols * self.extop_n_rows;
        total += pos_n_cols * self.pos_n_rows;

        self.stacked_n_vars = total.next_power_of_two().trailing_zeros() as usize;
        let stacked_size = 1usize << self.stacked_n_vars;

        // For now, build on CPU and upload (stacking is just memcpy with offsets).
        // TODO: implement as GPU kernel for zero-copy.
        let memory = self.gpu.stream.memcpy_dtov(&self.d_memory).unwrap();
        let memory_acc = self.gpu.stream.memcpy_dtov(&self.d_memory_acc).unwrap();
        let bytecode_acc = self.gpu.stream.memcpy_dtov(&self.d_bytecode_acc).unwrap();

        let mut stacked = vec![0u32; stacked_size];
        let mut offset = 0;
        stacked[offset..offset + self.memory_len].copy_from_slice(&memory);
        offset += self.memory_len;
        stacked[offset..offset + self.memory_len].copy_from_slice(&memory_acc);
        offset += self.memory_len;
        stacked[offset..offset + bytecode_acc.len()].copy_from_slice(&bytecode_acc);
        offset += self.exec_n_rows.max(1 << bytecode_log_size);

        // Stack table columns in height-sorted order (exec first as largest).
        for ci in 0..exec_n_cols.min(20) { // execution has 20 committed cols
            let col = self.gpu.stream.memcpy_dtov(&self.d_exec_cols[ci]).unwrap();
            stacked[offset..offset + self.exec_n_rows].copy_from_slice(&col[..self.exec_n_rows]);
            offset += self.exec_n_rows;
        }
        for ci in 0..extop_n_cols.min(29) { // extension_op has 29 cols
            let col = self.gpu.stream.memcpy_dtov(&self.d_extop_cols[ci]).unwrap();
            stacked[offset..offset + self.extop_n_rows].copy_from_slice(&col[..self.extop_n_rows]);
            offset += self.extop_n_rows;
        }
        for ci in 0..pos_n_cols {
            let col = self.gpu.stream.memcpy_dtov(&self.d_pos_cols[ci]).unwrap();
            stacked[offset..offset + self.pos_n_rows].copy_from_slice(&col[..self.pos_n_rows]);
            offset += self.pos_n_rows;
        }

        self.d_stacked = Some(self.gpu.stream.memcpy_stod(&stacked).unwrap());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 4: WHIR commit (DFT → Merkle)
    // ═══════════════════════════════════════════════════════════════════════

    /// WHIR commit: reorder → DFT → Merkle, all on device.
    /// Returns (root_digest, dft_device_buffer, merkle_layers).
    pub fn whir_commit(
        &self,
        folding_factor: usize,
        log_inv_rate: usize,
    ) -> (Vec<u32>, CudaSlice<u32>, Vec<Vec<u32>>) {
        let d_stacked = self.d_stacked.as_ref().expect("call stack_polynomial first");
        let n_evals = 1u32 << self.stacked_n_vars;
        let n_cols = 1u32 << folding_factor;

        // Reorder + DFT on device.
        let d_dft = self.gpu.ntt.reorder_and_dft_device(
            d_stacked, n_evals, folding_factor, log_inv_rate,
        );

        // Merkle on DFT output (stays on device).
        let full_len = (n_evals as u64) << log_inv_rate;
        let height = (full_len / n_cols as u64) as u32;
        let (root, layers) = self.gpu.merkle.build_tree_from_device(
            &d_dft, height, n_cols, n_cols,
        );

        (root, d_dft, layers)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 5: Product sumcheck (device-resident)
    // ═══════════════════════════════════════════════════════════════════════

    /// Run product sumcheck rounds with evals and weights on device.
    /// Only ~200 bytes per round crosses PCIe.
    pub fn product_sumcheck(
        &self,
        d_evals: CudaSlice<u32>,
        d_weights: CudaSlice<u32>,
        n_evals: usize,
        n_rounds: usize,
        evals_is_base: bool,
        mut on_round: impl FnMut([u32; 5], [u32; 5]) -> [u32; 5],
    ) -> (CudaSlice<u32>, CudaSlice<u32>, usize) {
        let mut d_e = d_evals;
        let mut d_w = d_weights;
        let mut n = n_evals;
        let mut is_base = evals_is_base;

        for _ in 0..n_rounds {
            let half = (n / 2) as u32;
            let (c0, c2) = if is_base {
                self.gpu.sumcheck.product_sumcheck_base_ext_device(&d_e, &d_w, half)
            } else {
                self.gpu.sumcheck.product_sumcheck_ext_ext_device(&d_e, &d_w, half)
            };

            let r = on_round(c0, c2);

            if is_base {
                d_e = self.gpu.fold.fold_base_to_ext_device(&d_e, half, &r);
                is_base = false;
            } else {
                d_e = self.gpu.fold.fold_ext_device(&d_e, half, &r);
            }
            d_w = self.gpu.fold.fold_ext_device(&d_w, half, &r);
            n /= 2;
        }

        (d_e, d_w, n)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 6: Eq polynomial (device-resident)
    // ═══════════════════════════════════════════════════════════════════════

    /// Build eq(point, x) on GPU for all x in {0,1}^n.
    pub fn eq_polynomial(&self, point: &[[u32; 5]]) -> CudaSlice<u32> {
        self.gpu.sumcheck.eq_polynomial_device(point)
    }

    /// Accumulate: weights[offset + j] += scalar * eq[j].
    pub fn eq_accumulate(
        &self,
        d_weights: &mut CudaSlice<u32>,
        d_eq: &CudaSlice<u32>,
        scalar: &[u32; 5],
        offset: u32,
        n: u32,
    ) {
        self.gpu.sumcheck.eq_accumulate_offset_device(d_weights, d_eq, scalar, offset, n);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 7: GKR quotient reduction (device-resident)
    // ═══════════════════════════════════════════════════════════════════════

    /// GKR layer reduction: sum pairs of quotients.
    pub fn gkr_reduce_layer(
        &self,
        d_nums: &CudaSlice<u32>,
        d_dens: &CudaSlice<u32>,
        n_pairs: u32,
    ) -> (CudaSlice<u32>, CudaSlice<u32>) {
        self.gpu.sumcheck.gkr_sum_quotients_device(d_nums, d_dens, n_pairs)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step 8: AIR constraint evaluation (device-resident)
    // ═══════════════════════════════════════════════════════════════════════

    /// Evaluate execution table constraints for one sumcheck round.
    pub fn air_eval_execution(
        &self,
        d_columns: &CudaSlice<u32>,
        d_down_cols: &CudaSlice<u32>,
        d_eq_factor: &CudaSlice<u32>,
        alphas: &[u32],
        n_rows: u32,
        n_pairs: u32,
    ) -> ([u32; 5], [u32; 5]) {
        self.gpu.sumcheck.air_sumcheck_execution_device(
            d_columns, d_down_cols, d_eq_factor, alphas, n_rows, n_pairs,
        )
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Utility: download data from GPU
    // ═══════════════════════════════════════════════════════════════════════

    pub fn download(&self, d: &CudaSlice<u32>) -> Vec<u32> {
        self.gpu.stream.memcpy_dtov(d).unwrap()
    }

    pub fn upload_slice(&self, data: &[u32]) -> CudaSlice<u32> {
        self.gpu.stream.memcpy_stod(data).unwrap()
    }

    pub fn alloc_zeros(&self, n: usize) -> CudaSlice<u32> {
        self.gpu.stream.alloc_zeros::<u32>(n).unwrap()
    }
}
