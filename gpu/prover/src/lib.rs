//! GPU-accelerated STARK prover for leanVM.
//!
//! Provides `GpuProverContext` which holds all GPU modules, and
//! `gpu_prove_execution` which reimplements the proving pipeline
//! with GPU acceleration.
//!
//! Architecture:
//! - CPU: VM execution, Fiat-Shamir (ProverState), proof assembly
//! - GPU: All heavy compute (Merkle, NTT, sumcheck, folding, PoW, logup)
//! - Sync points: ~200 bytes per sumcheck round (polynomial down, challenge up)

pub mod gpu_whir;
pub mod gpu_product_sumcheck;
pub mod gpu_prove_execution;
pub mod gpu_prover;
pub mod gpu_whir_protocol;
pub mod gpu_orchestrate;

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaStream};

pub struct GpuProverContext {
    pub stream: Arc<CudaStream>,
    pub merkle: gpu_merkle::GpuMerkle,
    pub ntt: gpu_ntt::GpuNtt,
    pub pow: gpu_pow_grind::GpuPowGrinder,
    pub sumcheck: gpu_sumcheck::GpuSumcheck,
    pub fold: gpu_poly_fold::GpuPolyFold,
    pub trace_ops: gpu_trace_ops::GpuTraceOps,
    pub logup: gpu_logup::GpuLogup,
}

impl GpuProverContext {
    /// Initialize all GPU modules on device 0.
    pub fn new() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA device required");
        let stream = ctx.default_stream();
        Self {
            merkle: gpu_merkle::GpuMerkle::new(stream.clone()),
            ntt: gpu_ntt::GpuNtt::new(stream.clone()),
            pow: gpu_pow_grind::GpuPowGrinder::new(stream.clone()),
            sumcheck: gpu_sumcheck::GpuSumcheck::new(stream.clone()),
            fold: gpu_poly_fold::GpuPolyFold::new(stream.clone()),
            trace_ops: gpu_trace_ops::GpuTraceOps::new(stream.clone()),
            logup: gpu_logup::GpuLogup::new(stream.clone()),
            stream,
        }
    }

    /// Build a Merkle tree on GPU from base field row data.
    /// Returns (root_digest, all_layer_digests).
    pub fn gpu_merkle_tree(
        &self,
        data: &[u32],
        height: u32,
        row_width: u32,
        row_stride: u32,
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        self.merkle.build_tree(data, height, row_width, row_stride)
    }

    /// Product sumcheck (base × ext): compute c0 and c2.
    pub fn gpu_product_sumcheck_base_ext(
        &self,
        pol_a: &[u32],
        pol_b: &[u32],
    ) -> ([u32; 5], [u32; 5]) {
        self.sumcheck.product_sumcheck_base_ext(pol_a, pol_b)
    }

    /// Product sumcheck (ext × ext).
    pub fn gpu_product_sumcheck_ext_ext(
        &self,
        pol_a: &[u32],
        pol_b: &[u32],
    ) -> ([u32; 5], [u32; 5]) {
        self.sumcheck.product_sumcheck_ext_ext(pol_a, pol_b)
    }

    /// PoW grinding on GPU.
    pub fn gpu_pow_grind(
        &self,
        challenger_state: &[u32; 16],
        nonce_slot: u32,
        target_bits: u32,
    ) -> Option<u32> {
        self.pow.grind(challenger_state, nonce_slot, target_bits, 1 << 28)
    }

    /// NTT (evals DFT) on GPU.
    pub fn gpu_dft(&self, data: &[u32], log_height: usize, width: usize) -> Vec<u32> {
        self.ntt.dft(data, log_height, width)
    }

    /// Fold base→ext on GPU.
    pub fn gpu_fold_base_to_ext(
        &self,
        data: &[u32],
        r_ext: &[u32; 5],
    ) -> Vec<u32> {
        self.fold.fold_base_to_ext(data, r_ext, gpu_poly_fold::FoldMode::Half)
    }

    /// Fold ext→ext on GPU.
    pub fn gpu_fold_ext(
        &self,
        data: &[u32],
        r_ext: &[u32; 5],
    ) -> Vec<u32> {
        self.fold.fold_ext(data, r_ext, gpu_poly_fold::FoldMode::Half)
    }

    /// Eq fold on GPU.
    pub fn gpu_eq_fold(&self, eq_data: &[u32], r_ext: &[u32; 5]) -> Vec<u32> {
        self.sumcheck.eq_fold(eq_data, r_ext)
    }

    /// Access count (histogram) on GPU.
    pub fn gpu_access_count(&self, column: &[u32], acc_size: usize) -> Vec<u32> {
        self.trace_ops.access_count_simple(column, acc_size)
    }

    /// MLE evaluation on GPU.
    pub fn gpu_mle_eval(&self, data: &[u32], point: &[[u32; 5]]) -> [u32; 5] {
        self.trace_ops.mle_eval(data, point)
    }

    /// Logup fingerprints on GPU.
    pub fn gpu_fingerprint(
        &self,
        columns_flat: &[u32],
        alphas: &[u32],
        c: &[u32; 5],
        n_rows: u32,
        n_cols: u32,
    ) -> Vec<u32> {
        self.logup.fingerprint(columns_flat, alphas, c, n_rows, n_cols)
    }

    /// Shift down on GPU.
    pub fn gpu_shift_down(&self, data: &[u32]) -> Vec<u32> {
        self.trace_ops.shift_down(data)
    }
}
