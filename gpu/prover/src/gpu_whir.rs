//! Full GPU-resident WHIR prover.
//!
//! Reimplements the WHIR product sumcheck loop with all data on GPU.
//! Upload polynomial + weights once → all rounds on GPU → download proof.
//!
//! PCIe traffic: ~200 bytes per round (polynomial coefficients down, challenge up).

use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream};

use gpu_poly_fold::GpuPolyFold;
use gpu_sumcheck::GpuSumcheck;

/// GPU-resident WHIR product sumcheck state.
///
/// Holds evals and weights as flat u32 device buffers.
/// All fold + sumcheck operations stay on GPU.
pub struct GpuWhirState {
    /// Polynomial evaluations on GPU.
    /// Base field: n u32s. Extension field: n * 5 u32s.
    pub d_evals: CudaSlice<u32>,
    /// Equality polynomial weights on GPU. Always extension field: n * 5 u32s.
    pub d_weights: CudaSlice<u32>,
    /// Number of evaluation points (scalar count for base, ext element count for ext).
    pub n_elements: usize,
    /// Whether evals is base field (true) or extension field (false).
    pub evals_is_base: bool,
    /// Reference to stream.
    pub stream: Arc<CudaStream>,
}

impl GpuWhirState {
    /// Upload base-field evals and ext-field weights to GPU.
    pub fn upload(
        stream: Arc<CudaStream>,
        evals_base_u32: &[u32],
        weights_ext_u32: &[u32],
    ) -> Self {
        let n = evals_base_u32.len();
        assert_eq!(weights_ext_u32.len(), n * 5);
        let d_evals = stream.memcpy_stod(evals_base_u32).unwrap();
        let d_weights = stream.memcpy_stod(weights_ext_u32).unwrap();
        Self {
            d_evals,
            d_weights,
            n_elements: n,
            evals_is_base: true,
            stream,
        }
    }

    /// Compute one product sumcheck round: (c0, c2) in ext field.
    /// Data stays on GPU; only 40 bytes downloaded.
    pub fn compute_round_poly(&self, sumcheck: &GpuSumcheck) -> ([u32; 5], [u32; 5]) {
        let half = (self.n_elements / 2) as u32;
        if self.evals_is_base {
            sumcheck.product_sumcheck_base_ext_device(&self.d_evals, &self.d_weights, half)
        } else {
            sumcheck.product_sumcheck_ext_ext_device(&self.d_evals, &self.d_weights, half)
        }
    }

    /// Fold both evals and weights at challenge r. Data stays on GPU.
    pub fn fold(&mut self, fold_engine: &GpuPolyFold, r: &[u32; 5]) {
        let half = (self.n_elements / 2) as u32;
        if self.evals_is_base {
            self.d_evals = fold_engine.fold_base_to_ext_device(&self.d_evals, half, r);
            self.evals_is_base = false;
        } else {
            self.d_evals = fold_engine.fold_ext_device(&self.d_evals, half, r);
        }
        self.d_weights = fold_engine.fold_ext_device(&self.d_weights, half, r);
        self.n_elements /= 2;
    }

    /// Download evals to host (for Merkle tree building, OOD evaluation, etc.).
    pub fn download_evals(&self) -> Vec<u32> {
        self.stream.memcpy_dtov(&self.d_evals).unwrap()
    }

    /// Download weights to host.
    pub fn download_weights(&self) -> Vec<u32> {
        self.stream.memcpy_dtov(&self.d_weights).unwrap()
    }

    /// Replace weights with a new device buffer (after eq accumulation).
    pub fn set_weights(&mut self, d_weights: CudaSlice<u32>) {
        self.d_weights = d_weights;
    }
}
