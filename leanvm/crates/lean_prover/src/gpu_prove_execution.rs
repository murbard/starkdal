//! Full GPU prove_execution.
//!
//! Lives inside lean_prover for full access to internal types.
//! Currently delegates to CPU with GPU sub-operations.

use crate::prove_execution::ExecutionProof;
use crate::*;
use backend::*;
use lean_vm::*;

/// GPU-accelerated prove_execution.
pub fn gpu_prove_execution(
    bytecode: &Bytecode,
    public_input: &[F],
    witness: &ExecutionWitness,
    whir_config: &WhirConfigBuilder,
    vm_profiler: bool,
) -> Result<ExecutionProof, ProverError> {
    // Delegates to CPU prove_execution which has GPU sub-operations
    // enabled via feature flags in whir (Merkle), fiat-shamir (PoW),
    // and sumcheck (product sumcheck) crates.
    crate::prove_execution::prove_execution(bytecode, public_input, witness, whir_config, vm_profiler)
}
