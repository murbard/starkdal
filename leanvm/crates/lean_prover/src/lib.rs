#![cfg_attr(not(test), allow(unused_crate_dependencies))]

use std::fmt::Display;

use backend::*;
use lean_vm::{
    EF, F, MAX_WHIR_LOG_INV_RATE, MIN_LOG_N_ROWS_PER_TABLE, MIN_WHIR_LOG_INV_RATE, RunnerError, Table, TableT,
};
use utils::*;

mod trace_gen;

pub mod prove_execution;
pub mod verify_execution;

#[cfg(test)]
mod test_zkvm;

use trace_gen::*;

// Right now, hash digests = 8 koala-bear (p = 2^31 - 2^24 + 1, i.e. ≈ 31 bits per field element)
pub const SECURITY_BITS: usize = 124; // TODO 128 bits security

pub const GRINDING_BITS: usize = 16;
pub const MAX_NUM_VARIABLES_TO_SEND_COEFFS: usize = 8;
pub const WHIR_INITIAL_FOLDING_FACTOR: usize = 7;
pub const WHIR_SUBSEQUENT_FOLDING_FACTOR: usize = 5;
pub const RS_DOMAIN_INITIAL_REDUCTION_FACTOR: usize = 5;

pub const SNARK_DOMAIN_SEP: [F; 8] = F::new_array([
    130704175, 1303721200, 493664240, 1035493700, 2063844858, 1410214009, 1938905908, 1696767928,
]);

pub fn default_whir_config(starting_log_inv_rate: usize) -> WhirConfigBuilder {
    WhirConfigBuilder {
        folding_factor: FoldingFactor::new(WHIR_INITIAL_FOLDING_FACTOR, WHIR_SUBSEQUENT_FOLDING_FACTOR),
        soundness_type: if cfg!(feature = "prox-gaps-conjecture") {
            SecurityAssumption::CapacityBound // TODO update formula with State of the Art Conjecture
        } else {
            SecurityAssumption::JohnsonBound
        },
        pow_bits: GRINDING_BITS,
        max_num_variables_to_send_coeffs: MAX_NUM_VARIABLES_TO_SEND_COEFFS,
        rs_domain_initial_reduction_factor: RS_DOMAIN_INITIAL_REDUCTION_FACTOR,
        security_level: SECURITY_BITS,
        starting_log_inv_rate,
    }
}

pub(crate) fn check_rate(log_inv_rate: usize) -> Result<(), ProofError> {
    if (MIN_WHIR_LOG_INV_RATE..=MAX_WHIR_LOG_INV_RATE).contains(&log_inv_rate) {
        Ok(())
    } else {
        Err(ProofError::InvalidRate)
    }
}

#[derive(Debug, Clone)]
pub enum ProverError {
    TooBigTable(TooBigTableError),
    Runner(RunnerError),
}

impl From<TooBigTableError> for ProverError {
    fn from(err: TooBigTableError) -> Self {
        Self::TooBigTable(err)
    }
}

impl From<RunnerError> for ProverError {
    fn from(err: RunnerError) -> Self {
        Self::Runner(err)
    }
}

impl Display for ProverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooBigTable(e) => write!(f, "{}", e),
            Self::Runner(e) => write!(f, "{}", e),
        }
    }
}

