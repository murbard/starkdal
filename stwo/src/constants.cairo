//! Field constants for the Stark252 coset FFT.
//!
//! Stark252 prime: P = 2^251 + 17 * 2^192 + 1.
//! Multiplicative group order P-1 = 2^192 * (2^59 + 17), so primitive 2^k-th roots of unity
//! exist for any k in [0, 192]. We pre-pick a generator of order 2^MAX_LOG; smaller-order
//! roots are obtained by squaring.

/// Maximum log2 of evaluation-domain size we support without recompiling constants.
/// Set to 20 — comfortably above the n=18, k=2 production target.
pub const MAX_LOG: u32 = 20;

/// Primitive 2^MAX_LOG-th root of unity in Stark252.
/// Computed offline as 3^((P-1) / 2^MAX_LOG) mod P, where 3 is a generator of (Z/P)*.
pub const OMEGA_MAX: felt252 =
    0x594beafca8a00d9581d81caee93dc85c727c9af7fc4c648e3d47b998574e81f;

/// Coset offset. Evaluations are at COSET_OFFSET * <omega>. Using 3 (a generator of (Z/P)*)
/// guarantees the coset is disjoint from the subgroup <omega> for any log_total < 192.
pub const COSET_OFFSET: felt252 = 3;
