// Poseidon16 table AIR constraints — CUDA device code (SKELETON).
//
// 80+ constraints, degree 10. The most complex AIR.
// Constrains the full Poseidon permutation round by round.
//
// This is a SKELETON — the full implementation requires:
// - MDS circulant matrix multiply as constraint (not compute)
// - S-box (x^3) verification at each round
// - Flag constraints
// - Output compression constraints
//
// For the first GPU integration, we'll evaluate the Poseidon16 table
// constraints on CPU and only GPU-accelerate the other two tables.
// The Poseidon16 table is typically the smallest (2^14-2^16 rows),
// so the CPU cost is manageable.
//
// TODO: implement full Poseidon16 constraint evaluation in CUDA.
// This requires:
// 1. Hardcoded MDS matrix constants (16×16 circulant)
// 2. Round constant arrays
// 3. Sparse matrix decomposition constants
// 4. ~80 assert_zero / assert_eq constraints unrolled

#pragma once
#include "../../field/koalabear_field.cuh"

// Placeholder: Poseidon16 constraint evaluation.
// Returns 0 for all constraints (to be replaced with actual implementation).
__device__ void eval_poseidon16_air_placeholder(
    const uint32_t* up,     // num_cols_poseidon_16 columns
    uint32_t* constraints,  // output: ~80 constraint values
    int n_constraints
) {
    for (int i = 0; i < n_constraints; i++) {
        constraints[i] = 0;
    }
}
