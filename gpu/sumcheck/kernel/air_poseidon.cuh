// Poseidon16 table AIR constraints — CUDA device code.
//
// 81 constraints, degree 10. Constrains the full Poseidon1-16 permutation.
//
// Column layout (100 committed):
//   0: flag_active, 1: index_b, 2: index_res, 3: flag_half_output,
//   4: flag_hardcoded_left, 5: offset_hardcoded_left,
//   6: effective_index_left_first, 7: effective_index_left_second,
//   8-23: inputs[16], 24-55: beginning_full_rounds[2][16],
//   56-75: partial_rounds[20], 76-91: ending_full_rounds[16],
//   92-99: outputs[8]
//
// MDS circulant first column: [1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1]
// S-box: x^3 (cube)
// 4 initial full rounds + 20 partial rounds + 4 final full rounds

#pragma once
#include "../../field/koalabear_field.cuh"

// MDS circulant matrix multiply: out = MDS * state
// MDS[i][j] = circ[(16+i-j) % 16]
__device__ void mds_circ_16(uint32_t state[16], const uint32_t circ[16]) {
    uint32_t out[16];
    for (int i = 0; i < 16; i++) {
        uint32_t acc = 0;
        for (int j = 0; j < 16; j++) {
            acc = kb_add(acc, kb_mul(circ[(16 + i - j) % 16], state[j]));
        }
        out[i] = acc;
    }
    for (int i = 0; i < 16; i++) state[i] = out[i];
}

// Apply S-box (cube) to all 16 elements.
__device__ void sbox_16(uint32_t state[16]) {
    for (int i = 0; i < 16; i++) {
        state[i] = kb_cube(state[i]);
    }
}

// Add round constants to state.
__device__ void add_rc(uint32_t state[16], const uint32_t rc[16]) {
    for (int i = 0; i < 16; i++) {
        state[i] = kb_add(state[i], rc[i]);
    }
}

// Evaluate Poseidon16 AIR constraints at one row.
//
// up: 100 column values at this row (base field, Montgomery form)
// rc: round constants, 28 * 16 = 448 values (Montgomery form)
// mds: MDS circulant first column, 16 values (Montgomery form)
// sparse_data: precomputed sparse matrix data for partial rounds
//   Layout: [m_i (16*16), first_row (20*16), v (20*15), scalar_rc (20)]
// constraints: output, 81 base field constraint values
//
// The caller (kernel) weights these by alpha powers and eq factor.
__device__ void eval_poseidon16_air(
    const uint32_t* up,          // 100 columns
    uint32_t* constraints,       // 81 output values
    const uint32_t* rc,          // 28 * 16 round constants
    const uint32_t* mds,         // 16 MDS circulant column
    const uint32_t* sparse_data, // sparse matrix data
    uint32_t* bus_data = nullptr, // optional: [0]=flag_active, [1]=precompile_data, [2]=index_a, [3]=index_b, [4]=index_res
    uint32_t* post_low_state = nullptr, // optional: 16 words after the low-degree block
    bool skip_low = false,
    const uint32_t* cached_low_state = nullptr
) {
    int ci = 0; // constraint index

    // ── Column aliases ──
    uint32_t flag_active = up[0];
    uint32_t index_b = up[1];
    uint32_t index_res = up[2];
    uint32_t flag_half_output = up[3];
    uint32_t flag_hardcoded_left = up[4];
    uint32_t offset_hardcoded_left = up[5];
    uint32_t eff_idx_left_first = up[6];
    uint32_t eff_idx_left_second = up[7];

    const uint32_t* inputs = up + 8;              // 16 elements
    const uint32_t* beg_full_0 = up + 24;         // 16 elements (after round pair 0)
    const uint32_t* beg_full_1 = up + 40;         // 16 elements (after round pair 1)
    const uint32_t* partial_cols = up + 56;        // 20 elements
    const uint32_t* end_full = up + 76;            // 16 elements
    const uint32_t* outputs = up + 92;             // 8 elements

    // ── Flag constraints ──

    // Boolean: flag_active * (1 - flag_active) = 0
    constraints[ci++] = kb_mul(flag_active, kb_sub(KB_MONTY_ONE, flag_active));
    // Boolean: flag_half_output
    constraints[ci++] = kb_mul(flag_half_output, kb_sub(KB_MONTY_ONE, flag_half_output));
    // Boolean: flag_hardcoded_left
    constraints[ci++] = kb_mul(flag_hardcoded_left, kb_sub(KB_MONTY_ONE, flag_hardcoded_left));

    // flag_hardcoded_left * (offset_hardcoded_left - eff_idx_left_first) = 0
    constraints[ci++] = kb_mul(flag_hardcoded_left, kb_sub(offset_hardcoded_left, eff_idx_left_first));

    // (1 - flag_hardcoded_left) * (index_a - eff_idx_left_first) = 0
    // where index_a = eff_idx_left_second - (1 - flag_hardcoded_left) * HALF_DIGEST_LEN
    // Simplified: (1 - fhl) * (eff_second - HALF_DIGEST_LEN - eff_first) = 0
    // HALF_DIGEST_LEN = 4
    uint32_t one_minus_fhl = kb_sub(KB_MONTY_ONE, flag_hardcoded_left);
    uint32_t half_digest_monty = kb_to_monty(4);
    uint32_t index_a = kb_sub(eff_idx_left_second, kb_mul(one_minus_fhl, half_digest_monty));
    constraints[ci++] = kb_mul(one_minus_fhl, kb_sub(index_a, eff_idx_left_first));

    // ── Poseidon permutation constraints ──

    // Initialize state from inputs.
    uint32_t state[16];
    for (int i = 0; i < 16; i++) state[i] = inputs[i];

    // Initial full rounds (4 rounds = 2 pairs).
    const uint32_t* rc_init = rc; // first 4 * 16 constants

    // Round pair 0: 2 full rounds
    add_rc(state, rc_init);        // round 0
    sbox_16(state);
    mds_circ_16(state, mds);
    add_rc(state, rc_init + 16);   // round 1
    sbox_16(state);
    mds_circ_16(state, mds);

    // Assert state == beg_full_0, then replace state with witness
    for (int i = 0; i < 16; i++) {
        constraints[ci++] = kb_sub(state[i], beg_full_0[i]);
        state[i] = beg_full_0[i];
    }

    // Round pair 1: 2 more full rounds
    add_rc(state, rc_init + 32);   // round 2
    sbox_16(state);
    mds_circ_16(state, mds);
    add_rc(state, rc_init + 48);   // round 3
    sbox_16(state);
    mds_circ_16(state, mds);

    // Assert state == beg_full_1, then replace state with witness
    for (int i = 0; i < 16; i++) {
        constraints[ci++] = kb_sub(state[i], beg_full_1[i]);
        state[i] = beg_full_1[i];
    }

    // ── Partial rounds (20 rounds) ──
    // Pre-multiply by m_i and add first-round constants.
    const uint32_t* m_i = sparse_data;                    // 16*16
    const uint32_t* sp_first_row = sparse_data + 256;     // 20*16
    const uint32_t* sp_v = sparse_data + 256 + 320;       // 20*15
    const uint32_t* sp_scalar_rc = sparse_data + 256 + 320 + 300; // 19 values (padded to 20)
    const uint32_t* sp_first_rc = sparse_data + 896;      // 16 values: sparse_first_round_constants

    if (skip_low) {
        for (int r = 0; r < 20; r++) constraints[ci++] = 0;
        for (int i = 0; i < 16; i++) state[i] = cached_low_state[i];
    } else {
        // Apply sparse first round constants (NOT raw rc[4*16]!)
        add_rc(state, sp_first_rc);

        // Multiply by m_i (dense 16x16)
        {
            uint32_t tmp[16];
            for (int i = 0; i < 16; i++) {
                uint32_t acc = 0;
                for (int j = 0; j < 16; j++) {
                    acc = kb_add(acc, kb_mul(m_i[i * 16 + j], state[j]));
                }
                tmp[i] = acc;
            }
            for (int i = 0; i < 16; i++) state[i] = tmp[i];
        }

        // 20 partial rounds
        for (int r = 0; r < 20; r++) {
            // S-box on state[0] only
            state[0] = kb_cube(state[0]);

            // Assert: state[0] == partial_cols[r] (low-degree constraint, degree 3)
            constraints[ci++] = kb_sub(state[0], partial_cols[r]);

            // Replace state[0] with witness column (low_degree_block reset — keeps degree bounded)
            state[0] = partial_cols[r];

            // Add scalar round constant (except last round)
            if (r < 19) {
                state[0] = kb_add(state[0], sp_scalar_rc[r]);
            }

            // Sparse matrix multiply (rank-1 update)
            uint32_t old_s0 = state[0];

            // New state[0] = dot(sp_first_row[r], state)
            uint32_t new_s0 = 0;
            for (int j = 0; j < 16; j++) {
                new_s0 = kb_add(new_s0, kb_mul(sp_first_row[r * 16 + j], state[j]));
            }
            state[0] = new_s0;

            // For i=1..15: state[i] += old_s0 * sp_v[r][i-1]
            for (int i = 1; i < 16; i++) {
                state[i] = kb_add(state[i], kb_mul(old_s0, sp_v[r * 15 + (i - 1)]));
            }
        }

        if (post_low_state) {
            for (int i = 0; i < 16; i++) post_low_state[i] = state[i];
        }
    }

    // ── Final full rounds (4 rounds = 2 pairs) ──
    const uint32_t* rc_final = rc + 64 + 320; // last 4 * 16 constants

    // Round pair 0 (rounds 24-25): assert after
    add_rc(state, rc_final);
    sbox_16(state);
    mds_circ_16(state, mds);
    add_rc(state, rc_final + 16);
    sbox_16(state);
    mds_circ_16(state, mds);

    // Assert state == end_full, then replace state with witness
    for (int i = 0; i < 16; i++) {
        constraints[ci++] = kb_sub(state[i], end_full[i]);
        state[i] = end_full[i];
    }

    // Round pair 1 (rounds 26-27): last 2 full rounds + compression
    add_rc(state, rc_final + 32);
    sbox_16(state);
    mds_circ_16(state, mds);
    add_rc(state, rc_final + 48);
    sbox_16(state);
    mds_circ_16(state, mds);

    // Compression: add inputs back
    for (int i = 0; i < 16; i++) {
        state[i] = kb_add(state[i], inputs[i]);
    }

    // Output constraints
    uint32_t one_minus_half = kb_sub(KB_MONTY_ONE, flag_half_output);
    for (int i = 0; i < 8; i++) {
        uint32_t diff = kb_sub(state[i], outputs[i]);
        if (i < 4) {
            // Always constrained
            constraints[ci++] = diff;
        } else {
            // Gated by (1 - flag_half_output)
            constraints[ci++] = kb_mul(one_minus_half, diff);
        }
    }

    // ci == 81

    // Bus data for bus constraint computation.
    if (bus_data) {
        bus_data[0] = flag_active; // selector (column 0)
        // precompile_data = 1 + flag_half_output*2 + flag_hardcoded_left*4 + flag_hardcoded_left*offset_hardcoded_left*8
        uint32_t TWO_M = kb_to_monty(2), FOUR_M = kb_to_monty(4), EIGHT_M = kb_to_monty(8);
        uint32_t pd = kb_add(KB_MONTY_ONE,
                     kb_add(kb_mul(flag_half_output, TWO_M),
                     kb_add(kb_mul(flag_hardcoded_left, FOUR_M),
                            kb_mul(kb_mul(flag_hardcoded_left, offset_hardcoded_left), EIGHT_M))));
        bus_data[1] = pd;
        // index_a = eff_idx_left_second - (1-flag_hardcoded_left) * HALF_DIGEST_LEN
        bus_data[2] = index_a; // already computed above as the index_a variable
        bus_data[3] = up[1]; // index_b = column 1
        bus_data[4] = up[2]; // index_res = column 2
    }
}
