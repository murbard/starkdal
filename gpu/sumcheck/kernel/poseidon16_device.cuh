// Poseidon16 permutation — device-callable version.
// Takes round constants as parameters (no __constant__ memory).
// For use in the sumcheck protocol kernel's Fiat-Shamir challenger.

#pragma once
#include "../../field/koalabear_field.cuh"

#define P16_WIDTH 16
#define P16_HALF_FULL 4
#define P16_PARTIAL_ROUNDS 20

// MDS circulant multiply (same as in poseidon16.cu but self-contained).
__device__ void p16_mds(uint32_t state[P16_WIDTH], const uint32_t mds_circ[P16_WIDTH]) {
    uint32_t out[P16_WIDTH];
    for (int i = 0; i < P16_WIDTH; i++) {
        uint32_t acc = 0;
        for (int j = 0; j < P16_WIDTH; j++)
            acc = kb_add(acc, kb_mul(mds_circ[(P16_WIDTH + i - j) & 15], state[j]));
        out[i] = acc;
    }
    for (int i = 0; i < P16_WIDTH; i++) state[i] = out[i];
}

// Full round: add RC, sbox (cube), MDS.
__device__ void p16_full_round(uint32_t state[P16_WIDTH], const uint32_t rc[P16_WIDTH], const uint32_t mds_circ[P16_WIDTH]) {
    for (int i = 0; i < P16_WIDTH; i++) {
        state[i] = kb_add(state[i], rc[i]);
        state[i] = kb_cube(state[i]);
    }
    p16_mds(state, mds_circ);
}

// Poseidon16 permutation with explicit constants.
// rc: 28 * 16 round constants (flat: rc[round * 16 + lane])
// mds_circ: 16-element MDS circulant first column
// sparse: layout [m_i(256), first_row(320), v(300), scalar_rc(19+pad), first_rc(16)] = 912 elements
__device__ void p16_permute(uint32_t state[P16_WIDTH],
    const uint32_t* rc, const uint32_t* mds_circ, const uint32_t* sparse)
{
    // Initial full rounds (4 rounds)
    for (int r = 0; r < P16_HALF_FULL; r++)
        p16_full_round(state, rc + r * P16_WIDTH, mds_circ);

    // Partial rounds
    const uint32_t* m_i = sparse;
    const uint32_t* sp_first_row = sparse + 256;
    const uint32_t* sp_v = sparse + 256 + 320;
    const uint32_t* sp_scalar_rc = sparse + 256 + 320 + 300;
    const uint32_t* sp_first_rc = sparse + 896;

    // Add sparse first round constants + multiply by m_i
    for (int i = 0; i < P16_WIDTH; i++)
        state[i] = kb_add(state[i], sp_first_rc[i]);
    {
        uint32_t tmp[P16_WIDTH];
        for (int i = 0; i < P16_WIDTH; i++) {
            uint32_t acc = 0;
            for (int j = 0; j < P16_WIDTH; j++)
                acc = kb_add(acc, kb_mul(m_i[i * P16_WIDTH + j], state[j]));
            tmp[i] = acc;
        }
        for (int i = 0; i < P16_WIDTH; i++) state[i] = tmp[i];
    }

    for (int r = 0; r < P16_PARTIAL_ROUNDS; r++) {
        state[0] = kb_cube(state[0]);
        if (r < P16_PARTIAL_ROUNDS - 1)
            state[0] = kb_add(state[0], sp_scalar_rc[r]);
        uint32_t old_s0 = state[0];
        uint32_t new_s0 = 0;
        for (int j = 0; j < P16_WIDTH; j++)
            new_s0 = kb_add(new_s0, kb_mul(sp_first_row[r * P16_WIDTH + j], state[j]));
        state[0] = new_s0;
        for (int i = 1; i < P16_WIDTH; i++)
            state[i] = kb_add(state[i], kb_mul(old_s0, sp_v[r * 15 + (i - 1)]));
    }

    // Terminal full rounds (4 rounds)
    for (int r = 0; r < P16_HALF_FULL; r++)
        p16_full_round(state, rc + (P16_HALF_FULL + P16_PARTIAL_ROUNDS + r) * P16_WIDTH, mds_circ);
}

// Poseidon16 compress: output = (permute(input) + input)[0..8]
// Operates in-place: state[0..15] → state[0..7] = compressed output
__device__ void p16_compress(uint32_t state[P16_WIDTH],
    const uint32_t* rc, const uint32_t* mds_circ, const uint32_t* sparse)
{
    uint32_t input[P16_WIDTH];
    for (int i = 0; i < P16_WIDTH; i++) input[i] = state[i];
    p16_permute(state, rc, mds_circ, sparse);
    for (int i = 0; i < 8; i++) state[i] = kb_add(state[i], input[i]);
}
