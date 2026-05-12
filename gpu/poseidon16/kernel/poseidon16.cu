// Poseidon16 permutation on KoalaBear field — CUDA kernel.
//
// Each thread computes one poseidon16_compress: output[0..8] = (perm(input) + input)[0..8].
// Uses shared field arithmetic from koalabear_field.cuh.
//
// Constants are uploaded to __constant__ memory by the host.

#include "../../field/koalabear_field.cuh"

// Aliases for backward compatibility within this file.
#define monty_add  kb_add
#define monty_sub  kb_sub
#define monty_mul  kb_mul
#define sbox       kb_cube

// ── Poseidon parameters ──────────────────────────────────────────────────
#define WIDTH           16
#define HALF_FULL       4
#define PARTIAL_ROUNDS  20
#define TOTAL_ROUNDS    (2 * HALF_FULL + PARTIAL_ROUNDS)

// ── Constant memory ──────────────────────────────────────────────────────
__constant__ uint32_t d_rc[TOTAL_ROUNDS][WIDTH];
__constant__ uint32_t d_mds[WIDTH][WIDTH];
__constant__ uint32_t d_sparse_first_rc[WIDTH];
__constant__ uint32_t d_sparse_m_i[WIDTH][WIDTH];
__constant__ uint32_t d_sparse_first_row[PARTIAL_ROUNDS][WIDTH];
__constant__ uint32_t d_sparse_v[PARTIAL_ROUNDS][WIDTH];
__constant__ uint32_t d_sparse_scalar_rc[PARTIAL_ROUNDS - 1];

// ── Dense MDS matrix–vector multiply ─────────────────────────────────────
// state ← MDS · state  (reads d_mds from constant memory).
__device__ void mds_multiply(uint32_t state[WIDTH]) {
    uint32_t tmp[WIDTH];
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        uint32_t acc = monty_mul(d_mds[i][0], state[0]);
        #pragma unroll
        for (int j = 1; j < WIDTH; j++) {
            acc = monty_add(acc, monty_mul(d_mds[i][j], state[j]));
        }
        tmp[i] = acc;
    }
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) state[i] = tmp[i];
}

// ── Full round: AddRC → S-box (all 16) → MDS ────────────────────────────
__device__ void full_round(uint32_t state[WIDTH], int round_idx) {
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        state[i] = monty_add(state[i], d_rc[round_idx][i]);
    }
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        state[i] = sbox(state[i]);
    }
    mds_multiply(state);
}

// ── Dense 16×16 matrix–vector multiply (for m_i) ────────────────────────
__device__ void sparse_m_i_multiply(uint32_t state[WIDTH]) {
    uint32_t tmp[WIDTH];
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        uint32_t acc = monty_mul(d_sparse_m_i[i][0], state[0]);
        #pragma unroll
        for (int j = 1; j < WIDTH; j++) {
            acc = monty_add(acc, monty_mul(d_sparse_m_i[i][j], state[j]));
        }
        tmp[i] = acc;
    }
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) state[i] = tmp[i];
}

// ── Partial rounds (sparse decomposition) ────────────────────────────────
__device__ void partial_rounds(uint32_t state[WIDTH]) {
    // Step 1: add first-round constants.
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        state[i] = monty_add(state[i], d_sparse_first_rc[i]);
    }

    // Step 2: dense m_i multiply (once).
    sparse_m_i_multiply(state);

    // Step 3: 20 partial rounds with sparse matmul.
    #pragma unroll
    for (int r = 0; r < PARTIAL_ROUNDS; r++) {
        // S-box on state[0] only.
        state[0] = sbox(state[0]);

        // Add scalar round constant (except last round).
        if (r < PARTIAL_ROUNDS - 1) {
            state[0] = monty_add(state[0], d_sparse_scalar_rc[r]);
        }

        // Sparse matrix multiply:
        //   new_s0 = dot(state, first_row[r])
        //   state[i] += old_s0 * v[r][i-1]  for i ∈ [1, 16)
        uint32_t old_s0 = state[0];

        // Dot product for new state[0].
        uint32_t dot = monty_mul(d_sparse_first_row[r][0], state[0]);
        #pragma unroll
        for (int j = 1; j < WIDTH; j++) {
            dot = monty_add(dot, monty_mul(d_sparse_first_row[r][j], state[j]));
        }
        state[0] = dot;

        // Rank-1 update for state[1..16].
        #pragma unroll
        for (int i = 1; i < WIDTH; i++) {
            state[i] = monty_add(state[i], monty_mul(old_s0, d_sparse_v[r][i - 1]));
        }
    }
}

// ── Poseidon16 permutation ───────────────────────────────────────────────
__device__ void poseidon16_permute(uint32_t state[WIDTH]) {
    // Initial full rounds.
    #pragma unroll
    for (int r = 0; r < HALF_FULL; r++) {
        full_round(state, r);
    }

    // Partial rounds (sparse decomposition).
    partial_rounds(state);

    // Terminal full rounds.
    #pragma unroll
    for (int r = 0; r < HALF_FULL; r++) {
        full_round(state, HALF_FULL + PARTIAL_ROUNDS + r);
    }
}

// ── Kernel: poseidon16_compress ──────────────────────────────────────────
// Each thread processes one state: output = (perm(input) + input)[0..8].
//
// Layout (AoS): input[tid * 16 + lane], output[tid * 8 + lane].
extern "C" __global__ void poseidon16_compress_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t n_states
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_states) return;

    // Load state from global memory.
    uint32_t state[WIDTH];
    uint32_t initial[WIDTH];
    const uint32_t* src = input + (uint64_t)tid * WIDTH;
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        state[i] = src[i];
        initial[i] = state[i];
    }

    // Permute.
    poseidon16_permute(state);

    // Compress: output = (perm(state) + state)[0..8].
    uint32_t* dst = output + (uint64_t)tid * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[i] = monty_add(state[i], initial[i]);
    }
}

// ── Kernel: poseidon16_permute (full 16-element output) ──────────────────
extern "C" __global__ void poseidon16_permute_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t n_states
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_states) return;

    uint32_t state[WIDTH];
    const uint32_t* src = input + (uint64_t)tid * WIDTH;
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        state[i] = src[i];
    }

    poseidon16_permute(state);

    uint32_t* dst = output + (uint64_t)tid * WIDTH;
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        dst[i] = state[i];
    }
}
