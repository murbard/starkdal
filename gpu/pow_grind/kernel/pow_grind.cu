// Proof-of-work grinding for WHIR — CUDA kernel.
//
// Each thread tests one candidate nonce. The Poseidon16 permutation is inlined
// (same algorithm as gpu/poseidon16/, using the shared field header).
// The first thread to find a valid nonce writes it atomically.

#include "../../field/koalabear_field.cuh"

// ── Poseidon16 constants (same as poseidon16.cu) ─────────────────────────
#define P16_WIDTH          16
#define P16_HALF_FULL      4
#define P16_PARTIAL_ROUNDS 20
#define P16_TOTAL_ROUNDS   (2 * P16_HALF_FULL + P16_PARTIAL_ROUNDS)

__constant__ uint32_t d_rc[P16_TOTAL_ROUNDS][P16_WIDTH];
__constant__ uint32_t d_mds[P16_WIDTH][P16_WIDTH];
__constant__ uint32_t d_sparse_first_rc[P16_WIDTH];
__constant__ uint32_t d_sparse_m_i[P16_WIDTH][P16_WIDTH];
__constant__ uint32_t d_sparse_first_row[P16_PARTIAL_ROUNDS][P16_WIDTH];
__constant__ uint32_t d_sparse_v[P16_PARTIAL_ROUNDS][P16_WIDTH];
__constant__ uint32_t d_sparse_scalar_rc[P16_PARTIAL_ROUNDS - 1];

// ── Poseidon16 permutation (inlined) ─────────────────────────────────────

__device__ void mds_multiply(uint32_t state[P16_WIDTH]) {
    uint32_t tmp[P16_WIDTH];
    #pragma unroll
    for (int i = 0; i < P16_WIDTH; i++) {
        uint32_t acc = kb_mul(d_mds[i][0], state[0]);
        #pragma unroll
        for (int j = 1; j < P16_WIDTH; j++)
            acc = kb_add(acc, kb_mul(d_mds[i][j], state[j]));
        tmp[i] = acc;
    }
    #pragma unroll
    for (int i = 0; i < P16_WIDTH; i++) state[i] = tmp[i];
}

__device__ void full_round(uint32_t state[P16_WIDTH], int round_idx) {
    #pragma unroll
    for (int i = 0; i < P16_WIDTH; i++)
        state[i] = kb_add(state[i], d_rc[round_idx][i]);
    #pragma unroll
    for (int i = 0; i < P16_WIDTH; i++)
        state[i] = kb_cube(state[i]);
    mds_multiply(state);
}

__device__ void sparse_m_i_multiply(uint32_t state[P16_WIDTH]) {
    uint32_t tmp[P16_WIDTH];
    #pragma unroll
    for (int i = 0; i < P16_WIDTH; i++) {
        uint32_t acc = kb_mul(d_sparse_m_i[i][0], state[0]);
        #pragma unroll
        for (int j = 1; j < P16_WIDTH; j++)
            acc = kb_add(acc, kb_mul(d_sparse_m_i[i][j], state[j]));
        tmp[i] = acc;
    }
    #pragma unroll
    for (int i = 0; i < P16_WIDTH; i++) state[i] = tmp[i];
}

__device__ void partial_rounds(uint32_t state[P16_WIDTH]) {
    #pragma unroll
    for (int i = 0; i < P16_WIDTH; i++)
        state[i] = kb_add(state[i], d_sparse_first_rc[i]);

    sparse_m_i_multiply(state);

    #pragma unroll
    for (int r = 0; r < P16_PARTIAL_ROUNDS; r++) {
        state[0] = kb_cube(state[0]);
        if (r < P16_PARTIAL_ROUNDS - 1)
            state[0] = kb_add(state[0], d_sparse_scalar_rc[r]);

        uint32_t old_s0 = state[0];
        uint32_t dot = kb_mul(d_sparse_first_row[r][0], state[0]);
        #pragma unroll
        for (int j = 1; j < P16_WIDTH; j++)
            dot = kb_add(dot, kb_mul(d_sparse_first_row[r][j], state[j]));
        state[0] = dot;

        #pragma unroll
        for (int i = 1; i < P16_WIDTH; i++)
            state[i] = kb_add(state[i], kb_mul(old_s0, d_sparse_v[r][i - 1]));
    }
}

__device__ void poseidon16_permute(uint32_t state[P16_WIDTH]) {
    #pragma unroll
    for (int r = 0; r < P16_HALF_FULL; r++)
        full_round(state, r);
    partial_rounds(state);
    #pragma unroll
    for (int r = 0; r < P16_HALF_FULL; r++)
        full_round(state, P16_HALF_FULL + P16_PARTIAL_ROUNDS + r);
}

// Poseidon16 compress: output = (perm(state) + state)[0..8].
__device__ void poseidon16_compress(const uint32_t input[16], uint32_t output[8]) {
    uint32_t state[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) state[i] = input[i];

    poseidon16_permute(state);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        output[i] = kb_add(state[i], input[i]);
}

// ── PoW grinding kernel ──────────────────────────────────────────────────
// Each thread tests one nonce. The challenger state occupies slots [0..15],
// with the nonce placed in a designated slot (typically state[8], matching
// the WHIR convention of absorbing the nonce into the capacity region).
//
// A valid nonce produces output[0] whose canonical value has >= target_bits
// trailing zero bits.
extern "C" __global__ void pow_grind_kernel(
    const uint32_t* __restrict__ challenger_state,  // 16 elements
    uint32_t* __restrict__ found_nonce,              // output: winning nonce (monty form)
    uint32_t* __restrict__ found_flag,               // atomic: set to 1 when found
    uint32_t target_bits,
    uint32_t nonce_slot,                             // which state slot to place nonce
    uint64_t batch_offset                            // starting nonce for this launch
) {
    if (*found_flag) return;  // early exit if another thread already found it

    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce_val = batch_offset + tid;

    // Build input state: copy challenger state, insert nonce.
    uint32_t input[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) input[i] = challenger_state[i];

    // Convert nonce to Montgomery form and place in designated slot.
    input[nonce_slot] = kb_to_monty((uint32_t)(nonce_val & 0x7FFFFFFFu));

    // Compress.
    uint32_t output[8];
    poseidon16_compress(input, output);

    // Check trailing zero bits of canonical output[0].
    uint32_t canonical = kb_from_monty(output[0]);
    uint32_t mask = (1u << target_bits) - 1u;
    if ((canonical & mask) == 0) {
        // Atomically claim the win.
        if (atomicCAS(found_flag, 0u, 1u) == 0u) {
            *found_nonce = input[nonce_slot];  // store in monty form
        }
    }
}

// ── Field test kernel (for property testing koalabear_field.cuh) ─────────
// Each thread tests field operations on one pair of random inputs.
// Results written to output buffer for host comparison.
extern "C" __global__ void field_test_kernel(
    const uint32_t* __restrict__ a_vals,   // n base field values
    const uint32_t* __restrict__ b_vals,   // n base field values
    uint32_t* __restrict__ results,        // n * 4 results: [add, sub, mul, cube]
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t a = a_vals[tid];
    uint32_t b = b_vals[tid];
    uint32_t* out = results + tid * 4;

    out[0] = kb_add(a, b);
    out[1] = kb_sub(a, b);
    out[2] = kb_mul(a, b);
    out[3] = kb_cube(a);
}

// Quintic extension test: each thread tests mul and square on one pair.
extern "C" __global__ void qe_test_kernel(
    const uint32_t* __restrict__ a_vals,   // n * 5 extension elements
    const uint32_t* __restrict__ b_vals,   // n * 5 extension elements
    uint32_t* __restrict__ results,        // n * 15: [add(5), mul(5), square_a(5)]
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const uint32_t* a = a_vals + tid * 5;
    const uint32_t* b = b_vals + tid * 5;
    uint32_t* out = results + tid * 15;

    qe_add(a, b, out);
    qe_mul(a, b, out + 5);
    qe_square(a, out + 10);
}
