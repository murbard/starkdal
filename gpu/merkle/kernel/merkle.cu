// Merkle tree construction — CUDA kernels.
//
// Two-phase construction matching leanVM's WHIR Merkle:
// 1. Leaf hashing: Poseidon16 sponge over each row of the DFT matrix → 8-element digest
// 2. Internal nodes: Poseidon16 compress pairs of sibling digests
//
// Sponge convention (RTL = right-to-left):
//   - State: [capacity(0..8) | rate(8..16)], WIDTH=16, RATE=8, DIGEST=8
//   - First chunk: fill state[15..0] from row data, compress
//   - Subsequent chunks: fill state[15..8] from row data, compress
//   - Output: state[0..8]
//   - "RTL" means data fills state positions from high to low index

#include "../../field/koalabear_field.cuh"

// ── Poseidon16 constants and permutation (same as pow_grind.cu) ──────────
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
    for (int r = 0; r < P16_HALF_FULL; r++) full_round(state, r);
    partial_rounds(state);
    #pragma unroll
    for (int r = 0; r < P16_HALF_FULL; r++)
        full_round(state, P16_HALF_FULL + P16_PARTIAL_ROUNDS + r);
}

// compress_in_place: state = perm(state) + state
__device__ void compress_in_place(uint32_t state[16]) {
    uint32_t initial[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) initial[i] = state[i];
    poseidon16_permute(state);
    #pragma unroll
    for (int i = 0; i < 16; i++) state[i] = kb_add(state[i], initial[i]);
}

// ── Leaf hashing kernel ──────────────────────────────────────────────────
// Each thread hashes one row of the matrix into an 8-element digest.
//
// Sponge (RTL):
//   First chunk: state[15..0] ← first 16 elements of row, then compress
//   Subsequent chunks: state[15..8] ← next 8 elements, then compress
//   Output: state[0..8]
//
// If row_width < 16, the remaining state positions stay as zeros.
extern "C" __global__ void merkle_leaf_hash_kernel(
    const uint32_t* __restrict__ matrix,  // height * row_stride elements (row-major)
    uint32_t* __restrict__ digests,       // height * 8 elements
    uint32_t height,
    uint32_t row_width,                   // number of elements to hash per row (may exceed row_stride)
    uint32_t row_stride                   // actual stride between rows in the matrix
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;

    const uint32_t* row_data = matrix + (uint64_t)row * row_stride;
    uint32_t state[16] = {0};

    // RTL (Right-To-Left): iterate the row in reverse order, matching the CPU convention.
    // Element at reverse position rpos maps to original position (row_width - 1 - rpos).
    // Positions beyond row_stride are zero (padding).
    #define RTL_READ(rpos) (((row_width - 1 - (rpos)) < row_stride) ? row_data[row_width - 1 - (rpos)] : 0u)

    // First chunk: fill state[15..0] RTL from row data (reversed).
    uint32_t pos = 0;
    for (int s = 15; s >= 0 && pos < row_width; s--, pos++) {
        state[s] = RTL_READ(pos);
    }
    compress_in_place(state);

    // Subsequent chunks: fill state[15..8] RTL from row data (reversed).
    while (pos < row_width) {
        for (int s = 15; s >= 8 && pos < row_width; s--, pos++) {
            state[s] = RTL_READ(pos);
        }
        compress_in_place(state);
    }
    #undef RTL_READ

    // Output: state[0..8].
    uint32_t* out = digests + (uint64_t)row * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = state[i];
}

// ── Internal node (binary reduction) kernel ──────────────────────────────
// Each thread compresses one pair of sibling digests.
// input = [left[0..8] || right[0..8]], output = compress(input)[0..8].
extern "C" __global__ void merkle_reduce_kernel(
    const uint32_t* __restrict__ children,  // n_pairs * 2 * 8 elements
    uint32_t* __restrict__ parents,          // n_pairs * 8 elements
    uint32_t n_pairs
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    // Load left and right digests into state[0..8] and state[8..16].
    uint32_t state[16];
    const uint32_t* left = children + (uint64_t)tid * 16;
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = left[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[8 + i] = left[8 + i];

    compress_in_place(state);

    uint32_t* out = parents + (uint64_t)tid * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = state[i];
}

extern "C" __global__ void merkle_eval_rows_at_randomness_kernel(
    const uint32_t* __restrict__ leaf_matrix, // height * full_leaf_base_width words
    const uint32_t* __restrict__ indices,     // n_samples raw indices
    const uint32_t* __restrict__ point_words, // n_coords * 5 ext coordinates
    uint32_t* __restrict__ out,               // n_samples * 5 ext outputs
    uint32_t n_samples,
    uint32_t full_leaf_base_width,
    uint32_t n_coords,
    uint32_t is_extension
) {
    uint32_t sample = blockIdx.x;
    if (sample >= n_samples || threadIdx.x != 0) return;

    extern __shared__ uint32_t scratch[];
    uint32_t index = indices[sample];
    const uint32_t* row = leaf_matrix + (uint64_t)index * full_leaf_base_width;

    if (n_coords == 0) {
        uint32_t* dst = out + sample * 5;
        if (is_extension) {
            #pragma unroll
            for (int k = 0; k < 5; k++) dst[k] = row[k];
        } else {
            dst[0] = row[0];
            #pragma unroll
            for (int k = 1; k < 5; k++) dst[k] = 0;
        }
        return;
    }

    if (!is_extension) {
        uint32_t current_len = full_leaf_base_width;
        for (uint32_t i = 0; i < current_len; i++) scratch[i] = row[i];

        uint32_t* current = scratch + full_leaf_base_width;
        const uint32_t* r0 = point_words;
        uint32_t n_pairs = current_len / 2;
        for (uint32_t j = 0; j < n_pairs; j++) {
            uint32_t lo = scratch[j];
            uint32_t hi = scratch[j + n_pairs];
            uint32_t diff = kb_sub(hi, lo);
            uint32_t* dst = current + j * 5;
            #pragma unroll
            for (int k = 0; k < 5; k++) dst[k] = kb_mul(r0[k], diff);
            dst[0] = kb_add(dst[0], lo);
        }
        current_len = n_pairs;

        for (uint32_t coord = 1; coord < n_coords; coord++) {
            const uint32_t* r = point_words + coord * 5;
            n_pairs = current_len / 2;
            for (uint32_t j = 0; j < n_pairs; j++) {
                uint32_t* lo_ptr = current + j * 5;
                uint32_t* hi_ptr = current + (j + n_pairs) * 5;
                uint32_t lo[5], hi[5], diff[5], prod[5];
                #pragma unroll
                for (int k = 0; k < 5; k++) {
                    lo[k] = lo_ptr[k];
                    hi[k] = hi_ptr[k];
                }
                qe_sub(hi, lo, diff);
                qe_mul(r, diff, prod);
                qe_add(lo, prod, lo_ptr);
            }
            current_len = n_pairs;
        }

        uint32_t* dst = out + sample * 5;
        #pragma unroll
        for (int k = 0; k < 5; k++) dst[k] = current[k];
        return;
    }

    uint32_t current_len = full_leaf_base_width / 5;
    for (uint32_t i = 0; i < full_leaf_base_width; i++) scratch[i] = row[i];

    for (uint32_t coord = 0; coord < n_coords; coord++) {
        const uint32_t* r = point_words + coord * 5;
        uint32_t n_pairs = current_len / 2;
        for (uint32_t j = 0; j < n_pairs; j++) {
            uint32_t* lo_ptr = scratch + j * 5;
            uint32_t* hi_ptr = scratch + (j + n_pairs) * 5;
            uint32_t lo[5], hi[5], diff[5], prod[5];
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                lo[k] = lo_ptr[k];
                hi[k] = hi_ptr[k];
            }
            qe_sub(hi, lo, diff);
            qe_mul(r, diff, prod);
            qe_add(lo, prod, lo_ptr);
        }
        current_len = n_pairs;
    }

    uint32_t* dst = out + sample * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) dst[k] = scratch[k];
}

extern "C" __global__ void merkle_gather_rows_kernel(
    const uint32_t* __restrict__ leaf_matrix,
    const uint32_t* __restrict__ indices,
    uint32_t* __restrict__ out,
    uint32_t n_samples,
    uint32_t row_width
) {
    uint32_t sample = blockIdx.x;
    if (sample >= n_samples) return;

    uint32_t index = indices[sample];
    const uint32_t* row = leaf_matrix + (uint64_t)index * row_width;
    uint32_t* dst = out + (uint64_t)sample * row_width;
    for (uint32_t i = threadIdx.x; i < row_width; i += blockDim.x) {
        dst[i] = row[i];
    }
}

extern "C" __global__ void merkle_gather_sibling_hashes_kernel(
    const uint32_t* __restrict__ layer,
    const uint32_t* __restrict__ indices,
    uint32_t* __restrict__ out,
    uint32_t n_samples,
    uint32_t level
) {
    uint32_t sample = blockIdx.x;
    if (sample >= n_samples || threadIdx.x >= 8) return;

    uint32_t sibling_index = (indices[sample] >> level) ^ 1u;
    out[sample * 8 + threadIdx.x] = layer[(uint64_t)sibling_index * 8 + threadIdx.x];
}
