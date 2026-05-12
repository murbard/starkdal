// Logup fingerprint computation — CUDA kernels.
//
// Computes fingerprints for the Logup/GKR lookup argument:
//   denom[i] = c - fingerprint(data_columns[i], alphas)
// where fingerprint(values, alphas) = Σ_j values[j] * alphas[j]
//
// Also: endianness reorder (bit-reversal within chunks).

#include "../../field/koalabear_field.cuh"

// ── Fingerprint kernel ───────────────────────────────────────────────────
// For each row i, compute:
//   fp = Σ_j (column_j[i] * alphas[j])     (base field dot product, result in ext)
//   denom[i] = c - fp                        (ext field subtraction)
//
// columns: n_cols column pointers, each pointing to n_rows base field elements.
// Since we can't pass arrays of pointers easily, columns are passed as a
// single flat buffer: columns[col * n_rows + row].
// alphas: n_cols × 5 (ext field per column).
// c: 5 elements (ext field challenge).
// denom: n_rows × 5 (ext field output).
extern "C" __global__ void fingerprint_kernel(
    const uint32_t* __restrict__ columns,   // n_cols * n_rows base elements (col-major)
    const uint32_t* __restrict__ alphas,    // n_cols * 5 ext elements
    const uint32_t* __restrict__ c,         // 5 ext elements
    uint32_t* __restrict__ denom,           // n_rows * 5 ext elements
    uint32_t n_rows,
    uint32_t n_cols
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Accumulate fingerprint in extension field.
    uint32_t fp[5] = {0, 0, 0, 0, 0};

    for (uint32_t col = 0; col < n_cols; col++) {
        uint32_t val = columns[col * n_rows + row];
        const uint32_t* alpha = alphas + col * 5;

        // fp += val * alpha (base × ext = scale each ext component)
        #pragma unroll
        for (int k = 0; k < 5; k++)
            fp[k] = kb_add(fp[k], kb_mul(val, alpha[k]));
    }

    // denom = c - fp
    uint32_t* out = denom + row * 5;
    uint32_t c_local[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) c_local[k] = c[k];

    qe_sub(c_local, fp, out);
}

// ── Endianness reorder (bit-reversal within chunks) ──────────────────────
// Reorders elements within power-of-two chunks using bit-reversal.
// This matches the `src_idx` function in logup.rs:
//   new_idx = (idx & ~mask) | bit_reverse(idx & mask, chunk_log)
extern "C" __global__ void endianness_reorder_kernel(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    uint32_t n,
    uint32_t chunk_log   // bit-reversal width within each chunk
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t mask = (1u << chunk_log) - 1;
    uint32_t lo = idx & mask;
    uint32_t hi = idx & ~mask;

    // Reverse chunk_log bits of lo.
    uint32_t rev = __brev(lo) >> (32 - chunk_log);
    uint32_t src_idx = hi | rev;

    dst[idx] = src[src_idx];
}

// ── Endianness reorder for ext field (5 u32s per element) ────────────────
extern "C" __global__ void endianness_reorder_ext_kernel(
    const uint32_t* __restrict__ src,   // n * 5 elements
    uint32_t* __restrict__ dst,          // n * 5 elements
    uint32_t n,
    uint32_t chunk_log
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t mask = (1u << chunk_log) - 1;
    uint32_t lo = idx & mask;
    uint32_t hi = idx & ~mask;
    uint32_t rev = __brev(lo) >> (32 - chunk_log);
    uint32_t src_idx = hi | rev;

    #pragma unroll
    for (int k = 0; k < 5; k++)
        dst[idx * 5 + k] = src[src_idx * 5 + k];
}
