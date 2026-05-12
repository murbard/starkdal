// Trace table operations — CUDA kernels.
//
// Glue kernels that keep data on GPU between heavy compute phases.
// Not compute-intensive themselves, but necessary to avoid PCIe round-trips.

#include "../../field/koalabear_field.cuh"

// ── Access count (histogram with atomics) ────────────────────────────────
// For each element in `column`, atomically increment acc[column[i]].
// This replaces the "TODO parallelize" loop in prove_execution.rs.
//
// column values are field elements used as memory/bytecode addresses.
// We use their canonical (non-Montgomery) form as the index.
//
// Output is plain u32 counts (NOT Montgomery form). The host converts
// to Montgomery field elements afterwards. This avoids the need for
// atomic modular addition on GPU.
extern "C" __global__ void access_count_kernel(
    const uint32_t* __restrict__ column,   // n elements (Montgomery form)
    uint32_t* __restrict__ acc,            // output: plain u32 counts
    uint32_t n,
    uint32_t n_values                      // number of consecutive addresses per access
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t addr = kb_from_monty(column[tid]);
    for (uint32_t j = 0; j < n_values; j++) {
        atomicAdd(&acc[addr + j], 1u);
    }
}

// Simpler variant: single-value access count (n_values=1, no inner loop).
extern "C" __global__ void access_count_simple_kernel(
    const uint32_t* __restrict__ column,
    uint32_t* __restrict__ acc,            // output: plain u32 counts
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t addr = kb_from_monty(column[tid]);
    atomicAdd(&acc[addr], 1u);
}

// ── Polynomial stacking (column copy with offsets) ───────────────────────
// Copy a column of n elements into a destination buffer at a given offset.
extern "C" __global__ void copy_column_kernel(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    uint32_t n,
    uint32_t dst_offset
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[dst_offset + tid] = src[tid];
}

// ── Shifted column (cyclic shift down by 1) ──────────────────────────────
// dst[i] = src[i+1] for i < n-1, dst[n-1] = src[n-1] (last element repeated).
// This matches the "down" shift in AIR constraints.
extern "C" __global__ void shift_down_kernel(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = (tid < n - 1) ? src[tid + 1] : src[n - 1];
}

// ── Bit-reversal permutation ─────────────────────────────────────────────
// In-place bit-reversal of n = 2^log_n elements.
// Each thread handles one swap (only when i < reversed(i) to avoid double-swap).
extern "C" __global__ void bit_reverse_kernel(
    uint32_t* __restrict__ data,
    uint32_t log_n,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Reverse log_n bits of i.
    uint32_t rev = __brev(i) >> (32 - log_n);
    if (i < rev) {
        uint32_t tmp = data[i];
        data[i] = data[rev];
        data[rev] = tmp;
    }
}

// ── Multilinear polynomial evaluation (MLE eval) ────────────────────────
// Evaluates a multilinear polynomial at a single point by iterated folding.
//
// Algorithm: for each variable k (0..log_n):
//   for each pair (j, j + stride) where stride = 1 << k:
//     data[j] = data[j] + point[k] * (data[j + stride] - data[j])
//   (halves the number of active elements each round)
// After log_n rounds, data[0] holds the result.
//
// For base-field data with extension-field point:
//   First fold: base → ext (data is u32, point component is u32[5])
//   Subsequent folds: ext → ext
//
// This kernel handles one fold round. Host calls it log_n times.
// For base→ext first round:
extern "C" __global__ void mle_fold_base_to_ext_kernel(
    const uint32_t* __restrict__ data,     // n_pairs * 2 base elements
    uint32_t* __restrict__ output,          // n_pairs * 5 ext elements
    const uint32_t* __restrict__ r_ext,     // 5 elements (point coordinate)
    uint32_t n_pairs
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t lo = data[2 * j];
    uint32_t hi = data[2 * j + 1];
    uint32_t diff = kb_sub(hi, lo);

    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];

    uint32_t* out = output + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++)
        out[k] = kb_mul(r[k], diff);
    out[0] = kb_add(out[0], lo);
}

// For ext→ext subsequent rounds:
extern "C" __global__ void mle_fold_ext_kernel(
    const uint32_t* __restrict__ data,     // n_pairs * 2 * 5 ext elements
    uint32_t* __restrict__ output,          // n_pairs * 5 ext elements
    const uint32_t* __restrict__ r_ext,     // 5 elements
    uint32_t n_pairs
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];

    const uint32_t* lo_ptr = data + (2 * j) * 5;
    const uint32_t* hi_ptr = data + (2 * j + 1) * 5;

    uint32_t lo[5], hi[5], diff[5], prod[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) { lo[k] = lo_ptr[k]; hi[k] = hi_ptr[k]; }

    qe_sub(hi, lo, diff);
    qe_mul(r, diff, prod);
    qe_add(lo, prod, output + j * 5);
}
