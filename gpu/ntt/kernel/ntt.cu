// Radix-2 "Evals DFT" kernel — matches leanVM's WHIR convention.
//
// The DFT transforms multilinear polynomial evaluations on {0,1}^n into
// evaluations at roots of unity. Layer-by-layer butterfly computation.
//
// Butterfly formula (EvalsButterfly):
//   For twiddle ω:
//     tmp = (x_lo - x_hi) * ω
//     y_hi = x_hi + tmp
//     y_lo = x_hi - tmp
//   For twiddle = 1 (TwiddleFree, i=0 in each block):
//     y_hi = x_lo
//     y_lo = 2 * x_hi - x_lo
//
// Where x_hi = data[i0] (first/upper half of block),
//       x_lo = data[i1] (second/lower half of block).
//
// Processing order: smallest blocks first (stride 1 → 2 → 4 → ... → n/2).

#include "../../field/koalabear_field.cuh"

// ── Single-layer butterfly kernel ────────────────────────────────────────
// Processes one DFT layer. Each thread handles one butterfly pair at one
// column within the row-major matrix.
//
// Parameters:
//   data:     row-major matrix, height * width elements, in-place
//   twiddles: m twiddle factors for this layer (m = block half-size)
//   m:        half-block size (number of twiddles)
//   width:    number of columns in the matrix
//   n_butterflies: total butterflies = (height / 2) * width
extern "C" __global__ void evals_dft_layer_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ twiddles,
    uint32_t m,
    uint32_t width,
    uint32_t n_butterflies
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_butterflies) return;

    // Decompose tid into (butterfly_idx, col).
    uint32_t col = tid % width;
    uint32_t bf_idx = tid / width;  // which butterfly pair (0..height/2)

    // Map butterfly index to row indices.
    // block = bf_idx / m, pos = bf_idx % m
    uint32_t block = bf_idx / m;
    uint32_t pos = bf_idx % m;
    uint32_t row_hi = block * 2 * m + pos;       // upper half row
    uint32_t row_lo = row_hi + m;                  // lower half row

    uint32_t idx_hi = row_hi * width + col;
    uint32_t idx_lo = row_lo * width + col;

    uint32_t x_hi = data[idx_hi];
    uint32_t x_lo = data[idx_lo];

    uint32_t y_hi, y_lo;

    if (pos == 0) {
        // Twiddle-free butterfly: (y_hi, y_lo) = (x_lo, 2*x_hi - x_lo)
        y_hi = x_lo;
        y_lo = kb_sub(kb_double(x_hi), x_lo);
    } else {
        // Standard butterfly with twiddle
        uint32_t tw = twiddles[pos];
        uint32_t tmp = kb_mul(kb_sub(x_lo, x_hi), tw);
        y_hi = kb_add(x_hi, tmp);
        y_lo = kb_sub(x_hi, tmp);
    }

    data[idx_hi] = y_hi;
    data[idx_lo] = y_lo;
}

// ── Fused multi-layer kernel ─────────────────────────────────────────────
// Processes multiple consecutive small layers in shared memory.
// Each thread block handles a contiguous chunk of 2^fused_layers elements,
// performing all fused_layers butterfly passes in shared memory before
// writing back to global memory. This eliminates per-layer kernel launch
// overhead and reduces global memory traffic by fused_layers×.
//
// Twiddle factors for all fused layers are passed as a flat array:
// [layer0_twiddles... | layer1_twiddles... | ... | layerK-1_twiddles...]
// with offsets[k] giving the start of layer k's twiddles.
//
// This kernel handles width=1 (scalar DFT). The single-layer kernel
// handles width>1.
extern "C" __global__ void evals_dft_fused_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ all_twiddles,  // flat: all layers' twiddles
    const uint32_t* __restrict__ tw_offsets,    // fused_layers+1 offsets
    uint32_t fused_layers,                       // number of layers to fuse
    uint32_t chunk_size,                         // 2^fused_layers (elements per block)
    uint32_t n_elements                          // total elements
) {
    extern __shared__ uint32_t smem[];

    uint32_t chunk_id = blockIdx.x;
    uint32_t local_tid = threadIdx.x;
    uint32_t base = chunk_id * chunk_size;

    // Load chunk into shared memory.
    for (uint32_t i = local_tid; i < chunk_size; i += blockDim.x) {
        uint32_t gidx = base + i;
        smem[i] = (gidx < n_elements) ? data[gidx] : 0;
    }
    __syncthreads();

    // Process fused_layers layers, from smallest blocks to largest.
    // Layer k (0-indexed within fused layers) has m = 2^k twiddles
    // and block size 2^(k+1).
    for (uint32_t layer = 0; layer < fused_layers; layer++) {
        uint32_t m = 1u << layer;           // half-block size
        uint32_t block_sz = m * 2;          // full block size
        uint32_t tw_base = tw_offsets[layer];
        uint32_t n_butterflies = chunk_size / 2;

        for (uint32_t bf = local_tid; bf < n_butterflies; bf += blockDim.x) {
            uint32_t block = bf / m;
            uint32_t pos = bf % m;
            uint32_t idx_hi = block * block_sz + pos;
            uint32_t idx_lo = idx_hi + m;

            uint32_t x_hi = smem[idx_hi];
            uint32_t x_lo = smem[idx_lo];
            uint32_t y_hi, y_lo;

            if (pos == 0) {
                y_hi = x_lo;
                y_lo = kb_sub(kb_double(x_hi), x_lo);
            } else {
                uint32_t tw = all_twiddles[tw_base + pos];
                uint32_t tmp = kb_mul(kb_sub(x_lo, x_hi), tw);
                y_hi = kb_add(x_hi, tmp);
                y_lo = kb_sub(x_hi, tmp);
            }

            smem[idx_hi] = y_hi;
            smem[idx_lo] = y_lo;
        }
        __syncthreads();
    }

    // Write back to global memory.
    for (uint32_t i = local_tid; i < chunk_size; i += blockDim.x) {
        uint32_t gidx = base + i;
        if (gidx < n_elements) data[gidx] = smem[i];
    }
}

// ── Fused inverse DFT kernel ─────────────────────────────────────────────
extern "C" __global__ void evals_idft_fused_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ all_twiddles,
    const uint32_t* __restrict__ all_inv_twiddles,
    const uint32_t* __restrict__ tw_offsets,
    uint32_t fused_layers,
    uint32_t chunk_size,
    uint32_t n_elements
) {
    extern __shared__ uint32_t smem[];

    uint32_t chunk_id = blockIdx.x;
    uint32_t local_tid = threadIdx.x;
    uint32_t base = chunk_id * chunk_size;

    for (uint32_t i = local_tid; i < chunk_size; i += blockDim.x) {
        uint32_t gidx = base + i;
        smem[i] = (gidx < n_elements) ? data[gidx] : 0;
    }
    __syncthreads();

    // Inverse: process layers from largest to smallest (reverse order).
    for (int layer = (int)fused_layers - 1; layer >= 0; layer--) {
        uint32_t m = 1u << layer;
        uint32_t block_sz = m * 2;
        uint32_t tw_base = tw_offsets[layer];
        uint32_t n_butterflies = chunk_size / 2;

        for (uint32_t bf = local_tid; bf < n_butterflies; bf += blockDim.x) {
            uint32_t block = bf / m;
            uint32_t pos = bf % m;
            uint32_t idx_hi = block * block_sz + pos;
            uint32_t idx_lo = idx_hi + m;

            uint32_t y_hi = smem[idx_hi];
            uint32_t y_lo = smem[idx_lo];
            uint32_t x_hi, x_lo;

            if (pos == 0) {
                x_hi = kb_halve(kb_add(y_hi, y_lo));
                x_lo = y_hi;
            } else {
                x_hi = kb_halve(kb_add(y_hi, y_lo));
                uint32_t diff_half = kb_halve(kb_sub(y_hi, y_lo));
                x_lo = kb_add(x_hi, kb_mul(diff_half, all_inv_twiddles[tw_base + pos]));
            }

            smem[idx_hi] = x_hi;
            smem[idx_lo] = x_lo;
        }
        __syncthreads();
    }

    for (uint32_t i = local_tid; i < chunk_size; i += blockDim.x) {
        uint32_t gidx = base + i;
        if (gidx < n_elements) data[gidx] = smem[i];
    }
}

// ── Prepare evals for FFT (reorder) ──────────────────────────────────────
// Reorders evaluations for the WHIR DFT convention.
// Maps: out[row * n_cols + col] = evals[(col * block_size + row) >> log_inv_rate]
// where block_size = (n_evals << log_inv_rate) / n_cols.
// Elements beyond n_evals map to zero.
extern "C" __global__ void prepare_evals_for_fft_kernel(
    const uint32_t* __restrict__ evals,
    uint32_t* __restrict__ out,
    uint32_t n_evals,
    uint32_t n_cols,
    uint32_t log_block_size,
    uint32_t log_inv_rate,
    uint32_t out_len
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_len) return;

    uint32_t col = i % n_cols;
    uint32_t row = i / n_cols;
    uint32_t src_index = ((col << log_block_size) + row) >> log_inv_rate;
    out[i] = (src_index < n_evals) ? evals[src_index] : 0;
}

// ── Inverse DFT layer (for round-trip testing) ──────────────────────────
// Inverts the evals butterfly. Given:
//   y_hi = x_hi + (x_lo - x_hi) * ω
//   y_lo = x_hi - (x_lo - x_hi) * ω
// Then:
//   x_hi = (y_hi + y_lo) / 2
//   x_lo = x_hi + (y_hi - x_hi) / ω = x_hi + (y_hi - y_lo) / (2ω)
//
// For twiddle-free (ω=1):
//   y_hi = x_lo, y_lo = 2*x_hi - x_lo
//   x_hi = (y_hi + y_lo) / 2, x_lo = y_hi
extern "C" __global__ void evals_idft_layer_kernel(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ twiddles,
    const uint32_t* __restrict__ inv_twiddles,  // 1/twiddle for each entry
    uint32_t m,
    uint32_t width,
    uint32_t n_butterflies
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_butterflies) return;

    uint32_t col = tid % width;
    uint32_t bf_idx = tid / width;
    uint32_t block = bf_idx / m;
    uint32_t pos = bf_idx % m;
    uint32_t row_hi = block * 2 * m + pos;
    uint32_t row_lo = row_hi + m;

    uint32_t idx_hi = row_hi * width + col;
    uint32_t idx_lo = row_lo * width + col;

    uint32_t y_hi = data[idx_hi];
    uint32_t y_lo = data[idx_lo];

    uint32_t x_hi, x_lo;

    if (pos == 0) {
        // Inverse of twiddle-free: x_hi = (y_hi + y_lo) / 2, x_lo = y_hi
        x_hi = kb_halve(kb_add(y_hi, y_lo));
        x_lo = y_hi;
    } else {
        // Inverse of standard butterfly:
        // x_hi = (y_hi + y_lo) / 2
        // x_lo = x_hi + (y_hi - y_lo) / (2 * ω)
        x_hi = kb_halve(kb_add(y_hi, y_lo));
        uint32_t diff_half = kb_halve(kb_sub(y_hi, y_lo));
        x_lo = kb_add(x_hi, kb_mul(diff_half, inv_twiddles[pos]));
    }

    data[idx_hi] = x_hi;
    data[idx_lo] = x_lo;
}
