// Multilinear polynomial folding — CUDA kernels.
//
// Core sumcheck primitive: given evaluations of a multilinear polynomial on
// {0,1}^n, fold one variable at challenge point r to produce evaluations on
// {0,1}^{n-1}.
//
// Formula: out[j] = data[i0] + r * (data[i1] - data[i0])
//
// Three addressing modes:
//   - LSB fold (bit=0): i0 = 2j, i1 = 2j+1 (contiguous pairs)
//   - Half fold: i0 = j, i1 = j + n/2 (used by fold_multilinear)
//   - Arbitrary bit fold: i0 = (j_hi << (bit+1)) | j_lo, i1 = i0 | (1<<bit)
//
// Three type combinations:
//   - base→base: data and r are base field, output is base field
//   - base→ext:  data is base field, r is quintic ext, output is ext (first fold)
//   - ext→ext:   data and r are quintic ext, output is ext (subsequent folds)

#include "../../field/koalabear_field.cuh"

// ── Fold: base field → base field (r is base field) ─────────────────────
// out[j] = data[i0] + r * (data[i1] - data[i0])

// LSB fold (bit=0): pairs are (2j, 2j+1).
extern "C" __global__ void fold_base_lsb_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ output,
    uint32_t r,
    uint32_t n_pairs
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t lo = data[2 * j];
    uint32_t hi = data[2 * j + 1];
    output[j] = kb_add(lo, kb_mul(r, kb_sub(hi, lo)));
}

// Half fold: pairs are (j, j + n_pairs).
extern "C" __global__ void fold_base_half_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ output,
    uint32_t r,
    uint32_t n_pairs
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t lo = data[j];
    uint32_t hi = data[j + n_pairs];
    output[j] = kb_add(lo, kb_mul(r, kb_sub(hi, lo)));
}

// Arbitrary bit fold.
extern "C" __global__ void fold_base_at_bit_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ output,
    uint32_t r,
    uint32_t n_pairs,
    uint32_t bit
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t stride = 1u << bit;
    uint32_t lo_mask = stride - 1;
    uint32_t i_hi = j >> bit;
    uint32_t i_lo = j & lo_mask;
    uint32_t i0 = (i_hi << (bit + 1)) | i_lo;
    uint32_t i1 = i0 | stride;

    uint32_t lo = data[i0];
    uint32_t hi = data[i1];
    output[j] = kb_add(lo, kb_mul(r, kb_sub(hi, lo)));
}

// ── Fold: base field → quintic extension (r is ext, first fold) ─────────
// out[j] (5 elements) = embed(data[i0]) + r * embed(data[i1] - data[i0])
// where embed(x) = [x, 0, 0, 0, 0]

// LSB fold.
extern "C" __global__ void fold_base_to_ext_lsb_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ output,      // n_pairs * 5 elements
    const uint32_t* __restrict__ r_ext, // 5 elements
    uint32_t n_pairs
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];

    uint32_t lo = data[2 * j];
    uint32_t hi = data[2 * j + 1];
    uint32_t diff = kb_sub(hi, lo);

    // r * diff (base scalar times ext vector = scale each component)
    uint32_t* out = output + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++)
        out[k] = kb_mul(r[k], diff);
    out[0] = kb_add(out[0], lo);
}

// Half fold.
extern "C" __global__ void fold_base_to_ext_half_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ r_ext,
    uint32_t n_pairs
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];

    uint32_t lo = data[j];
    uint32_t hi = data[j + n_pairs];
    uint32_t diff = kb_sub(hi, lo);

    uint32_t* out = output + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++)
        out[k] = kb_mul(r[k], diff);
    out[0] = kb_add(out[0], lo);
}

// Arbitrary bit fold.
extern "C" __global__ void fold_base_to_ext_at_bit_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ r_ext,
    uint32_t n_pairs,
    uint32_t bit
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];

    uint32_t stride = 1u << bit;
    uint32_t lo_mask = stride - 1;
    uint32_t i0 = ((j >> bit) << (bit + 1)) | (j & lo_mask);
    uint32_t i1 = i0 | stride;

    uint32_t lo = data[i0];
    uint32_t hi = data[i1];
    uint32_t diff = kb_sub(hi, lo);

    uint32_t* out = output + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++)
        out[k] = kb_mul(r[k], diff);
    out[0] = kb_add(out[0], lo);
}

// ── Fold: ext → ext (both data and r are quintic extension) ─────────────
// out[j] = data[i0] + r * (data[i1] - data[i0])
// All elements are 5 u32s wide.

// LSB fold.
extern "C" __global__ void fold_ext_lsb_kernel(
    const uint32_t* __restrict__ data,   // n_pairs * 2 * 5 elements
    uint32_t* __restrict__ output,        // n_pairs * 5 elements
    const uint32_t* __restrict__ r_ext,   // 5 elements
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

// Half fold.
extern "C" __global__ void fold_ext_half_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ r_ext,
    uint32_t n_pairs
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];

    const uint32_t* lo_ptr = data + j * 5;
    const uint32_t* hi_ptr = data + (j + n_pairs) * 5;

    uint32_t lo[5], hi[5], diff[5], prod[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) { lo[k] = lo_ptr[k]; hi[k] = hi_ptr[k]; }

    qe_sub(hi, lo, diff);
    qe_mul(r, diff, prod);
    qe_add(lo, prod, output + j * 5);
}

// Arbitrary bit fold.
extern "C" __global__ void fold_ext_at_bit_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ r_ext,
    uint32_t n_pairs,
    uint32_t bit
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];

    uint32_t stride = 1u << bit;
    uint32_t lo_mask = stride - 1;
    uint32_t i0 = ((j >> bit) << (bit + 1)) | (j & lo_mask);
    uint32_t i1 = i0 | stride;

    uint32_t lo[5], hi[5], diff[5], prod[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) { lo[k] = data[i0 * 5 + k]; hi[k] = data[i1 * 5 + k]; }

    qe_sub(hi, lo, diff);
    qe_mul(r, diff, prod);
    qe_add(lo, prod, output + j * 5);
}
