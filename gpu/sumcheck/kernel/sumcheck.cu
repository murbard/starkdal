// Sumcheck computation — CUDA kernels.
//
// Two kernel types:
// 1. Product sumcheck: c0 = Σ a_lo·b_lo, c2 = Σ a_hi·b_hi (degree-2).
//    Used in every WHIR round for polynomial commitment.
//
// 2. Generic column interpolation + constraint accumulation:
//    For each row-pair (i0, i1), interpolate columns at z=0,2,3,...,degree,
//    evaluate a weighted sum, multiply by eq_factor, accumulate.
//    Used for AIR sumcheck rounds.
//
// Both use block-level parallel reduction.

#include "../../field/koalabear_field.cuh"

// ── Warp-level reduction ─────────────────────────────────────────────────

__device__ __forceinline__ void warp_reduce_add_ext(uint32_t val[5]) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            uint32_t other = __shfl_down_sync(0xFFFFFFFF, val[k], offset);
            val[k] = kb_add(val[k], other);
        }
    }
}

// ── Product sumcheck kernel (degree 2) ───────────────────────────────────
// Computes c0 = Σ_i pol_a[i] * pol_b[i]  (lower halves)
//          c2 = Σ_i pol_a[i+half] * pol_b[i+half]  (upper halves)
// where pol_a is base field, pol_b is quintic ext.
//
// Each thread handles one pair. Block-reduce to partial sums.
// Output: n_blocks partial sums for c0 and c2 (each 5 elements).
extern "C" __global__ void product_sumcheck_base_ext_kernel(
    const uint32_t* __restrict__ pol_a,     // n elements (base field)
    const uint32_t* __restrict__ pol_b,     // n * 5 elements (ext field)
    uint32_t* __restrict__ partial_c0,      // n_blocks * 5 elements
    uint32_t* __restrict__ partial_c2,      // n_blocks * 5 elements
    uint32_t half                           // n / 2
) {
    extern __shared__ uint32_t smem[];
    // Layout: smem[0..blockDim.x*5] = c0, smem[blockDim.x*5..blockDim.x*10] = c2

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    uint32_t c0_local[5] = {0, 0, 0, 0, 0};
    uint32_t c2_local[5] = {0, 0, 0, 0, 0};

    if (gid < half) {
        // c0: a_lo * b_lo
        uint32_t a_lo = pol_a[gid];
        uint32_t a_hi = pol_a[gid + half];
        const uint32_t* b_lo = pol_b + gid * 5;
        const uint32_t* b_hi = pol_b + (gid + half) * 5;

        #pragma unroll
        for (int k = 0; k < 5; k++)
            c0_local[k] = kb_mul(a_lo, b_lo[k]);

        // c2: (a_hi - a_lo) * (b_hi[k] - b_lo[k])
        uint32_t a_diff = kb_sub(a_hi, a_lo);
        #pragma unroll
        for (int k = 0; k < 5; k++)
            c2_local[k] = kb_mul(a_diff, kb_sub(b_hi[k], b_lo[k]));
    }

    // Warp-level reduction.
    warp_reduce_add_ext(c0_local);
    warp_reduce_add_ext(c2_local);

    // Write warp results to shared memory.
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            smem[warp_id * 5 + k] = c0_local[k];
            smem[(blockDim.x / 32 + warp_id) * 5 + k] = c2_local[k];
        }
    }
    __syncthreads();

    // Final reduction across warps (first warp only).
    if (tid < 32) {
        uint32_t n_warps = blockDim.x / 32;
        uint32_t final_c0[5] = {0, 0, 0, 0, 0};
        uint32_t final_c2[5] = {0, 0, 0, 0, 0};
        for (uint32_t w = tid; w < n_warps; w += 32) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                final_c0[k] = kb_add(final_c0[k], smem[w * 5 + k]);
                final_c2[k] = kb_add(final_c2[k], smem[(n_warps + w) * 5 + k]);
            }
        }
        warp_reduce_add_ext(final_c0);
        warp_reduce_add_ext(final_c2);
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                partial_c0[blockIdx.x * 5 + k] = final_c0[k];
                partial_c2[blockIdx.x * 5 + k] = final_c2[k];
            }
        }
    }
}

// ── Product sumcheck kernel: ext × ext ───────────────────────────────────
extern "C" __global__ void product_sumcheck_ext_ext_kernel(
    const uint32_t* __restrict__ pol_a,     // n * 5 elements (ext field)
    const uint32_t* __restrict__ pol_b,     // n * 5 elements (ext field)
    uint32_t* __restrict__ partial_c0,
    uint32_t* __restrict__ partial_c2,
    uint32_t half
) {
    extern __shared__ uint32_t smem[];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    uint32_t c0_local[5] = {0, 0, 0, 0, 0};
    uint32_t c2_local[5] = {0, 0, 0, 0, 0};

    if (gid < half) {
        const uint32_t* a_lo_p = pol_a + gid * 5;
        const uint32_t* b_lo_p = pol_b + gid * 5;
        const uint32_t* a_hi_p = pol_a + (gid + half) * 5;
        const uint32_t* b_hi_p = pol_b + (gid + half) * 5;

        uint32_t a_lo[5], b_lo[5], a_hi[5], b_hi[5];
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            a_lo[k] = a_lo_p[k]; b_lo[k] = b_lo_p[k];
            a_hi[k] = a_hi_p[k]; b_hi[k] = b_hi_p[k];
        }

        // c0 = a_lo * b_lo
        qe_mul(a_lo, b_lo, c0_local);

        // c2 = (a_hi - a_lo) * (b_hi - b_lo)
        uint32_t a_diff[5], b_diff[5];
        qe_sub(a_hi, a_lo, a_diff);
        qe_sub(b_hi, b_lo, b_diff);
        qe_mul(a_diff, b_diff, c2_local);
    }

    warp_reduce_add_ext(c0_local);
    warp_reduce_add_ext(c2_local);

    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            smem[warp_id * 5 + k] = c0_local[k];
            smem[(blockDim.x / 32 + warp_id) * 5 + k] = c2_local[k];
        }
    }
    __syncthreads();

    if (tid < 32) {
        uint32_t n_warps = blockDim.x / 32;
        uint32_t final_c0[5] = {0, 0, 0, 0, 0};
        uint32_t final_c2[5] = {0, 0, 0, 0, 0};
        for (uint32_t w = tid; w < n_warps; w += 32) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                final_c0[k] = kb_add(final_c0[k], smem[w * 5 + k]);
                final_c2[k] = kb_add(final_c2[k], smem[(n_warps + w) * 5 + k]);
            }
        }
        warp_reduce_add_ext(final_c0);
        warp_reduce_add_ext(final_c2);
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                partial_c0[blockIdx.x * 5 + k] = final_c0[k];
                partial_c2[blockIdx.x * 5 + k] = final_c2[k];
            }
        }
    }
}

// ── Block-level reduction of partial sums ────────────────────────────────
// Sums n_blocks partial ext-field values into a single result.
extern "C" __global__ void reduce_ext_kernel(
    const uint32_t* __restrict__ partials,  // n_blocks * 5 elements
    uint32_t* __restrict__ result,           // 5 elements
    uint32_t n_blocks
) {
    extern __shared__ uint32_t smem[];
    uint32_t tid = threadIdx.x;

    uint32_t acc[5] = {0, 0, 0, 0, 0};
    for (uint32_t i = tid; i < n_blocks; i += blockDim.x) {
        #pragma unroll
        for (int k = 0; k < 5; k++)
            acc[k] = kb_add(acc[k], partials[i * 5 + k]);
    }

    warp_reduce_add_ext(acc);

    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 5; k++)
            smem[warp_id * 5 + k] = acc[k];
    }
    __syncthreads();

    if (tid < 32) {
        uint32_t n_warps = blockDim.x / 32;
        uint32_t final_acc[5] = {0, 0, 0, 0, 0};
        for (uint32_t w = tid; w < n_warps; w += 32) {
            #pragma unroll
            for (int k = 0; k < 5; k++)
                final_acc[k] = kb_add(final_acc[k], smem[w * 5 + k]);
        }
        warp_reduce_add_ext(final_acc);
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 5; k++)
                result[k] = final_acc[k];
        }
    }
}

// ── Packed→flat transpose for extension field ────────────────────────────
// Converts SIMD-packed layout to flat scalar layout.
// Packed: [packed_idx * (DIM * WIDTH) + comp * WIDTH + lane]
// Flat:   [(packed_idx * WIDTH + lane) * DIM + comp]
extern "C" __global__ void transpose_packed_ext_kernel(
    const uint32_t* __restrict__ packed,   // n_packed * DIM * WIDTH elements
    uint32_t* __restrict__ flat,            // n_packed * DIM * WIDTH elements
    uint32_t n_packed,
    uint32_t dim,       // extension dimension (5)
    uint32_t width      // SIMD width (4 for NEON)
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n_packed * width;  // total scalar elements
    if (tid >= total) return;

    uint32_t pi = tid / width;    // packed index
    uint32_t lane = tid % width;

    for (uint32_t comp = 0; comp < dim; comp++) {
        flat[tid * dim + comp] = packed[pi * (dim * width) + comp * width + lane];
    }
}

// ── Eq polynomial kernel ─────────────────────────────────────────────────
// Builds eq(point, x) for all x in {0,1}^n.
// Algorithm: start with out[0] = 1. For each variable k (0..n_vars):
//   For each existing element j (0..2^k):
//     out[2j]   = out[j] * (1 - point[k])
//     out[2j+1] = out[j] * point[k]
// After n_vars steps, out has 2^n_vars extension field elements.
//
// This kernel does ONE step of the expansion. The host calls it n_vars times.
// Input: src (2^k ext elements), point_k (1 ext element, 5 u32s).
// Output: dst (2^(k+1) ext elements).
extern "C" __global__ void eq_expand_step_kernel(
    const uint32_t* __restrict__ src,    // 2^k * 5 ext elements
    uint32_t* __restrict__ dst,           // 2^(k+1) * 5 ext elements
    const uint32_t* __restrict__ point_k, // 5 u32s (ext field coordinate)
    uint32_t n_src                        // 2^k
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_src) return;

    uint32_t pk[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) pk[i] = point_k[i];

    // one_minus_pk = 1 - pk (ext field subtraction)
    uint32_t one_minus_pk[5];
    one_minus_pk[0] = kb_sub(KB_MONTY_ONE, pk[0]);
    for (int i = 1; i < 5; i++) one_minus_pk[i] = kb_neg(pk[i]);

    uint32_t src_val[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) src_val[i] = src[j * 5 + i];

    // dst[2j] = src[j] * (1 - pk)
    uint32_t even[5];
    qe_mul(src_val, one_minus_pk, even);
    #pragma unroll
    for (int i = 0; i < 5; i++) dst[(2 * j) * 5 + i] = even[i];

    // dst[2j+1] = src[j] * pk
    uint32_t odd[5];
    qe_mul(src_val, pk, odd);
    #pragma unroll
    for (int i = 0; i < 5; i++) dst[(2 * j + 1) * 5 + i] = odd[i];
}

// Accumulate: weights[x] += scalar * eq_val[x] for all x.
// Both weights and eq_val are n * 5 ext elements.
extern "C" __global__ void eq_accumulate_kernel(
    uint32_t* __restrict__ weights,       // n * 5 ext, modified in place
    const uint32_t* __restrict__ eq_val,  // n * 5 ext
    const uint32_t* __restrict__ scalar,  // 5 ext
    uint32_t n
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    uint32_t s[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) s[i] = scalar[i];

    uint32_t eq[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) eq[i] = eq_val[j * 5 + i];

    uint32_t prod[5];
    qe_mul(s, eq, prod);

    #pragma unroll
    for (int i = 0; i < 5; i++)
        weights[j * 5 + i] = kb_add(weights[j * 5 + i], prod[i]);
}

// Accumulate with offset: weights[offset + j] += scalar * eq_val[j].
extern "C" __global__ void eq_accumulate_offset_kernel(
    uint32_t* __restrict__ weights,       // total_n * 5 ext, modified in place
    const uint32_t* __restrict__ eq_val,  // n * 5 ext
    const uint32_t* __restrict__ scalar,  // 5 ext
    uint32_t offset,                       // element offset into weights
    uint32_t n
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    uint32_t s[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) s[i] = scalar[i];

    uint32_t eq[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) eq[i] = eq_val[j * 5 + i];

    uint32_t prod[5];
    qe_mul(s, eq, prod);

    uint32_t dst = (offset + j) * 5;
    #pragma unroll
    for (int i = 0; i < 5; i++)
        weights[dst + i] = kb_add(weights[dst + i], prod[i]);
}

// ── AIR constraint evaluation kernel ─────────────────────────────────────
// For each row pair (i0, i1), evaluates constraints at z=0 and z=2,
// multiplied by alpha powers, accumulated into partial sums.
// This handles the Execution table (13 constraints, 20 up + 2 down columns).
//
// For each pair:
//   point[k] = col[i0][k]  (column value at lower row)
//   diff[k] = col[i1][k] - col[i0][k]
//   At z=0: eval constraints with point values
//   At z=2: eval constraints with (point + 2*diff) values
//   result[z] += sum(alpha[c] * constraint[c]) * eq_factor
//
// partial_sums: n_blocks * (degree+1) * 5 ext field elements.
#include "air_constraints.cuh"

extern "C" __global__ void air_sumcheck_execution_kernel(
    const uint32_t* __restrict__ columns,   // n_cols * n_rows base elements (col-major)
    const uint32_t* __restrict__ down_cols,  // 2 * n_rows base elements (col-major)
    const uint32_t* __restrict__ eq_factor,  // n_pairs * 5 ext elements
    const uint32_t* __restrict__ alphas,     // 13 * 5 ext elements (alpha powers)
    uint32_t* __restrict__ partial_z0,       // n_blocks * 5 ext
    uint32_t* __restrict__ partial_z2,       // n_blocks * 5 ext
    uint32_t n_rows,
    uint32_t n_pairs                         // n_rows / 2
) {
    extern __shared__ uint32_t smem[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    uint32_t z0_acc[5] = {0,0,0,0,0};
    uint32_t z2_acc[5] = {0,0,0,0,0};

    if (gid < n_pairs) {
        uint32_t i0 = gid;
        uint32_t i1 = gid + n_pairs;

        // Load column values at (i0, i1).
        uint32_t up_0[20], up_2[20];
        uint32_t down_0[2], down_2[2];

        for (int c = 0; c < 20; c++) {
            uint32_t v0 = columns[c * n_rows + i0];
            uint32_t v1 = columns[c * n_rows + i1];
            up_0[c] = v0;
            // z=2: point + 2*diff = point + 2*(v1-v0) = 2*v1 - v0
            up_2[c] = kb_sub(kb_double(v1), v0);
        }
        for (int c = 0; c < 2; c++) {
            uint32_t v0 = down_cols[c * n_rows + i0];
            uint32_t v1 = down_cols[c * n_rows + i1];
            down_0[c] = v0;
            down_2[c] = kb_sub(kb_double(v1), v0);
        }

        // Evaluate constraints at z=0 and z=2.
        uint32_t c0[13], c2[13];
        eval_execution_air(up_0, down_0, c0);
        eval_execution_air(up_2, down_2, c2);

        // Accumulate: z_acc += sum(alphas[c] * constraints[c]) * eq_factor[pair]
        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];

        uint32_t weighted_z0[5] = {0,0,0,0,0};
        uint32_t weighted_z2[5] = {0,0,0,0,0};
        for (int c = 0; c < 13; c++) {
            // alpha[c] is ext field, constraint is base field.
            // Product: alpha[c] * constraint[c] (base × ext = ext)
            for (int k = 0; k < 5; k++) {
                uint32_t a = alphas[c * 5 + k];
                weighted_z0[k] = kb_add(weighted_z0[k], kb_mul(a, c0[c]));
                weighted_z2[k] = kb_add(weighted_z2[k], kb_mul(a, c2[c]));
            }
        }

        // Multiply by eq factor (ext × ext = ext).
        uint32_t prod_z0[5], prod_z2[5];
        qe_mul(weighted_z0, eq, prod_z0);
        qe_mul(weighted_z2, eq, prod_z2);

        for (int k = 0; k < 5; k++) {
            z0_acc[k] = prod_z0[k];
            z2_acc[k] = prod_z2[k];
        }
    }

    // Warp-level reduction.
    warp_reduce_add_ext(z0_acc);
    warp_reduce_add_ext(z2_acc);

    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) {
        for (int k = 0; k < 5; k++) {
            smem[warp_id * 5 + k] = z0_acc[k];
            smem[(blockDim.x / 32 + warp_id) * 5 + k] = z2_acc[k];
        }
    }
    __syncthreads();

    if (tid < 32) {
        uint32_t n_warps = blockDim.x / 32;
        uint32_t final_z0[5] = {0,0,0,0,0};
        uint32_t final_z2[5] = {0,0,0,0,0};
        for (uint32_t w = tid; w < n_warps; w += 32) {
            for (int k = 0; k < 5; k++) {
                final_z0[k] = kb_add(final_z0[k], smem[w * 5 + k]);
                final_z2[k] = kb_add(final_z2[k], smem[(n_warps + w) * 5 + k]);
            }
        }
        warp_reduce_add_ext(final_z0);
        warp_reduce_add_ext(final_z2);
        if (tid == 0) {
            for (int k = 0; k < 5; k++) {
                partial_z0[blockIdx.x * 5 + k] = final_z0[k];
                partial_z2[blockIdx.x * 5 + k] = final_z2[k];
            }
        }
    }
}

// ── Split-eq update kernel ───────────────────────────────────────────────
// After a sumcheck round with challenge r, update the eq factor:
// For each pair (j, j+stride): eq[j] = eq[j] * (1 - r) + eq[j+stride] * r
// This folds one variable of the eq polynomial.
extern "C" __global__ void split_eq_fold_kernel(
    const uint32_t* __restrict__ eq_in,     // n * 5 ext elements
    uint32_t* __restrict__ eq_out,           // (n/2) * 5 ext elements
    const uint32_t* __restrict__ r_ext,      // 5 elements (challenge)
    uint32_t n_pairs
) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_pairs) return;

    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];

    const uint32_t* lo = eq_in + (2 * j) * 5;
    const uint32_t* hi = eq_in + (2 * j + 1) * 5;

    uint32_t lo_v[5], hi_v[5], diff[5], prod[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) { lo_v[k] = lo[k]; hi_v[k] = hi[k]; }

    // out = lo + r * (hi - lo)
    qe_sub(hi_v, lo_v, diff);
    qe_mul(r, diff, prod);
    qe_add(lo_v, prod, eq_out + j * 5);
}
