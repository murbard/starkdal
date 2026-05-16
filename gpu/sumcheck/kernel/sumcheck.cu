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
#include "fiat_shamir.cuh"

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

// ── Sum of quotients kernel: Σ nums[i] / dens[i] ────────────────────────
extern "C" __global__ void sum_quotients_ext_kernel(
    const uint32_t* __restrict__ nums,      // n * 5 elements (ext field)
    const uint32_t* __restrict__ dens,      // n * 5 elements (ext field)
    uint32_t* __restrict__ partial_sums,    // n_blocks * 5 elements
    uint32_t n
) {
    extern __shared__ uint32_t smem[];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    uint32_t acc[5] = {0, 0, 0, 0, 0};
    if (gid < n) {
        uint32_t num[5], den[5], den_inv[5];
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            num[k] = nums[gid * 5 + k];
            den[k] = dens[gid * 5 + k];
        }
        qe_inv(den, den_inv);
        qe_mul(num, den_inv, acc);
    }

    warp_reduce_add_ext(acc);

    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            smem[warp_id * 5 + k] = acc[k];
        }
    }
    __syncthreads();

    if (tid < 32) {
        uint32_t n_warps = blockDim.x / 32;
        uint32_t final_acc[5] = {0, 0, 0, 0, 0};
        for (uint32_t w = tid; w < n_warps; w += 32) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                final_acc[k] = kb_add(final_acc[k], smem[w * 5 + k]);
            }
        }
        warp_reduce_add_ext(final_acc);
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                partial_sums[blockIdx.x * 5 + k] = final_acc[k];
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

extern "C" __global__ void expand_univariate_points_kernel(
    const uint32_t* __restrict__ univariate_points, // n_points * 5 ext elements
    uint32_t* __restrict__ expanded_points,         // n_points * n_vars * 5 ext elements
    uint32_t n_points,
    uint32_t n_vars
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    const uint32_t* src = univariate_points + idx * 5;
    uint32_t* dst = expanded_points + idx * n_vars * 5;

    uint32_t cur[5];
    #pragma unroll
    for (int i = 0; i < 5; i++) cur[i] = src[i];

    for (uint32_t var = 0; var < n_vars; var++) {
        #pragma unroll
        for (int i = 0; i < 5; i++) dst[var * 5 + i] = cur[i];
        uint32_t next[5];
        qe_square(cur, next);
        #pragma unroll
        for (int i = 0; i < 5; i++) cur[i] = next[i];
    }
}

extern "C" __global__ void expand_sampled_base_query_points_kernel(
    const uint32_t* __restrict__ sampled_words, // n_samples base elements in Montgomery form
    uint32_t* __restrict__ expanded_points,     // n_samples * n_vars * 5 ext elements
    uint32_t* __restrict__ indices_out,         // n_samples raw integer indices
    uint32_t n_samples,
    uint32_t bits,
    uint32_t domain_gen,
    uint32_t n_vars
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    uint32_t mask = (bits == 0) ? 0u : ((1u << bits) - 1u);
    uint32_t raw_index = kb_from_monty(sampled_words[idx]) & mask;
    indices_out[idx] = raw_index;

    if (n_vars == 0) return;

    uint32_t cur = kb_exp(domain_gen, raw_index);
    uint32_t* dst = expanded_points + idx * n_vars * 5;
    for (uint32_t var = 0; var < n_vars; var++) {
        dst[var * 5 + 0] = cur;
        #pragma unroll
        for (int k = 1; k < 5; k++) dst[var * 5 + k] = 0;
        cur = kb_square(cur);
    }
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
#include "air_ext_op.cuh"
#include "air_poseidon.cuh"
#include "air_constraints_ext.cuh"

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

// ── Multi-z AIR constraint evaluation kernel (Execution table) ──────────
// Evaluates constraints at z=0, 2, 3, 4 (degree-1 = 4 evaluations for degree 5).
// For each pair (i0, i1):
//   point(z) = v0 + z*(v1 - v0) for each column
//   Evaluate 13 constraints at z, weight by alpha powers and eq factor.
// Output: 4 partial sums (one per z-value), each n_blocks * 5 ext elements.
extern "C" __global__ void air_execution_multi_z_kernel(
    const uint32_t* __restrict__ columns,   // 20 * n_rows base (col-major)
    const uint32_t* __restrict__ down_cols,  // 2 * n_rows base (col-major)
    const uint32_t* __restrict__ eq_factor,  // n_pairs * 5 ext
    const uint32_t* __restrict__ alphas,     // 13 * 5 ext
    uint32_t* __restrict__ partial_z0,       // n_blocks * 5 ext
    uint32_t* __restrict__ partial_z2,       // n_blocks * 5 ext
    uint32_t* __restrict__ partial_z3,       // n_blocks * 5 ext
    uint32_t* __restrict__ partial_z4,       // n_blocks * 5 ext
    uint32_t n_rows,
    uint32_t n_pairs
) {
    extern __shared__ uint32_t smem[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    uint32_t acc_z[4][5];
    for (int z = 0; z < 4; z++)
        for (int k = 0; k < 5; k++) acc_z[z][k] = 0;

    if (gid < n_pairs) {
        uint32_t i0 = gid;
        uint32_t i1 = gid + n_pairs;

        // Load columns and compute diff.
        uint32_t v0_up[20], diff_up[20];
        uint32_t v0_dn[2], diff_dn[2];
        for (int c = 0; c < 20; c++) {
            v0_up[c] = columns[c * n_rows + i0];
            uint32_t v1 = columns[c * n_rows + i1];
            diff_up[c] = kb_sub(v1, v0_up[c]);
        }
        for (int c = 0; c < 2; c++) {
            v0_dn[c] = down_cols[c * n_rows + i0];
            uint32_t v1 = down_cols[c * n_rows + i1];
            diff_dn[c] = kb_sub(v1, v0_dn[c]);
        }

        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];

        // z=0: point = v0
        {
            uint32_t constraints[13];
            eval_execution_air(v0_up, v0_dn, constraints);
            uint32_t weighted[5] = {0,0,0,0,0};
            for (int c = 0; c < 13; c++)
                for (int k = 0; k < 5; k++)
                    weighted[k] = kb_add(weighted[k], kb_mul(alphas[c*5+k], constraints[c]));
            uint32_t prod[5]; qe_mul(weighted, eq, prod);
            for (int k = 0; k < 5; k++) acc_z[0][k] = prod[k];
        }

        // z=2,3,4: point = v0 + z*diff
        uint32_t up[20], dn[2];
        // Start at z=1 (add diff once), but we skip z=1.
        for (int c = 0; c < 20; c++) up[c] = kb_add(v0_up[c], diff_up[c]); // z=1 (skip)
        for (int c = 0; c < 2; c++) dn[c] = kb_add(v0_dn[c], diff_dn[c]);

        for (int zi = 0; zi < 3; zi++) { // z=2,3,4
            for (int c = 0; c < 20; c++) up[c] = kb_add(up[c], diff_up[c]);
            for (int c = 0; c < 2; c++) dn[c] = kb_add(dn[c], diff_dn[c]);
            uint32_t constraints[13];
            eval_execution_air(up, dn, constraints);
            uint32_t weighted[5] = {0,0,0,0,0};
            for (int c = 0; c < 13; c++)
                for (int k = 0; k < 5; k++)
                    weighted[k] = kb_add(weighted[k], kb_mul(alphas[c*5+k], constraints[c]));
            uint32_t prod[5]; qe_mul(weighted, eq, prod);
            for (int k = 0; k < 5; k++) acc_z[1+zi][k] = prod[k];
        }
    }

    // Warp + block reduction for all 4 z-values.
    for (int z = 0; z < 4; z++) warp_reduce_add_ext(acc_z[z]);

    uint32_t warp_id = tid / 32, lane = tid % 32;
    uint32_t n_warps = blockDim.x / 32;
    if (lane == 0) {
        for (int z = 0; z < 4; z++)
            for (int k = 0; k < 5; k++)
                smem[(z * n_warps + warp_id) * 5 + k] = acc_z[z][k];
    }
    __syncthreads();

    if (tid < 32) {
        uint32_t final_z[4][5];
        for (int z = 0; z < 4; z++)
            for (int k = 0; k < 5; k++) final_z[z][k] = 0;
        for (uint32_t w = tid; w < n_warps; w += 32)
            for (int z = 0; z < 4; z++)
                for (int k = 0; k < 5; k++)
                    final_z[z][k] = kb_add(final_z[z][k], smem[(z * n_warps + w) * 5 + k]);
        for (int z = 0; z < 4; z++) warp_reduce_add_ext(final_z[z]);
        if (tid == 0) {
            for (int k = 0; k < 5; k++) {
                partial_z0[blockIdx.x * 5 + k] = final_z[0][k];
                partial_z2[blockIdx.x * 5 + k] = final_z[1][k];
                partial_z3[blockIdx.x * 5 + k] = final_z[2][k];
                partial_z4[blockIdx.x * 5 + k] = final_z[3][k];
            }
        }
    }
}

// ── GKR quotient sum (2-by-2 reduction) ──────────────────────────────────
// For each pair (i, i+1):
//   new_num[i/2] = num[i]*den[i+1] + num[i+1]*den[i]
//   new_den[i/2] = den[i]*den[i+1]
// Both num and den are ext field elements (5 u32s each).
extern "C" __global__ void gkr_sum_quotients_kernel(
    const uint32_t* __restrict__ nums,    // n * 5 ext elements
    const uint32_t* __restrict__ dens,    // n * 5 ext elements
    uint32_t* __restrict__ new_nums,       // (n/2) * 5 ext elements
    uint32_t* __restrict__ new_dens,       // (n/2) * 5 ext elements
    uint32_t n_pairs                       // n / 2
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    uint32_t i0 = 2 * tid;
    uint32_t i1 = 2 * tid + 1;

    uint32_t num0[5], num1[5], den0[5], den1[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        num0[k] = nums[i0 * 5 + k];
        num1[k] = nums[i1 * 5 + k];
        den0[k] = dens[i0 * 5 + k];
        den1[k] = dens[i1 * 5 + k];
    }

    // new_num = num0 * den1 + num1 * den0
    uint32_t prod0[5], prod1[5], sum_num[5];
    qe_mul(num0, den1, prod0);
    qe_mul(num1, den0, prod1);
    qe_add(prod0, prod1, sum_num);

    // new_den = den0 * den1
    uint32_t prod_den[5];
    qe_mul(den0, den1, prod_den);

    #pragma unroll
    for (int k = 0; k < 5; k++) {
        new_nums[tid * 5 + k] = sum_num[k];
        new_dens[tid * 5 + k] = prod_den[k];
    }
}

// ── GKR quotient sum at arbitrary fold bit ───────────────────────────────
// Same as gkr_sum_quotients_kernel but pairs at stride 2^fold_bit:
//   i0 = (tid_hi << (fold_bit+1)) | tid_lo;  i1 = i0 | (1 << fold_bit)
// Output: new[tid] = nums[i0]*dens[i1] + nums[i1]*dens[i0], new_den[tid] = dens[i0]*dens[i1]
extern "C" __global__ void gkr_sum_quotients_at_bit_kernel(
    const uint32_t* __restrict__ nums,
    const uint32_t* __restrict__ dens,
    uint32_t* __restrict__ new_nums,
    uint32_t* __restrict__ new_dens,
    uint32_t n_pairs,
    uint32_t fold_bit
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    uint32_t stride = 1u << fold_bit;
    uint32_t lo_mask = stride - 1;
    uint32_t i_hi = tid >> fold_bit;
    uint32_t i_lo = tid & lo_mask;
    uint32_t i0 = (i_hi << (fold_bit + 1)) | i_lo;
    uint32_t i1 = i0 | stride;

    uint32_t num0[5], num1[5], den0[5], den1[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        num0[k] = nums[i0 * 5 + k];
        num1[k] = nums[i1 * 5 + k];
        den0[k] = dens[i0 * 5 + k];
        den1[k] = dens[i1 * 5 + k];
    }

    uint32_t prod0[5], prod1[5], sum_num[5];
    qe_mul(num0, den1, prod0);
    qe_mul(num1, den0, prod1);
    qe_add(prod0, prod1, sum_num);

    uint32_t prod_den[5];
    qe_mul(den0, den1, prod_den);

    #pragma unroll
    for (int k = 0; k < 5; k++) {
        new_nums[tid * 5 + k] = sum_num[k];
        new_dens[tid * 5 + k] = prod_den[k];
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

// ── GKR quotient sumcheck kernel ────────────────────────────────────────
//
// For each pair j (folded at stride `half`):
//   nl_lo, nl_hi = nums_left[j], nums_left[j+half]
//   nr_lo, nr_hi = nums_right[j], nums_right[j+half]
//   dl_lo, dl_hi = dens_left[j], dens_left[j+half]
//   dr_lo, dr_hi = dens_right[j], dens_right[j+half]
//
// Outputs (summed over all j, weighted by eq[j]):
//   c0_num = Σ (nl_lo*dr_lo + nr_lo*dl_lo) * eq[j]
//   c2_num = Σ ((nl_hi-nl_lo)*(dr_hi-dr_lo) + (nr_hi-nr_lo)*(dl_hi-dl_lo)) * eq[j]
//   c0_den = Σ (dl_lo*dr_lo) * eq[j]
//   c2_den = Σ ((dl_hi-dl_lo)*(dr_hi-dr_lo)) * eq[j]
//
// All arrays are ext field (5 u32s per element).
// This is the core sumcheck for GKR quotient layer proving.
// Parameterized by fold_bit: pairs at (j, j | (1 << fold_bit)) where j has bit `fold_bit` clear.
// fold_bit=0: adjacent pairs (2j, 2j+1). fold_bit=k: stride 2^k.
// The `new_j` index iterates over positions with fold_bit clear:
//   i_hi = new_j >> fold_bit;  i_lo = new_j & ((1<<fold_bit)-1);
//   i0 = (i_hi << (fold_bit+1)) | i_lo;  i1 = i0 | (1 << fold_bit);
extern "C" __global__ void gkr_quotient_sumcheck_kernel(
    const uint32_t* __restrict__ nums_l,
    const uint32_t* __restrict__ nums_r,
    const uint32_t* __restrict__ dens_l,
    const uint32_t* __restrict__ dens_r,
    const uint32_t* __restrict__ eq_vals, // n_pairs * 5 ext
    uint32_t* __restrict__ partial_c0_num,
    uint32_t* __restrict__ partial_c2_num,
    uint32_t* __restrict__ partial_c0_den,
    uint32_t* __restrict__ partial_c2_den,
    uint32_t n_pairs,                      // number of pair positions
    uint32_t fold_bit                      // which bit to fold on
) {
    extern __shared__ uint32_t smem[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    uint32_t c0n[5] = {0,0,0,0,0};
    uint32_t c2n[5] = {0,0,0,0,0};
    uint32_t c0d[5] = {0,0,0,0,0};
    uint32_t c2d[5] = {0,0,0,0,0};

    if (gid < n_pairs) {
        uint32_t stride = 1u << fold_bit;
        uint32_t lo_mask = stride - 1;
        uint32_t i_hi = gid >> fold_bit;
        uint32_t i_lo = gid & lo_mask;
        uint32_t i0 = (i_hi << (fold_bit + 1)) | i_lo;
        uint32_t i1 = i0 | stride;

        uint32_t nl_lo[5], nl_hi[5], nr_lo[5], nr_hi[5];
        uint32_t dl_lo[5], dl_hi[5], dr_lo[5], dr_hi[5];
        uint32_t eq[5];

        #pragma unroll
        for (int k = 0; k < 5; k++) {
            nl_lo[k] = nums_l[i0 * 5 + k];
            nl_hi[k] = nums_l[i1 * 5 + k];
            nr_lo[k] = nums_r[i0 * 5 + k];
            nr_hi[k] = nums_r[i1 * 5 + k];
            dl_lo[k] = dens_l[i0 * 5 + k];
            dl_hi[k] = dens_l[i1 * 5 + k];
            dr_lo[k] = dens_r[i0 * 5 + k];
            dr_hi[k] = dens_r[i1 * 5 + k];
            eq[k] = eq_vals[gid * 5 + k];
        }

        // c0_num = (nl_lo * dr_lo + nr_lo * dl_lo) * eq
        uint32_t tmp1[5], tmp2[5], raw_c0n[5];
        qe_mul(nl_lo, dr_lo, tmp1);
        qe_mul(nr_lo, dl_lo, tmp2);
        qe_add(tmp1, tmp2, raw_c0n);
        qe_mul(raw_c0n, eq, c0n);

        // c2_num = ((nl_hi-nl_lo)*(dr_hi-dr_lo) + (nr_hi-nr_lo)*(dl_hi-dl_lo)) * eq
        uint32_t nl_d[5], dr_d[5], nr_d[5], dl_d[5];
        qe_sub(nl_hi, nl_lo, nl_d);
        qe_sub(dr_hi, dr_lo, dr_d);
        qe_sub(nr_hi, nr_lo, nr_d);
        qe_sub(dl_hi, dl_lo, dl_d);
        qe_mul(nl_d, dr_d, tmp1);
        qe_mul(nr_d, dl_d, tmp2);
        uint32_t raw_c2n[5];
        qe_add(tmp1, tmp2, raw_c2n);
        qe_mul(raw_c2n, eq, c2n);

        // c0_den = dl_lo * dr_lo * eq
        uint32_t raw_c0d[5];
        qe_mul(dl_lo, dr_lo, raw_c0d);
        qe_mul(raw_c0d, eq, c0d);

        // c2_den = (dl_hi-dl_lo) * (dr_hi-dr_lo) * eq
        uint32_t raw_c2d[5];
        qe_mul(dl_d, dr_d, raw_c2d);
        qe_mul(raw_c2d, eq, c2d);
    }

    // Warp-level reduction for all 4 accumulators
    warp_reduce_add_ext(c0n);
    warp_reduce_add_ext(c2n);
    warp_reduce_add_ext(c0d);
    warp_reduce_add_ext(c2d);

    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    uint32_t n_warps = blockDim.x / 32;
    // smem: [c0n warps | c2n warps | c0d warps | c2d warps]
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            smem[(0 * n_warps + warp_id) * 5 + k] = c0n[k];
            smem[(1 * n_warps + warp_id) * 5 + k] = c2n[k];
            smem[(2 * n_warps + warp_id) * 5 + k] = c0d[k];
            smem[(3 * n_warps + warp_id) * 5 + k] = c2d[k];
        }
    }
    __syncthreads();

    if (tid < 32) {
        uint32_t f_c0n[5]={0,0,0,0,0}, f_c2n[5]={0,0,0,0,0};
        uint32_t f_c0d[5]={0,0,0,0,0}, f_c2d[5]={0,0,0,0,0};
        for (uint32_t w = tid; w < n_warps; w += 32) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                f_c0n[k] = kb_add(f_c0n[k], smem[(0 * n_warps + w) * 5 + k]);
                f_c2n[k] = kb_add(f_c2n[k], smem[(1 * n_warps + w) * 5 + k]);
                f_c0d[k] = kb_add(f_c0d[k], smem[(2 * n_warps + w) * 5 + k]);
                f_c2d[k] = kb_add(f_c2d[k], smem[(3 * n_warps + w) * 5 + k]);
            }
        }
        warp_reduce_add_ext(f_c0n);
        warp_reduce_add_ext(f_c2n);
        warp_reduce_add_ext(f_c0d);
        warp_reduce_add_ext(f_c2d);
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                partial_c0_num[blockIdx.x * 5 + k] = f_c0n[k];
                partial_c2_num[blockIdx.x * 5 + k] = f_c2n[k];
                partial_c0_den[blockIdx.x * 5 + k] = f_c0d[k];
                partial_c2_den[blockIdx.x * 5 + k] = f_c2d[k];
            }
        }
    }
}

// ── GKR quotient sumcheck (packed bit-reversed, full layer) ─────────────
//
// Handles the ENTIRE layer quotient sumcheck for the GKR protocol.
// Data layout: arrays are in natural order (after conversion from packed-BR).
// Each thread handles one pair at stride `half` (fold-at-half mode).
//
// For each pair j (j < half, elements at j and j+half):
//   nl_lo, nl_hi = nums_l[j], nums_l[j+half]
//   ... (same for nr, dl, dr)
//   c0_num += (nl_lo*dr_lo + nr_lo*dl_lo) * eq[j]
//   c2_num += ((nl_hi-nl_lo)*(dr_hi-dr_lo) + (nr_hi-nr_lo)*(dl_hi-dl_lo)) * eq[j]
//   c0_den += dl_lo*dr_lo * eq[j]
//   c2_den += (dl_hi-dl_lo)*(dr_hi-dr_lo) * eq[j]
//
// This is the fold-at-half variant. Data arrays have 2*half elements.
// eq has half elements.
extern "C" __global__ void gkr_quotient_sc_half_kernel(
    const uint32_t* __restrict__ nums_l,  // 2*half * 5 ext
    const uint32_t* __restrict__ nums_r,  // 2*half * 5 ext
    const uint32_t* __restrict__ dens_l,  // 2*half * 5 ext
    const uint32_t* __restrict__ dens_r,  // 2*half * 5 ext
    const uint32_t* __restrict__ eq_vals, // half * 5 ext
    uint32_t* __restrict__ partial_c0_num,
    uint32_t* __restrict__ partial_c2_num,
    uint32_t* __restrict__ partial_c0_den,
    uint32_t* __restrict__ partial_c2_den,
    uint32_t half
) {
    extern __shared__ uint32_t smem[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    uint32_t c0n[5]={0,0,0,0,0}, c2n[5]={0,0,0,0,0};
    uint32_t c0d[5]={0,0,0,0,0}, c2d[5]={0,0,0,0,0};

    if (gid < half) {
        uint32_t i0 = gid;
        uint32_t i1 = gid + half;
        uint32_t nl_lo[5], nl_hi[5], nr_lo[5], nr_hi[5];
        uint32_t dl_lo[5], dl_hi[5], dr_lo[5], dr_hi[5];
        uint32_t eq[5];

        #pragma unroll
        for (int k = 0; k < 5; k++) {
            nl_lo[k] = nums_l[i0 * 5 + k]; nl_hi[k] = nums_l[i1 * 5 + k];
            nr_lo[k] = nums_r[i0 * 5 + k]; nr_hi[k] = nums_r[i1 * 5 + k];
            dl_lo[k] = dens_l[i0 * 5 + k]; dl_hi[k] = dens_l[i1 * 5 + k];
            dr_lo[k] = dens_r[i0 * 5 + k]; dr_hi[k] = dens_r[i1 * 5 + k];
            eq[k] = eq_vals[gid * 5 + k];
        }

        uint32_t tmp1[5], tmp2[5], raw[5];
        // c0_num = (nl_lo*dr_lo + nr_lo*dl_lo) * eq
        qe_mul(nl_lo, dr_lo, tmp1); qe_mul(nr_lo, dl_lo, tmp2);
        qe_add(tmp1, tmp2, raw); qe_mul(raw, eq, c0n);
        // c2_num
        uint32_t nl_d[5], dr_d[5], nr_d[5], dl_d[5];
        qe_sub(nl_hi, nl_lo, nl_d); qe_sub(dr_hi, dr_lo, dr_d);
        qe_sub(nr_hi, nr_lo, nr_d); qe_sub(dl_hi, dl_lo, dl_d);
        qe_mul(nl_d, dr_d, tmp1); qe_mul(nr_d, dl_d, tmp2);
        qe_add(tmp1, tmp2, raw); qe_mul(raw, eq, c2n);
        // c0_den = dl_lo*dr_lo * eq
        qe_mul(dl_lo, dr_lo, raw); qe_mul(raw, eq, c0d);
        // c2_den = (dl_d * dr_d) * eq
        qe_mul(dl_d, dr_d, raw); qe_mul(raw, eq, c2d);
    }

    warp_reduce_add_ext(c0n); warp_reduce_add_ext(c2n);
    warp_reduce_add_ext(c0d); warp_reduce_add_ext(c2d);

    uint32_t warp_id = tid / 32, lane = tid % 32;
    uint32_t n_warps = blockDim.x / 32;
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            smem[(0*n_warps+warp_id)*5+k] = c0n[k];
            smem[(1*n_warps+warp_id)*5+k] = c2n[k];
            smem[(2*n_warps+warp_id)*5+k] = c0d[k];
            smem[(3*n_warps+warp_id)*5+k] = c2d[k];
        }
    }
    __syncthreads();

    if (tid < 32) {
        uint32_t f0n[5]={0,0,0,0,0}, f2n[5]={0,0,0,0,0};
        uint32_t f0d[5]={0,0,0,0,0}, f2d[5]={0,0,0,0,0};
        for (uint32_t w = tid; w < n_warps; w += 32) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                f0n[k] = kb_add(f0n[k], smem[(0*n_warps+w)*5+k]);
                f2n[k] = kb_add(f2n[k], smem[(1*n_warps+w)*5+k]);
                f0d[k] = kb_add(f0d[k], smem[(2*n_warps+w)*5+k]);
                f2d[k] = kb_add(f2d[k], smem[(3*n_warps+w)*5+k]);
            }
        }
        warp_reduce_add_ext(f0n); warp_reduce_add_ext(f2n);
        warp_reduce_add_ext(f0d); warp_reduce_add_ext(f2d);
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                partial_c0_num[blockIdx.x*5+k] = f0n[k];
                partial_c2_num[blockIdx.x*5+k] = f2n[k];
                partial_c0_den[blockIdx.x*5+k] = f0d[k];
                partial_c2_den[blockIdx.x*5+k] = f2d[k];
            }
        }
    }
}

// ── GKR quotient sumcheck on interleaved data ───────────────────────────
//
// Operates on a SINGLE nums and dens array (not split into left/right).
// Within each chunk of `2^chunk_log` elements:
//   Bit (chunk_log-1): left (0) vs right (1)
//   Bit (chunk_log-2): lo (0) vs hi (1)  — this is the sumcheck fold variable
//   Remaining bits: inner position (the pair index)
//
// For each pair position j (quarter of a chunk):
//   nl_lo = nums[chunk_base + j]
//   nl_hi = nums[chunk_base + j + quarter]
//   nr_lo = nums[chunk_base + j + half]
//   nr_hi = nums[chunk_base + j + half + quarter]
//
// eq_outer[chunk_idx] weights the chunk, eq_within[j] weights the inner position.
// Output: partial sums for (c0_num, c2_num, c0_den, c2_den).
extern "C" __global__ void gkr_quotient_sc_interleaved_kernel(
    const uint32_t* __restrict__ nums,       // full array, n_total * 5 ext
    const uint32_t* __restrict__ dens,       // full array, n_total * 5 ext
    const uint32_t* __restrict__ eq_within,  // quarter * 5 ext (inner eq table)
    const uint32_t* __restrict__ eq_outer,   // n_chunks * 5 ext (outer eq table, as base broadcast to ext)
    uint32_t* __restrict__ partial_c0_num,
    uint32_t* __restrict__ partial_c2_num,
    uint32_t* __restrict__ partial_c0_den,
    uint32_t* __restrict__ partial_c2_den,
    uint32_t n_total,        // total elements
    uint32_t chunk_size,     // 2^chunk_log
    uint32_t n_chunks        // n_total / chunk_size
) {
    extern __shared__ uint32_t smem[];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    uint32_t half = chunk_size / 2;
    uint32_t quarter = chunk_size / 4;
    uint32_t total_pairs = n_chunks * quarter; // pairs across all chunks

    uint32_t c0n[5]={0,0,0,0,0}, c2n[5]={0,0,0,0,0};
    uint32_t c0d[5]={0,0,0,0,0}, c2d[5]={0,0,0,0,0};

    if (gid < total_pairs) {
        uint32_t chunk_idx = gid / quarter;
        uint32_t inner = gid % quarter;
        uint32_t base = chunk_idx * chunk_size;

        // Read 8 values: nl_lo, nl_hi, nr_lo, nr_hi, dl_lo, dl_hi, dr_lo, dr_hi
        uint32_t nl_lo[5], nl_hi[5], nr_lo[5], nr_hi[5];
        uint32_t dl_lo[5], dl_hi[5], dr_lo[5], dr_hi[5];
        uint32_t eq_w[5], eq_o[5];

        #pragma unroll
        for (int k = 0; k < 5; k++) {
            nl_lo[k] = nums[(base + inner) * 5 + k];
            nl_hi[k] = nums[(base + inner + quarter) * 5 + k];
            nr_lo[k] = nums[(base + inner + half) * 5 + k];
            nr_hi[k] = nums[(base + inner + half + quarter) * 5 + k];
            dl_lo[k] = dens[(base + inner) * 5 + k];
            dl_hi[k] = dens[(base + inner + quarter) * 5 + k];
            dr_lo[k] = dens[(base + inner + half) * 5 + k];
            dr_hi[k] = dens[(base + inner + half + quarter) * 5 + k];
            eq_w[k] = eq_within[inner * 5 + k];
            eq_o[k] = (chunk_idx < n_chunks) ? eq_outer[chunk_idx * 5 + k] : 0;
        }

        // pair_coeffs: c0_num = (nl_lo*dr_lo + nr_lo*dl_lo), c0_den = dl_lo*dr_lo
        uint32_t tmp1[5], tmp2[5], raw[5];
        qe_mul(nl_lo, dr_lo, tmp1); qe_mul(nr_lo, dl_lo, tmp2);
        qe_add(tmp1, tmp2, raw);
        // Weight by eq_within * eq_outer
        uint32_t eq_combined[5];
        qe_mul(eq_w, eq_o, eq_combined);
        qe_mul(raw, eq_combined, c0n);

        // c2_num
        uint32_t nl_d[5], dr_d[5], nr_d[5], dl_d[5];
        qe_sub(nl_hi, nl_lo, nl_d); qe_sub(dr_hi, dr_lo, dr_d);
        qe_sub(nr_hi, nr_lo, nr_d); qe_sub(dl_hi, dl_lo, dl_d);
        qe_mul(nl_d, dr_d, tmp1); qe_mul(nr_d, dl_d, tmp2);
        qe_add(tmp1, tmp2, raw);
        qe_mul(raw, eq_combined, c2n);

        // c0_den = dl_lo*dr_lo * eq
        qe_mul(dl_lo, dr_lo, raw);
        qe_mul(raw, eq_combined, c0d);

        // c2_den = (dl_d*dr_d) * eq
        qe_mul(dl_d, dr_d, raw);
        qe_mul(raw, eq_combined, c2d);
    }

    // Standard warp + block reduction (same as other kernels)
    warp_reduce_add_ext(c0n); warp_reduce_add_ext(c2n);
    warp_reduce_add_ext(c0d); warp_reduce_add_ext(c2d);

    uint32_t warp_id = tid / 32, lane = tid % 32;
    uint32_t n_warps = blockDim.x / 32;
    if (lane == 0) {
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            smem[(0*n_warps+warp_id)*5+k] = c0n[k];
            smem[(1*n_warps+warp_id)*5+k] = c2n[k];
            smem[(2*n_warps+warp_id)*5+k] = c0d[k];
            smem[(3*n_warps+warp_id)*5+k] = c2d[k];
        }
    }
    __syncthreads();
    if (tid < 32) {
        uint32_t f0n[5]={0,0,0,0,0}, f2n[5]={0,0,0,0,0};
        uint32_t f0d[5]={0,0,0,0,0}, f2d[5]={0,0,0,0,0};
        for (uint32_t w = tid; w < n_warps; w += 32) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                f0n[k] = kb_add(f0n[k], smem[(0*n_warps+w)*5+k]);
                f2n[k] = kb_add(f2n[k], smem[(1*n_warps+w)*5+k]);
                f0d[k] = kb_add(f0d[k], smem[(2*n_warps+w)*5+k]);
                f2d[k] = kb_add(f2d[k], smem[(3*n_warps+w)*5+k]);
            }
        }
        warp_reduce_add_ext(f0n); warp_reduce_add_ext(f2n);
        warp_reduce_add_ext(f0d); warp_reduce_add_ext(f2d);
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                partial_c0_num[blockIdx.x*5+k] = f0n[k];
                partial_c2_num[blockIdx.x*5+k] = f2n[k];
                partial_c0_den[blockIdx.x*5+k] = f0d[k];
                partial_c2_den[blockIdx.x*5+k] = f2d[k];
            }
        }
    }
}

// ── Logup fingerprint kernel ────────────────────────────────────────────
//
// Computes denominators for the logup argument:
//   denom[i] = c - (contrib + data[0][i] * alpha[0] + data[1][i] * alpha[1] + ... + index * alpha[n_data])
//
// where data[k][i] are base field columns, c and contrib are ext field constants,
// alpha[k] are ext field weights, and index = i (the row index, base field).
//
// Each thread computes one denominator. Input columns are col-major (n_rows per column).
// Output is ext field (5 u32s per element).
extern "C" __global__ void logup_fingerprint_kernel(
    const uint32_t* __restrict__ columns,    // n_cols * n_rows base elements (col-major)
    const uint32_t* __restrict__ c_ext,      // 5 ext (the random challenge)
    const uint32_t* __restrict__ contrib,    // 5 ext (domain sep contribution)
    const uint32_t* __restrict__ alphas,     // (n_cols + 1) * 5 ext (alpha eq poly, last = index weight)
    uint32_t* __restrict__ denoms,           // n_rows * 5 ext output
    uint32_t n_rows,
    uint32_t n_cols                          // number of data columns
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;

    // Start with contrib
    uint32_t acc[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) acc[k] = contrib[k];

    // Add data[col] * alpha[col] for each column
    for (uint32_t col = 0; col < n_cols; col++) {
        uint32_t val = columns[col * n_rows + i];
        const uint32_t* alpha = alphas + col * 5;
        // base * ext: multiply base field value by ext field alpha
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            acc[k] = kb_add(acc[k], kb_mul(val, alpha[k]));
        }
    }

    // Add index * alpha[n_cols]
    {
        uint32_t idx = kb_to_monty(i);  // convert index to Montgomery form
        const uint32_t* alpha_idx = alphas + n_cols * 5;
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            acc[k] = kb_add(acc[k], kb_mul(idx, alpha_idx[k]));
        }
    }

    // denom = c - acc
    uint32_t c[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) c[k] = c_ext[k];

    uint32_t result[5];
    qe_sub(c, acc, result);

    #pragma unroll
    for (int k = 0; k < 5; k++) denoms[i * 5 + k] = result[k];
}

extern "C" __global__ void logup_prepare_constants_kernel(
    const uint32_t* __restrict__ alpha_eq,
    const uint32_t* __restrict__ alpha_indices,
    const uint32_t* __restrict__ alpha_negated,
    const uint32_t* __restrict__ contrib_indices,
    const uint32_t* __restrict__ contrib_coeffs,
    uint32_t* __restrict__ out_contrib,
    uint32_t* __restrict__ out_alphas,
    uint32_t n_alphas,
    uint32_t n_contrib
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    qe_zero(out_contrib);
    for (uint32_t i = 0; i < n_contrib; i++) {
        uint32_t idx = contrib_indices[i];
        if (idx == 0xFFFFFFFFu) continue;
        uint32_t term[5];
        qe_base_mul(alpha_eq + idx * 5, contrib_coeffs[i], term);
        qe_add(out_contrib, term, out_contrib);
    }

    for (uint32_t i = 0; i < n_alphas; i++) {
        uint32_t* dst = out_alphas + i * 5;
        uint32_t idx = alpha_indices[i];
        if (idx == 0xFFFFFFFFu) {
            qe_zero(dst);
            continue;
        }
        if (alpha_negated[i]) {
            qe_neg(alpha_eq + idx * 5, dst);
        } else {
            #pragma unroll
            for (int k = 0; k < 5; k++) dst[k] = alpha_eq[idx * 5 + k];
        }
    }
}

// ── Batch multi-column fold kernels ──────────────────────────────────────
// Fold all columns at half in one kernel launch. Col-major layout.

// Base→Ext fold: data[col * n_rows + j], output[col * n_pairs * 5 + j * 5 + k]
extern "C" __global__ void fold_multi_col_b2e_half_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ out,
    const uint32_t* __restrict__ r_ext,
    uint32_t n_rows,
    uint32_t n_pairs,
    uint32_t n_cols
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cols * n_pairs) return;
    uint32_t col = tid / n_pairs;
    uint32_t j = tid % n_pairs;
    uint32_t lo = data[col * n_rows + j];
    uint32_t diff = kb_sub(data[col * n_rows + j + n_pairs], lo);
    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];
    // out = EF::from(lo) + diff * r
    uint32_t res[5];
    qe_from_base(lo, res);
    #pragma unroll
    for (int k = 0; k < 5; k++) res[k] = kb_add(res[k], kb_mul(r[k], diff));
    uint32_t out_off = col * n_pairs * 5 + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) out[out_off + k] = res[k];
}

// First base->ext fold of one polynomial into many point columns.
extern "C" __global__ void fold_single_b2e_many_points_half_kernel(
    const uint32_t* __restrict__ data,
    const uint32_t* __restrict__ points,
    uint32_t* __restrict__ out,
    uint32_t point_stride_words,
    uint32_t coord_idx,
    uint32_t n_elems,
    uint32_t n_pairs,
    uint32_t n_cols
) {
    (void)n_elems;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cols * n_pairs) return;
    uint32_t col = tid / n_pairs;
    uint32_t j = tid % n_pairs;
    const uint32_t* r_ext = points + col * point_stride_words + coord_idx * 5;
    uint32_t lo = data[j];
    uint32_t diff = kb_sub(data[j + n_pairs], lo);
    uint32_t out_off = col * n_pairs * 5 + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) out[out_off + k] = kb_mul(r_ext[k], diff);
    out[out_off] = kb_add(out[out_off], lo);
}

// Ext→Ext fold: data[col * n_elems * 5 + elem * 5 + k]
extern "C" __global__ void fold_multi_col_ext_half_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ out,
    const uint32_t* __restrict__ r_ext,
    uint32_t n_elems,
    uint32_t n_pairs,
    uint32_t n_cols
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cols * n_pairs) return;
    uint32_t col = tid / n_pairs;
    uint32_t j = tid % n_pairs;
    uint32_t base_off = col * n_elems * 5;
    uint32_t lo[5], hi[5], r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        lo[k] = data[base_off + j * 5 + k];
        hi[k] = data[base_off + (j + n_pairs) * 5 + k];
        r[k] = r_ext[k];
    }
    uint32_t diff[5]; qe_sub(hi, lo, diff);
    uint32_t prod[5]; qe_mul(r, diff, prod);
    uint32_t res[5]; qe_add(lo, prod, res);
    uint32_t out_off = col * n_pairs * 5 + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) out[out_off + k] = res[k];
}

// First ext->ext fold of one polynomial into many point columns.
extern "C" __global__ void fold_single_ext_many_points_half_kernel(
    const uint32_t* __restrict__ data,
    const uint32_t* __restrict__ points,
    uint32_t* __restrict__ out,
    uint32_t point_stride_words,
    uint32_t coord_idx,
    uint32_t n_elems,
    uint32_t n_pairs,
    uint32_t n_cols
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cols * n_pairs) return;
    uint32_t col = tid / n_pairs;
    uint32_t j = tid % n_pairs;
    const uint32_t* r_ext = points + col * point_stride_words + coord_idx * 5;
    uint32_t lo[5], hi[5], r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        lo[k] = data[j * 5 + k];
        hi[k] = data[(j + n_pairs) * 5 + k];
        r[k] = r_ext[k];
    }
    uint32_t diff[5]; qe_sub(hi, lo, diff);
    uint32_t prod[5]; qe_mul(r, diff, prod);
    uint32_t res[5]; qe_add(lo, prod, res);
    uint32_t out_off = col * n_pairs * 5 + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) out[out_off + k] = res[k];
}

// Subsequent ext->ext folds of many point columns with per-point challenges.
extern "C" __global__ void fold_multi_col_ext_half_per_point_kernel(
    const uint32_t* __restrict__ data,
    const uint32_t* __restrict__ points,
    uint32_t* __restrict__ out,
    uint32_t point_stride_words,
    uint32_t coord_idx,
    uint32_t n_elems,
    uint32_t n_pairs,
    uint32_t n_cols
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cols * n_pairs) return;
    uint32_t col = tid / n_pairs;
    uint32_t j = tid % n_pairs;
    uint32_t base_off = col * n_elems * 5;
    const uint32_t* r_ext = points + col * point_stride_words + coord_idx * 5;
    uint32_t lo[5], hi[5], r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        lo[k] = data[base_off + j * 5 + k];
        hi[k] = data[base_off + (j + n_pairs) * 5 + k];
        r[k] = r_ext[k];
    }
    uint32_t diff[5]; qe_sub(hi, lo, diff);
    uint32_t prod[5]; qe_mul(r, diff, prod);
    uint32_t res[5]; qe_add(lo, prod, res);
    uint32_t out_off = col * n_pairs * 5 + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) out[out_off + k] = res[k];
}

// Repeat one device-resident extension value into an output column.
extern "C" __global__ void repeat_ext_value_kernel(
    const uint32_t* __restrict__ value,
    uint32_t* __restrict__ out,
    uint32_t n_values
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;
    uint32_t out_off = idx * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) out[out_off + k] = value[k];
}

// Repeat one device-resident base value as extension values.
extern "C" __global__ void repeat_base_value_as_ext_kernel(
    const uint32_t* __restrict__ value,
    uint32_t* __restrict__ out,
    uint32_t n_values
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;
    uint32_t out_off = idx * 5;
    out[out_off] = value[0];
    #pragma unroll
    for (int k = 1; k < 5; k++) out[out_off + k] = 0;
}

// ── Bit-reversal within chunks kernel ────────────────────────────────────
// Permutes elements within each chunk of 2^chunk_log, in-place on base field.
// For each chunk: out[br(i)] = in[i] where br reverses chunk_log bits.
extern "C" __global__ void bit_reverse_within_chunks_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ out,
    uint32_t n,             // total elements
    uint32_t chunk_log      // log2 of chunk size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t chunk_size = 1u << chunk_log;
    uint32_t chunk_mask = chunk_size - 1;
    uint32_t chunk_idx = tid / chunk_size;
    uint32_t pos_in_chunk = tid & chunk_mask;
    // Bit-reverse pos_in_chunk within chunk_log bits
    uint32_t rev = __brev(pos_in_chunk) >> (32 - chunk_log);
    out[chunk_idx * chunk_size + rev] = data[tid];
}

// ── Batch fold at arbitrary bit (base→ext) ──────────────────────────────
extern "C" __global__ void fold_multi_col_b2e_at_bit_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ out,
    const uint32_t* __restrict__ r_ext,
    uint32_t n_rows,       // current row count (2 * n_pairs for fold_bit=0, or general)
    uint32_t n_pairs,      // output size per column
    uint32_t n_cols,
    uint32_t fold_bit      // which bit to fold
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cols * n_pairs) return;
    uint32_t col = tid / n_pairs;
    uint32_t j = tid % n_pairs;
    // Compute i0, i1 from fold_bit
    uint32_t stride = 1u << fold_bit;
    uint32_t lo_mask = stride - 1;
    uint32_t i_hi = j >> fold_bit;
    uint32_t i_lo = j & lo_mask;
    uint32_t i0 = (i_hi << (fold_bit + 1)) | i_lo;
    uint32_t i1 = i0 | stride;
    uint32_t lo = data[col * n_rows + i0];
    uint32_t diff = kb_sub(data[col * n_rows + i1], lo);
    uint32_t r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) r[k] = r_ext[k];
    uint32_t res[5];
    qe_from_base(lo, res);
    #pragma unroll
    for (int k = 0; k < 5; k++) res[k] = kb_add(res[k], kb_mul(r[k], diff));
    uint32_t out_off = col * n_pairs * 5 + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) out[out_off + k] = res[k];
}

// ── Batch fold at arbitrary bit (ext→ext) ───────────────────────────────
extern "C" __global__ void fold_multi_col_ext_at_bit_kernel(
    const uint32_t* __restrict__ data,
    uint32_t* __restrict__ out,
    const uint32_t* __restrict__ r_ext,
    uint32_t n_elems,
    uint32_t n_pairs,
    uint32_t n_cols,
    uint32_t fold_bit
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cols * n_pairs) return;
    uint32_t col = tid / n_pairs;
    uint32_t j = tid % n_pairs;
    uint32_t stride = 1u << fold_bit;
    uint32_t lo_mask = stride - 1;
    uint32_t i_hi = j >> fold_bit;
    uint32_t i_lo = j & lo_mask;
    uint32_t i0 = (i_hi << (fold_bit + 1)) | i_lo;
    uint32_t i1 = i0 | stride;
    uint32_t base_off = col * n_elems * 5;
    uint32_t lo[5], hi[5], r[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        lo[k] = data[base_off + i0 * 5 + k];
        hi[k] = data[base_off + i1 * 5 + k];
        r[k] = r_ext[k];
    }
    uint32_t diff[5]; qe_sub(hi, lo, diff);
    uint32_t prod[5]; qe_mul(r, diff, prod);
    uint32_t res[5]; qe_add(lo, prod, res);
    uint32_t out_off = col * n_pairs * 5 + j * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) out[out_off + k] = res[k];
}

// ── Helper: fold-at-bit pair indexing ────────────────────────────────────
// Given pair index j and fold_bit, compute (i0, i1).
#define FOLD_BIT_PAIR(j, fold_bit, i0_out, i1_out) do { \
    uint32_t _stride = 1u << (fold_bit); \
    uint32_t _lo_mask = _stride - 1; \
    uint32_t _i_hi = (j) >> (fold_bit); \
    uint32_t _i_lo = (j) & _lo_mask; \
    (i0_out) = (_i_hi << ((fold_bit) + 1)) | _i_lo; \
    (i1_out) = (i0_out) | _stride; \
} while(0)

// ── Multi-z warp+block reduction helper macro ─────────────────────────────
// Reduces N_Z accumulators (each uint32_t[5]) across warps in a block.
// Writes results to partial_sums at stride n_blocks * 5 per z-point.
// partial_sums[z_idx * gridDim.x * 5 + blockIdx.x * 5 + k]
#define MULTI_Z_REDUCE(N_Z, acc_array, partial_sums) do { \
    for (int z = 0; z < (N_Z); z++) warp_reduce_add_ext((acc_array)[z]); \
    uint32_t wid = threadIdx.x / 32, lane = threadIdx.x % 32; \
    uint32_t nw = blockDim.x / 32; \
    if (lane == 0) { \
        for (int z = 0; z < (N_Z); z++) \
            for (int k = 0; k < 5; k++) \
                smem[(z * nw + wid) * 5 + k] = (acc_array)[z][k]; \
    } \
    __syncthreads(); \
    if (threadIdx.x < 32) { \
        uint32_t fz[(N_Z)][5]; \
        for (int z = 0; z < (N_Z); z++) \
            for (int k = 0; k < 5; k++) fz[z][k] = 0; \
        for (uint32_t w = threadIdx.x; w < nw; w += 32) \
            for (int z = 0; z < (N_Z); z++) \
                for (int k = 0; k < 5; k++) \
                    fz[z][k] = kb_add(fz[z][k], smem[(z * nw + w) * 5 + k]); \
        for (int z = 0; z < (N_Z); z++) warp_reduce_add_ext(fz[z]); \
        if (threadIdx.x == 0) { \
            for (int z = 0; z < (N_Z); z++) \
                for (int k = 0; k < 5; k++) \
                    (partial_sums)[(z * gridDim.x + blockIdx.x) * 5 + k] = fz[z][k]; \
        } \
    } \
} while(0)

// ── Multi-z ExtensionOp kernel (base field columns, 5 z-points) ────────
extern "C" __global__ void air_ext_op_multi_z_kernel(
    const uint32_t* __restrict__ columns,   // 29 * n_rows base (col-major)
    const uint32_t* __restrict__ down_cols,  // 13 * n_rows base (col-major)
    const uint32_t* __restrict__ eq_factor,  // n_pairs * 5 ext
    const uint32_t* __restrict__ alphas,     // 33 * 5 ext
    uint32_t* __restrict__ partial_sums,     // 5 * n_blocks * 5 ext
    uint32_t n_rows,
    uint32_t n_pairs
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t acc[6][5];
    for (int z = 0; z < 6; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;

    if (gid < n_pairs) {
        uint32_t i0 = gid, i1 = gid + n_pairs;

        uint32_t v0_up[29], diff_up[29];
        for (int c = 0; c < 29; c++) {
            v0_up[c] = columns[c * n_rows + i0];
            diff_up[c] = kb_sub(columns[c * n_rows + i1], v0_up[c]);
        }
        uint32_t v0_dn[13], diff_dn[13];
        for (int c = 0; c < 13; c++) {
            v0_dn[c] = down_cols[c * n_rows + i0];
            diff_dn[c] = kb_sub(down_cols[c * n_rows + i1], v0_dn[c]);
        }

        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];

        // z=0: point = v0
        {
            uint32_t constraints[33];
            eval_extension_op_air(v0_up, v0_dn, constraints);
            uint32_t w[5] = {0,0,0,0,0};
            for (int c = 0; c < 33; c++)
                for (int k = 0; k < 5; k++)
                    w[k] = kb_add(w[k], kb_mul(alphas[c*5+k], constraints[c]));
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }

        // Advance to z=1 (skip)
        uint32_t up_cur[29], dn_cur[13];
        for (int c = 0; c < 29; c++) up_cur[c] = kb_add(v0_up[c], diff_up[c]);
        for (int c = 0; c < 13; c++) dn_cur[c] = kb_add(v0_dn[c], diff_dn[c]);

        // z=2,3,4,5
        for (int zi = 0; zi < 5; zi++) {
            for (int c = 0; c < 29; c++) up_cur[c] = kb_add(up_cur[c], diff_up[c]);
            for (int c = 0; c < 13; c++) dn_cur[c] = kb_add(dn_cur[c], diff_dn[c]);
            uint32_t constraints[33];
            eval_extension_op_air(up_cur, dn_cur, constraints);
            uint32_t w[5] = {0,0,0,0,0};
            for (int c = 0; c < 33; c++)
                for (int k = 0; k < 5; k++)
                    w[k] = kb_add(w[k], kb_mul(alphas[c*5+k], constraints[c]));
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(6, acc, partial_sums);
}

// ── Multi-z Poseidon16 kernel (base field columns, 10 z-points) ─────────
extern "C" __global__ void air_poseidon16_multi_z_kernel(
    const uint32_t* __restrict__ columns,    // 100 * n_rows base (col-major)
    const uint32_t* __restrict__ eq_factor,  // n_pairs * 5 ext
    const uint32_t* __restrict__ alphas,     // 81 * 5 ext
    const uint32_t* __restrict__ rc,         // 28 * 16 base field round constants
    const uint32_t* __restrict__ mds,        // 16 base field MDS circulant column
    const uint32_t* __restrict__ sparse,     // sparse matrix data
    uint32_t* __restrict__ partial_sums,     // 9 * n_blocks * 5 ext
    uint32_t n_rows,
    uint32_t n_pairs
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t acc[10][5];
    for (int z = 0; z < 10; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;

    if (gid < n_pairs) {
        uint32_t i0 = gid, i1 = gid + n_pairs;

        uint32_t v0[100], diff[100];
        for (int c = 0; c < 100; c++) {
            v0[c] = columns[c * n_rows + i0];
            diff[c] = kb_sub(columns[c * n_rows + i1], v0[c]);
        }

        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];

        // z=0: eval using v0 directly
        {
            uint32_t constraints[81];
            eval_poseidon16_air(v0, constraints, rc, mds, sparse);
            uint32_t w[5] = {0,0,0,0,0};
            for (int c = 0; c < 81; c++)
                for (int k = 0; k < 5; k++)
                    w[k] = kb_add(w[k], kb_mul(alphas[c*5+k], constraints[c]));
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }

        // Advance to z=1 (skip)
        uint32_t cur[100];
        for (int c = 0; c < 100; c++) cur[c] = kb_add(v0[c], diff[c]);

        // z=2,3,...,9
        for (int zi = 0; zi < 9; zi++) {
            for (int c = 0; c < 100; c++) cur[c] = kb_add(cur[c], diff[c]);
            uint32_t constraints[81];
            eval_poseidon16_air(cur, constraints, rc, mds, sparse);
            uint32_t w[5] = {0,0,0,0,0};
            for (int c = 0; c < 81; c++)
                for (int k = 0; k < 5; k++)
                    w[k] = kb_add(w[k], kb_mul(alphas[c*5+k], constraints[c]));
            uint32_t prod[5]; qe_mul(w, eq, prod);

            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(10, acc, partial_sums);
}

// ── Multi-z Execution kernel with fold_bit (base field, 4 z-points) ─────
// Same as air_execution_multi_z_kernel but pairs at arbitrary fold_bit.
// Generic bus constraint computation macro.
// bus_data: [flag, data0, data1, data2, data3] (5 base field values)
// logup_a: 5 ext values = [alpha0..alpha3 for data, alpha_domainsep]
// bus_b: ext value (bus_beta)
// Result stored in bus_out (5 u32 ext field).
#define COMPUTE_BUS_VALUE(bus_data, logup_a, bus_b, bus_out) do { \
    uint32_t _bi[5] = {0,0,0,0,0}; \
    for (int _i = 0; _i < 4; _i++) \
        for (int _k = 0; _k < 5; _k++) \
            _bi[_k] = kb_add(_bi[_k], kb_mul((logup_a)[_i*5+_k], (bus_data)[1+_i])); \
    for (int _k = 0; _k < 5; _k++) \
        _bi[_k] = kb_add(_bi[_k], kb_mul((logup_a)[4*5+_k], KB_MONTY_ONE)); \
    uint32_t _bp[5]; qe_mul(_bi, bus_b, _bp); \
    _bp[0] = kb_add(_bp[0], (bus_data)[0]); \
    for (int _k = 0; _k < 5; _k++) (bus_out)[_k] = _bp[_k]; \
} while(0)

// Ext-field bus constraint: bus_data values are ext field (5 u32s each, 25 total).
// bus_data layout: [flag(5), data0(5), data1(5), data2(5), data3(5)]
#define COMPUTE_BUS_VALUE_EXT(bus_data_ext, logup_a, bus_b, bus_out) do { \
    uint32_t _bi2[5] = {0,0,0,0,0}; \
    uint32_t _tmp2[5]; \
    for (int _i = 0; _i < 4; _i++) { \
        qe_mul((logup_a) + _i*5, (bus_data_ext) + (_i+1)*5, _tmp2); \
        qe_add(_bi2, _tmp2, _bi2); \
    } \
    qe_base_mul((logup_a) + 4*5, KB_MONTY_ONE, _tmp2); \
    qe_add(_bi2, _tmp2, _bi2); \
    uint32_t _bp2[5]; qe_mul(_bi2, bus_b, _bp2); \
    qe_add(_bp2, (bus_data_ext), _bp2); \
    for (int _k = 0; _k < 5; _k++) (bus_out)[_k] = _bp2[_k]; \
} while(0)

// Ext-field weighted: bus at alpha[0] + non-bus already weighted sum.
#define ADD_BUS_TO_WEIGHTED_EXT(bus_val, alphas, nonbus_w, w_out) do { \
    uint32_t _bw2[5]; qe_mul((alphas), bus_val, _bw2); \
    qe_add(_bw2, nonbus_w, w_out); \
} while(0)

// Weighted sum: bus (ext) at alpha[0] + N_NONBUS base-field constraints at alpha[1..N_NONBUS].
#define WEIGHTED_SUM_WITH_BUS(bus_val, constraints, alphas, n_nonbus, w_out) do { \
    uint32_t _bw[5]; qe_mul((alphas), bus_val, _bw); \
    uint32_t _ow[5] = {0,0,0,0,0}; \
    for (int _c = 0; _c < (n_nonbus); _c++) \
        for (int _k = 0; _k < 5; _k++) \
            _ow[_k] = kb_add(_ow[_k], kb_mul((alphas)[(_c+1)*5+_k], (constraints)[_c])); \
    for (int _k = 0; _k < 5; _k++) (w_out)[_k] = kb_add(_bw[_k], _ow[_k]); \
} while(0)

extern "C" __global__ void air_sumcheck_pad_eval_kernel(
    const uint32_t* __restrict__ all_cols,
    const uint32_t* __restrict__ alphas,
    const uint32_t* __restrict__ logup_alphas,
    const uint32_t* __restrict__ bus_beta,
    const uint32_t* __restrict__ rc,
    const uint32_t* __restrict__ mds,
    const uint32_t* __restrict__ sparse,
    uint32_t table_index,
    uint32_t n_rows,
    uint32_t* __restrict__ out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t row = n_rows - 1;
    uint32_t weighted[5];
    qe_zero(weighted);

    if (table_index == 0) {
        uint32_t up[20];
        uint32_t down[2];
        for (int c = 0; c < 20; c++) up[c] = all_cols[c * n_rows + row];
        down[0] = all_cols[0 * n_rows + row];
        down[1] = all_cols[1 * n_rows + row];

        uint32_t constraints[13], bus_data[5], bus_value[5];
        eval_execution_air(up, down, constraints, bus_data);
        COMPUTE_BUS_VALUE(bus_data, logup_alphas, bus_beta, bus_value);
        WEIGHTED_SUM_WITH_BUS(bus_value, constraints, alphas, 12, weighted);
    } else if (table_index == 1) {
        uint32_t up[29];
        uint32_t down[13];
        const uint32_t down_cols[13] = {1, 0, 5, 2, 3, 4, 6, 7, 24, 25, 26, 27, 28};
        for (int c = 0; c < 29; c++) up[c] = all_cols[c * n_rows + row];
        for (int c = 0; c < 13; c++) down[c] = all_cols[down_cols[c] * n_rows + row];

        uint32_t constraints[33], bus_data[5], bus_value[5];
        eval_extension_op_air(up, down, constraints, bus_data);
        COMPUTE_BUS_VALUE(bus_data, logup_alphas, bus_beta, bus_value);
        WEIGHTED_SUM_WITH_BUS(bus_value, constraints, alphas, 33, weighted);
    } else if (table_index == 2) {
        uint32_t up[100];
        for (int c = 0; c < 100; c++) up[c] = all_cols[c * n_rows + row];

        uint32_t constraints[81], bus_data[5], bus_value[5];
        eval_poseidon16_air(up, constraints, rc, mds, sparse, bus_data);
        COMPUTE_BUS_VALUE(bus_data, logup_alphas, bus_beta, bus_value);
        WEIGHTED_SUM_WITH_BUS(bus_value, constraints, alphas, 81, weighted);
    }

    #pragma unroll
    for (int k = 0; k < 5; k++) out[k] = weighted[k];
}

#define POSEIDON16_LOW_START 37
#define POSEIDON16_LOW_COUNT 20

__device__ __forceinline__ uint32_t poseidon16_low_lagrange_coeff(int hi_idx, int low_idx) {
    const uint32_t coeffs[6][4] = {
        {532676608u, 1065353219u, 2130706428u, 532676612u},
        {2130706432u, 9u, 2130706417u, 9u},
        {1065353214u, 21u, 2130706398u, 1065353234u},
        {2130706428u, 40u, 2130706369u, 30u},
        {1598029816u, 1065353284u, 2130706328u, 1598029872u},
        {2130706419u, 105u, 2130706273u, 70u},
    };
    return kb_to_monty(coeffs[hi_idx][low_idx]);
}

__device__ __forceinline__ void poseidon16_base_low_weighted(
    const uint32_t* constraints,
    const uint32_t* alphas,
    uint32_t out[5]
) {
    qe_zero(out);
    for (int c = POSEIDON16_LOW_START; c < POSEIDON16_LOW_START + POSEIDON16_LOW_COUNT; c++) {
        for (int k = 0; k < 5; k++) {
            out[k] = kb_add(out[k], kb_mul(alphas[(c + 1) * 5 + k], constraints[c]));
        }
    }
}

__device__ __forceinline__ void poseidon16_interpolate_low_weight(
    const uint32_t low_weighted[4][5],
    int hi_idx,
    uint32_t out[5]
) {
    qe_zero(out);
    for (int i = 0; i < 4; i++) {
        uint32_t term[5];
        qe_base_mul(low_weighted[i], poseidon16_low_lagrange_coeff(hi_idx, i), term);
        qe_add(out, term, out);
    }
}

__device__ __forceinline__ void poseidon16_interpolate_base_state(
    const uint32_t state0[16],
    const uint32_t state2[16],
    int hi_z,
    uint32_t out[16]
) {
    uint32_t z_half = kb_halve(kb_to_monty((uint32_t)hi_z));
    for (int i = 0; i < 16; i++) {
        out[i] = kb_add(state0[i], kb_mul(kb_sub(state2[i], state0[i]), z_half));
    }
}

__device__ __forceinline__ void poseidon16_interpolate_ext_state(
    const uint32_t* state0,
    const uint32_t* state2,
    int hi_z,
    uint32_t* out
) {
    uint32_t z_half = kb_halve(kb_to_monty((uint32_t)hi_z));
    for (int i = 0; i < 16; i++) {
        uint32_t diff[5], term[5];
        qe_sub(state2 + i * 5, state0 + i * 5, diff);
        qe_base_mul(diff, z_half, term);
        qe_add(state0 + i * 5, term, out + i * 5);
    }
}

extern "C" __global__ void air_execution_multi_z_fb_kernel(
    const uint32_t* __restrict__ columns,
    const uint32_t* __restrict__ down_cols,
    const uint32_t* __restrict__ eq_factor,
    const uint32_t* __restrict__ alphas,     // 13 * 5 ext (alpha[0]=bus, alpha[1..12]=non-bus)
    const uint32_t* __restrict__ logup_alphas, // 5 * 5 ext (logup alpha eq poly)
    const uint32_t* __restrict__ bus_beta_p, // 5 ext
    uint32_t* __restrict__ partial_sums,
    uint32_t n_rows,
    uint32_t n_pairs,
    uint32_t fold_bit
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc[5][5]; // degree 5 → 5 z-points: z=0,2,3,4,5
    for (int z = 0; z < 5; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;

    // Load bus constants into registers
    uint32_t la[25]; for (int i = 0; i < 25; i++) la[i] = logup_alphas[i];
    uint32_t bb[5]; for (int k = 0; k < 5; k++) bb[k] = bus_beta_p[k];

    if (gid < n_pairs) {
        uint32_t i0, i1; FOLD_BIT_PAIR(gid, fold_bit, i0, i1);
        uint32_t v0_up[20], diff_up[20];
        for (int c = 0; c < 20; c++) {
            v0_up[c] = columns[c * n_rows + i0];
            diff_up[c] = kb_sub(columns[c * n_rows + i1], v0_up[c]);
        }
        uint32_t v0_dn[2], diff_dn[2];
        for (int c = 0; c < 2; c++) {
            v0_dn[c] = down_cols[c * n_rows + i0];
            diff_dn[c] = kb_sub(down_cols[c * n_rows + i1], v0_dn[c]);
        }
        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];
        // z=0
        {
            uint32_t constraints[13], bd[5];
            eval_execution_air(v0_up, v0_dn, constraints, bd);
            uint32_t bv[5]; COMPUTE_BUS_VALUE(bd, la, bb, bv);
            uint32_t w[5]; WEIGHTED_SUM_WITH_BUS(bv, constraints, alphas, 12, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }
        // z=1 skip, z=2,3,4,5
        uint32_t up[20], dn[2];
        for (int c = 0; c < 20; c++) up[c] = kb_add(v0_up[c], diff_up[c]);
        for (int c = 0; c < 2; c++) dn[c] = kb_add(v0_dn[c], diff_dn[c]);
        for (int zi = 0; zi < 4; zi++) { // 4 iterations: z=2,3,4,5
            for (int c = 0; c < 20; c++) up[c] = kb_add(up[c], diff_up[c]);
            for (int c = 0; c < 2; c++) dn[c] = kb_add(dn[c], diff_dn[c]);
            uint32_t constraints[13], bd[5];
            eval_execution_air(up, dn, constraints, bd);
            uint32_t bv[5]; COMPUTE_BUS_VALUE(bd, la, bb, bv);
            uint32_t w[5]; WEIGHTED_SUM_WITH_BUS(bv, constraints, alphas, 12, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(5, acc, partial_sums);
}

// ── Multi-z ExtensionOp with fold_bit (base field, 5 z-points) ──────────
extern "C" __global__ void air_ext_op_multi_z_fb_kernel(
    const uint32_t* __restrict__ columns,
    const uint32_t* __restrict__ down_cols,
    const uint32_t* __restrict__ eq_factor,
    const uint32_t* __restrict__ alphas,
    const uint32_t* __restrict__ logup_alphas,
    const uint32_t* __restrict__ bus_beta_p,
    uint32_t* __restrict__ partial_sums,
    uint32_t n_rows,
    uint32_t n_pairs,
    uint32_t fold_bit
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc[6][5];
    for (int z = 0; z < 6; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;
    uint32_t la[25]; for (int i = 0; i < 25; i++) la[i] = logup_alphas[i];
    uint32_t bb[5]; for (int k = 0; k < 5; k++) bb[k] = bus_beta_p[k];
    if (gid < n_pairs) {
        uint32_t i0, i1; FOLD_BIT_PAIR(gid, fold_bit, i0, i1);
        uint32_t v0_up[29], diff_up[29];
        for (int c = 0; c < 29; c++) {
            v0_up[c] = columns[c * n_rows + i0];
            diff_up[c] = kb_sub(columns[c * n_rows + i1], v0_up[c]);
        }
        uint32_t v0_dn[13], diff_dn[13];
        for (int c = 0; c < 13; c++) {
            v0_dn[c] = down_cols[c * n_rows + i0];
            diff_dn[c] = kb_sub(down_cols[c * n_rows + i1], v0_dn[c]);
        }
        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];
        {
            uint32_t constraints[33], bd[5];
            eval_extension_op_air(v0_up, v0_dn, constraints, bd);
            uint32_t bv[5]; COMPUTE_BUS_VALUE(bd, la, bb, bv);
            uint32_t w[5]; WEIGHTED_SUM_WITH_BUS(bv, constraints, alphas, 33, w);

            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }
        uint32_t up_cur[29], dn_cur[13];
        for (int c = 0; c < 29; c++) up_cur[c] = kb_add(v0_up[c], diff_up[c]);
        for (int c = 0; c < 13; c++) dn_cur[c] = kb_add(v0_dn[c], diff_dn[c]);
        for (int zi = 0; zi < 5; zi++) {
            for (int c = 0; c < 29; c++) up_cur[c] = kb_add(up_cur[c], diff_up[c]);
            for (int c = 0; c < 13; c++) dn_cur[c] = kb_add(dn_cur[c], diff_dn[c]);
            uint32_t constraints[33], bd[5];
            eval_extension_op_air(up_cur, dn_cur, constraints, bd);
            uint32_t bv[5]; COMPUTE_BUS_VALUE(bd, la, bb, bv);
            uint32_t w[5]; WEIGHTED_SUM_WITH_BUS(bv, constraints, alphas, 33, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(6, acc, partial_sums);
}

// ── Multi-z Poseidon16 with fold_bit (base field, 10 z-points) ──────────
extern "C" __global__ void air_poseidon16_multi_z_fb_kernel(
    const uint32_t* __restrict__ columns,
    const uint32_t* __restrict__ eq_factor,
    const uint32_t* __restrict__ alphas,
    const uint32_t* __restrict__ rc,
    const uint32_t* __restrict__ mds,
    const uint32_t* __restrict__ sparse,
    const uint32_t* __restrict__ logup_alphas,
    const uint32_t* __restrict__ bus_beta_p,
    uint32_t* __restrict__ partial_sums,
    uint32_t n_rows,
    uint32_t n_pairs,
    uint32_t fold_bit
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc[10][5];
    for (int z = 0; z < 10; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;
    uint32_t la[25]; for (int i = 0; i < 25; i++) la[i] = logup_alphas[i];
    uint32_t bb[5]; for (int k = 0; k < 5; k++) bb[k] = bus_beta_p[k];
    if (gid < n_pairs) {
        uint32_t i0, i1; FOLD_BIT_PAIR(gid, fold_bit, i0, i1);
        uint32_t v0[100], diff[100];
        for (int c = 0; c < 100; c++) {
            v0[c] = columns[c * n_rows + i0];
            diff[c] = kb_sub(columns[c * n_rows + i1], v0[c]);
        }
        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];
        uint32_t low_weighted[4][5];
        uint32_t state0[16], state2[16];
        {
            uint32_t constraints[81], bd[5];
            eval_poseidon16_air(v0, constraints, rc, mds, sparse, bd, state0);
            poseidon16_base_low_weighted(constraints, alphas, low_weighted[0]);
            uint32_t bv[5]; COMPUTE_BUS_VALUE(bd, la, bb, bv);
            uint32_t w[5]; WEIGHTED_SUM_WITH_BUS(bv, constraints, alphas, 81, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }
        uint32_t cur[100];
        for (int c = 0; c < 100; c++) cur[c] = kb_add(v0[c], diff[c]);
        for (int zi = 0; zi < 3; zi++) {
            for (int c = 0; c < 100; c++) cur[c] = kb_add(cur[c], diff[c]);
            uint32_t constraints[81], bd[5];
            uint32_t* post_state = (zi == 0) ? state2 : nullptr;
            eval_poseidon16_air(cur, constraints, rc, mds, sparse, bd, post_state);
            poseidon16_base_low_weighted(constraints, alphas, low_weighted[zi + 1]);
            uint32_t bv[5]; COMPUTE_BUS_VALUE(bd, la, bb, bv);
            uint32_t w[5]; WEIGHTED_SUM_WITH_BUS(bv, constraints, alphas, 81, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
        for (int hi = 0; hi < 6; hi++) {
            for (int c = 0; c < 100; c++) cur[c] = kb_add(cur[c], diff[c]);
            uint32_t cached_state[16], constraints[81], bd[5];
            poseidon16_interpolate_base_state(state0, state2, 5 + hi, cached_state);
            eval_poseidon16_air(cur, constraints, rc, mds, sparse, bd, nullptr, true, cached_state);
            uint32_t bv[5]; COMPUTE_BUS_VALUE(bd, la, bb, bv);
            uint32_t high_w[5]; WEIGHTED_SUM_WITH_BUS(bv, constraints, alphas, 81, high_w);
            uint32_t low_interp[5], w[5];
            poseidon16_interpolate_low_weight(low_weighted, hi, low_interp);
            qe_add(high_w, low_interp, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[4+hi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(10, acc, partial_sums);
}

// ── Ext-field multi-z kernels with fold_bit ─────────────────────────────

extern "C" __global__ void air_execution_multi_z_ext_fb_kernel(
    const uint32_t* __restrict__ columns,
    const uint32_t* __restrict__ down_cols,
    const uint32_t* __restrict__ eq_factor,
    const uint32_t* __restrict__ alphas,
    const uint32_t* __restrict__ logup_alphas,
    const uint32_t* __restrict__ bus_beta_p,
    uint32_t* __restrict__ partial_sums,
    uint32_t n_elems,
    uint32_t n_pairs,
    uint32_t fold_bit
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc[5][5];
    for (int z = 0; z < 5; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;
    uint32_t la[25]; for (int i = 0; i < 25; i++) la[i] = logup_alphas[i];
    uint32_t bb[5]; for (int k = 0; k < 5; k++) bb[k] = bus_beta_p[k];
    if (gid < n_pairs) {
        uint32_t i0, i1; FOLD_BIT_PAIR(gid, fold_bit, i0, i1);
        uint32_t v0_up[20*5], diff_up[20*5];
        for (int c = 0; c < 20; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0_up[c*5+k] = columns[base_off + i0*5 + k];
                diff_up[c*5+k] = kb_sub(columns[base_off + i1*5 + k], v0_up[c*5+k]);
            }
        }
        uint32_t v0_dn[2*5], diff_dn[2*5];
        for (int c = 0; c < 2; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0_dn[c*5+k] = down_cols[base_off + i0*5 + k];
                diff_dn[c*5+k] = kb_sub(down_cols[base_off + i1*5 + k], v0_dn[c*5+k]);
            }
        }
        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];
        {
            uint32_t nonbus_w[5], bd[25];
            eval_execution_air_ext_weighted(v0_up, v0_dn, alphas, nonbus_w, bd);
            uint32_t bv[5]; COMPUTE_BUS_VALUE_EXT(bd, la, bb, bv);
            uint32_t w[5]; ADD_BUS_TO_WEIGHTED_EXT(bv, alphas, nonbus_w, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }
        uint32_t cur_up[20*5], cur_dn[2*5];
        for (int i = 0; i < 20*5; i++) cur_up[i] = kb_add(v0_up[i], diff_up[i]);
        for (int i = 0; i < 2*5; i++) cur_dn[i] = kb_add(v0_dn[i], diff_dn[i]);
        for (int zi = 0; zi < 4; zi++) {
            for (int i = 0; i < 20*5; i++) cur_up[i] = kb_add(cur_up[i], diff_up[i]);
            for (int i = 0; i < 2*5; i++) cur_dn[i] = kb_add(cur_dn[i], diff_dn[i]);
            uint32_t nonbus_w[5], bd[25];
            eval_execution_air_ext_weighted(cur_up, cur_dn, alphas, nonbus_w, bd);
            uint32_t bv[5]; COMPUTE_BUS_VALUE_EXT(bd, la, bb, bv);
            uint32_t w[5]; ADD_BUS_TO_WEIGHTED_EXT(bv, alphas, nonbus_w, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(5, acc, partial_sums);
}

extern "C" __global__ void air_ext_op_multi_z_ext_fb_kernel(
    const uint32_t* __restrict__ columns,
    const uint32_t* __restrict__ down_cols,
    const uint32_t* __restrict__ eq_factor,
    const uint32_t* __restrict__ alphas,
    const uint32_t* __restrict__ logup_alphas,
    const uint32_t* __restrict__ bus_beta_p,
    uint32_t* __restrict__ partial_sums,
    uint32_t n_elems,
    uint32_t n_pairs,
    uint32_t fold_bit
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc[6][5];
    for (int z = 0; z < 6; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;
    uint32_t la[25]; for (int i = 0; i < 25; i++) la[i] = logup_alphas[i];
    uint32_t bb[5]; for (int k = 0; k < 5; k++) bb[k] = bus_beta_p[k];
    if (gid < n_pairs) {
        uint32_t i0, i1; FOLD_BIT_PAIR(gid, fold_bit, i0, i1);
        uint32_t v0_up[29*5], diff_up[29*5];
        for (int c = 0; c < 29; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0_up[c*5+k] = columns[base_off + i0*5 + k];
                diff_up[c*5+k] = kb_sub(columns[base_off + i1*5 + k], v0_up[c*5+k]);
            }
        }
        uint32_t v0_dn[13*5], diff_dn[13*5];
        for (int c = 0; c < 13; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0_dn[c*5+k] = down_cols[base_off + i0*5 + k];
                diff_dn[c*5+k] = kb_sub(down_cols[base_off + i1*5 + k], v0_dn[c*5+k]);
            }
        }
        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];
        {
            uint32_t nonbus_w[5], bd[25];
            eval_ext_op_air_ext_weighted(v0_up, v0_dn, alphas, nonbus_w, bd);
            uint32_t bv[5]; COMPUTE_BUS_VALUE_EXT(bd, la, bb, bv);
            uint32_t w[5]; ADD_BUS_TO_WEIGHTED_EXT(bv, alphas, nonbus_w, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }
        uint32_t cur_up[29*5], cur_dn[13*5];
        for (int i = 0; i < 29*5; i++) cur_up[i] = kb_add(v0_up[i], diff_up[i]);
        for (int i = 0; i < 13*5; i++) cur_dn[i] = kb_add(v0_dn[i], diff_dn[i]);
        for (int zi = 0; zi < 5; zi++) {
            for (int i = 0; i < 29*5; i++) cur_up[i] = kb_add(cur_up[i], diff_up[i]);
            for (int i = 0; i < 13*5; i++) cur_dn[i] = kb_add(cur_dn[i], diff_dn[i]);
            uint32_t nonbus_w[5], bd[25];
            eval_ext_op_air_ext_weighted(cur_up, cur_dn, alphas, nonbus_w, bd);
            uint32_t bv[5]; COMPUTE_BUS_VALUE_EXT(bd, la, bb, bv);
            uint32_t w[5]; ADD_BUS_TO_WEIGHTED_EXT(bv, alphas, nonbus_w, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(6, acc, partial_sums);
}

extern "C" __global__ void air_poseidon16_multi_z_ext_fb_kernel(
    const uint32_t* __restrict__ columns,
    const uint32_t* __restrict__ eq_factor,
    const uint32_t* __restrict__ alphas,
    const uint32_t* __restrict__ rc,
    const uint32_t* __restrict__ mds,
    const uint32_t* __restrict__ sparse,
    const uint32_t* __restrict__ logup_alphas,
    const uint32_t* __restrict__ bus_beta_p,
    uint32_t* __restrict__ partial_sums,
    uint32_t n_elems,
    uint32_t n_pairs,
    uint32_t fold_bit
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t acc[10][5];
    for (int z = 0; z < 10; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;
    uint32_t la[25]; for (int i = 0; i < 25; i++) la[i] = logup_alphas[i];
    uint32_t bb[5]; for (int k = 0; k < 5; k++) bb[k] = bus_beta_p[k];
    if (gid < n_pairs) {
        uint32_t i0, i1; FOLD_BIT_PAIR(gid, fold_bit, i0, i1);
        uint32_t v0[100*5], diff[100*5];
        for (int c = 0; c < 100; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0[c*5+k] = columns[base_off + i0*5 + k];
                diff[c*5+k] = kb_sub(columns[base_off + i1*5 + k], v0[c*5+k]);
            }
        }
        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];
        uint32_t low_weighted[4][5];
        uint32_t state0[16 * 5], state2[16 * 5];
        {
            uint32_t nonbus_w[5], bd[25];
            eval_poseidon16_air_ext_weighted(v0, alphas, rc, mds, sparse, nonbus_w, bd, low_weighted[0], state0);
            uint32_t bv[5]; COMPUTE_BUS_VALUE_EXT(bd, la, bb, bv);
            uint32_t w[5]; ADD_BUS_TO_WEIGHTED_EXT(bv, alphas, nonbus_w, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }
        uint32_t cur[100*5];
        for (int i = 0; i < 100*5; i++) cur[i] = kb_add(v0[i], diff[i]);
        for (int zi = 0; zi < 3; zi++) {
            for (int i = 0; i < 100*5; i++) cur[i] = kb_add(cur[i], diff[i]);
            uint32_t nonbus_w[5], bd[25];
            uint32_t* post_state = (zi == 0) ? state2 : nullptr;
            eval_poseidon16_air_ext_weighted(cur, alphas, rc, mds, sparse, nonbus_w, bd, low_weighted[zi + 1], post_state);
            uint32_t bv[5]; COMPUTE_BUS_VALUE_EXT(bd, la, bb, bv);
            uint32_t w[5]; ADD_BUS_TO_WEIGHTED_EXT(bv, alphas, nonbus_w, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
        for (int hi = 0; hi < 6; hi++) {
            for (int i = 0; i < 100*5; i++) cur[i] = kb_add(cur[i], diff[i]);
            uint32_t cached_state[16 * 5], nonbus_w[5], bd[25];
            poseidon16_interpolate_ext_state(state0, state2, 5 + hi, cached_state);
            eval_poseidon16_air_ext_weighted(cur, alphas, rc, mds, sparse, nonbus_w, bd, nullptr, nullptr, true, cached_state);
            uint32_t bv[5]; COMPUTE_BUS_VALUE_EXT(bd, la, bb, bv);
            uint32_t high_w[5]; ADD_BUS_TO_WEIGHTED_EXT(bv, alphas, nonbus_w, high_w);
            uint32_t low_interp[5], w[5];
            poseidon16_interpolate_low_weight(low_weighted, hi, low_interp);
            qe_add(high_w, low_interp, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[4+hi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(10, acc, partial_sums);
}

// ── Ext-field multi-z Execution kernel (4 z-points) ─────────────────────
extern "C" __global__ void air_execution_multi_z_ext_kernel(
    const uint32_t* __restrict__ columns,   // 20 * n_elems * 5 ext (col-major)
    const uint32_t* __restrict__ down_cols,  // 2 * n_elems * 5 ext (col-major)
    const uint32_t* __restrict__ eq_factor,  // n_pairs * 5 ext
    const uint32_t* __restrict__ alphas,     // 13 * 5 ext (actually 12+bus)
    uint32_t* __restrict__ partial_sums,     // 4 * n_blocks * 5 ext
    uint32_t n_elems,                        // current element count (2*n_pairs)
    uint32_t n_pairs
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t acc[4][5];
    for (int z = 0; z < 4; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;

    if (gid < n_pairs) {
        uint32_t i0 = gid, i1 = gid + n_pairs;

        // Load ext-field column values: each column c has n_elems ext elements (5 u32s each)
        // Layout: columns[c * n_elems * 5 + elem * 5 + k]
        uint32_t v0_up[20*5], diff_up[20*5];
        for (int c = 0; c < 20; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0_up[c*5+k] = columns[base_off + i0*5 + k];
                uint32_t hi = columns[base_off + i1*5 + k];
                diff_up[c*5+k] = kb_sub(hi, v0_up[c*5+k]);
            }
        }
        uint32_t v0_dn[2*5], diff_dn[2*5];
        for (int c = 0; c < 2; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0_dn[c*5+k] = down_cols[base_off + i0*5 + k];
                uint32_t hi = down_cols[base_off + i1*5 + k];
                diff_dn[c*5+k] = kb_sub(hi, v0_dn[c*5+k]);
            }
        }

        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];

        // z=0
        {
            uint32_t w[5];
            eval_execution_air_ext_weighted(v0_up, v0_dn, alphas, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }

        // Advance to z=1 (skip)
        uint32_t cur_up[20*5], cur_dn[2*5];
        for (int i = 0; i < 20*5; i++) cur_up[i] = kb_add(v0_up[i], diff_up[i]);
        for (int i = 0; i < 2*5; i++) cur_dn[i] = kb_add(v0_dn[i], diff_dn[i]);

        // z=2,3,4
        for (int zi = 0; zi < 3; zi++) {
            for (int i = 0; i < 20*5; i++) cur_up[i] = kb_add(cur_up[i], diff_up[i]);
            for (int i = 0; i < 2*5; i++) cur_dn[i] = kb_add(cur_dn[i], diff_dn[i]);
            uint32_t w[5];
            eval_execution_air_ext_weighted(cur_up, cur_dn, alphas, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(4, acc, partial_sums);
}

// ── Ext-field multi-z ExtensionOp kernel (5 z-points) ───────────────────
extern "C" __global__ void air_ext_op_multi_z_ext_kernel(
    const uint32_t* __restrict__ columns,    // 29 * n_elems * 5 ext
    const uint32_t* __restrict__ down_cols,   // 13 * n_elems * 5 ext
    const uint32_t* __restrict__ eq_factor,   // n_pairs * 5 ext
    const uint32_t* __restrict__ alphas,      // 33 * 5 ext
    uint32_t* __restrict__ partial_sums,      // 5 * n_blocks * 5 ext
    uint32_t n_elems,
    uint32_t n_pairs
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t acc[6][5];
    for (int z = 0; z < 6; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;

    if (gid < n_pairs) {
        uint32_t i0 = gid, i1 = gid + n_pairs;

        uint32_t v0_up[29*5], diff_up[29*5];
        for (int c = 0; c < 29; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0_up[c*5+k] = columns[base_off + i0*5 + k];
                diff_up[c*5+k] = kb_sub(columns[base_off + i1*5 + k], v0_up[c*5+k]);
            }
        }
        uint32_t v0_dn[13*5], diff_dn[13*5];
        for (int c = 0; c < 13; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0_dn[c*5+k] = down_cols[base_off + i0*5 + k];
                diff_dn[c*5+k] = kb_sub(down_cols[base_off + i1*5 + k], v0_dn[c*5+k]);
            }
        }

        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];

        // z=0
        {
            uint32_t w[5];
            eval_ext_op_air_ext_weighted(v0_up, v0_dn, alphas, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }

        uint32_t cur_up[29*5], cur_dn[13*5];
        for (int i = 0; i < 29*5; i++) cur_up[i] = kb_add(v0_up[i], diff_up[i]);
        for (int i = 0; i < 13*5; i++) cur_dn[i] = kb_add(v0_dn[i], diff_dn[i]);

        for (int zi = 0; zi < 5; zi++) {
            for (int i = 0; i < 29*5; i++) cur_up[i] = kb_add(cur_up[i], diff_up[i]);
            for (int i = 0; i < 13*5; i++) cur_dn[i] = kb_add(cur_dn[i], diff_dn[i]);
            uint32_t w[5];
            eval_ext_op_air_ext_weighted(cur_up, cur_dn, alphas, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(6, acc, partial_sums);
}

// ── Ext-field multi-z Poseidon16 kernel (10 z-points) ───────────────────
extern "C" __global__ void air_poseidon16_multi_z_ext_kernel(
    const uint32_t* __restrict__ columns,    // 100 * n_elems * 5 ext
    const uint32_t* __restrict__ eq_factor,  // n_pairs * 5 ext
    const uint32_t* __restrict__ alphas,     // 81 * 5 ext
    const uint32_t* __restrict__ rc,         // 28 * 16 base field
    const uint32_t* __restrict__ mds,        // 16 base field
    const uint32_t* __restrict__ sparse,     // sparse matrix data (base)
    uint32_t* __restrict__ partial_sums,     // 9 * n_blocks * 5 ext
    uint32_t n_elems,
    uint32_t n_pairs
) {
    extern __shared__ uint32_t smem[];
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t acc[10][5];
    for (int z = 0; z < 10; z++) for (int k = 0; k < 5; k++) acc[z][k] = 0;

    if (gid < n_pairs) {
        uint32_t i0 = gid, i1 = gid + n_pairs;

        // Load ext-field columns. 100 cols × 5 components each.
        // Use local memory (will spill, but it's cached).
        uint32_t v0[100*5], diff[100*5];
        for (int c = 0; c < 100; c++) {
            uint32_t base_off = c * n_elems * 5;
            for (int k = 0; k < 5; k++) {
                v0[c*5+k] = columns[base_off + i0*5 + k];
                diff[c*5+k] = kb_sub(columns[base_off + i1*5 + k], v0[c*5+k]);
            }
        }

        uint32_t eq[5];
        for (int k = 0; k < 5; k++) eq[k] = eq_factor[gid * 5 + k];

        // z=0
        {
            uint32_t w[5];
            eval_poseidon16_air_ext_weighted(v0, alphas, rc, mds, sparse, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[0][k] = prod[k];
        }

        // z=1 skip
        uint32_t cur[100*5];
        for (int i = 0; i < 100*5; i++) cur[i] = kb_add(v0[i], diff[i]);

        for (int zi = 0; zi < 9; zi++) {
            for (int i = 0; i < 100*5; i++) cur[i] = kb_add(cur[i], diff[i]);
            uint32_t w[5];
            eval_poseidon16_air_ext_weighted(cur, alphas, rc, mds, sparse, w);
            uint32_t prod[5]; qe_mul(w, eq, prod);
            for (int k = 0; k < 5; k++) acc[1+zi][k] = prod[k];
        }
    }
    MULTI_Z_REDUCE(10, acc, partial_sums);
}

// Debug kernel: evaluate Poseidon16 ext-field constraints, return weighted + bus
extern "C" __global__ void debug_poseidon16_ext_eval_kernel(
    const uint32_t* __restrict__ up_vals,  // 100 * 5 ext field values
    const uint32_t* __restrict__ alphas,   // (1+81) * 5 ext
    const uint32_t* __restrict__ rc,
    const uint32_t* __restrict__ mds,
    const uint32_t* __restrict__ sparse,
    const uint32_t* __restrict__ logup_alphas,
    const uint32_t* __restrict__ bus_beta_p,
    uint32_t* __restrict__ result          // [0..5]=nonbus_w, [5..10]=total_w, [10..35]=bus_data
) {
    uint32_t w[5], bd[25];
    eval_poseidon16_air_ext_weighted(up_vals, alphas, rc, mds, sparse, w, bd);
    for (int k = 0; k < 5; k++) result[k] = w[k]; // non-bus weighted sum
    // Compute bus
    uint32_t la[25]; for (int i = 0; i < 25; i++) la[i] = logup_alphas[i];
    uint32_t bb[5]; for (int k = 0; k < 5; k++) bb[k] = bus_beta_p[k];
    uint32_t bv[5]; COMPUTE_BUS_VALUE_EXT(bd, la, bb, bv);
    uint32_t total[5]; ADD_BUS_TO_WEIGHTED_EXT(bv, alphas, w, total);
    for (int k = 0; k < 5; k++) result[5 + k] = total[k]; // total weighted sum
    for (int i = 0; i < 25; i++) result[10 + i] = bd[i]; // bus data
}

// ── Protocol step kernel ─────────────────────────────────────────────────
// Runs on a SINGLE thread. Replaces all CPU round-loop glue:
// pe construction → Lagrange interpolation → expand_bare_to_full →
// accumulate cc → Fiat-Shamir hash → sample challenge → update state.
//
// Per-table state stored in device memory. Fiat-Shamir state on device.
// After this kernel, the fold kernels can run with the challenge.
//
// Inputs:
//   ze_all: per-table ze values (flattened: table0_ze[nz0], table1_ze[nz1], ...)
//   table_state: per-table {sum(5), mmf(5), ea(5), nz, eta_k(5), pad_contrib(5), joined}
//   n_tables: number of tables
//   mfd: max full degree (length of cc vector)
//   challenger_state: GpuChallenger (Fiat-Shamir state)
//   rc, mds, sparse: Poseidon16 constants for Fiat-Shamir
//
// Outputs:
//   cc: combined polynomial coefficients (mfd+1 ext elements)
//   challenge: sampled challenge (5 u32s)
//   transcript_chunk: the scalars added to transcript this round
//   updated table_state: sum, mmf updated from challenge
//
// Layout of table_state per table (35 u32s):
//   [0..5]  = sum (ext)
//   [5..10] = mmf (ext)
//   [10..15] = ea (ext) — eq_alpha for this round
//   [15]    = nz (degree, as base field monty — or just raw u32)
//   [16..21] = eta_k (eta^i * k[i], ext)
//   [21..26] = pad_contrib (ext, 0 if no padding)
//   [26]    = joined (1 if table has joined this round, 0 if still waiting)
//   Total: 27 u32s per table (padded to 32 for alignment)

#define TS_SUM    0
#define TS_MMF    5
#define TS_EA    10
#define TS_NZ    15
#define TS_ETAK  16
#define TS_PAD   21
#define TS_JOIN  26
#define TS_EA_INV 27
#define TS_STRIDE 32
#define AIR_NO_PADDING_SENTINEL 0xFFFFFFFFu

extern "C" __global__ void air_sumcheck_patch_state_kernel(
    const uint32_t* __restrict__ round_meta,
    uint32_t* __restrict__ table_state,
    uint32_t n_tables
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tables) return;

    const uint32_t* meta = round_meta + tid * TS_STRIDE;
    uint32_t* state = table_state + tid * TS_STRIDE;

    for (int k = 0; k < 5; k++) state[TS_EA + k] = meta[TS_EA + k];
    state[TS_NZ] = meta[TS_NZ];
    for (int k = 0; k < 5; k++) state[TS_PAD + k] = meta[TS_PAD + k];
    state[TS_JOIN] = meta[TS_JOIN];
    for (int k = 0; k < 5; k++) state[TS_EA_INV + k] = meta[TS_EA_INV + k];
}

__device__ __forceinline__ bool qe_is_zero_dev(const uint32_t* a) {
    return (a[0] | a[1] | a[2] | a[3] | a[4]) == 0;
}

__device__ void mle_zeros_then_ones_permuted_suffix_dev(
    uint32_t n_zeros,
    const uint32_t* __restrict__ point_words,
    uint32_t total_coords,
    uint32_t n_vars,
    uint32_t local_round,
    uint32_t pivot,
    uint32_t out[5]
) {
    uint32_t ONE[5]; qe_one(ONE);
    uint32_t ZERO[5]; qe_zero(ZERO);

    uint32_t scale[5];
    uint32_t offset[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        scale[k] = ONE[k];
        offset[k] = ZERO[k];
    }

    uint32_t len = n_vars - local_round;
    uint32_t n_values = 1u << len;
    uint32_t fold_bit = local_round < pivot ? pivot - 1 - local_round : 0;
    uint32_t zero_idx = len - 1 - fold_bit;
    uint32_t head_len = n_vars > pivot ? n_vars - pivot : 0;
    if (head_len > len) head_len = len;

    for (uint32_t i = 0; i < len; i++) {
        if (n_zeros == 0) break;
        if (n_zeros == n_values) {
            #pragma unroll
            for (int k = 0; k < 5; k++) out[k] = offset[k];
            return;
        }

        uint32_t p[5];
        if (i == zero_idx) {
            qe_zero(p);
        } else {
            uint32_t src_rel = i < head_len ? i : len - 1 - (i - head_len);
            uint32_t src_coord = total_coords - n_vars + src_rel;
            const uint32_t* src = point_words + src_coord * 5;
            #pragma unroll
            for (int k = 0; k < 5; k++) p[k] = src[k];
        }

        uint32_t half = n_values >> 1;
        if (n_zeros < half) {
            uint32_t term[5], one_m_p[5], new_scale[5], new_offset[5];
            qe_mul(scale, p, term);
            qe_add(offset, term, new_offset);
            qe_sub(ONE, p, one_m_p);
            qe_mul(scale, one_m_p, new_scale);
            #pragma unroll
            for (int k = 0; k < 5; k++) {
                scale[k] = new_scale[k];
                offset[k] = new_offset[k];
            }
        } else {
            uint32_t new_scale[5];
            qe_mul(scale, p, new_scale);
            #pragma unroll
            for (int k = 0; k < 5; k++) scale[k] = new_scale[k];
            n_zeros -= half;
        }
        n_values = half;
    }

    if (n_zeros == 0) {
        uint32_t sum[5];
        qe_add(offset, scale, sum);
        #pragma unroll
        for (int k = 0; k < 5; k++) out[k] = sum[k];
    } else {
        #pragma unroll
        for (int k = 0; k < 5; k++) out[k] = offset[k];
    }
}

extern "C" __global__ void air_sumcheck_patch_state_from_gkr_kernel(
    const uint32_t* __restrict__ round_meta,
    const uint32_t* __restrict__ round_extra,
    const uint32_t* __restrict__ point_words,
    uint32_t total_coords,
    uint32_t* __restrict__ table_state,
    uint32_t n_tables
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tables) return;

    const uint32_t* meta = round_meta + tid * TS_STRIDE;
    const uint32_t* extra = round_extra + tid * 4;
    uint32_t* state = table_state + tid * TS_STRIDE;

    state[TS_NZ] = meta[TS_NZ];
    state[TS_JOIN] = meta[TS_JOIN];
    if (!meta[TS_JOIN]) {
        #pragma unroll
        for (int k = 0; k < 5; k++) {
            state[TS_EA + k] = 0;
            state[TS_PAD + k] = 0;
            state[TS_EA_INV + k] = 0;
        }
        return;
    }

    uint32_t n_vars = extra[0];
    uint32_t local_round = extra[1];
    uint32_t current_unpadded_len = extra[2];
    uint32_t pivot = extra[3];

    const uint32_t* eq_alpha = point_words + (total_coords - 1 - local_round) * 5;
    #pragma unroll
    for (int k = 0; k < 5; k++) state[TS_EA + k] = eq_alpha[k];

    if (qe_is_zero_dev(eq_alpha)) {
        #pragma unroll
        for (int k = 0; k < 5; k++) state[TS_EA_INV + k] = 0;
    } else {
        qe_inv(eq_alpha, state + TS_EA_INV);
    }

    if (current_unpadded_len == AIR_NO_PADDING_SENTINEL) {
        #pragma unroll
        for (int k = 0; k < 5; k++) state[TS_PAD + k] = 0;
    } else {
        uint32_t padding_eval[5];
        mle_zeros_then_ones_permuted_suffix_dev(
            current_unpadded_len,
            point_words,
            total_coords,
            n_vars,
            local_round,
            pivot,
            padding_eval
        );
        qe_mul(meta + TS_PAD, padding_eval, state + TS_PAD);
    }
}

extern "C" __global__ void air_sumcheck_patch_pad_beta_kernel(
    const uint32_t* __restrict__ pad_coeffs,
    const uint32_t* __restrict__ bus_beta,
    uint32_t* __restrict__ round_meta,
    uint32_t n_tables
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tables) return;

    const uint32_t* coeff = pad_coeffs + tid * 5;
    uint32_t* meta = round_meta + tid * TS_STRIDE;

    uint32_t beta_term[5];
    qe_mul(bus_beta, coeff, beta_term);

    uint32_t patched[5];
    qe_add(meta + TS_PAD, beta_term, patched);
    #pragma unroll
    for (int k = 0; k < 5; k++) meta[TS_PAD + k] = patched[k];
}

extern "C" __global__ void air_sumcheck_patch_pad_evals_kernel(
    const uint32_t* __restrict__ pad_evals,
    const uint32_t* __restrict__ round_extra,
    uint32_t* __restrict__ round_meta,
    uint32_t n_tables
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tables) return;

    const uint32_t* pad_eval = pad_evals + tid * 5;
    const uint32_t* extra = round_extra + tid * 4;
    uint32_t* meta = round_meta + tid * TS_STRIDE;

    if (extra[2] == AIR_NO_PADDING_SENTINEL) {
        #pragma unroll
        for (int k = 0; k < 5; k++) meta[TS_PAD + k] = 0;
        return;
    }

    #pragma unroll
    for (int k = 0; k < 5; k++) meta[TS_PAD + k] = pad_eval[k];
}

extern "C" __global__ void air_sumcheck_init_sums_from_logup_kernel(
    const uint32_t* __restrict__ bus_numerators,
    const uint32_t* __restrict__ bus_denominators,
    const uint32_t* __restrict__ logup_c,
    const uint32_t* __restrict__ bus_beta,
    const uint32_t* __restrict__ bus_directions,
    uint32_t* __restrict__ table_state,
    uint32_t n_tables
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tables) return;

    const uint32_t* numer = bus_numerators + tid * 5;
    const uint32_t* denom = bus_denominators + tid * 5;
    uint32_t* state = table_state + tid * TS_STRIDE;

    uint32_t signed_numer[5];
    if (bus_directions[tid] == 0) {
        #pragma unroll
        for (int k = 0; k < 5; k++) signed_numer[k] = kb_neg(numer[k]);
    } else {
        #pragma unroll
        for (int k = 0; k < 5; k++) signed_numer[k] = numer[k];
    }

    uint32_t denom_minus_c[5];
    qe_sub(denom, logup_c, denom_minus_c);

    uint32_t beta_term[5];
    qe_mul(bus_beta, denom_minus_c, beta_term);

    uint32_t sum[5];
    qe_add(signed_numer, beta_term, sum);

    #pragma unroll
    for (int k = 0; k < 5; k++) state[TS_SUM + k] = sum[k];
}

// Lagrange interpolation: given n points (x_i, y_i) with x_i = i, compute polynomial coefficients.
// Uses the standard O(n^2) algorithm. n <= 12 typically (max_full_degree + 1).
__device__ void lagrange_interp(const uint32_t* pe, int n, uint32_t* coeffs) {
    // pe: n ext-field values (pe[i*5..i*5+5] = y_i)
    // coeffs: n ext-field values (output polynomial coefficients)
    // x_i = i in base field.
    for (int i = 0; i < n; i++) {
        qe_zero(coeffs + i * 5);
    }

    for (int i = 0; i < n; i++) {
        uint32_t basis[12][5];
        uint32_t next_basis[12][5];
        for (int d = 0; d < n; d++) {
            qe_zero(basis[d]);
            qe_zero(next_basis[d]);
        }
        qe_one(basis[0]);
        int deg = 0;

        uint32_t denom[5];
        qe_one(denom);

        for (int j = 0; j < n; j++) {
            if (j == i) continue;

            uint32_t j_ext[5];
            qe_from_base(kb_to_monty(j), j_ext);
            for (int d = 0; d < n; d++) {
                qe_zero(next_basis[d]);
            }

            for (int d = 0; d <= deg; d++) {
                uint32_t neg_j_term[5];
                qe_mul(basis[d], j_ext, neg_j_term);
                qe_neg(neg_j_term, neg_j_term);
                qe_add(next_basis[d], neg_j_term, next_basis[d]);
                qe_add(next_basis[d + 1], basis[d], next_basis[d + 1]);
            }
            deg++;
            for (int d = 0; d <= deg; d++) {
                for (int k = 0; k < 5; k++) basis[d][k] = next_basis[d][k];
            }

            int diff_ij = i - j;
            uint32_t diff_base = diff_ij >= 0 ? kb_to_monty(diff_ij) : kb_neg(kb_to_monty(-diff_ij));
            uint32_t diff_ext[5];
            uint32_t new_denom[5];
            qe_from_base(diff_base, diff_ext);
            qe_mul(denom, diff_ext, new_denom);
            for (int k = 0; k < 5; k++) denom[k] = new_denom[k];
        }

        uint32_t yi[5];
        uint32_t denom_inv[5];
        uint32_t scale[5];
        for (int k = 0; k < 5; k++) yi[k] = pe[i * 5 + k];
        qe_inv(denom, denom_inv);
        qe_mul(yi, denom_inv, scale);

        for (int d = 0; d <= deg; d++) {
            uint32_t term[5];
            qe_mul(scale, basis[d], term);
            qe_add(coeffs + d * 5, term, coeffs + d * 5);
        }
    }
}

// expand_bare_to_full: bare[0..d] → full[0..d+1]
// full[0] = (1-alpha) * bare[0]
// full[k] = (1-alpha) * bare[k] + (2*alpha-1) * bare[k-1]  for 1<=k<=d
// full[d+1] = (2*alpha-1) * bare[d]
__device__ void expand_bare_to_full_dev(
    const uint32_t* bare, int d_plus_1, const uint32_t alpha[5],
    uint32_t* full)
{
    uint32_t ONE[5]; qe_one(ONE);
    uint32_t one_m_a[5]; qe_sub(ONE, alpha, one_m_a);
    uint32_t two_a[5]; qe_add(alpha, alpha, two_a);
    uint32_t two_a_m_1[5]; qe_sub(two_a, ONE, two_a_m_1);
    
    int d = d_plus_1 - 1;
    // full[0] = (1-alpha) * bare[0]
    qe_mul(one_m_a, bare, full);
    // full[k] = (1-alpha)*bare[k] + (2*alpha-1)*bare[k-1]
    for (int k = 1; k <= d; k++) {
        uint32_t t1[5], t2[5];
        qe_mul(one_m_a, bare + k * 5, t1);
        qe_mul(two_a_m_1, bare + (k - 1) * 5, t2);
        qe_add(t1, t2, full + k * 5);
    }
    // full[d+1] = (2*alpha-1) * bare[d]
    qe_mul(two_a_m_1, bare + d * 5, full + (d + 1) * 5);
}

// Evaluate polynomial at a point: p(x) = sum coeffs[i] * x^i
__device__ void poly_eval_dev(const uint32_t* coeffs, int n, const uint32_t x[5], uint32_t result[5]) {
    // Horner: result = coeffs[n-1]; for i = n-2..0: result = result*x + coeffs[i]
    for (int k = 0; k < 5; k++) result[k] = coeffs[(n - 1) * 5 + k];
    for (int i = n - 2; i >= 0; i--) {
        uint32_t t[5];
        qe_mul(result, x, t);
        qe_add(t, coeffs + i * 5, result);
    }
}

extern "C" __global__ void air_sumcheck_build_bare_kernel(
    const uint32_t* __restrict__ ze_all,
    const uint32_t* __restrict__ ze_offsets,
    const uint32_t* __restrict__ table_state,
    uint32_t n_tables,
    uint32_t mfd,
    uint32_t* __restrict__ bare_scratch
) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tables) return;

    const uint32_t* ts = table_state + t * TS_STRIDE;
    if (!ts[TS_JOIN]) return;

    uint32_t nz = ts[TS_NZ];
    const uint32_t* ze = ze_all + ze_offsets[t] * 5;
    const uint32_t* sum = ts + TS_SUM;
    const uint32_t* mmf = ts + TS_MMF;
    const uint32_t* ea = ts + TS_EA;
    const uint32_t* pad = ts + TS_PAD;

    uint32_t ze_adj[12 * 5];
    for (uint32_t i = 0; i < nz; i++) {
        qe_add(ze + i * 5, pad, ze_adj + i * 5);
    }

    uint32_t pe[13 * 5];
    qe_mul(ze_adj, mmf, pe);
    {
        uint32_t ONE[5]; qe_one(ONE);
        uint32_t one_m_ea[5]; qe_sub(ONE, ea, one_m_ea);
        uint32_t t1[5]; qe_mul(one_m_ea, pe, t1);
        uint32_t t2[5]; qe_sub(sum, t1, t2);
        const uint32_t* ea_inv = ts + TS_EA_INV;
        qe_mul(t2, ea_inv, pe + 5);
    }
    for (uint32_t i = 1; i < nz; i++) {
        qe_mul(ze_adj + i * 5, mmf, pe + (i + 1) * 5);
    }

    uint32_t* bare_dst = bare_scratch + t * mfd * 5;
    lagrange_interp(pe, nz + 1, bare_dst);
}

// Evaluates the MLE of [0, ..., 0, 1, ..., 1] with `n_zeros` zeros.
__device__ void mle_of_zeros_then_ones_dev(
    uint32_t n_zeros,
    const uint32_t* point,   // eq_prefix_len * 5 words
    uint32_t eq_prefix_len,
    uint32_t out[5]
) {
    uint32_t ONE[5]; qe_one(ONE);
    uint32_t ZERO[5]; qe_zero(ZERO);

    uint32_t scale[5];
    uint32_t offset[5];
    for (int k = 0; k < 5; k++) {
        scale[k] = ONE[k];
        offset[k] = ZERO[k];
    }

    uint32_t n_values = 1u << eq_prefix_len;
    for (uint32_t i = 0; i < eq_prefix_len; i++) {
        if (n_zeros == 0) {
            break;
        }
        if (n_zeros == n_values) {
            for (int k = 0; k < 5; k++) out[k] = offset[k];
            return;
        }

        uint32_t half = n_values >> 1;
        const uint32_t* p = point + i * 5;
        if (n_zeros < half) {
            uint32_t term[5], one_m_p[5], new_scale[5], new_offset[5];
            qe_mul(scale, p, term);
            qe_add(offset, term, new_offset);
            qe_sub(ONE, p, one_m_p);
            qe_mul(scale, one_m_p, new_scale);
            for (int k = 0; k < 5; k++) {
                scale[k] = new_scale[k];
                offset[k] = new_offset[k];
            }
        } else {
            uint32_t new_scale[5];
            qe_mul(scale, p, new_scale);
            for (int k = 0; k < 5; k++) scale[k] = new_scale[k];
            n_zeros -= half;
        }
        n_values = half;
    }

    if (n_zeros == 0) {
        uint32_t sum[5];
        qe_add(offset, scale, sum);
        for (int k = 0; k < 5; k++) out[k] = sum[k];
    } else {
        for (int k = 0; k < 5; k++) out[k] = offset[k];
    }
}

extern "C" __global__ void gkr_round_protocol_step_kernel(
    const uint32_t* __restrict__ c0_num,
    const uint32_t* __restrict__ c2_num,
    const uint32_t* __restrict__ c0_den,
    const uint32_t* __restrict__ c2_den,
    const uint32_t* __restrict__ alpha,
    const uint32_t* __restrict__ eq_alpha,
    const uint32_t* __restrict__ eq_prefix,
    uint32_t eq_prefix_len,
    uint32_t active_pairs,
    uint32_t* __restrict__ sum_inout,
    uint32_t* __restrict__ mmf_inout,
    uint32_t* __restrict__ challenge_out,
    uint32_t* __restrict__ transcript_tail_out,
    uint32_t* __restrict__ challenger_state,
    const uint32_t* __restrict__ p16_rc,
    const uint32_t* __restrict__ p16_mds,
    const uint32_t* __restrict__ p16_sparse
) {
    uint32_t padding_eval[5];
    mle_of_zeros_then_ones_dev(active_pairs, eq_prefix, eq_prefix_len, padding_eval);

    uint32_t alpha_c0_den[5], alpha_c2_den[5], padding_sum[5];
    qe_mul(alpha, c0_den, alpha_c0_den);
    qe_mul(alpha, c2_den, alpha_c2_den);
    qe_mul(alpha, padding_eval, padding_sum);

    uint32_t c0_raw[5], c2_raw[5], tmp[5];
    qe_add(c0_num, alpha_c0_den, tmp);
    qe_add(tmp, padding_sum, c0_raw);
    qe_add(c2_num, alpha_c2_den, c2_raw);

    uint32_t c0_mmf[5], c1_mmf[5], c2_mmf[5];
    qe_mul(c0_raw, mmf_inout, c0_mmf);
    qe_mul(c2_raw, mmf_inout, c2_mmf);

    uint32_t ONE[5]; qe_one(ONE);
    uint32_t one_m_eq_alpha[5];
    qe_sub(ONE, eq_alpha, one_m_eq_alpha);
    uint32_t lhs[5], numer[5], eq_alpha_inv[5], h1_mmf[5];
    qe_mul(one_m_eq_alpha, c0_mmf, lhs);
    qe_sub(sum_inout, lhs, numer);
    qe_inv(eq_alpha, eq_alpha_inv);
    qe_mul(numer, eq_alpha_inv, h1_mmf);
    uint32_t c0_plus_c2[5];
    qe_add(c0_mmf, c2_mmf, c0_plus_c2);
    qe_sub(h1_mmf, c0_plus_c2, c1_mmf);

    uint32_t bare[15];
    for (int k = 0; k < 5; k++) {
        bare[k] = c0_mmf[k];
        bare[5 + k] = c1_mmf[k];
        bare[10 + k] = c2_mmf[k];
        transcript_tail_out[k] = c1_mmf[k];
        transcript_tail_out[5 + k] = c2_mmf[k];
    }

    uint32_t full[20];
    expand_bare_to_full_dev(bare, 3, eq_alpha, full);

    GpuChallenger ch;
    for (int i = 0; i < FS_RATE; i++) ch.state[i] = challenger_state[i];
    ch.rc = p16_rc;
    ch.mds = p16_mds;
    ch.sparse = p16_sparse;
    challenger_observe_scalars(&ch, full, 20);
    challenger_sample_ext(&ch, challenge_out);
    for (int i = 0; i < FS_RATE; i++) challenger_state[i] = ch.state[i];

    uint32_t one_m_r[5], eq_term0[5], eq_term1[5], eq_eval[5];
    qe_sub(ONE, challenge_out, one_m_r);
    qe_mul(one_m_eq_alpha, one_m_r, eq_term0);
    qe_mul(eq_alpha, challenge_out, eq_term1);
    qe_add(eq_term0, eq_term1, eq_eval);

    uint32_t bare_eval[5];
    poly_eval_dev(bare, 3, challenge_out, bare_eval);
    qe_mul(eq_eval, bare_eval, sum_inout);
    uint32_t new_mmf[5];
    qe_mul(mmf_inout, eq_eval, new_mmf);
    for (int k = 0; k < 5; k++) mmf_inout[k] = new_mmf[k];
}

extern "C" __global__ void product_sumcheck_observe_round_poly_kernel(
    const uint32_t* __restrict__ c0,
    const uint32_t* __restrict__ c2,
    const uint32_t* __restrict__ sum_in,
    uint32_t* __restrict__ poly_out,
    uint32_t* __restrict__ transcript_tail_out,
    uint32_t* __restrict__ challenger_state,
    const uint32_t* __restrict__ p16_rc,
    const uint32_t* __restrict__ p16_mds,
    const uint32_t* __restrict__ p16_sparse
) {
    uint32_t two_c0[5], c1[5];
    qe_add(c0, c0, two_c0);
    qe_sub(sum_in, two_c0, c1);
    qe_sub(c1, c2, c1);

    for (int k = 0; k < 5; k++) {
        poly_out[k] = c0[k];
        poly_out[5 + k] = c1[k];
        poly_out[10 + k] = c2[k];
        transcript_tail_out[k] = c1[k];
        transcript_tail_out[5 + k] = c2[k];
    }

    GpuChallenger ch;
    for (int i = 0; i < FS_RATE; i++) ch.state[i] = challenger_state[i];
    ch.rc = p16_rc;
    ch.mds = p16_mds;
    ch.sparse = p16_sparse;
    challenger_observe_scalars(&ch, poly_out, 15);
    for (int i = 0; i < FS_RATE; i++) challenger_state[i] = ch.state[i];
}

extern "C" __global__ void product_sumcheck_update_sum_kernel(
    const uint32_t* __restrict__ poly,
    const uint32_t* __restrict__ challenge,
    uint32_t* __restrict__ sum_out
) {
    poly_eval_dev(poly, 3, challenge, sum_out);
}

extern "C" __global__ void ext_affine_combine_kernel(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    const uint32_t* __restrict__ alpha,
    uint32_t* __restrict__ out
) {
    uint32_t alpha_b[5];
    qe_mul(alpha, b, alpha_b);
    qe_add(a, alpha_b, out);
}

extern "C" __global__ void ext_add_kernel(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t* __restrict__ out
) {
    uint32_t res[5];
    qe_add(a, b, res);
    #pragma unroll
    for (int k = 0; k < 5; k++) out[k] = res[k];
}

extern "C" __global__ void extension_powers_kernel(
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ out,
    uint32_t n_powers
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t cur[5];
    qe_one(cur);

    for (uint32_t i = 0; i < n_powers; i++) {
        #pragma unroll
        for (int k = 0; k < 5; k++) out[i * 5 + k] = cur[k];
        uint32_t next[5];
        qe_mul(cur, base, next);
        #pragma unroll
        for (int k = 0; k < 5; k++) cur[k] = next[k];
    }
}

extern "C" __global__ void dense_eq_accumulate_from_points_kernel(
    uint32_t* __restrict__ weights,
    const uint32_t* __restrict__ points,
    const uint32_t* __restrict__ scalars,
    uint32_t n_points,
    uint32_t n_vars,
    uint32_t n_total
) {
    // Use shared memory to cache points+scalars if they fit (< 12 KB to keep occupancy).
    // For small n_points (OOD, ≤ 4 samples), this avoids repeated global memory reads.
    extern __shared__ uint32_t smem[];
    uint32_t total_point_words = n_points * n_vars * 5;
    uint32_t total_scalar_words = n_points * 5;
    uint32_t total_smem_words = total_point_words + total_scalar_words;
    // smem_bytes is set by the host launch; 0 means "don't use shared memory".
    bool use_smem = (total_smem_words * 4 <= 12288);  // 12 KB threshold
    if (use_smem) {
        for (uint32_t i = threadIdx.x; i < total_smem_words; i += blockDim.x) {
            if (i < total_point_words)
                smem[i] = points[i];
            else
                smem[i] = scalars[i - total_point_words];
        }
        __syncthreads();
    }
    const uint32_t* eff_points = use_smem ? smem : points;
    const uint32_t* eff_scalars = use_smem ? (smem + total_point_words) : scalars;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;

    uint32_t acc[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) acc[k] = weights[idx * 5 + k];

    for (uint32_t sample = 0; sample < n_points; sample++) {
        uint32_t eq[5];
        qe_one(eq);
        const uint32_t* sample_points = eff_points + sample * n_vars * 5;

        for (uint32_t var = 0; var < n_vars; var++) {
            const uint32_t* pk = sample_points + var * 5;
            uint32_t factor[5];
            if ((idx >> (n_vars - 1 - var)) & 1u) {
                #pragma unroll
                for (int k = 0; k < 5; k++) factor[k] = pk[k];
            } else {
                factor[0] = kb_sub(KB_MONTY_ONE, pk[0]);
                #pragma unroll
                for (int k = 1; k < 5; k++) factor[k] = kb_neg(pk[k]);
            }

            uint32_t next_eq[5];
            qe_mul(eq, factor, next_eq);
            #pragma unroll
            for (int k = 0; k < 5; k++) eq[k] = next_eq[k];
        }

        uint32_t term[5];
        qe_mul(eq, eff_scalars + sample * 5, term);
        uint32_t new_acc[5];
        qe_add(acc, term, new_acc);
        #pragma unroll
        for (int k = 0; k < 5; k++) acc[k] = new_acc[k];
    }

    #pragma unroll
    for (int k = 0; k < 5; k++) weights[idx * 5 + k] = acc[k];
}

extern "C" __global__ void ext_dot_accumulate_kernel(
    const uint32_t* __restrict__ base_sum,
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ scalars,
    uint32_t n_terms,
    uint32_t* __restrict__ out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t acc[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) acc[k] = base_sum[k];

    for (uint32_t i = 0; i < n_terms; i++) {
        uint32_t term[5];
        qe_mul(values + i * 5, scalars + i * 5, term);
        uint32_t new_acc[5];
        qe_add(acc, term, new_acc);
        #pragma unroll
        for (int k = 0; k < 5; k++) acc[k] = new_acc[k];
    }

    #pragma unroll
    for (int k = 0; k < 5; k++) out[k] = acc[k];
}

extern "C" __global__ void eq_polynomial_from_flat_points_kernel(
    const uint32_t* __restrict__ points,
    uint32_t n_vars,
    uint32_t* __restrict__ out,
    uint32_t n_total
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;

    uint32_t eq[5];
    qe_one(eq);

    for (uint32_t var = 0; var < n_vars; var++) {
        const uint32_t* pk = points + var * 5;
        uint32_t factor[5];
        if ((idx >> (n_vars - 1 - var)) & 1u) {
            #pragma unroll
            for (int k = 0; k < 5; k++) factor[k] = pk[k];
        } else {
            factor[0] = kb_sub(KB_MONTY_ONE, pk[0]);
            #pragma unroll
            for (int k = 1; k < 5; k++) factor[k] = kb_neg(pk[k]);
        }

        uint32_t next_eq[5];
        qe_mul(eq, factor, next_eq);
        #pragma unroll
        for (int k = 0; k < 5; k++) eq[k] = next_eq[k];
    }

    #pragma unroll
    for (int k = 0; k < 5; k++) out[idx * 5 + k] = eq[k];
}

extern "C" __global__ void extract_permuted_suffix_ext_kernel(
    const uint32_t* __restrict__ point_words,
    uint32_t total_coords,
    uint32_t suffix_len,
    uint32_t len,
    uint32_t pivot,
    uint32_t* __restrict__ out
) {
    uint32_t coord = blockIdx.x * blockDim.x + threadIdx.x;
    if (coord >= len) return;

    uint32_t head_len = suffix_len > pivot ? suffix_len - pivot : 0;
    if (head_len > len) head_len = len;

    uint32_t src_rel;
    if (coord < head_len) {
        src_rel = coord;
    } else {
        src_rel = len - 1 - (coord - head_len);
    }
    uint32_t src_coord = total_coords - suffix_len + src_rel;

    #pragma unroll
    for (int k = 0; k < 5; k++) {
        out[coord * 5 + k] = point_words[src_coord * 5 + k];
    }
}

extern "C" __global__ void extract_reversed_suffix_ext_kernel(
    const uint32_t* __restrict__ point_words,
    uint32_t total_coords,
    uint32_t suffix_len,
    uint32_t* __restrict__ out
) {
    uint32_t coord = blockIdx.x * blockDim.x + threadIdx.x;
    if (coord >= suffix_len) return;

    uint32_t src_coord = total_coords - 1u - coord;
    #pragma unroll
    for (int k = 0; k < 5; k++) {
        out[coord * 5 + k] = point_words[src_coord * 5 + k];
    }
}

extern "C" __global__ void next_mle_from_point_kernel(
    const uint32_t* __restrict__ point_words,
    uint32_t n_vars,
    uint32_t* __restrict__ out,
    uint32_t n_total
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;

    uint32_t prod_all[5];
    qe_one(prod_all);
    for (uint32_t i = 0; i < n_vars; i++) {
        uint32_t next_prod[5];
        qe_mul(prod_all, point_words + i * 5, next_prod);
        #pragma unroll
        for (int k = 0; k < 5; k++) prod_all[k] = next_prod[k];
    }

    uint32_t sum[5] = {0, 0, 0, 0, 0};
    for (uint32_t arr = 0; arr < n_vars; arr++) {
        bool y_arr = ((idx >> (n_vars - 1u - arr)) & 1u) != 0;
        if (!y_arr) continue;

        uint32_t term[5];
        qe_one(term);

        for (uint32_t i = 0; i < arr; i++) {
            bool y_i = ((idx >> (n_vars - 1u - i)) & 1u) != 0;
            uint32_t factor[5];
            if (y_i) {
                #pragma unroll
                for (int k = 0; k < 5; k++) factor[k] = point_words[i * 5 + k];
            } else {
                factor[0] = kb_sub(KB_MONTY_ONE, point_words[i * 5]);
                #pragma unroll
                for (int k = 1; k < 5; k++) factor[k] = kb_neg(point_words[i * 5 + k]);
            }
            uint32_t next_term[5];
            qe_mul(term, factor, next_term);
            #pragma unroll
            for (int k = 0; k < 5; k++) term[k] = next_term[k];
        }

        uint32_t carry[5];
        carry[0] = kb_sub(KB_MONTY_ONE, point_words[arr * 5]);
        #pragma unroll
        for (int k = 1; k < 5; k++) carry[k] = kb_neg(point_words[arr * 5 + k]);
        uint32_t with_carry[5];
        qe_mul(term, carry, with_carry);
        #pragma unroll
        for (int k = 0; k < 5; k++) term[k] = with_carry[k];

        for (uint32_t i = arr + 1; i < n_vars; i++) {
            bool y_i = ((idx >> (n_vars - 1u - i)) & 1u) != 0;
            if (y_i) {
                #pragma unroll
                for (int k = 0; k < 5; k++) term[k] = 0;
                break;
            }
            uint32_t next_term[5];
            qe_mul(term, point_words + i * 5, next_term);
            #pragma unroll
            for (int k = 0; k < 5; k++) term[k] = next_term[k];
        }

        #pragma unroll
        for (int k = 0; k < 5; k++) sum[k] = kb_add(sum[k], term[k]);
    }

    if (idx == n_total - 1u) {
        #pragma unroll
        for (int k = 0; k < 5; k++) sum[k] = kb_add(sum[k], prod_all[k]);
    }

    #pragma unroll
    for (int k = 0; k < 5; k++) out[idx * 5 + k] = sum[k];
}

extern "C" __global__ void reverse_gkr_challenges_into_point_kernel(
    const uint32_t* __restrict__ round_challenges,
    uint32_t n_rounds,
    uint32_t* __restrict__ out
) {
    uint32_t coord = blockIdx.x * blockDim.x + threadIdx.x;
    if (coord > n_rounds) return;

    uint32_t src_coord = (coord < n_rounds) ? (n_rounds - 1 - coord) : n_rounds;
    const uint32_t* src = round_challenges + src_coord * 5;
    uint32_t* dst = out + coord * 5;

    #pragma unroll
    for (int k = 0; k < 5; k++) dst[k] = src[k];
}

extern "C" __global__ void gkr_finalize_layer_kernel(
    const uint32_t* __restrict__ nl,
    const uint32_t* __restrict__ nr,
    const uint32_t* __restrict__ dl,
    const uint32_t* __restrict__ dr,
    uint32_t* __restrict__ beta_out,
    uint32_t* __restrict__ claim_num_out,
    uint32_t* __restrict__ claim_den_out,
    uint32_t* __restrict__ transcript_out,
    uint32_t* __restrict__ challenger_state,
    const uint32_t* __restrict__ p16_rc,
    const uint32_t* __restrict__ p16_mds,
    const uint32_t* __restrict__ p16_sparse
) {
    for (int k = 0; k < 5; k++) {
        transcript_out[k] = nl[k];
        transcript_out[5 + k] = nr[k];
        transcript_out[10 + k] = dl[k];
        transcript_out[15 + k] = dr[k];
    }

    GpuChallenger ch;
    for (int i = 0; i < FS_RATE; i++) ch.state[i] = challenger_state[i];
    ch.rc = p16_rc;
    ch.mds = p16_mds;
    ch.sparse = p16_sparse;
    challenger_observe_scalars(&ch, transcript_out, 20);
    challenger_sample_ext(&ch, beta_out);
    for (int i = 0; i < FS_RATE; i++) challenger_state[i] = ch.state[i];

    uint32_t ONE[5], one_m_beta[5];
    qe_one(ONE);
    qe_sub(ONE, beta_out, one_m_beta);

    uint32_t left_num[5], right_num[5], left_den[5], right_den[5];
    qe_mul(one_m_beta, nl, left_num);
    qe_mul(beta_out, nr, right_num);
    qe_add(left_num, right_num, claim_num_out);

    qe_mul(one_m_beta, dl, left_den);
    qe_mul(beta_out, dr, right_den);
    qe_add(left_den, right_den, claim_den_out);
}


// Protocol step kernel: runs on 1 thread, does ALL per-round glue.
// Called after multi-z kernels complete, before fold kernels.
extern "C" __global__ void air_sumcheck_protocol_step_kernel(
    const uint32_t* __restrict__ ze_all,       // all tables' ze values concatenated
    const uint32_t* __restrict__ ze_offsets,    // ze_all offset for each table, in ext elements
    uint32_t* __restrict__ table_state,         // per-table state (TS_STRIDE * n_tables u32s)
    uint32_t n_tables,
    uint32_t mfd,                               // max full degree
    uint32_t* __restrict__ cc_out,              // combined polynomial (mfd+1) * 5 u32s
    uint32_t* __restrict__ bare_scratch,        // per-table bare coeffs, n_tables * mfd * 5 u32s
    uint32_t* __restrict__ challenge_out,       // sampled challenge (5 u32s)
    uint32_t* __restrict__ transcript_out,      // transcript chunk for this round
    uint32_t* __restrict__ transcript_len_out,  // length of transcript chunk
    uint32_t* __restrict__ challenger_state,    // persistent Fiat-Shamir state (8 u32s)
    const uint32_t* __restrict__ p16_rc,        // Poseidon16 constants for Fiat-Shamir
    const uint32_t* __restrict__ p16_mds,
    const uint32_t* __restrict__ p16_sparse
) {
    // Initialize cc to zero
    for (uint32_t i = 0; i < (mfd + 1) * 5; i++) cc_out[i] = 0;
    
    // Process each table
    for (uint32_t t = 0; t < n_tables; t++) {
        uint32_t* ts = table_state + t * TS_STRIDE;
        uint32_t joined = ts[TS_JOIN];
        if (!joined) {
            // Table hasn't joined yet — contribute sum * eta_k to cc[1]
            uint32_t contrib[5];
            qe_mul(ts + TS_ETAK, ts + TS_SUM, contrib);
            for (int k = 0; k < 5; k++)
                cc_out[1 * 5 + k] = kb_add(cc_out[1 * 5 + k], contrib[k]);
            continue;
        }
        
        uint32_t nz = ts[TS_NZ]; // bare degree (number of z-points excl z=1)
        uint32_t* sum = ts + TS_SUM;
        uint32_t* mmf = ts + TS_MMF;
        uint32_t* ea = ts + TS_EA;
        uint32_t* eta_k = ts + TS_ETAK;
        uint32_t* bare_dst = bare_scratch + t * mfd * 5;
        
        // expand_bare_to_full: bare → full (degree n_pe → n_pe+1 coefficients... wait, 
        // bare has n_pe coefficients, full has n_pe+1? No — bare has n_pe coefficients
        // (degree n_pe-1), full has n_pe coefficients (degree n_pe-1... hmm).
        // Actually: bare polynomial has degree nz (= n_pe - 1). expand adds 1 to degree.
        // So full has nz+1 = n_pe coefficients... wait, expand_bare_to_full(bare, alpha)
        // takes bare of length d+1 and returns full of length d+2.
        // bare has n_pe coefficients (degree n_pe-1 = nz).
        // full has n_pe+1 coefficients (degree nz+1 = n_pe).
        uint32_t full[13 * 5];
        expand_bare_to_full_dev(bare_dst, nz + 1, ea, full);
        int n_full = nz + 2;

        // Accumulate into cc: cc[i] += eta_k * full[i]
        for (int i = 0; i < n_full && i <= (int)mfd; i++) {
            uint32_t term[5];
            qe_mul(eta_k, full + i * 5, term);
            uint32_t s[5];
            qe_add(cc_out + i * 5, term, s);
            for (int k = 0; k < 5; k++) cc_out[i * 5 + k] = s[k];
        }
    }
    
    // Fiat-Shamir: observe cc, sample challenge
    GpuChallenger ch;
    for (int i = 0; i < FS_RATE; i++) ch.state[i] = challenger_state[i];
    ch.rc = p16_rc;
    ch.mds = p16_mds;
    ch.sparse = p16_sparse;
    
    // flatten cc to base scalars and observe
    // cc has (mfd+1) ext elements = (mfd+1)*5 base elements
    challenger_observe_scalars(&ch, cc_out, (mfd + 1) * 5);
    
    // Write transcript: cc scalars skipping first EF::DIMENSION=5 elements
    uint32_t tlen = (mfd + 1) * 5 - 5;
    for (uint32_t i = 0; i < tlen; i++)
        transcript_out[i] = cc_out[5 + i];
    *transcript_len_out = tlen;
    
    // Sample challenge (1 ext element = 5 base elements)
    challenger_sample_ext(&ch, challenge_out);
    
    // Save challenger state
    for (int i = 0; i < FS_RATE; i++) challenger_state[i] = ch.state[i];
    
    // Update per-table state with the challenge
    for (uint32_t t = 0; t < n_tables; t++) {
        uint32_t* ts = table_state + t * TS_STRIDE;
        if (!ts[TS_JOIN]) {
            // Not joined: k[idx] *= ch
            uint32_t new_etak[5];
            qe_mul(ts + TS_ETAK, challenge_out, new_etak);
            for (int k = 0; k < 5; k++) ts[TS_ETAK + k] = new_etak[k];
            continue;
        }
        
        uint32_t nz = ts[TS_NZ];
        uint32_t* sum = ts + TS_SUM;
        uint32_t* mmf = ts + TS_MMF;
        uint32_t* ea = ts + TS_EA;
        
        // ee = (1-ea)*(1-ch) + ea*ch
        uint32_t ONE[5]; qe_one(ONE);
        uint32_t one_m_ea[5]; qe_sub(ONE, ea, one_m_ea);
        uint32_t one_m_ch[5]; qe_sub(ONE, challenge_out, one_m_ch);
        uint32_t t1[5], t2[5], ee[5];
        qe_mul(one_m_ea, one_m_ch, t1);
        qe_mul(ea, challenge_out, t2);
        qe_add(t1, t2, ee);
        
        // Evaluate bare polynomial at challenge
        const uint32_t* bare_saved = bare_scratch + t * mfd * 5;
        uint32_t bp_ch[5];
        poly_eval_dev(bare_saved, nz + 1, challenge_out, bp_ch);
        
        // sum = bp(ch) * ee
        qe_mul(bp_ch, ee, sum);
        
        // mmf *= ee
        uint32_t new_mmf[5];
        qe_mul(mmf, ee, new_mmf);
        for (int k = 0; k < 5; k++) mmf[k] = new_mmf[k];
    }
}

// Test kernel: verify GPU Fiat-Shamir matches CPU by hashing known data.
extern "C" __global__ void test_fiat_shamir_kernel(
    const uint32_t* __restrict__ p16_rc,
    const uint32_t* __restrict__ p16_mds,
    const uint32_t* __restrict__ p16_sparse,
    const uint32_t* __restrict__ input_scalars,
    uint32_t n_scalars,
    uint32_t* __restrict__ output_state,  // 8 u32s: challenger state after observe
    uint32_t* __restrict__ output_sample  // 5 u32s: sampled ext element
) {
    GpuChallenger ch;
    challenger_init(&ch, p16_rc, p16_mds, p16_sparse);
    challenger_observe_scalars(&ch, input_scalars, n_scalars);
    for (int i = 0; i < 8; i++) output_state[i] = ch.state[i];
    challenger_sample_ext(&ch, output_sample);
}

extern "C" __global__ void challenger_observe_scalars_kernel(
    uint32_t* __restrict__ challenger_state,
    const uint32_t* __restrict__ p16_rc,
    const uint32_t* __restrict__ p16_mds,
    const uint32_t* __restrict__ p16_sparse,
    const uint32_t* __restrict__ observe_scalars,
    uint32_t n_observe_scalars
) {
    GpuChallenger ch;
    for (int i = 0; i < FS_RATE; i++) ch.state[i] = challenger_state[i];
    ch.rc = p16_rc;
    ch.mds = p16_mds;
    ch.sparse = p16_sparse;

    if (n_observe_scalars > 0) {
        challenger_observe_scalars(&ch, observe_scalars, n_observe_scalars);
    }

    for (int i = 0; i < FS_RATE; i++) challenger_state[i] = ch.state[i];
}

extern "C" __global__ void challenger_sample_base_scalars_kernel(
    uint32_t* __restrict__ challenger_state,
    const uint32_t* __restrict__ p16_rc,
    const uint32_t* __restrict__ p16_mds,
    const uint32_t* __restrict__ p16_sparse,
    uint32_t* __restrict__ samples_out,
    uint32_t n_base_samples
) {
    GpuChallenger ch;
    for (int i = 0; i < FS_RATE; i++) ch.state[i] = challenger_state[i];
    ch.rc = p16_rc;
    ch.mds = p16_mds;
    ch.sparse = p16_sparse;

    uint32_t n_blocks = (n_base_samples + FS_RATE - 1) / FS_RATE;
    uint32_t written = 0;
    for (uint32_t block = 0; block <= n_blocks; block++) {
        uint32_t buf[FS_WIDTH];
        buf[0] = kb_to_monty(block);
        for (int j = 1; j < FS_RATE; j++) buf[j] = 0;
        for (int j = 0; j < FS_RATE; j++) buf[FS_RATE + j] = ch.state[j];
        p16_compress(buf, ch.rc, ch.mds, ch.sparse);
        if (block < n_blocks) {
            for (int j = 0; j < FS_RATE && written < n_base_samples; j++, written++) {
                samples_out[written] = buf[j];
            }
        } else {
            for (int j = 0; j < FS_RATE; j++) challenger_state[j] = buf[j];
        }
    }
}

extern "C" __global__ void challenger_observe_and_sample_kernel(
    uint32_t* __restrict__ challenger_state,
    const uint32_t* __restrict__ p16_rc,
    const uint32_t* __restrict__ p16_mds,
    const uint32_t* __restrict__ p16_sparse,
    const uint32_t* __restrict__ observe_scalars,
    uint32_t n_observe_scalars,
    uint32_t* __restrict__ samples_out,
    uint32_t n_sample_exts
) {
    GpuChallenger ch;
    for (int i = 0; i < FS_RATE; i++) ch.state[i] = challenger_state[i];
    ch.rc = p16_rc;
    ch.mds = p16_mds;
    ch.sparse = p16_sparse;

    if (n_observe_scalars > 0) {
        challenger_observe_scalars(&ch, observe_scalars, n_observe_scalars);
    }

    uint32_t n_base_samples = n_sample_exts * 5;
    uint32_t n_blocks = (n_base_samples + FS_RATE - 1) / FS_RATE;
    uint32_t written = 0;
    for (uint32_t block = 0; block <= n_blocks; block++) {
        uint32_t buf[FS_WIDTH];
        buf[0] = kb_to_monty(block);
        for (int j = 1; j < FS_RATE; j++) buf[j] = 0;
        for (int j = 0; j < FS_RATE; j++) buf[FS_RATE + j] = ch.state[j];
        p16_compress(buf, ch.rc, ch.mds, ch.sparse);
        if (block < n_blocks) {
            for (int j = 0; j < FS_RATE && written < n_base_samples; j++, written++) {
                samples_out[written] = buf[j];
            }
        } else {
            for (int j = 0; j < FS_RATE; j++) challenger_state[j] = buf[j];
        }
    }
}

// Test kernel for Lagrange interpolation verification
extern "C" __global__ void test_lagrange_interp_kernel(
    const uint32_t* __restrict__ pe,  // n * 5 ext values
    uint32_t n,                        // number of points
    uint32_t* __restrict__ coeffs_out  // n * 5 ext output
) {
    lagrange_interp(pe, n, coeffs_out);
}

// Debug kernel: dump table_state values for verification
extern "C" __global__ void debug_dump_table_state_kernel(
    const uint32_t* __restrict__ table_state,
    uint32_t n_tables,
    uint32_t* __restrict__ output  // n_tables * TS_STRIDE u32s
) {
    for (uint32_t t = 0; t < n_tables; t++) {
        for (int i = 0; i < TS_STRIDE; i++) {
            output[t * TS_STRIDE + i] = table_state[t * TS_STRIDE + i];
        }
    }
}

// Test kernel for expand_bare_to_full_dev
extern "C" __global__ void test_expand_bare_to_full_kernel(
    const uint32_t* __restrict__ bare,  // n * 5 ext values
    uint32_t n,
    const uint32_t alpha[5],
    uint32_t* __restrict__ full_out  // (n+1) * 5 ext output
) {
    expand_bare_to_full_dev(bare, n, alpha, full_out);
}

// Test kernel for qe_inv
extern "C" __global__ void test_qe_inv_kernel(
    const uint32_t a[5],
    uint32_t* __restrict__ out  // 5 u32s
) {
    qe_inv(a, out);
}

// Debug kernel: compute pe from ze+state like protocol_step, output pe directly
extern "C" __global__ void debug_compute_pe_kernel(
    const uint32_t* __restrict__ ze_all,
    const uint32_t* __restrict__ ze_offsets,
    const uint32_t* __restrict__ table_state,
    uint32_t table_idx,
    uint32_t* __restrict__ pe_out,      // (nz+1)*5 u32s
    uint32_t* __restrict__ nz_out       // 1 u32
) {
    const uint32_t* ts = table_state + table_idx * TS_STRIDE;
    uint32_t nz = ts[TS_NZ];
    *nz_out = nz;
    const uint32_t* ze = ze_all + ze_offsets[table_idx] * 5;
    const uint32_t* sum = ts + TS_SUM;
    const uint32_t* mmf = ts + TS_MMF;
    const uint32_t* ea = ts + TS_EA;
    const uint32_t* pad = ts + TS_PAD;

    uint32_t ze_adj[12*5];
    for (uint32_t i = 0; i < nz; i++) {
        uint32_t t[5]; qe_add(ze + i*5, pad, t);
        for (int k = 0; k < 5; k++) ze_adj[i*5+k] = t[k];
    }
    qe_mul(ze_adj, mmf, pe_out);
    {
        uint32_t ONE[5]; qe_one(ONE);
        uint32_t one_m_ea[5]; qe_sub(ONE, ea, one_m_ea);
        uint32_t t1[5]; qe_mul(one_m_ea, pe_out, t1);
        uint32_t t2[5]; qe_sub(sum, t1, t2);
        // Use precomputed ea_inv from table_state
        const uint32_t* ea_inv = ts + TS_EA_INV;
        qe_mul(t2, ea_inv, pe_out + 5);
    }
    for (uint32_t i = 1; i < nz; i++)
        qe_mul(ze_adj + i*5, mmf, pe_out + (i+1)*5);
}
