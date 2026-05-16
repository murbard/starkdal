// KoalaBear field arithmetic — shared CUDA device library.
//
// Base field: p = 0x7F000001 (2^31 - 2^24 + 1), Montgomery form with R = 2^32.
// Quintic extension: F[X] / (X^5 + X^2 - 1), dimension 5 over base field.
//
// Include this header in every GPU kernel that does field arithmetic.

#pragma once
#include <stdint.h>

// ── Base field constants ─────────────────────────────────────────────────
#define KB_P          0x7F000001u
#define KB_MONTY_MU   0x81000001u   // P^{-1} mod 2^32

// Montgomery form of 1: (1 << 32) % P = 2^25 - 2
#define KB_MONTY_ONE  0x01fffffeu

// ── Montgomery reduction ─────────────────────────────────────────────────
// x * R^{-1} mod P, where x ∈ [0, R·P).
// Uses positive-MU convention: MU = P^{-1} mod R.
__device__ __forceinline__ uint32_t kb_reduce(uint64_t x) {
    uint32_t t = (uint32_t)x * KB_MONTY_MU;
    uint64_t u = (uint64_t)t * (uint64_t)KB_P;
    uint64_t diff = x - u;
    uint32_t hi = (uint32_t)(diff >> 32);
    if (x < u) hi += KB_P;
    return hi;
}

// ── Base field arithmetic ────────────────────────────────────────────────

__device__ __forceinline__ uint32_t kb_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    if (s >= KB_P) s -= KB_P;
    return s;
}

__device__ __forceinline__ uint32_t kb_sub(uint32_t a, uint32_t b) {
    uint32_t d = a - b;
    if (a < b) d += KB_P;
    return d;
}

__device__ __forceinline__ uint32_t kb_mul(uint32_t a, uint32_t b) {
    return kb_reduce((uint64_t)a * (uint64_t)b);
}

__device__ __forceinline__ uint32_t kb_neg(uint32_t a) {
    return (a == 0) ? 0 : (KB_P - a);
}

__device__ __forceinline__ uint32_t kb_double(uint32_t a) {
    return kb_add(a, a);
}

__device__ __forceinline__ uint32_t kb_square(uint32_t a) {
    return kb_mul(a, a);
}

// x / 2 mod P. If x is odd, add (P+1)/2.
__device__ __forceinline__ uint32_t kb_halve(uint32_t a) {
    uint32_t shr = a >> 1;
    // (KB_P + 1) / 2 = 0x3F800001
    return (a & 1) ? (shr + 0x3F800001u) : shr;
}

// x^3 (S-box used in Poseidon).
__device__ __forceinline__ uint32_t kb_cube(uint32_t x) {
    uint32_t x2 = kb_mul(x, x);
    return kb_mul(x2, x);
}

// Convert canonical u32 → Montgomery form.
__device__ __forceinline__ uint32_t kb_to_monty(uint32_t v) {
    // (v << 32) % P — but we need 64-bit intermediate
    return (uint32_t)(((uint64_t)v << 32) % KB_P);
}

// Convert Montgomery form → canonical u32.
__device__ __forceinline__ uint32_t kb_from_monty(uint32_t v) {
    return kb_reduce((uint64_t)v);
}

// a^exp via square-and-multiply (exp as u32).
__device__ __forceinline__ uint32_t kb_exp(uint32_t base, uint32_t exp) {
    uint32_t result = KB_MONTY_ONE;
    uint32_t b = base;
    while (exp > 0) {
        if (exp & 1) result = kb_mul(result, b);
        b = kb_mul(b, b);
        exp >>= 1;
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^{P-2} mod P.
__device__ __forceinline__ uint32_t kb_inv(uint32_t a) {
    return kb_exp(a, KB_P - 2);
}

// 5-element dot product: sum(a[i] * b[i]) for i in [0,5).
__device__ __forceinline__ uint32_t kb_dot5(
    const uint32_t a[5], const uint32_t b[5]
) {
    uint32_t acc = kb_mul(a[0], b[0]);
    acc = kb_add(acc, kb_mul(a[1], b[1]));
    acc = kb_add(acc, kb_mul(a[2], b[2]));
    acc = kb_add(acc, kb_mul(a[3], b[3]));
    acc = kb_add(acc, kb_mul(a[4], b[4]));
    return acc;
}

// ── Quintic extension field: F[X] / (X^5 + X^2 - 1) ────────────────────
// Elements are uint32_t[5] = {c0, c1, c2, c3, c4} representing
// c0 + c1*X + c2*X^2 + c3*X^3 + c4*X^4.
// All coefficients in Montgomery form.

__device__ __forceinline__ void qe_add(
    const uint32_t a[5], const uint32_t b[5], uint32_t out[5]
) {
    #pragma unroll
    for (int i = 0; i < 5; i++) out[i] = kb_add(a[i], b[i]);
}

__device__ __forceinline__ void qe_sub(
    const uint32_t a[5], const uint32_t b[5], uint32_t out[5]
) {
    #pragma unroll
    for (int i = 0; i < 5; i++) out[i] = kb_sub(a[i], b[i]);
}

__device__ __forceinline__ void qe_neg(const uint32_t a[5], uint32_t out[5]) {
    #pragma unroll
    for (int i = 0; i < 5; i++) out[i] = kb_neg(a[i]);
}

// Scalar multiplication: out = a * scalar (base field element).
__device__ __forceinline__ void qe_base_mul(
    const uint32_t a[5], uint32_t scalar, uint32_t out[5]
) {
    #pragma unroll
    for (int i = 0; i < 5; i++) out[i] = kb_mul(a[i], scalar);
}

// Quintic extension multiplication.
// Matches the formula from extension.rs::quintic_mul:
//   res[0] = dot(a, [b0,  b4,       b3,      b2,      b1-b4])
//   res[1] = dot(a, [b1,  b0,       b4,      b3,      b2])
//   res[2] = dot(a, [b2,  b1-b4,    b0-b3,   b4-b2,   b3-(b1-b4)])
//   res[3] = dot(a, [b3,  b2,       b1-b4,   b0-b3,   b4-b2])
//   res[4] = dot(a, [b4,  b3,       b2,      b1-b4,   b0-b3])
__device__ __forceinline__ void qe_mul(
    const uint32_t a[5], const uint32_t b[5], uint32_t out[5]
) {
    uint32_t b0_m3 = kb_sub(b[0], b[3]);
    uint32_t b1_m4 = kb_sub(b[1], b[4]);
    uint32_t b4_m2 = kb_sub(b[4], b[2]);
    uint32_t b3_m_b1_m4 = kb_sub(b[3], b1_m4);

    uint32_t r0[5] = {b[0], b[4],  b[3],  b[2],   b1_m4};
    uint32_t r1[5] = {b[1], b[0],  b[4],  b[3],   b[2]};
    uint32_t r2[5] = {b[2], b1_m4, b0_m3, b4_m2,  b3_m_b1_m4};
    uint32_t r3[5] = {b[3], b[2],  b1_m4, b0_m3,  b4_m2};
    uint32_t r4[5] = {b[4], b[3],  b[2],  b1_m4,  b0_m3};

    out[0] = kb_dot5(a, r0);
    out[1] = kb_dot5(a, r1);
    out[2] = kb_dot5(a, r2);
    out[3] = kb_dot5(a, r3);
    out[4] = kb_dot5(a, r4);
}

// Quintic extension squaring (optimized, fewer multiplications than mul).
// Matches extension.rs::quintic_square.
__device__ __forceinline__ void qe_square(const uint32_t a[5], uint32_t out[5]) {
    uint32_t two_a0 = kb_double(a[0]);
    uint32_t two_a1 = kb_double(a[1]);
    uint32_t two_a2 = kb_double(a[2]);
    uint32_t two_a3 = kb_double(a[3]);

    uint32_t two_a1_a4 = kb_mul(two_a1, a[4]);
    uint32_t two_a2_a3 = kb_mul(two_a2, a[3]);
    uint32_t two_a2_a4 = kb_mul(two_a2, a[4]);
    uint32_t two_a3_a4 = kb_mul(two_a3, a[4]);

    uint32_t a3_sq = kb_square(a[3]);
    uint32_t a4_sq = kb_square(a[4]);

    // res[0] = a0^2 + 2*a1*a4 + 2*a2*a3 - a4^2
    out[0] = kb_sub(kb_add(kb_add(kb_square(a[0]), two_a1_a4), two_a2_a3), a4_sq);

    // res[1] = 2*a0*a1 + a3^2 + 2*a2*a4
    out[1] = kb_add(kb_add(kb_mul(two_a0, a[1]), a3_sq), two_a2_a4);

    // res[2] = a1^2 + 2*a0*a2 - 2*a1*a4 - 2*a2*a3 + 2*a3*a4 + a4^2
    out[2] = kb_add(kb_sub(kb_sub(kb_add(kb_square(a[1]), kb_mul(two_a0, a[2])),
             two_a1_a4), two_a2_a3), kb_add(two_a3_a4, a4_sq));

    // res[3] = 2*a0*a3 + 2*a1*a2 - a3^2 - 2*a2*a4 + a4^2
    out[3] = kb_add(kb_sub(kb_sub(kb_add(kb_mul(two_a0, a[3]), kb_mul(two_a1, a[2])),
             a3_sq), two_a2_a4), a4_sq);

    // res[4] = a2^2 + 2*a0*a4 + 2*a1*a3 - 2*a3*a4
    out[4] = kb_sub(kb_add(kb_add(kb_square(a[2]), kb_mul(two_a0, a[4])),
             kb_mul(two_a1, a[3])), two_a3_a4);
}

// Quintic extension halving: out = a / 2 (component-wise).
__device__ __forceinline__ void qe_halve(const uint32_t a[5], uint32_t out[5]) {
    #pragma unroll
    for (int i = 0; i < 5; i++) out[i] = kb_halve(a[i]);
}

// Quintic extension cube: out = a^3.
__device__ __forceinline__ void qe_cube(const uint32_t a[5], uint32_t out[5]) {
    uint32_t sq[5];
    qe_square(a, sq);
    qe_mul(sq, a, out);
}

// Quintic extension: set to zero.
__device__ __forceinline__ void qe_zero(uint32_t out[5]) {
    #pragma unroll
    for (int i = 0; i < 5; i++) out[i] = 0;
}

// Quintic extension: set to one (Montgomery form).
__device__ __forceinline__ void qe_one(uint32_t out[5]) {
    out[0] = KB_MONTY_ONE;
    out[1] = 0; out[2] = 0; out[3] = 0; out[4] = 0;
}

// Quintic extension: embed base field element.
__device__ __forceinline__ void qe_from_base(uint32_t scalar, uint32_t out[5]) {
    out[0] = scalar;
    out[1] = 0; out[2] = 0; out[3] = 0; out[4] = 0;
}

// Quintic extension Frobenius matrix, matching
// lean-da/crates/backend/koala-bear/src/quintic_extension/mod.rs.
// Constants are stored in Montgomery form.
__device__ __constant__ uint32_t KB_QE_FROBENIUS[4][5] = {
    {765015189u, 235704805u, 564529442u, 1275025315u, 1102401726u},
    {169616684u, 1188396601u, 806656646u, 992929951u, 830547243u},
    {905752085u, 405622337u, 1280056543u, 122670283u, 1249984505u},
    {688953779u, 980828988u, 565273709u, 1858491776u, 932120269u},
};

// Frobenius endomorphism over F[X] / (X^5 + X^2 - 1), matching the CPU
// implementation in quintic_extension/extension.rs exactly.
__device__ __forceinline__ void qe_frobenius(const uint32_t a[5], uint32_t out[5]) {
    out[0] = a[0];
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t ai = a[i + 1];
        out[0] = kb_add(out[0], kb_mul(ai, KB_QE_FROBENIUS[i][0]));
        out[1] = kb_add(out[1], kb_mul(ai, KB_QE_FROBENIUS[i][1]));
        out[2] = kb_add(out[2], kb_mul(ai, KB_QE_FROBENIUS[i][2]));
        out[3] = kb_add(out[3], kb_mul(ai, KB_QE_FROBENIUS[i][3]));
        out[4] = kb_add(out[4], kb_mul(ai, KB_QE_FROBENIUS[i][4]));
    }
}

__device__ __forceinline__ void qe_repeated_frobenius(
    const uint32_t a[5], int count, uint32_t out[5]
) {
    if (count <= 0) {
        #pragma unroll
        for (int i = 0; i < 5; i++) out[i] = a[i];
        return;
    }

    count %= 5;
    if (count == 0) {
        #pragma unroll
        for (int i = 0; i < 5; i++) out[i] = a[i];
        return;
    }

    qe_frobenius(a, out);
    for (int i = 1; i < count; i++) {
        uint32_t next[5];
        qe_frobenius(out, next);
        #pragma unroll
        for (int k = 0; k < 5; k++) out[k] = next[k];
    }
}

// Exact quintic inverse formula from the CPU prover.
__device__ __forceinline__ void qe_inv(const uint32_t a[5], uint32_t out[5]) {
    uint32_t a_exp_q[5];
    qe_frobenius(a, a_exp_q);

    uint32_t a_mul_aq[5];
    qe_mul(a, a_exp_q, a_mul_aq);

    uint32_t a_exp_q_plus_q_sq[5];
    qe_frobenius(a_mul_aq, a_exp_q_plus_q_sq);

    uint32_t a_exp_q3_plus_q4[5];
    qe_repeated_frobenius(a_exp_q_plus_q_sq, 2, a_exp_q3_plus_q4);

    uint32_t prod_conj[5];
    qe_mul(a_exp_q_plus_q_sq, a_exp_q3_plus_q4, prod_conj);

    uint32_t norm_weights[5] = {
        prod_conj[0],
        prod_conj[4],
        prod_conj[3],
        prod_conj[2],
        kb_sub(prod_conj[1], prod_conj[4]),
    };
    uint32_t norm = kb_dot5(a, norm_weights);
    uint32_t norm_inv = kb_inv(norm);
    qe_base_mul(prod_conj, norm_inv, out);
}
