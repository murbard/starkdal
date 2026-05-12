// ExtensionOp table AIR constraints — CUDA device code.
// 33 constraints, degree 6, 29 up columns + 13 down columns.
//
// Uses quintic extension field multiplication from koalabear_field.cuh.

#pragma once
#include "../../field/koalabear_field.cuh"

// Column indices.
#define EO_COL_IS_BE   0
#define EO_COL_START   1
#define EO_COL_FLAG_ADD 2
#define EO_COL_FLAG_MUL 3
#define EO_COL_FLAG_PEQ 4
#define EO_COL_LEN     5
#define EO_COL_IDX_A   6
#define EO_COL_IDX_B   7
#define EO_COL_IDX_RES 8
#define EO_COL_VA      9   // 5 columns
#define EO_COL_VB      14  // 5 columns
#define EO_COL_VRES    19  // 5 columns
#define EO_COL_COMP    24  // 5 columns

// Quintic mul helper (uses qe_mul from shared header).
__device__ void quintic_mul_air(const uint32_t a[5], const uint32_t b[5], uint32_t out[5]) {
    qe_mul(a, b, out);
}

__device__ void eval_extension_op_air(
    const uint32_t up[29],
    const uint32_t down[13],  // start_down, is_be_down, len_down, flag_add_down, flag_mul_down, flag_peq_down, idx_a_down, idx_b_down, comp_down[5]
    uint32_t constraints[33]  // output: 33 constraint values
) {
    uint32_t ONE = KB_MONTY_ONE;
    // DIMENSION = 5
    uint32_t FIVE = kb_to_monty(5);

    uint32_t is_be = up[EO_COL_IS_BE];
    uint32_t start = up[EO_COL_START];
    uint32_t flag_add = up[EO_COL_FLAG_ADD];
    uint32_t flag_mul = up[EO_COL_FLAG_MUL];
    uint32_t flag_peq = up[EO_COL_FLAG_PEQ];
    uint32_t len = up[EO_COL_LEN];

    uint32_t va[5], vb[5], vres[5], comp[5];
    for (int k = 0; k < 5; k++) {
        va[k] = up[EO_COL_VA + k];
        vb[k] = up[EO_COL_VB + k];
        vres[k] = up[EO_COL_VRES + k];
        comp[k] = up[EO_COL_COMP + k];
    }

    uint32_t start_down = down[0];
    uint32_t is_be_down = down[1];
    uint32_t len_down = down[2];
    uint32_t flag_add_down = down[3];
    uint32_t flag_mul_down = down[4];
    uint32_t flag_peq_down = down[5];
    uint32_t idx_a_down = down[6];
    uint32_t idx_b_down = down[7];
    uint32_t comp_down[5];
    for (int k = 0; k < 5; k++) comp_down[k] = down[8 + k];

    uint32_t is_ee = kb_neg(kb_sub(is_be, ONE));
    uint32_t not_start_down = kb_neg(kb_sub(start_down, ONE));

    uint32_t va_f_or_ef[5];
    va_f_or_ef[0] = va[0];
    for (int k = 1; k < 5; k++) va_f_or_ef[k] = kb_mul(va[k], is_ee);

    uint32_t comp_tail[5];
    for (int k = 0; k < 5; k++) comp_tail[k] = kb_mul(comp_down[k], not_start_down);

    int ci = 0;

    // Boolean constraints (5)
    constraints[ci++] = kb_mul(is_be, kb_sub(is_be, ONE));
    constraints[ci++] = kb_mul(start, kb_sub(start, ONE));
    constraints[ci++] = kb_mul(flag_add, kb_sub(flag_add, ONE));
    constraints[ci++] = kb_mul(flag_mul, kb_sub(flag_mul, ONE));
    constraints[ci++] = kb_mul(flag_peq, kb_sub(flag_peq, ONE));

    // Add constraints (5): (comp[k] - (va_f_or_ef[k] + vb[k] + comp_tail[k])) * flag_add
    for (int k = 0; k < 5; k++) {
        uint32_t expr = kb_sub(comp[k], kb_add(kb_add(va_f_or_ef[k], vb[k]), comp_tail[k]));
        constraints[ci++] = kb_mul(expr, flag_add);
    }

    // Mul constraints (5): (comp[k] - (va_times_vb[k] + comp_tail[k])) * flag_mul
    uint32_t va_times_vb[5];
    quintic_mul_air(va_f_or_ef, vb, va_times_vb);
    for (int k = 0; k < 5; k++) {
        uint32_t expr = kb_sub(comp[k], kb_add(va_times_vb[k], comp_tail[k]));
        constraints[ci++] = kb_mul(expr, flag_mul);
    }

    // PolyEq constraints (5)
    uint32_t poly_eq_val[5];
    for (int k = 0; k < 5; k++) {
        uint32_t base = kb_sub(kb_double(va_times_vb[k]), kb_add(va_f_or_ef[k], vb[k]));
        poly_eq_val[k] = (k == 0) ? kb_add(base, ONE) : base;
    }
    uint32_t comp_down_or_one[5];
    comp_down_or_one[0] = kb_add(kb_mul(comp_down[0], not_start_down), start_down);
    for (int k = 1; k < 5; k++) comp_down_or_one[k] = kb_mul(comp_down[k], not_start_down);
    uint32_t poly_eq_result[5];
    quintic_mul_air(poly_eq_val, comp_down_or_one, poly_eq_result);
    for (int k = 0; k < 5; k++) {
        constraints[ci++] = kb_mul(kb_sub(comp[k], poly_eq_result[k]), flag_peq);
    }

    // Result assignment (5): (comp[k] - vres[k]) * start
    for (int k = 0; k < 5; k++) {
        constraints[ci++] = kb_mul(kb_sub(comp[k], vres[k]), start);
    }

    // Counter/index constraints (8)
    constraints[ci++] = kb_mul(not_start_down, kb_sub(len, kb_sub(len_down, ONE))); // len - len_down - 1
    constraints[ci++] = kb_mul(not_start_down, kb_sub(is_be, is_be_down));
    constraints[ci++] = kb_mul(not_start_down, kb_sub(flag_add, flag_add_down));
    constraints[ci++] = kb_mul(not_start_down, kb_sub(flag_mul, flag_mul_down));
    constraints[ci++] = kb_mul(not_start_down, kb_sub(flag_peq, flag_peq_down));

    uint32_t a_incr = kb_add(is_be, kb_mul(is_ee, FIVE));
    constraints[ci++] = kb_mul(not_start_down, kb_sub(idx_a_down, kb_add(up[EO_COL_IDX_A], a_incr)));
    constraints[ci++] = kb_mul(not_start_down, kb_sub(idx_b_down, kb_add(up[EO_COL_IDX_B], FIVE)));

    // Last constraint: start_down * (len - 1)
    constraints[ci++] = kb_mul(start_down, kb_sub(len, ONE));

    // ci should be 33
}
