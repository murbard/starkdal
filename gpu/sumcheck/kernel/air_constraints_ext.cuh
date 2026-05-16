// Extension-field AIR constraint evaluation — CUDA device code.
//
// Same constraints as air_constraints.cuh, air_ext_op.cuh, air_poseidon.cuh
// but operating on quintic extension field values (uint32_t[5] per element).
// Used for sumcheck rounds 1+ after the first fold converts columns to ext field.

#pragma once
#include "../../field/koalabear_field.cuh"

// ── Execution table (13 constraints) ────────────────────────────────────
// Each up/down value is ext field (5 u32s). Constraints produce ext field values.
// alphas: 13 * 5 ext. weighted: output 5 ext = Σ alpha[c] * constraint[c].
__device__ void eval_execution_air_ext_weighted(
    const uint32_t* up,        // 20 * 5 ext (flat)
    const uint32_t* down,      // 2 * 5 ext (flat)
    const uint32_t* alphas,    // (1+12) * 5 ext, alpha[0] = bus, alpha[1..12] = non-bus
    uint32_t weighted[5],      // output
    uint32_t* bus_data = nullptr  // optional: 5 ext values [flag, data0..3]
) {
    // Aliases (each is 5 u32s)
    const uint32_t *pc = up, *fp = up+5;
    const uint32_t *addr_a = up+10, *addr_b = up+15, *addr_c = up+20;
    const uint32_t *val_a = up+25, *val_b = up+30, *val_c = up+35;
    const uint32_t *op_a = up+40, *op_b = up+45, *op_c = up+50;
    const uint32_t *flag_a = up+55, *flag_b = up+60, *flag_c = up+65;
    const uint32_t *flag_c_fp = up+70, *flag_ab_fp = up+75;
    const uint32_t *mul_flag = up+80, *jump = up+85, *aux = up+90;
    const uint32_t *next_pc = down, *next_fp = down+5;

    uint32_t ONE[5]; qe_one(ONE);
    uint32_t TWO[5]; qe_add(ONE, ONE, TWO);

    // Intermediates
    uint32_t t1[5], t2[5], t3[5];

    // one_m_fa_fab = -(flag_a + flag_ab_fp - 1)
    uint32_t one_m_fa_fab[5];
    qe_add(flag_a, flag_ab_fp, t1); qe_sub(t1, ONE, t1); qe_neg(t1, one_m_fa_fab);

    uint32_t one_m_fb_fab[5];
    qe_add(flag_b, flag_ab_fp, t1); qe_sub(t1, ONE, t1); qe_neg(t1, one_m_fb_fab);

    uint32_t one_m_fc_fcfp[5];
    qe_add(flag_c, flag_c_fp, t1); qe_sub(t1, ONE, t1); qe_neg(t1, one_m_fc_fcfp);

    uint32_t fp_op_a[5], fp_op_b[5], fp_op_c[5];
    qe_add(fp, op_a, fp_op_a);
    qe_add(fp, op_b, fp_op_b);
    qe_add(fp, op_c, fp_op_c);

    // nu_a = flag_a * op_a + one_m_fa_fab * val_a + flag_ab_fp * fp_op_a
    uint32_t nu_a[5], nu_b[5], nu_c[5];
    qe_mul(flag_a, op_a, t1); qe_mul(one_m_fa_fab, val_a, t2); qe_mul(flag_ab_fp, fp_op_a, t3);
    qe_add(t1, t2, nu_a); qe_add(nu_a, t3, nu_a);

    qe_mul(flag_b, op_b, t1); qe_mul(one_m_fb_fab, val_b, t2); qe_mul(flag_ab_fp, fp_op_b, t3);
    qe_add(t1, t2, nu_b); qe_add(nu_b, t3, nu_b);

    qe_mul(flag_c, op_c, t1); qe_mul(one_m_fc_fcfp, val_c, t2); qe_mul(flag_c_fp, fp_op_c, t3);
    qe_add(t1, t2, nu_c); qe_add(nu_c, t3, nu_c);

    // add_flag = aux * 2 - aux * aux
    uint32_t add_flag[5], aux_sq[5];
    qe_mul(aux, TWO, t1); qe_mul(aux, aux, aux_sq); qe_sub(t1, aux_sq, add_flag);

    // deref = halve(aux * (aux - 1))
    uint32_t deref[5], aux_m1[5];
    qe_sub(aux, ONE, aux_m1); qe_mul(aux, aux_m1, t1); qe_halve(t1, deref);

    uint32_t pc_p1[5]; qe_add(pc, ONE, pc_p1);
    uint32_t nu_a_m1[5]; qe_sub(nu_a, ONE, nu_a_m1);

    uint32_t jump_cond[5]; qe_mul(jump, nu_a, jump_cond);
    uint32_t not_jc[5]; qe_sub(jump_cond, ONE, t1); qe_neg(t1, not_jc);

    // Accumulate constraints weighted by alphas
    qe_zero(weighted);
    int ci = 0;
    uint32_t c[5], prod[5];

    // c0: one_m_fa_fab * (addr_a - fp_op_a)
    qe_sub(addr_a, fp_op_a, t1); qe_mul(one_m_fa_fab, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c1: one_m_fb_fab * (addr_b - fp_op_b)
    qe_sub(addr_b, fp_op_b, t1); qe_mul(one_m_fb_fab, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c2: one_m_fc_fcfp * (addr_c - fp_op_c)
    qe_sub(addr_c, fp_op_c, t1); qe_mul(one_m_fc_fcfp, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c3: add_flag * (nu_b - (nu_a + nu_c))
    qe_add(nu_a, nu_c, t1); qe_sub(nu_b, t1, t2); qe_mul(add_flag, t2, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c4: mul_flag * (nu_b - nu_a * nu_c)
    qe_mul(nu_a, nu_c, t1); qe_sub(nu_b, t1, t2); qe_mul(mul_flag, t2, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c5: deref * (addr_b - (val_a + op_b))
    qe_add(val_a, op_b, t1); qe_sub(addr_b, t1, t2); qe_mul(deref, t2, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c6: deref * (val_b - nu_c)
    qe_sub(val_b, nu_c, t1); qe_mul(deref, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c7: jump_cond * (nu_a - 1)
    qe_mul(jump_cond, nu_a_m1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c8: jump_cond * (next_pc - nu_b)
    qe_sub(next_pc, nu_b, t1); qe_mul(jump_cond, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c9: jump_cond * (next_fp - nu_c)
    qe_sub(next_fp, nu_c, t1); qe_mul(jump_cond, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c10: not_jc * (next_pc - pc - 1)
    qe_sub(next_pc, pc_p1, t1); qe_mul(not_jc, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // c11: not_jc * (next_fp - fp)
    qe_sub(next_fp, fp, t1); qe_mul(not_jc, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // Bus data output for ext-field bus constraint computation.
    // The bus constraint is at alpha[0], computed separately in the kernel.
    if (bus_data) {
        // is_precompile = -(add_flag + mul_flag + deref + jump - 1) — ext field
        uint32_t is_precomp[5];
        qe_add(add_flag, mul_flag, t1); qe_add(deref, jump, t2);
        qe_add(t1, t2, t1); qe_sub(t1, ONE, t1); qe_neg(t1, is_precomp);
        // bus_data layout: [flag(5), data0(5), data1(5), data2(5), data3(5)] = 25 u32s
        for (int k = 0; k < 5; k++) {
            bus_data[0*5+k] = is_precomp[k]; // flag
            bus_data[1*5+k] = up[19*5+k];    // precompile_data = up[19]
            bus_data[2*5+k] = nu_a[k];
            bus_data[3*5+k] = nu_b[k];
            bus_data[4*5+k] = nu_c[k];
        }
    }
}

// ── ExtensionOp table (33 constraints) ──────────────────────────────────
__device__ void eval_ext_op_air_ext_weighted(
    const uint32_t* up,        // 29 * 5 ext (flat)
    const uint32_t* down,      // 13 * 5 ext (flat)
    const uint32_t* alphas,    // (1+33) * 5 ext, alpha[0]=bus, alpha[1..33]=non-bus
    uint32_t weighted[5],      // output
    uint32_t* bus_data = nullptr  // optional: 25 u32s [flag(5), data0(5)..data3(5)]
) {
    uint32_t ONE[5]; qe_one(ONE);
    uint32_t FIVE[5]; qe_from_base(kb_to_monty(5), FIVE);

    // Column aliases
    const uint32_t *is_be = up + 0*5;
    const uint32_t *start = up + 1*5;
    const uint32_t *flag_add = up + 2*5;
    const uint32_t *flag_mul = up + 3*5;
    const uint32_t *flag_peq = up + 4*5;
    const uint32_t *len = up + 5*5;
    const uint32_t *idx_a = up + 6*5;
    const uint32_t *idx_b = up + 7*5;
    // idx_res = up + 8*5 (not used in constraints directly)
    const uint32_t *va = up + 9*5;   // 5 coords
    const uint32_t *vb = up + 14*5;  // 5 coords
    const uint32_t *vres = up + 19*5; // 5 coords
    const uint32_t *comp = up + 24*5; // 5 coords

    const uint32_t *start_down = down + 0*5;
    const uint32_t *is_be_down = down + 1*5;
    const uint32_t *len_down = down + 2*5;
    const uint32_t *flag_add_down = down + 3*5;
    const uint32_t *flag_mul_down = down + 4*5;
    const uint32_t *flag_peq_down = down + 5*5;
    const uint32_t *idx_a_down = down + 6*5;
    const uint32_t *idx_b_down = down + 7*5;
    const uint32_t *comp_down = down + 8*5; // 5 coords = 25 u32s

    uint32_t t1[5], t2[5], t3[5], c[5], prod[5];

    // is_ee = -(is_be - 1) = 1 - is_be
    uint32_t is_ee[5]; qe_sub(is_be, ONE, t1); qe_neg(t1, is_ee);
    // not_start_down = 1 - start_down
    uint32_t nsd[5]; qe_sub(start_down, ONE, t1); qe_neg(t1, nsd);

    // va_f_or_ef: va[0] unchanged, va[k>=1] *= is_ee
    uint32_t va_foe[5][5];
    for (int k = 0; k < 5; k++) {
        for (int i = 0; i < 5; i++) va_foe[k][i] = va[k*5+i];
    }
    // va_foe[0] stays as va[0]
    for (int k = 1; k < 5; k++) {
        qe_mul(va_foe[k], is_ee, t1);
        for (int i = 0; i < 5; i++) va_foe[k][i] = t1[i];
    }

    // comp_tail[k] = comp_down[k] * not_start_down
    uint32_t comp_tail[5][5];
    for (int k = 0; k < 5; k++) {
        qe_mul(comp_down + k*5, nsd, comp_tail[k]);
    }

    // Quintic mul: va_times_vb = va_foe * vb (quintic × quintic in ext field)
    // va_foe and vb are each 5 "coordinates" of the quintic extension, each coordinate is ext field.
    // This is extension-over-extension multiplication, NOT plain qe_mul.
    // However, the AIR evaluates the same formula: we need to compute the quintic dot products.
    // Since each coordinate is already ext field, the quintic mul formula is the same
    // but each scalar op becomes an ext-field op.
    uint32_t va_times_vb[5][5];
    {
        // Precompute differences
        uint32_t b1_m4[5]; qe_sub(vb + 1*5, vb + 4*5, b1_m4);
        uint32_t b0_m3[5]; qe_sub(vb + 0*5, vb + 3*5, b0_m3);
        uint32_t b4_m2[5]; qe_sub(vb + 4*5, vb + 2*5, b4_m2);
        uint32_t b3_m_b1m4[5]; qe_sub(vb + 3*5, b1_m4, b3_m_b1m4);

        // res[0] = Σ va_foe[j] * row0[j]
        // row0 = [b0, b4, b3, b2, b1_m4]
        uint32_t acc[5]; qe_zero(acc);
        qe_mul(va_foe[0], vb + 0*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[1], vb + 4*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[2], vb + 3*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[3], vb + 2*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[4], b1_m4, t1); qe_add(acc, t1, acc);
        for (int i = 0; i < 5; i++) va_times_vb[0][i] = acc[i];

        // res[1]: row1 = [b1, b0, b4, b3, b2]
        qe_zero(acc);
        qe_mul(va_foe[0], vb + 1*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[1], vb + 0*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[2], vb + 4*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[3], vb + 3*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[4], vb + 2*5, t1); qe_add(acc, t1, acc);
        for (int i = 0; i < 5; i++) va_times_vb[1][i] = acc[i];

        // res[2]: row2 = [b2, b1_m4, b0_m3, b4_m2, b3_m_b1m4]
        qe_zero(acc);
        qe_mul(va_foe[0], vb + 2*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[1], b1_m4, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[2], b0_m3, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[3], b4_m2, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[4], b3_m_b1m4, t1); qe_add(acc, t1, acc);
        for (int i = 0; i < 5; i++) va_times_vb[2][i] = acc[i];

        // res[3]: row3 = [b3, b2, b1_m4, b0_m3, b4_m2]
        qe_zero(acc);
        qe_mul(va_foe[0], vb + 3*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[1], vb + 2*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[2], b1_m4, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[3], b0_m3, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[4], b4_m2, t1); qe_add(acc, t1, acc);
        for (int i = 0; i < 5; i++) va_times_vb[3][i] = acc[i];

        // res[4]: row4 = [b4, b3, b2, b1_m4, b0_m3]
        qe_zero(acc);
        qe_mul(va_foe[0], vb + 4*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[1], vb + 3*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[2], vb + 2*5, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[3], b1_m4, t1); qe_add(acc, t1, acc);
        qe_mul(va_foe[4], b0_m3, t1); qe_add(acc, t1, acc);
        for (int i = 0; i < 5; i++) va_times_vb[4][i] = acc[i];
    }

    qe_zero(weighted);
    int ci = 0;

    // Boolean constraints (5): x * (x - 1)
    #define BOOL_CONSTRAINT(x) do { \
        qe_sub(ONE, x, t1); qe_mul(x, t1, c); /* x*(1-x) */ \
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++; \
    } while(0)
    BOOL_CONSTRAINT(is_be);
    BOOL_CONSTRAINT(start);
    BOOL_CONSTRAINT(flag_add);
    BOOL_CONSTRAINT(flag_mul);
    BOOL_CONSTRAINT(flag_peq);
    #undef BOOL_CONSTRAINT

    // Add constraints (5): (comp[k] - (va_foe[k] + vb[k] + comp_tail[k])) * flag_add
    for (int k = 0; k < 5; k++) {
        qe_add(va_foe[k], vb + k*5, t1);
        qe_add(t1, comp_tail[k], t1);
        qe_sub(comp + k*5, t1, t2);
        qe_mul(t2, flag_add, c);
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    }

    // Mul constraints (5): (comp[k] - (va_times_vb[k] + comp_tail[k])) * flag_mul
    for (int k = 0; k < 5; k++) {
        qe_add(va_times_vb[k], comp_tail[k], t1);
        qe_sub(comp + k*5, t1, t2);
        qe_mul(t2, flag_mul, c);
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    }

    // PolyEq constraints (5)
    {
        uint32_t pev[5][5]; // poly_eq_val
        for (int k = 0; k < 5; k++) {
            qe_add(va_times_vb[k], va_times_vb[k], t1); // 2 * va_times_vb
            qe_add(va_foe[k], vb + k*5, t2);
            qe_sub(t1, t2, pev[k]);
            if (k == 0) qe_add(pev[k], ONE, pev[k]);
        }
        // comp_down_or_one[0] = comp_down[0]*nsd + start_down
        // comp_down_or_one[k>0] = comp_down[k]*nsd
        uint32_t cdo[5][5];
        qe_mul(comp_down + 0*5, nsd, t1);
        qe_add(t1, start_down, cdo[0]);
        for (int k = 1; k < 5; k++) qe_mul(comp_down + k*5, nsd, cdo[k]);

        // poly_eq_result = quintic_mul(pev, cdo)
        uint32_t per[5][5];
        {
            uint32_t d1m4[5]; qe_sub(cdo[1], cdo[4], d1m4);
            uint32_t d0m3[5]; qe_sub(cdo[0], cdo[3], d0m3);
            uint32_t d4m2[5]; qe_sub(cdo[4], cdo[2], d4m2);
            uint32_t d3m_d1m4[5]; qe_sub(cdo[3], d1m4, d3m_d1m4);

            uint32_t acc[5];
            // res[0]
            qe_zero(acc);
            qe_mul(pev[0], cdo[0], t1); qe_add(acc, t1, acc);
            qe_mul(pev[1], cdo[4], t1); qe_add(acc, t1, acc);
            qe_mul(pev[2], cdo[3], t1); qe_add(acc, t1, acc);
            qe_mul(pev[3], cdo[2], t1); qe_add(acc, t1, acc);
            qe_mul(pev[4], d1m4, t1); qe_add(acc, t1, acc);
            for (int i = 0; i < 5; i++) per[0][i] = acc[i];
            // res[1]
            qe_zero(acc);
            qe_mul(pev[0], cdo[1], t1); qe_add(acc, t1, acc);
            qe_mul(pev[1], cdo[0], t1); qe_add(acc, t1, acc);
            qe_mul(pev[2], cdo[4], t1); qe_add(acc, t1, acc);
            qe_mul(pev[3], cdo[3], t1); qe_add(acc, t1, acc);
            qe_mul(pev[4], cdo[2], t1); qe_add(acc, t1, acc);
            for (int i = 0; i < 5; i++) per[1][i] = acc[i];
            // res[2]
            qe_zero(acc);
            qe_mul(pev[0], cdo[2], t1); qe_add(acc, t1, acc);
            qe_mul(pev[1], d1m4, t1); qe_add(acc, t1, acc);
            qe_mul(pev[2], d0m3, t1); qe_add(acc, t1, acc);
            qe_mul(pev[3], d4m2, t1); qe_add(acc, t1, acc);
            qe_mul(pev[4], d3m_d1m4, t1); qe_add(acc, t1, acc);
            for (int i = 0; i < 5; i++) per[2][i] = acc[i];
            // res[3]
            qe_zero(acc);
            qe_mul(pev[0], cdo[3], t1); qe_add(acc, t1, acc);
            qe_mul(pev[1], cdo[2], t1); qe_add(acc, t1, acc);
            qe_mul(pev[2], d1m4, t1); qe_add(acc, t1, acc);
            qe_mul(pev[3], d0m3, t1); qe_add(acc, t1, acc);
            qe_mul(pev[4], d4m2, t1); qe_add(acc, t1, acc);
            for (int i = 0; i < 5; i++) per[3][i] = acc[i];
            // res[4]
            qe_zero(acc);
            qe_mul(pev[0], cdo[4], t1); qe_add(acc, t1, acc);
            qe_mul(pev[1], cdo[3], t1); qe_add(acc, t1, acc);
            qe_mul(pev[2], cdo[2], t1); qe_add(acc, t1, acc);
            qe_mul(pev[3], d1m4, t1); qe_add(acc, t1, acc);
            qe_mul(pev[4], d0m3, t1); qe_add(acc, t1, acc);
            for (int i = 0; i < 5; i++) per[4][i] = acc[i];
        }

        for (int k = 0; k < 5; k++) {
            qe_sub(comp + k*5, per[k], t1);
            qe_mul(t1, flag_peq, c);
            qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
        }
    }

    // Result assignment (5): (comp[k] - vres[k]) * start
    for (int k = 0; k < 5; k++) {
        qe_sub(comp + k*5, vres + k*5, t1);
        qe_mul(t1, start, c);
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    }

    // Counter constraints (7): all multiplied by nsd (not_start_down)
    // len - len_down - 1
    qe_sub(len, len_down, t1); qe_sub(t1, ONE, t1); qe_mul(nsd, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    // is_be - is_be_down
    qe_sub(is_be, is_be_down, t1); qe_mul(nsd, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    // flag_add - flag_add_down
    qe_sub(flag_add, flag_add_down, t1); qe_mul(nsd, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    // flag_mul - flag_mul_down
    qe_sub(flag_mul, flag_mul_down, t1); qe_mul(nsd, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    // flag_peq - flag_peq_down
    qe_sub(flag_peq, flag_peq_down, t1); qe_mul(nsd, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    // idx_a_down - idx_a - (is_be + is_ee * 5)
    qe_mul(is_ee, FIVE, t1); qe_add(is_be, t1, t1);
    qe_add(idx_a, t1, t2); qe_sub(idx_a_down, t2, t1); qe_mul(nsd, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    // idx_b_down - idx_b - 5
    qe_add(idx_b, FIVE, t1); qe_sub(idx_b_down, t1, t2); qe_mul(nsd, t2, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // Last: start_down * (len - 1)
    qe_sub(len, ONE, t1); qe_mul(start_down, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    // ci == 33

    if (bus_data) {
        // activation_flag = start * (flag_add + flag_mul + flag_poly_eq)
        uint32_t active[5]; qe_add(flag_add, flag_mul, t1); qe_add(t1, flag_peq, active);
        uint32_t af[5]; qe_mul(start, active, af);
        // aux = is_be*4 + flag_add*8 + flag_mul*16 + flag_poly_eq*32 + len*64
        uint32_t FOUR_E[5]; qe_from_base(kb_to_monty(4), FOUR_E);
        uint32_t EIGHT_E[5]; qe_from_base(kb_to_monty(8), EIGHT_E);
        uint32_t SIXTEEN_E[5]; qe_from_base(kb_to_monty(16), SIXTEEN_E);
        uint32_t THIRTYTWO_E[5]; qe_from_base(kb_to_monty(32), THIRTYTWO_E);
        uint32_t SIXTYFOUR_E[5]; qe_from_base(kb_to_monty(64), SIXTYFOUR_E);
        uint32_t aux_v[5];
        qe_mul(is_be, FOUR_E, t1); qe_mul(flag_add, EIGHT_E, t2);
        qe_add(t1, t2, aux_v);
        qe_mul(flag_mul, SIXTEEN_E, t1); qe_add(aux_v, t1, aux_v);
        qe_mul(flag_peq, THIRTYTWO_E, t1); qe_add(aux_v, t1, aux_v);
        qe_mul(len, SIXTYFOUR_E, t1); qe_add(aux_v, t1, aux_v);
        for (int k = 0; k < 5; k++) {
            bus_data[0*5+k] = af[k];       // flag
            bus_data[1*5+k] = aux_v[k];    // data[0]
            bus_data[2*5+k] = idx_a[k];    // data[1] = up[6]
            bus_data[3*5+k] = idx_b[k];    // data[2] = up[7]
            bus_data[4*5+k] = up[8*5+k];   // data[3] = idx_res = up[8]
        }
    }
}

// ── Poseidon16 table (81 constraints) ───────────────────────────────────
// state evolves through the full permutation, all in ext field.
// Round constants are base field (embedded into ext as needed).
__device__ void eval_poseidon16_air_ext_weighted(
    const uint32_t* up,            // 100 * 5 ext (flat)
    const uint32_t* alphas,        // (1+81) * 5 ext, alpha[0]=bus, alpha[1..81]=non-bus
    const uint32_t* rc_base,       // 28 * 16 base field round constants
    const uint32_t* mds_base,      // 16 base field MDS circulant column
    const uint32_t* sparse_data,   // sparse matrix data (base field)
    uint32_t weighted[5],          // output
    uint32_t* bus_data = nullptr,  // optional: 25 u32s [flag(5), data0..3(5 each)]
    uint32_t* low_weighted = nullptr, // optional: weighted low-degree block contribution
    uint32_t* post_low_state = nullptr, // optional: 16 ext elements after the low-degree block
    bool skip_low = false,
    const uint32_t* cached_low_state = nullptr
) {
    uint32_t ONE[5]; qe_one(ONE);
    uint32_t t1[5], t2[5], c[5], prod[5];

    // Column aliases (each 5 u32s)
    const uint32_t *flag_active = up + 0*5;
    const uint32_t *flag_half_output = up + 3*5;
    const uint32_t *flag_hardcoded_left = up + 4*5;
    const uint32_t *offset_hardcoded_left = up + 5*5;
    const uint32_t *eff_idx_left_first = up + 6*5;
    const uint32_t *eff_idx_left_second = up + 7*5;
    const uint32_t *inputs_p = up + 8*5;       // 16 × 5
    const uint32_t *beg_full_0_p = up + 24*5;  // 16 × 5
    const uint32_t *beg_full_1_p = up + 40*5;  // 16 × 5
    const uint32_t *partial_cols_p = up + 56*5; // 20 × 5
    const uint32_t *end_full_p = up + 76*5;    // 16 × 5
    const uint32_t *outputs_p = up + 92*5;     // 8 × 5

    qe_zero(weighted);
    if (low_weighted) qe_zero(low_weighted);
    int ci = 0;

    // Flag constraints (5)
    #define BOOL_C(x) do { \
        qe_sub(ONE, x, t1); qe_mul(x, t1, c); /* x*(1-x) */ \
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++; \
    } while(0)
    BOOL_C(flag_active);
    BOOL_C(flag_half_output);
    BOOL_C(flag_hardcoded_left);
    #undef BOOL_C

    // fhl * (offset_hardcoded_left - eff_idx_left_first) = 0
    qe_sub(offset_hardcoded_left, eff_idx_left_first, t1);
    qe_mul(flag_hardcoded_left, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // (1 - fhl) * (index_a - eff_idx_left_first) = 0
    uint32_t one_m_fhl[5]; qe_sub(ONE, flag_hardcoded_left, one_m_fhl);
    uint32_t four[5]; qe_from_base(kb_to_monty(4), four);
    uint32_t idx_a[5];
    qe_mul(one_m_fhl, four, t1); qe_sub(eff_idx_left_second, t1, idx_a);
    qe_sub(idx_a, eff_idx_left_first, t1);
    qe_mul(one_m_fhl, t1, c);
    qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;

    // ── Poseidon permutation constraints ──
    // Initialize state from inputs (ext field).
    uint32_t state[16][5];
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 5; k++)
            state[i][k] = inputs_p[i*5 + k];

    // Helper: MDS multiply on ext-field state with base-field MDS constants
    // out[i] = Σ_j mds_circ[(16+i-j)%16] * state[j]
    // mds_circ[k] is base field → qe_base_mul
    #define MDS_EXT() do { \
        uint32_t tmp[16][5]; \
        for (int i = 0; i < 16; i++) { \
            qe_zero(tmp[i]); \
            for (int j = 0; j < 16; j++) { \
                uint32_t mds_val = mds_base[(16+i-j) & 15]; \
                qe_base_mul(state[j], mds_val, t1); \
                qe_add(tmp[i], t1, tmp[i]); \
            } \
        } \
        for (int i = 0; i < 16; i++) \
            for (int k = 0; k < 5; k++) \
                state[i][k] = tmp[i][k]; \
    } while(0)

    // Helper: add base-field round constants to ext-field state
    #define ADD_RC_EXT(rc_off) do { \
        for (int i = 0; i < 16; i++) { \
            uint32_t rc_val = rc_base[(rc_off)*16 + i]; \
            state[i][0] = kb_add(state[i][0], rc_val); \
        } \
    } while(0)

    // Helper: sbox (cube) all 16 state elements in ext field
    #define SBOX_EXT() do { \
        for (int i = 0; i < 16; i++) qe_cube(state[i], state[i]); \
    } while(0)

    // Initial full rounds (4 rounds = 2 pairs)
    ADD_RC_EXT(0); SBOX_EXT(); MDS_EXT();
    ADD_RC_EXT(1); SBOX_EXT(); MDS_EXT();

    // Assert state == beg_full_0, then replace state with witness (degree reduction)
    for (int i = 0; i < 16; i++) {
        qe_sub(state[i], beg_full_0_p + i*5, c);
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
        for (int k = 0; k < 5; k++) state[i][k] = beg_full_0_p[i*5 + k];
    }

    ADD_RC_EXT(2); SBOX_EXT(); MDS_EXT();
    ADD_RC_EXT(3); SBOX_EXT(); MDS_EXT();

    // Assert state == beg_full_1, then replace state with witness (degree reduction)
    for (int i = 0; i < 16; i++) {
        qe_sub(state[i], beg_full_1_p + i*5, c);
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
        for (int k = 0; k < 5; k++) state[i][k] = beg_full_1_p[i*5 + k];
    }

    // ── Partial rounds (20 rounds) ──
    const uint32_t* m_i = sparse_data;
    const uint32_t* sp_first_row = sparse_data + 256;
    const uint32_t* sp_v = sparse_data + 256 + 320;
    const uint32_t* sp_scalar_rc = sparse_data + 256 + 320 + 300;
    const uint32_t* sp_first_rc = sparse_data + 896;  // sparse_first_round_constants (16 values)

    if (skip_low) {
        ci += 20;
        for (int i = 0; i < 16; i++)
            for (int k = 0; k < 5; k++)
                state[i][k] = cached_low_state[i * 5 + k];
    } else {
        // Add sparse first round constants (NOT raw rc_base+64!)
        for (int i = 0; i < 16; i++) {
            state[i][0] = kb_add(state[i][0], sp_first_rc[i]);
        }

        // Multiply by m_i (16x16 base field matrix on ext field state)
        {
            uint32_t tmp[16][5];
            for (int i = 0; i < 16; i++) {
                qe_zero(tmp[i]);
                for (int j = 0; j < 16; j++) {
                    qe_base_mul(state[j], m_i[i*16+j], t1);
                    qe_add(tmp[i], t1, tmp[i]);
                }
            }
            for (int i = 0; i < 16; i++)
                for (int k = 0; k < 5; k++)
                    state[i][k] = tmp[i][k];
        }

        for (int r = 0; r < 20; r++) {
            // S-box on state[0] only
            qe_cube(state[0], state[0]);

            // Assert: state[0] == partial_cols[r]
            qe_sub(state[0], partial_cols_p + r*5, c);
            qe_mul(alphas + (ci+1)*5, c, prod);
            qe_add(weighted, prod, weighted);
            if (low_weighted) qe_add(low_weighted, prod, low_weighted);
            ci++;

            // Replace state[0] with witness column (low_degree_block reset — keeps degree bounded)
            for (int k = 0; k < 5; k++) state[0][k] = partial_cols_p[r*5 + k];

            // Add scalar round constant (except last round)
            if (r < 19) {
                state[0][0] = kb_add(state[0][0], sp_scalar_rc[r]);
            }

            // Sparse matrix multiply
            uint32_t old_s0[5];
            for (int k = 0; k < 5; k++) old_s0[k] = state[0][k];
            {
                uint32_t new_s0[5]; qe_zero(new_s0);
                qe_base_mul(old_s0, sp_first_row[r*16+0], t1); qe_add(new_s0, t1, new_s0);
                for (int j = 1; j < 16; j++) {
                    qe_base_mul(state[j], sp_first_row[r*16+j], t1);
                    qe_add(new_s0, t1, new_s0);
                }
                for (int k = 0; k < 5; k++) state[0][k] = new_s0[k];
            }

            // state[i] += old_s0 * sp_v[r][i-1] for i=1..15
            for (int i = 1; i < 16; i++) {
                qe_base_mul(old_s0, sp_v[r*15 + (i-1)], t1);
                qe_add(state[i], t1, state[i]);
            }
        }

        if (post_low_state) {
            for (int i = 0; i < 16; i++)
                for (int k = 0; k < 5; k++)
                    post_low_state[i * 5 + k] = state[i][k];
        }
    }

    // ── Final full rounds ──
    int rc_final_off = 24; // rounds 24-27
    ADD_RC_EXT(rc_final_off); SBOX_EXT(); MDS_EXT();
    ADD_RC_EXT(rc_final_off+1); SBOX_EXT(); MDS_EXT();

    // Assert state == end_full, then replace state with witness (degree reduction)
    for (int i = 0; i < 16; i++) {
        qe_sub(state[i], end_full_p + i*5, c);
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
        for (int k = 0; k < 5; k++) state[i][k] = end_full_p[i*5 + k];
    }

    ADD_RC_EXT(rc_final_off+2); SBOX_EXT(); MDS_EXT();
    ADD_RC_EXT(rc_final_off+3); SBOX_EXT(); MDS_EXT();

    // Compression: add inputs back
    for (int i = 0; i < 16; i++)
        qe_add(state[i], inputs_p + i*5, state[i]);

    // Output constraints
    uint32_t one_m_half[5]; qe_sub(ONE, flag_half_output, one_m_half);
    for (int i = 0; i < 8; i++) {
        qe_sub(state[i], outputs_p + i*5, t1);
        if (i < 4) {
            for (int k = 0; k < 5; k++) c[k] = t1[k]; // always constrained
        } else {
            qe_mul(one_m_half, t1, c); // gated
        }
        qe_mul(alphas + (ci+1)*5, c, prod); qe_add(weighted, prod, weighted); ci++;
    }
    // ci == 81

    if (bus_data) {
        // flag = flag_active (column 0)
        for (int k = 0; k < 5; k++) bus_data[0*5+k] = flag_active[k];
        // precompile_data = 1 + flag_half_output*2 + flag_hardcoded_left*4 + fhl*offset_hcl*8
        uint32_t TWO_E[5]; qe_from_base(kb_to_monty(2), TWO_E);
        uint32_t FOUR_E[5]; qe_from_base(kb_to_monty(4), FOUR_E);
        uint32_t EIGHT_E[5]; qe_from_base(kb_to_monty(8), EIGHT_E);
        uint32_t pd[5];
        qe_mul(flag_half_output, TWO_E, t1);
        qe_mul(flag_hardcoded_left, FOUR_E, t2);
        qe_add(ONE, t1, pd); qe_add(pd, t2, pd);
        qe_mul(flag_hardcoded_left, offset_hardcoded_left, t1);
        qe_mul(t1, EIGHT_E, t1); qe_add(pd, t1, pd);
        for (int k = 0; k < 5; k++) bus_data[1*5+k] = pd[k];
        // index_a = eff_idx_left_second - (1-fhl)*4
        for (int k = 0; k < 5; k++) bus_data[2*5+k] = idx_a[k];
        // index_b = up[1], index_res = up[2]
        for (int k = 0; k < 5; k++) bus_data[3*5+k] = up[1*5+k];
        for (int k = 0; k < 5; k++) bus_data[4*5+k] = up[2*5+k];
    }

    #undef MDS_EXT
    #undef ADD_RC_EXT
    #undef SBOX_EXT
}
