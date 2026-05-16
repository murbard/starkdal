// AIR constraint evaluation functions — CUDA device code.
//
// Ports the 3 AIR table constraint formulas from leanVM to CUDA.
// Each function evaluates all constraints for one table type at one
// row, accumulating into an extension-field result with alpha powers.
//
// All field arithmetic uses the shared koalabear_field.cuh header.

#pragma once
#include "../../field/koalabear_field.cuh"

// ── Execution table AIR (13 constraints, degree 5) ──────────────────────
// Columns: 20 up + 2 down (PC, FP shifted).
// up[0..20] = [PC, FP, addr_a, addr_b, addr_c, val_a, val_b, val_c,
//              op_a, op_b, op_c, flag_a, flag_b, flag_c, flag_c_fp, flag_ab_fp,
//              mul, jump, aux, precompile_data]
// down[0..2] = [next_pc, next_fp]
//
// Returns the accumulated constraint value (base field, to be multiplied by
// alpha powers and eq factor on the host/caller side).
__device__ void eval_execution_air(
    const uint32_t up[20],
    const uint32_t down[2],
    uint32_t constraints[13],  // output: 12 constraint expressions (base field) + slot 12 unused
    uint32_t* bus_data = nullptr  // optional output: 5 values for bus constraint
) {
    // Column aliases.
    uint32_t pc = up[0], fp = up[1];
    uint32_t addr_a = up[2], addr_b = up[3], addr_c = up[4];
    uint32_t val_a = up[5], val_b = up[6], val_c = up[7];
    uint32_t op_a = up[8], op_b = up[9], op_c = up[10];
    uint32_t flag_a = up[11], flag_b = up[12], flag_c = up[13];
    uint32_t flag_c_fp = up[14], flag_ab_fp = up[15];
    uint32_t mul_flag = up[16], jump = up[17], aux = up[18];
    // up[19] = precompile_data (used for bus, skipped here)
    uint32_t next_pc = down[0], next_fp = down[1];

    // ONE in Montgomery form.
    uint32_t ONE = KB_MONTY_ONE;
    uint32_t TWO = kb_add(ONE, ONE);

    // Intermediate values.
    uint32_t one_m_fa_fab = kb_neg(kb_sub(kb_add(flag_a, flag_ab_fp), ONE));
    uint32_t one_m_fb_fab = kb_neg(kb_sub(kb_add(flag_b, flag_ab_fp), ONE));
    uint32_t one_m_fc_fcfp = kb_neg(kb_sub(kb_add(flag_c, flag_c_fp), ONE));

    uint32_t fp_op_a = kb_add(fp, op_a);
    uint32_t fp_op_b = kb_add(fp, op_b);
    uint32_t fp_op_c = kb_add(fp, op_c);

    uint32_t nu_a = kb_add(kb_add(kb_mul(flag_a, op_a), kb_mul(one_m_fa_fab, val_a)),
                           kb_mul(flag_ab_fp, fp_op_a));
    uint32_t nu_b = kb_add(kb_add(kb_mul(flag_b, op_b), kb_mul(one_m_fb_fab, val_b)),
                           kb_mul(flag_ab_fp, fp_op_b));
    uint32_t nu_c = kb_add(kb_add(kb_mul(flag_c, op_c), kb_mul(one_m_fc_fcfp, val_c)),
                           kb_mul(flag_c_fp, fp_op_c));

    uint32_t add_flag = kb_sub(kb_mul(aux, TWO), kb_mul(aux, aux));
    uint32_t aux_m_one = kb_sub(aux, ONE);
    uint32_t deref = kb_halve(kb_mul(aux, aux_m_one));
    // is_precompile = -(add + mul + deref + jump - 1)
    // uint32_t is_precompile = kb_neg(kb_sub(kb_add(kb_add(add_flag, mul_flag), kb_add(deref, jump)), ONE));

    uint32_t pc_plus_one = kb_add(pc, ONE);
    uint32_t nu_a_m_one = kb_sub(nu_a, ONE);
    uint32_t jump_cond = kb_mul(jump, nu_a);
    uint32_t not_jump_cond = kb_neg(kb_sub(jump_cond, ONE));

    // 13 constraints.
    constraints[0] = kb_mul(one_m_fa_fab, kb_sub(addr_a, fp_op_a));
    constraints[1] = kb_mul(one_m_fb_fab, kb_sub(addr_b, fp_op_b));
    constraints[2] = kb_mul(one_m_fc_fcfp, kb_sub(addr_c, fp_op_c));

    constraints[3] = kb_mul(add_flag, kb_sub(nu_b, kb_add(nu_a, nu_c)));
    constraints[4] = kb_mul(mul_flag, kb_sub(nu_b, kb_mul(nu_a, nu_c)));

    constraints[5] = kb_mul(deref, kb_sub(addr_b, kb_add(val_a, op_b)));
    constraints[6] = kb_mul(deref, kb_sub(val_b, nu_c));

    constraints[7] = kb_mul(jump_cond, nu_a_m_one);
    constraints[8] = kb_mul(jump_cond, kb_sub(next_pc, nu_b));
    constraints[9] = kb_mul(jump_cond, kb_sub(next_fp, nu_c));
    constraints[10] = kb_mul(not_jump_cond, kb_sub(next_pc, pc_plus_one));
    constraints[11] = kb_mul(not_jump_cond, kb_sub(next_fp, fp));

    // Constraint 12 slot: NOT USED — the bus constraint is at index 0 in the
    // CPU (before the 12 assert_zero constraints). It's ext-field valued and
    // computed separately in the kernel using bus_data_out below.
    constraints[12] = 0;

    // Output bus data for the bus constraint computation.
    // bus_data[0] = is_precompile (flag)
    // bus_data[1] = precompile_data = up[19]
    // bus_data[2] = nu_a, bus_data[3] = nu_b, bus_data[4] = nu_c
    if (bus_data) {
        bus_data[0] = kb_neg(kb_sub(kb_add(kb_add(add_flag, mul_flag), kb_add(deref, jump)), ONE));
        bus_data[1] = up[19]; // precompile_data
        bus_data[2] = nu_a;
        bus_data[3] = nu_b;
        bus_data[4] = nu_c;
    }
}
