"""
DAL recursive aggregation — minimal version for debugging.
Just verify one child proof and return.
"""
from recursion import *
from hashing import *
from utils import *
from fiat_shamir import *

MAX_CHILDREN = MAX_CHILDREN_PLACEHOLDER
NUM_SYNDROME_CHECKS = 4
CHILD_DATA_SIZE = 24
CHILD_DATA_CHUNKS = 3

BYTECODE_CLAIM_OFFSET = BYTECODE_CLAIM_OFFSET_PLACEHOLDER
BYTECODE_HASH_DOMSEP_OFFSET = BYTECODE_HASH_DOMSEP_OFFSET_PLACEHOLDER
INPUT_DATA_SIZE_PADDED = INPUT_DATA_SIZE_PADDED_PLACEHOLDER
INPUT_DATA_NUM_CHUNKS = INPUT_DATA_SIZE_PADDED / DIGEST_LEN
BYTECODE_SUMCHECK_PROOF_SIZE = BYTECODE_SUMCHECK_PROOF_SIZE_PLACEHOLDER


def main():
    pub_mem = 0
    build_preamble_memory()

    # Load our input data
    data_buf = Array(INPUT_DATA_SIZE_PADDED)
    hint_witness("input_data", data_buf)
    bytecode_claim_output = data_buf + BYTECODE_CLAIM_OFFSET
    bytecode_hash_domsep = data_buf + BYTECODE_HASH_DOMSEP_OFFSET

    # Meta
    meta = Array(1)
    hint_witness("meta", meta)

    # Process first child only
    child_data = Array(CHILD_DATA_SIZE)
    hint_witness("child_pi", child_data)

    # Hash child_data → child PI hash
    ch0 = Array(DIGEST_LEN)
    poseidon16_compress(ZERO_VEC_PTR, child_data, ch0)
    ch1 = Array(DIGEST_LEN)
    poseidon16_compress(ch0, child_data + DIGEST_LEN, ch1)
    child_pi_hash = Array(DIGEST_LEN)
    poseidon16_compress(ch1, child_data + 2 * DIGEST_LEN, child_pi_hash)

    # Inner bytecode claim
    inner_claim = Array(BYTECODE_CLAIM_SIZE_PADDED)
    hint_witness("inner_bytecode_claim", inner_claim)

    # Verify child proof
    recursion_claim = recursion(child_pi_hash, bytecode_hash_domsep)

    # Bytecode reduction for 2 claims (inner + recursion)
    claim_ptrs = Array(2)
    claim_ptrs[0] = inner_claim
    claim_ptrs[1] = recursion_claim
    reduce_bytecode_claims(claim_ptrs, 2, bytecode_claim_output)

    # Hash input data and assert matches public memory
    outer_hash = slice_hash_with_iv(data_buf, INPUT_DATA_NUM_CHUNKS)
    copy_8(outer_hash, pub_mem)
    return


def reduce_bytecode_claims(bytecode_claims, n_bytecode_claims, bytecode_claim_output):
    bytecode_claims_hash: Mut = ZERO_VEC_PTR
    for i in range(0, n_bytecode_claims):
        claim_ptr = bytecode_claims[i]
        for k in unroll(BYTECODE_CLAIM_SIZE, BYTECODE_CLAIM_SIZE_PADDED):
            assert claim_ptr[k] == 0
        claim_hash = slice_hash(claim_ptr, BYTECODE_CLAIM_SIZE_PADDED / DIGEST_LEN)
        new_hash = Array(DIGEST_LEN)
        poseidon16_compress(bytecode_claims_hash, claim_hash, new_hash)
        bytecode_claims_hash = new_hash

    bytecode_sumcheck_proof = Array(BYTECODE_SUMCHECK_PROOF_SIZE)
    hint_witness("bytecode_sumcheck_proof", bytecode_sumcheck_proof)
    reduction_fs: Mut = fs_new(bytecode_sumcheck_proof)
    reduction_fs, received_claims_hash = fs_receive_chunks(reduction_fs, 1)
    copy_8(bytecode_claims_hash, received_claims_hash)

    reduction_fs, alpha = fs_sample_ef(reduction_fs)
    alpha_powers = powers(alpha, n_bytecode_claims)

    all_values = Array(n_bytecode_claims * DIM)
    for i in range(0, n_bytecode_claims):
        claim_ptr = bytecode_claims[i]
        copy_5(claim_ptr + BYTECODE_POINT_N_VARS * DIM, all_values + i * DIM)

    claimed_sum = Array(DIM)
    dot_product_ee_dynamic(all_values, alpha_powers, claimed_sum, n_bytecode_claims)

    reduction_fs, challenges, final_eval = sumcheck_verify(reduction_fs, BYTECODE_POINT_N_VARS, claimed_sum, 2)

    eq_evals = Array(n_bytecode_claims * DIM)
    for i in range(0, n_bytecode_claims):
        claim_ptr = bytecode_claims[i]
        poly_eq_ee(claim_ptr, challenges, eq_evals + i * DIM, BYTECODE_POINT_N_VARS)
    w_r = Array(DIM)
    dot_product_ee_dynamic(eq_evals, alpha_powers, w_r, n_bytecode_claims)

    bytecode_value_at_r = div_extension_ret(final_eval, w_r)

    copy_many_ef(challenges, bytecode_claim_output, BYTECODE_POINT_N_VARS)
    copy_5(bytecode_value_at_r, bytecode_claim_output + BYTECODE_POINT_N_VARS * DIM)
    return
