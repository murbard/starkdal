"""
DAL recursive aggregation: verifies exactly K child proofs, checks syndrome sums = 0.

Fixed arity — always processes MAX_CHILDREN children. No runtime branching.

Each child's public input: hash(leaf_data) where leaf_data is:
  [subtree_root(8), betas(4), partial_sums(4), x_start, w_start, sign_start, padding]
  padded to 24 FE (3 Poseidon chunks).

This circuit's public input: hash(input_data) = 8 FE.
"""
from recursion import *
from hashing import *
from utils import *
from fiat_shamir import *

MAX_CHILDREN = MAX_CHILDREN_PLACEHOLDER
NUM_SYNDROME_CHECKS = 4
CHILD_DATA_SIZE = 24  # padded size of each child's leaf_data
CHILD_DATA_CHUNKS = 3  # 24/8

BYTECODE_CLAIM_OFFSET = BYTECODE_CLAIM_OFFSET_PLACEHOLDER
BYTECODE_HASH_DOMSEP_OFFSET = BYTECODE_HASH_DOMSEP_OFFSET_PLACEHOLDER
INPUT_DATA_SIZE_PADDED = INPUT_DATA_SIZE_PADDED_PLACEHOLDER
INPUT_DATA_NUM_CHUNKS = INPUT_DATA_SIZE_PADDED / DIGEST_LEN
BYTECODE_SUMCHECK_PROOF_SIZE = BYTECODE_SUMCHECK_PROOF_SIZE_PLACEHOLDER


def main():
    pub_mem = 0
    build_preamble_memory()

    # Load our input data (hinted, then hashed to match public input)
    data_buf = Array(INPUT_DATA_SIZE_PADDED)
    hint_witness("input_data", data_buf)

    full_root = data_buf
    betas = data_buf + DIGEST_LEN

    bytecode_claim_output = data_buf + BYTECODE_CLAIM_OFFSET
    bytecode_hash_domsep = data_buf + BYTECODE_HASH_DOMSEP_OFFSET

    # Accumulators for syndrome partial sums
    sum_0: Mut = 0
    sum_1: Mut = 0
    sum_2: Mut = 0
    sum_3: Mut = 0

    # Storage for subtree roots (for Merkle composition)
    subtree_roots = Array(MAX_CHILDREN * DIGEST_LEN)

    # Bytecode claims: each child produces two claims (from input_data + from recursion)
    # Store them as an array of pointers
    all_claims = Array(MAX_CHILDREN * 2 * BYTECODE_CLAIM_SIZE_PADDED)
    claim_ptrs = Array(MAX_CHILDREN * 2)

    for child_idx in unroll(0, MAX_CHILDREN):
        # Load child's leaf_data (24 FE)
        child_data = Array(CHILD_DATA_SIZE)
        hint_witness("child_pi", child_data)

        # Hash child_data using Poseidon sponge (3 chunks)
        ch0 = Array(DIGEST_LEN)
        poseidon16_compress(ZERO_VEC_PTR, child_data, ch0)
        ch1 = Array(DIGEST_LEN)
        poseidon16_compress(ch0, child_data + DIGEST_LEN, ch1)
        child_pi_hash = Array(DIGEST_LEN)
        poseidon16_compress(ch1, child_data + 2 * DIGEST_LEN, child_pi_hash)

        # Load inner bytecode claim
        inner_claim_ptr = all_claims + child_idx * 2 * BYTECODE_CLAIM_SIZE_PADDED
        hint_witness("inner_bytecode_claim", inner_claim_ptr)
        claim_ptrs[child_idx * 2] = inner_claim_ptr

        # Verify child proof in-circuit
        recursion_claim = recursion(child_pi_hash, bytecode_hash_domsep)
        outer_claim_ptr = all_claims + child_idx * 2 * BYTECODE_CLAIM_SIZE_PADDED + BYTECODE_CLAIM_SIZE_PADDED
        for k in unroll(0, BYTECODE_CLAIM_SIZE_PADDED):
            outer_claim_ptr[k] = recursion_claim[k]
        claim_ptrs[child_idx * 2 + 1] = outer_claim_ptr

        # Accumulate partial sums (at child_data offsets 12..16)
        sum_0 = sum_0 + child_data[12]
        sum_1 = sum_1 + child_data[13]
        sum_2 = sum_2 + child_data[14]
        sum_3 = sum_3 + child_data[15]

        # Store subtree root (at child_data offsets 0..8)
        for k in unroll(0, DIGEST_LEN):
            subtree_roots[child_idx * DIGEST_LEN + k] = child_data[k]

    # Assert partial sums add to zero
    assert sum_0 == 0
    assert sum_1 == 0
    assert sum_2 == 0
    assert sum_3 == 0

    # Verify subtree roots compose to full_root via binary hashing
    # Hash pairs: roots[0]+roots[1] → h[0], roots[2]+roots[3] → h[1], etc.
    # Then hash h[0]+h[1] → hh[0], etc. until one root.
    composed = Array(MAX_CHILDREN * DIGEST_LEN)
    for k in unroll(0, MAX_CHILDREN * DIGEST_LEN):
        composed[k] = subtree_roots[k]
    n_nodes: Mut = MAX_CHILDREN
    src_layer: Mut = composed
    for level in unroll(0, log2_ceil(MAX_CHILDREN)):
        dst_layer = Array(MAX_CHILDREN * DIGEST_LEN / 2)
        for pair in unroll(0, MAX_CHILDREN / 2):
            poseidon16_compress(
                src_layer + 2 * pair * DIGEST_LEN,
                src_layer + (2 * pair + 1) * DIGEST_LEN,
                dst_layer + pair * DIGEST_LEN
            )
        src_layer = dst_layer
    # src_layer now points to the single root
    for k in unroll(0, DIGEST_LEN):
        assert src_layer[k] == full_root[k]

    # Reduce bytecode claims
    n_claims = MAX_CHILDREN * 2
    reduce_bytecode_claims(claim_ptrs, n_claims, bytecode_claim_output)

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
