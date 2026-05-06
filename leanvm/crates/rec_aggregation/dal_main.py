"""
DAL recursive aggregation: verifies K child proofs and checks syndrome sums = 0.

Each child proof is a syndrome-check leaf proof with public input:
  [subtree_root(8), betas(4), partial_sums(4), x_start, w_start, sign_start]
  padded to 32 FE.

This circuit's public input (hashed to a single digest):
  [full_root(8), betas(4), n_children(1), bytecode_claim, bytecode_hash_domsep(8)]
"""
from recursion import *
from hashing import *

MAX_CHILDREN = MAX_CHILDREN_PLACEHOLDER
NUM_SYNDROME_CHECKS = 4
CHILD_PI_SIZE = 32  # padded size of each child's public input

BYTECODE_CLAIM_OFFSET = BYTECODE_CLAIM_OFFSET_PLACEHOLDER
BYTECODE_HASH_DOMSEP_OFFSET = BYTECODE_HASH_DOMSEP_OFFSET_PLACEHOLDER
INPUT_DATA_SIZE_PADDED = INPUT_DATA_SIZE_PADDED_PLACEHOLDER
INPUT_DATA_NUM_CHUNKS = INPUT_DATA_SIZE_PADDED / DIGEST_LEN


def main():
    pub_mem = 0
    build_preamble_memory()

    # Load our input data (hinted, then hashed to match public input)
    data_buf = Array(INPUT_DATA_SIZE_PADDED)
    hint_witness("input_data", data_buf)

    full_root = data_buf
    betas = data_buf + DIGEST_LEN
    n_children_field = data_buf + DIGEST_LEN + NUM_SYNDROME_CHECKS

    bytecode_claim_output = data_buf + BYTECODE_CLAIM_OFFSET
    bytecode_hash_domsep = data_buf + BYTECODE_HASH_DOMSEP_OFFSET

    # Meta
    meta = Array(1)
    hint_witness("meta", meta)
    n_children = meta[0]
    assert n_children == n_children_field[0]
    assert 0 < n_children
    assert n_children <= MAX_CHILDREN

    # Accumulators for syndrome partial sums
    sum_0: Mut = 0
    sum_1: Mut = 0
    sum_2: Mut = 0
    sum_3: Mut = 0

    # Subtree roots for Merkle composition
    subtree_roots = Array(MAX_CHILDREN * DIGEST_LEN)

    # Bytecode claims (2 per child: the one from input_data + the one from recursion)
    bytecode_claims_buf = Array(MAX_CHILDREN * 2 * BYTECODE_CLAIM_SIZE_PADDED)

    for child_idx in unroll(0, MAX_CHILDREN):
        if child_idx < n_children:
            # Load child's public input (19 FE padded to 32)
            child_pi = Array(CHILD_PI_SIZE)
            hint_witness("child_pi", child_pi)

            # Hash child PI to get the public memory digest the inner proof was verified against
            child_pi_hash = Array(DIGEST_LEN)
            poseidon16_compress(ZERO_VEC_PTR, child_pi, child_pi_hash)
            for chunk in unroll(1, CHILD_PI_SIZE / DIGEST_LEN - 1):
                poseidon16_compress(child_pi_hash, child_pi + (chunk + 1) * DIGEST_LEN, child_pi_hash)

            child_pub_mem = Array(PUB_INPUT_SIZE)
            copy_8(child_pi_hash, child_pub_mem)

            # Load child's bytecode claim from its input data
            inner_claim = bytecode_claims_buf + child_idx * 2 * BYTECODE_CLAIM_SIZE_PADDED
            hint_witness("inner_bytecode_claim", inner_claim)

            # Verify child proof in-circuit — returns the bytecode evaluation claim
            recursion_claim = recursion(child_pub_mem, bytecode_hash_domsep)
            # Copy recursion claim to our buffer
            for k in unroll(0, BYTECODE_CLAIM_SIZE_PADDED):
                bytecode_claims_buf[child_idx * 2 * BYTECODE_CLAIM_SIZE_PADDED + BYTECODE_CLAIM_SIZE_PADDED + k] = recursion_claim[k]

            # Accumulate partial syndrome sums from child PI
            # Child PI layout: [root(8), betas(4), sums(4), ...]
            child_sums = child_pi + 12  # offset to partial sums
            sum_0 = sum_0 + child_sums[0]
            sum_1 = sum_1 + child_sums[1]
            sum_2 = sum_2 + child_sums[2]
            sum_3 = sum_3 + child_sums[3]

            # Store subtree root for composition check
            for k in unroll(0, DIGEST_LEN):
                subtree_roots[child_idx * DIGEST_LEN + k] = child_pi[k]

    # Assert partial sums add to zero
    assert sum_0 == 0
    assert sum_1 == 0
    assert sum_2 == 0
    assert sum_3 == 0

    # Verify subtree roots compose to full_root
    # Hash subtree roots pairwise up the tree
    # (for now: just hash all subtree roots sequentially and assert == full_root)
    computed_root = Array(DIGEST_LEN)
    poseidon16_compress(subtree_roots, subtree_roots + DIGEST_LEN, computed_root)
    for pair in unroll(1, MAX_CHILDREN / 2):
        if pair * 2 < n_children:
            poseidon16_compress(computed_root, subtree_roots + pair * 2 * DIGEST_LEN, computed_root)
            # This is a simplification — proper tree composition needed for production
    for k in unroll(0, DIGEST_LEN):
        assert computed_root[k] == full_root[k]

    # Reduce bytecode claims (batch-verify all children used the same bytecode)
    claims_ptr = Array(n_children * 2)
    for c in unroll(0, MAX_CHILDREN):
        if c < n_children:
            claims_ptr[c * 2] = bytecode_claims_buf + c * 2 * BYTECODE_CLAIM_SIZE_PADDED
            claims_ptr[c * 2 + 1] = bytecode_claims_buf + c * 2 * BYTECODE_CLAIM_SIZE_PADDED + BYTECODE_CLAIM_SIZE_PADDED

    if n_children > 0:
        reduce_bytecode_claims(claims_ptr, n_children * 2, bytecode_claim_output)

    # Hash input data and assert matches public memory
    outer_hash = slice_hash_with_iv(data_buf, INPUT_DATA_NUM_CHUNKS)
    copy_8(outer_hash, pub_mem)
    return
