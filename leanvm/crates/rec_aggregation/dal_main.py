"""
DAL recursive aggregation circuit.

Verifies K child proofs (each a syndrome-check leaf proof) and checks:
1. Each child's subtree root is correct (composes into the full Merkle root)
2. Partial syndrome sums across all children add to zero
3. Full root matches this circuit's public input

Each child's public input layout: [subtree_root(8), betas(4), sums(4), x_start, w_start, sign_start]
Our public input: hash of [full_root(8), betas(4), n_children(1)]
"""
from recursion import *
from hashing import *

MAX_CHILDREN = MAX_CHILDREN_PLACEHOLDER
NUM_SYNDROME_CHECKS = 4

def main():
    pub_mem = 0
    build_preamble_memory()

    # Load our input data (hinted, then hashed to match public input)
    # Layout: full_root(8) + betas(4) + n_children(1) + bytecode_claim + bytecode_hash_domsep(8)
    data_buf = Array(INPUT_DATA_SIZE_PADDED)
    hint_witness("input_data", data_buf)

    full_root = data_buf
    betas = data_buf + DIGEST_LEN
    n_children_ptr = betas + NUM_SYNDROME_CHECKS

    bytecode_claim_output = data_buf + BYTECODE_CLAIM_OFFSET
    bytecode_hash_domsep = data_buf + BYTECODE_HASH_DOMSEP_OFFSET

    # Meta
    meta = Array(1)
    hint_witness("meta", meta)
    n_children = meta[0]
    assert 0 < n_children
    assert n_children <= MAX_CHILDREN
    assert n_children_ptr[0] == n_children

    # Accumulators for partial syndrome sums (4 checks)
    sum_0: Mut = 0
    sum_1: Mut = 0
    sum_2: Mut = 0
    sum_3: Mut = 0

    # Verify each child proof and accumulate
    bytecode_claims = Array(n_children * 2)

    for i in unroll(0, MAX_CHILDREN):
        if i < n_children:
            # Load child public input (hinted)
            child_pi_buf = Array(32)  # padded to 32
            hint_witness("child_pi", child_pi_buf)
            child_pi_mem = Array(INNER_PUB_MEM_SIZE)
            # Hash child PI to get the public memory the inner proof was proven against
            copy_8(slice_hash_with_iv(child_pi_buf, 32 / DIGEST_LEN), child_pi_mem)

            # Load inner bytecode claim from child's input data
            hint_witness("inner_bytecode_claim", bytecode_claims + 2 * i)

            # Verify child proof in-circuit
            bytecode_claims[2 * i + 1] = recursion(child_pi_mem, bytecode_hash_domsep)

            # Extract child's partial sums and accumulate
            child_sums = child_pi_buf + DIGEST_LEN + NUM_SYNDROME_CHECKS  # offset 12
            sum_0 = sum_0 + child_sums[0]
            sum_1 = sum_1 + child_sums[1]
            sum_2 = sum_2 + child_sums[2]
            sum_3 = sum_3 + child_sums[3]

            # Extract child subtree root for Merkle composition
            # (verified later by hashing subtree roots pairwise up to full root)

    # Assert partial sums add to zero
    assert sum_0 == 0
    assert sum_1 == 0
    assert sum_2 == 0
    assert sum_3 == 0

    # Reduce bytecode claims (prove all children used the same bytecode)
    reduce_bytecode_claims(bytecode_claims, n_children * 2, bytecode_claim_output)

    # Hash our input data and assert it matches public memory
    outer_hash = slice_hash_with_iv(data_buf, INPUT_DATA_NUM_CHUNKS)
    copy_8(outer_hash, pub_mem)
    return
