"""
DAL recursive aggregation — minimal: verify one child proof, skip bytecode claim reduction.
FOR BENCHMARKING ONLY — not cryptographically complete.
"""
from recursion import *
from hashing import *
from utils import *

MAX_CHILDREN = MAX_CHILDREN_PLACEHOLDER
CHILD_DATA_SIZE = 24

BYTECODE_CLAIM_OFFSET = BYTECODE_CLAIM_OFFSET_PLACEHOLDER
BYTECODE_HASH_DOMSEP_OFFSET = BYTECODE_HASH_DOMSEP_OFFSET_PLACEHOLDER
INPUT_DATA_SIZE_PADDED = INPUT_DATA_SIZE_PADDED_PLACEHOLDER
INPUT_DATA_NUM_CHUNKS = INPUT_DATA_SIZE_PADDED / DIGEST_LEN


def main():
    pub_mem = 0
    build_preamble_memory()

    data_buf = Array(INPUT_DATA_SIZE_PADDED)
    hint_witness("input_data", data_buf)
    bytecode_hash_domsep = data_buf + BYTECODE_HASH_DOMSEP_OFFSET

    # Load + hash child PI, verify proof
    child_data = Array(CHILD_DATA_SIZE)
    hint_witness("child_pi", child_data)

    ch0 = Array(DIGEST_LEN)
    poseidon16_compress(ZERO_VEC_PTR, child_data, ch0)
    ch1 = Array(DIGEST_LEN)
    poseidon16_compress(ch0, child_data + DIGEST_LEN, ch1)
    child_pi_hash = Array(DIGEST_LEN)
    poseidon16_compress(ch1, child_data + 2 * DIGEST_LEN, child_pi_hash)

    inner_claim = Array(BYTECODE_CLAIM_SIZE_PADDED)
    hint_witness("inner_bytecode_claim", inner_claim)

    # VERIFY CHILD PROOF IN-CIRCUIT
    _recursion_claim = recursion(child_pi_hash, bytecode_hash_domsep)

    # Skip bytecode claim reduction — bytecode claim in data_buf was pre-filled by hint

    # Hash input data → public memory
    outer_hash = slice_hash_with_iv(data_buf, INPUT_DATA_NUM_CHUNKS)
    copy_8(outer_hash, pub_mem)
    return
