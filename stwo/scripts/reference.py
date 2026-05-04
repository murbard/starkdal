#!/usr/bin/env python3
"""Reference implementation for Cairo coset-FFT + Blake2s Merkle commitment.

Field: Stark252 (p = 2^251 + 17*2^192 + 1).
Algorithm:
  1. Take 2^LOG_N coefficients c[0..N).
  2. Twist by powers of coset offset g: c'[i] = c[i] * g^i.
  3. NTT of size N_eval = 2^(LOG_N + LOG_BLOWUP) over Stark252, primitive
     N_eval-th root of unity OMEGA. Pads input with zeros to length N_eval.
     Output: evaluations P(g * OMEGA^k) for k = 0..N_eval.
  4. Build Blake2s Merkle tree over the evaluations. Leaves are the 32-byte
     little-endian byte representation of each felt; internal nodes are
     blake2s(left || right). Tree size = N_eval (must be power of 2).
  5. Print root.

Endianness: felts serialize as 32-byte little-endian. This matches packing into
the Cairo blake2s primitive (which reads each 4-byte chunk as little-endian u32).
"""
import hashlib
import sys
import json

# Stark252 prime
P = (1 << 251) + 17 * (1 << 192) + 1
GENERATOR = 3  # 3 is a generator of (Z/P)*

# Coset offset.
COSET_OFFSET = 3


def primitive_root_of_unity(log_order: int) -> int:
    """Return a primitive 2^log_order-th root of unity in Stark252."""
    assert 0 <= log_order <= 192
    return pow(GENERATOR, (P - 1) // (1 << log_order), P)


def bit_reverse(x: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def ntt(a: list, omega: int) -> list:
    """Iterative Cooley-Tukey NTT (decimation in time) over Stark252.
    Input length must be a power of 2; omega must be a primitive len(a)-th root of unity.
    Output[k] = sum_i a[i] * omega^(i*k) mod P.
    """
    n = len(a)
    log_n = n.bit_length() - 1
    assert 1 << log_n == n
    a = list(a)
    # Bit-reverse permutation
    for i in range(n):
        j = bit_reverse(i, log_n)
        if i < j:
            a[i], a[j] = a[j], a[i]
    # Butterflies
    m = 2
    while m <= n:
        half = m >> 1
        omega_m = pow(omega, n // m, P)
        for k in range(0, n, m):
            w = 1
            for j in range(half):
                t = (w * a[k + j + half]) % P
                u = a[k + j]
                a[k + j] = (u + t) % P
                a[k + j + half] = (u - t) % P
                w = (w * omega_m) % P
        m <<= 1
    return a


def coset_fft(coeffs: list, log_n: int, log_blowup: int) -> list:
    """Evaluate polynomial with given coeffs at coset COSET_OFFSET * <omega>
    of size 2^(log_n + log_blowup).
    """
    n_eval = 1 << (log_n + log_blowup)
    omega = primitive_root_of_unity(log_n + log_blowup)
    twisted = [0] * n_eval
    g_pow = 1
    for i, c in enumerate(coeffs):
        twisted[i] = (c * g_pow) % P
        g_pow = (g_pow * COSET_OFFSET) % P
    return ntt(twisted, omega)


def felt_to_le_bytes(x: int) -> bytes:
    assert 0 <= x < P
    return x.to_bytes(32, "little")


def blake2s_chunk(felts: list) -> bytes:
    """Hash a chunk of f felts as concatenated 32-byte LE encodings.
    The chunk size = number of evaluations packed per Merkle leaf.
    Cairo will reproduce this by feeding 2 felts (16 u32s = 64 bytes) per blake2s
    block, calling blake2s_compress for non-final blocks and blake2s_finalize on the
    last. For len(felts) odd, the last block contains one felt + 32 zero-bytes.
    """
    assert len(felts) > 0
    payload = b"".join(felt_to_le_bytes(x) for x in felts)
    return hashlib.blake2s(payload).digest()


def blake2s_pair(left: bytes, right: bytes) -> bytes:
    assert len(left) == 32 and len(right) == 32
    return hashlib.blake2s(left + right).digest()


def merkle_root(evals: list, log_felts_per_leaf: int) -> bytes:
    """Build binary Blake2s Merkle tree over evals chunked into leaves of size
    2^log_felts_per_leaf. Return the root."""
    f = 1 << log_felts_per_leaf
    n_total = len(evals)
    assert n_total % f == 0, "evals length must be divisible by leaf size"
    n_leaves = n_total // f
    assert n_leaves > 0 and (n_leaves & (n_leaves - 1)) == 0, "n_leaves must be power of 2"
    layer = [blake2s_chunk(evals[i * f : (i + 1) * f]) for i in range(n_leaves)]
    while len(layer) > 1:
        layer = [blake2s_pair(layer[2 * i], layer[2 * i + 1]) for i in range(len(layer) // 2)]
    return layer[0]


def le_bytes_to_felt(b: bytes) -> int:
    """Inverse of felt_to_le_bytes — convert 32 LE bytes to a felt."""
    assert len(b) == 32
    return int.from_bytes(b, "little")


def root_as_words(root: bytes) -> list:
    """Convert 32-byte digest to 8 little-endian u32 words.
    This matches what Cairo's blake2s_finalize returns directly."""
    assert len(root) == 32
    return [int.from_bytes(root[4 * i : 4 * (i + 1)], "little") for i in range(8)]


def commit(coeffs: list, log_n: int, log_blowup: int, log_felts_per_leaf: int) -> dict:
    assert len(coeffs) == 1 << log_n
    evals = coset_fft(coeffs, log_n, log_blowup)
    root = merkle_root(evals, log_felts_per_leaf)
    n_leaves = len(evals) >> log_felts_per_leaf
    return {
        "log_n": log_n,
        "log_blowup": log_blowup,
        "log_felts_per_leaf": log_felts_per_leaf,
        "n_eval": len(evals),
        "n_leaves": n_leaves,
        "tree_depth": (n_leaves.bit_length() - 1),
        "first_eval": evals[0],
        "last_eval": evals[-1],
        "root_hex": root.hex(),
        "root_words": root_as_words(root),
    }


def deterministic_coeffs(log_n: int, seed: int = 0) -> list:
    """Deterministic test vector: c[i] = (seed + i + 1) mod P."""
    n = 1 << log_n
    return [(seed + i + 1) % P for i in range(n)]


def pick_log_felts_per_leaf(log_n_total: int) -> int:
    """Smallest k such that 2^k + k >= log_n_total (leaf size >= proof size)."""
    k = 0
    while (1 << k) + k < log_n_total:
        k += 1
    return k


def main():
    log_n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    log_blowup = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if len(sys.argv) > 4:
        log_felts_per_leaf = int(sys.argv[4])
    else:
        log_felts_per_leaf = pick_log_felts_per_leaf(log_n + log_blowup)
    coeffs = deterministic_coeffs(log_n, seed)
    result = commit(coeffs, log_n, log_blowup, log_felts_per_leaf)
    # Also expose constants useful for Cairo embedding.
    result["P"] = P
    result["coset_offset"] = COSET_OFFSET
    result["omega"] = primitive_root_of_unity(log_n + log_blowup)
    result["seed"] = seed
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
