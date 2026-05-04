# starkdal

Experimental prototype for a "dumb" Data Availability Layer (DAL) commitment scheme. The idea: prove the full polynomial evaluation + Merkle commitment inside a STARK, so every leaf under the root is *exactly* correct — not just δ-close as with FRI/FRIDA-style approaches. This matters for DAS systems where nodes must rebroadcast leaves with valid Merkle proofs.

**This is a playground for benchmarking, not production code.**

## What it does

1. Start with 2^n coefficients over a prime field (the data to commit)
2. Coset FFT to 2^(n+k) evaluations (Reed-Solomon encoding with blowup factor 2^k)
3. Build a Merkle tree over chunks of evaluations (leaf size chosen so leaf ≥ proof path)
4. Prove the whole thing in a zkVM — the proof attests that the committed root is the *exact* RS encoding of the original data

## Implementations

### `leanvm/` — primary

Built on [leanEthereum/leanMultisig](https://github.com/leanEthereum/leanMultisig)'s minimal zkVM (WHIR + SuperSpartan, KoalaBear field, native Poseidon16 precompile).

```bash
cd leanvm
cargo run --release --example fft_merkle_bench -- --log-n 8
cargo test --example fft_merkle_bench --release  # 23 tests
```

### `stwo/` — reference

Cairo 2.18 circuit proven with Stwo (Circle STARKs, Stark252 field, Blake2s hashing).

```bash
cd stwo
scarb cairo-test          # 11 tests
scarb execute --arguments-file <args.json>
python3 scripts/bench.py  # full benchmark with proving
```

## Benchmarks

Coset FFT + Merkle commitment, k=1 blowup:

| n | payload | leanVM prove (Graviton3) | leanVM proof | Stwo prove | Stwo proof |
|---|---------|--------------------------|--------------|------------|------------|
| 8 | 1 KB | 0.08s | 184 KB | 32s | 1.09 MB |
| 10 | 4 KB | 0.17s | 251 KB | 41s | 1.09 MB |
| 12 | 16 KB | 0.80s | 280 KB | 84s | 1.18 MB |
| 14 | 64 KB | 1.0s | 313 KB | 270s | 1.27 MB |
| 16 | 256 KB | 3.8s | 362 KB | — | — |
| 18 | 1 MB | 14s | 415 KB | — | — |

leanVM is ~100x faster and produces ~4x smaller proofs, primarily due to the native Poseidon16 precompile (one AIR row per hash vs hundreds for Blake2s).

## Why "the dumb way"

FRIDA and similar FRI-based commitment schemes give you *proximity* to a low-degree polynomial, not an *exact* degree bound. In a DAS network, nodes that hold samples need to gossip them with Merkle proofs. If the commitment is over a proximate-but-not-exact codeword, a node that error-decodes to recover the "correct" values can't authenticate them — the corrected leaves don't match the root. By proving the FFT explicitly, every leaf under the root is by construction a true evaluation, and any node can rebroadcast with a valid proof.
