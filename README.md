# starkdal

Experimental prototype for a "dumb" Data Availability Layer (DAL) commitment scheme. The idea: prove the full polynomial evaluation + Merkle commitment inside a STARK, so every leaf under the root is *exactly* correct — not just delta-close as with FRI/FRIDA-style approaches. This matters for DAS systems where nodes must rebroadcast leaves with valid Merkle proofs.

**This is a playground for benchmarking, not production code.**

## What it does

1. Start with polynomials over a prime field (the data to commit)
2. RS-encode: coset FFT to evaluation domain (rate 1/2)
3. Commit: Poseidon Merkle tree over the evaluations
4. Prove inside a zkVM: the proof attests the committed root is the *exact* RS encoding

## Approaches

Three circuit strategies, each trading off complexity for throughput:

### 1. FFT + Merkle (baseline)

Prove the coset FFT and Merkle tree construction directly in-circuit. Simple but expensive: O(n log n) multiplications, unrolled into millions of program lines for large n.

```bash
cd leanvm
cargo run --release --example bench_fft_unroll -- --log-n 12
```

### 2. Syndrome check

Instead of proving the FFT, verify RS membership via syndrome checking: for random beta, check that the batched syndrome sum is zero. O(n) multiplications per check, 4 checks for 80-bit security. Supports chunked parallel proving with shared bytecode.

```bash
cargo run --release --example bench_syndrome_unroll -- --log-n 12
cargo run --release --example bench_chunked -- --log-n 22 --log-chunk 12 --concurrency 8
```

### 3. RLC + fold (fastest)

Batch m codewords via Random Linear Combination using extension-field precompiles (`dot_product_be`), then fold with random challenges (`dot_product_ee` + `add_ee`). Inspired by the leanDAS/STARS paper's approach of mapping bulk computation to precompile calls.

```bash
# Single proof (m=510 saturates 2^21 ext-op table)
cargo run --release --example bench_fri_fold -- --log-poly 11 --m 510

# Parallel batches
cargo run --release --example bench_fri_fold -- --log-poly 11 --m 510 --batches 4 --concurrency 4

# Larger proofs (ext-op table bumped to 2^23)
cargo run --release --example bench_fri_fold -- --log-poly 11 --m 2046
```

## Recursive aggregation

The syndrome approach includes a full recursive pipeline: prove leaf chunks in parallel, then aggregate via in-circuit WHIR proof verification (`recursion.py`).

```bash
cargo run --release --example bench_recursive -- --log-n 22 --arity 8
```

## Benchmarks

All results on AWS Graviton4 (c8g.8xlarge, 32 cores, 64 GB RAM).

### Single-proof throughput

| Approach | Config | Prove | Data | Throughput |
|---|---|---|---|---|
| Syndrome (unrolled) | n=12 | 0.64s | 16 KB | 25 KB/s |
| RLC+fold | m=510, n=4096 | 3.7s | 4.0 MB | 1,037 KB/s |
| RLC+fold | m=1022, n=4096 | 7.5s | 8.0 MB | 1,097 KB/s |
| RLC+fold | m=2046, n=4096 | 15.0s | 16.0 MB | 1,088 KB/s |

### Parallel proving (RLC+fold, m=510)

| Concurrency | Batches | Data | Prove wall | Throughput |
|---|---|---|---|---|
| 1 | 4 | 16 MB | 14.9s | 1,094 KB/s |
| 2 | 4 | 16 MB | 12.7s | 1,284 KB/s |
| 4 | 4 | 16 MB | 11.6s | 1,393 KB/s |
| 4 | 8 | 33 MB | 23.2s | 1,407 KB/s |

### 16 MB payload strategies

| Strategy | Proofs | Prove wall | Agg proofs needed |
|---|---|---|---|
| 1 x m=2046 | 1 | 15.0s | 0 |
| 2 x m=1022 c=2 | 2 | 12.9s | 1 |
| 4 x m=510 c=4 | 4 | 11.6s | ~3 |

### Historical: syndrome recursive pipeline (16 MB)

| Phase | Time |
|---|---|
| Leaf proving (1024 proofs, c=8) | 257s |
| Aggregation (1024 proofs, c=8) | 167s |
| **Total** | **425s (38 KB/s)** |

The RLC+fold approach is **37x faster** than the syndrome recursive pipeline.

## Implementation

Built on [leanEthereum/leanMultisig](https://github.com/leanEthereum/leanMultisig)'s minimal zkVM:
- WHIR + SuperSpartan proving system
- KoalaBear field (p = 2^31 - 2^24 + 1), quintic extension (|E| ~ 2^155)
- Native Poseidon16 precompile
- Extension-field precompiles: `dot_product_be`, `dot_product_ee`, `add_ee`

### Key files

| File | Description |
|---|---|
| `examples/bench_fri_fold.rs` | RLC+fold benchmark (parallel batching) |
| `examples/bench_syndrome_unroll.rs` | Unrolled syndrome check |
| `examples/bench_recursive.rs` | Full recursive pipeline |
| `examples/bench_chunked.rs` | Chunked parallel syndrome |
| `src/lib.rs` | Shared infrastructure (CircuitParams, reference computations, codegen helpers) |
| `crates/rec_aggregation/` | Recursive proof aggregation (recursion.py, dal_main.py) |

### Extension-op table limit

The ext-op table is capped (default 2^21, bumped to 2^23 in this repo). Data per proof ~ 2 x ext_op_rows bytes, so the cap directly determines max payload per proof. The commitment surface budget (2^30 total) permits up to 2^23.

## Why "the dumb way"

FRIDA and similar FRI-based commitment schemes give you *proximity* to a low-degree polynomial, not an *exact* degree bound. In a DAS network, nodes that hold samples need to gossip them with Merkle proofs. If the commitment is over a proximate-but-not-exact codeword, a node that error-decodes to recover the "correct" values can't authenticate them — the corrected leaves don't match the root. By proving the FFT explicitly, every leaf under the root is by construction a true evaluation, and any node can rebroadcast with a valid proof.
