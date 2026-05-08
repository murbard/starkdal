# starkdal

Experimental prototypes for Data Availability Layer (DAL) commitment schemes. Two fundamentally different approaches:

1. **STARK-based**: prove RS encoding inside a zkVM (exact correctness, compact proof)
2. **ZODA**: the encoding itself is the proof (zero overhead, high throughput)

**This is a playground for benchmarking, not production code.**

## ZODA (`zoda/`)

Implementation of the ZODA tensor variation (Appendix E of [Evans, Mohnblatt, Angeris 2025/034](https://eprint.iacr.org/2025/034)). The standard tensor code Z = G·X̃·G'^T is entirely over the base field — **4× expansion, zero encoding overhead**. Proof of correct encoding via two short vectors in the extension field.

Uses [concrete-ntt](https://crates.io/crates/concrete-ntt) for SIMD-optimized negacyclic NTTs over KoalaBear (p = 2^31 - 2^24 + 1). BLAKE3 for Merkle commitments.

```bash
cd zoda
cargo run --release -- --n 4096              # 64 MB encode
cargo run --release -- --n 4096 --decode     # + roundtrip decode verification
cargo run --release -- --n 8192              # 256 MB encode
```

### Benchmarks (Graviton4, c8g.8xlarge, 32 cores)

| Data | Encoded | Encode time | **Throughput** | NTT throughput |
|---|---|---|---|---|
| 16 MB | 64 MB | 0.023s | **701 MB/s** | 1.6 GB/s |
| 64 MB | 256 MB | 0.081s | **787 MB/s** | 1.8 GB/s |
| 256 MB | 1 GB | 0.334s | **768 MB/s** | 1.8 GB/s |
| 1 GB | 4 GB | 1.684s | **608 MB/s** | 1.6 GB/s |

### Why the tensor variation (Appendix E)?

The original ZODA field-extension variant (Section 5) stores Y over a quintic extension field, causing **12× data expansion** (3× worse than the standard 4×). The tensor variation keeps everything in the base field: Z = G·X̃·G'^T with 4× expansion, and the proof is just two short EF vectors (negligible network cost). See the [paper](https://eprint.iacr.org/2025/034) Section 5 vs Appendix E.

### Architecture

| Component | Implementation |
|---|---|
| RS encoding | concrete-ntt negacyclic NTT (SIMD: AVX2/NEON) |
| Merkle commitment | BLAKE3, raw Montgomery byte hashing |
| Fiat-Shamir | BLAKE3 sponge for random EF vectors |
| Field | KoalaBear (31-bit) + quintic extension for proof vectors |

Full protocol: encode, open (row/column Merkle proofs), verify (3 consistency checks), decode.

## STARK-based (`leanvm/`)

Built on [leanEthereum/leanMultisig](https://github.com/leanEthereum/leanMultisig)'s minimal zkVM (WHIR + SuperSpartan, KoalaBear field, Poseidon16 precompile).

Three circuit strategies:

### 1. FFT + Merkle (baseline)

```bash
cd leanvm
cargo run --release --example bench_fft_unroll -- --log-n 12
```

### 2. Syndrome check

```bash
cargo run --release --example bench_syndrome_unroll -- --log-n 12
cargo run --release --example bench_chunked -- --log-n 22 --log-chunk 12
```

### 3. RLC + fold (fastest STARK approach)

Batch m codewords via Random Linear Combination using extension-field precompiles, then fold with random challenges.

```bash
cargo run --release --example bench_fri_fold -- --log-poly 11 --m 510
cargo run --release --example bench_fri_fold -- --log-poly 11 --m 510 --batches 4 --concurrency 4
```

### STARK benchmarks (Graviton4, c8g.8xlarge, 32 cores)

| Approach | Config | Data | Prove | Throughput |
|---|---|---|---|---|
| Syndrome (recursive) | n=22, 1024 leaves | 16 MB | 425s | 38 KB/s |
| RLC+fold (single) | m=510 | 4 MB | 3.7s | 1,037 KB/s |
| RLC+fold (parallel c=4) | m=510, 4 batches | 16 MB | 11.6s | 1,393 KB/s |

### Comparison: STARK vs ZODA

| | STARK (RLC+fold) | ZODA (tensor) |
|---|---|---|
| 64 MB encode | 11.6s | **0.081s** (140× faster) |
| Throughput | 1.4 MB/s | **787 MB/s** |
| Proof size | ~400 KB | N/A (encoding IS proof) |
| Verifier downloads | 400 KB proof | ~3 MB (sampled rows+cols) |
| Data expansion | 4× (bumped ext-op limit) | 4× (standard tensor code) |
| Security model | STARK proof (succinct) | Sampling (probabilistic) |

ZODA is 140× faster but requires the verifier to download rows and columns of the encoding. The STARK approach produces a compact proof that any third party can verify without the full encoding.

## Why "the dumb way"

FRIDA and similar FRI-based commitment schemes give you *proximity* to a low-degree polynomial, not an *exact* degree bound. In a DAS network, nodes that hold samples need to gossip them with Merkle proofs. If the commitment is over a proximate-but-not-exact codeword, a node that error-decodes to recover the "correct" values can't authenticate them — the corrected leaves don't match the root. By proving the encoding explicitly, every leaf under the root is by construction a true evaluation, and any node can rebroadcast with a valid proof.
