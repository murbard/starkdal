# starkdal

Data Availability Layer (DAL) commitment schemes and GPU-accelerated STARK proving.

1. **GPU STARK Prover**: monolithic GPU implementation of the leanVM prover (1.6 MiB/s on A100)
2. **STARK-based circuits**: prove RS encoding inside a zkVM (exact correctness, compact proof)
3. **ZODA**: the encoding itself is the proof (zero overhead, high throughput)

## ZODA (`zoda/`)

Implementation of the ZODA tensor variation (Appendix E of [Evans, Mohnblatt, Angeris 2025/034](https://eprint.iacr.org/2025/034)). The standard tensor code Z = G·X̃·G'^T is entirely over the base field — **4× expansion, zero encoding overhead**. Proof of correct encoding via two short vectors in the extension field.

Uses [concrete-ntt](https://crates.io/crates/concrete-ntt) for SIMD-optimized negacyclic NTTs over KoalaBear (p = 2^31 - 2^24 + 1). BLAKE3 for Merkle commitments (domain-separated leaf/internal nodes).

```bash
cd zoda
cargo run --release -- --n 4096              # 64 MB encode
cargo run --release -- --n 4096 --decode     # + roundtrip decode verification
cargo run --release -- --n 8192 --samples 50 # 256 MB, 50 samples per dimension
```

### Benchmarks (Graviton4, c8g.8xlarge, 32 cores)

| Data | Encoded | Encode | **Throughput** | NTT | Verify |
|---|---|---|---|---|---|
| 16 MB | 64 MB | 0.030s | **526 MB/s** | 1.7 GB/s | 10ms |
| 64 MB | 256 MB | 0.097s | **658 MB/s** | 1.9 GB/s | 19ms |
| 256 MB | 1 GB | 0.360s | **711 MB/s** | 1.9 GB/s | 39ms |
| 1 GB | 4 GB | 1.635s | **627 MB/s** | 1.7 GB/s | 78ms |

### Why the tensor variation (Appendix E)?

The original ZODA field-extension variant (Section 5) stores Y over a quintic extension field, causing **12× data expansion** (3× worse than the standard 4×). The paper explicitly notes: *"this does not result in a zero-overhead protocol since the field extension's elements do not carry additional information that can reconstruct the corresponding rows."*

The tensor variation keeps everything in the base field: Z = G·X̃·G'^T with 4× expansion, and the proof is just two short EF vectors (negligible network cost).

### Architecture

| Component | Implementation |
|---|---|
| RS encoding | concrete-ntt negacyclic NTT (SIMD: AVX2/NEON) |
| Merkle commitment | BLAKE3, domain-separated (TAG_ROW/TAG_COL/TAG_INTERNAL) |
| Fiat-Shamir | BLAKE3 XOF seeded from both Merkle roots |
| Random sampling | BLAKE3 XOF with rejection sampling |
| Field | KoalaBear (31-bit) + quintic extension for proof vectors only |

Full protocol: encode, open (row/column Merkle proofs), verify (row check, column check, cross check), decode.

The verifier receives only the `ZodaCommitment` (two Merkle roots + two proof vectors) and re-derives random vectors via Fiat-Shamir — no trust in the prover beyond the commitment.

### Performance notes

The NTT runs at **1.9 GB/s** (concrete-ntt with NEON). The encode throughput is lower because of:
- Column gather: reading columns from row-major intermediate results (strided memory access)
- Proof vectors: additional column NTTs + extension-field dot products
- Merkle hashing: BLAKE3 over ~1 GB of encoded data

The NTT expansion ratio is 6× (rate-1/2 row encoding × rate-1/2 column encoding × 1.5 for proof vector NTTs). At 1.9 GB/s NTT throughput, the theoretical peak is ~317 MB/s per machine — we achieve ~70% of that.

## GPU STARK Prover (`gpu/` + `lean-da/`)

Full monolithic GPU implementation of the leanVM STARK prover. Everything between trace upload and proof download runs on GPU — no CPU orchestration of protocol rounds.

Built on [leanEthereum/leanMultisig](https://github.com/leanEthereum/leanMultisig)'s minimal zkVM (WHIR + SuperSpartan, KoalaBear field, Poseidon16 precompile).

| Crate | Role |
|-------|------|
| `gpu/poseidon16` | Poseidon16 compress (106M/s RTX 3060) |
| `gpu/sumcheck` | AIR + product + GKR sumcheck (CUDA graph captured) |
| `gpu/ntt` | Radix-2 NTT with fused shared-memory layers |
| `gpu/merkle` | Leaf hash + binary tree reduction |
| `gpu/poly_fold` | Multilinear polynomial folding |
| `gpu/pow_grind` | Proof-of-work grinding |
| `gpu/trace_ops` | Column manipulation utilities |

### Benchmarks

| Hardware | Workload | Throughput | vs CPU |
|----------|----------|------------|--------|
| RTX 3060 | lean-da 10 blobs | 812 KiB/s | 2.7x |
| A100 80GB | lean-da 48 blobs | 1656 KiB/s | 2.0x |
| A100 80GB | lean-da 56 blobs | 1448 KiB/s | 2.4x |
| RTX 3060 | XMSS 780 sigs | 502 XMSS/s | 2.9x |

```bash
# Run lean-da with GPU
cd lean-da && cargo run --release -p lean-da --features gpu -- --n-blobs 48

# Deploy to cloud GPU and benchmark
bash bench/deploy_and_bench.sh ubuntu@<host> ~/.ssh/key
```

See [GPU_PROVER_HANDOVER.md](GPU_PROVER_HANDOVER.md) for full architecture details.

## STARK-based circuits (`leanvm/`)

Built on [leanEthereum/leanMultisig](https://github.com/leanEthereum/leanMultisig)'s minimal zkVM.

Three circuit strategies, numbered by progression:

### 1. FFT + Merkle (baseline)

Prove the coset FFT and Merkle tree directly in-circuit. Simple but expensive.

```bash
cd leanvm
cargo run --release --example bench_fft_unroll -- --log-n 12
```

### 2. Syndrome check

Verify RS membership via batched syndrome check (Schwartz-Zippel). Uses barycentric-style 1/(β - x_j) weights. Supports chunked parallel proving with shared bytecode.

```bash
cargo run --release --example bench_syndrome_unroll -- --log-n 12
cargo run --release --example bench_chunked -- --log-n 22 --log-chunk 12 --concurrency 8
cargo run --release --example bench_recursive -- --log-n 22 --arity 8
```

### 3. RLC + fold (fastest STARK approach)

Batch m codewords via Random Linear Combination using extension-field precompiles (`dot_product_be`), then fold with random challenges (`dot_product_ee` + `add_ee`).

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
| 64 MB throughput | 1.4 MB/s | **658 MB/s** (470× faster) |
| Proof size | ~400 KB | N/A (encoding IS proof) |
| Verifier downloads | 400 KB proof | ~3 MB (sampled rows+cols) |
| Data expansion | 4× | 4× |
| Security model | STARK proof (succinct) | Sampling (probabilistic) |

ZODA is ~500× faster but requires the verifier to download rows and columns of the encoding. The STARK approach produces a compact proof that any third party can verify without the full encoding.

## Why "the dumb way"

FRIDA and similar FRI-based commitment schemes give you *proximity* to a low-degree polynomial, not an *exact* degree bound. In a DAS network, nodes that hold samples need to gossip them with Merkle proofs. If the commitment is over a proximate-but-not-exact codeword, a node that error-decodes to recover the "correct" values can't authenticate them — the corrected leaves don't match the root. By proving the encoding explicitly, every leaf under the root is by construction a true evaluation, and any node can rebroadcast with a valid proof.
