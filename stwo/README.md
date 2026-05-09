# stwo — Cairo reference implementation

Cairo 2.18 circuit proven with Stwo (Circle STARKs, Stark252 field, Blake2s hashing). This is a reference implementation — the primary work is in `../leanvm/` (STARK-based) and `../zoda/` (ZODA).

## What it does

Proves a coset FFT + Blake2s Merkle commitment inside a Stwo STARK circuit. The data is encoded as polynomial coefficients over Stark252, evaluated via FFT on a coset, and committed via a Blake2s Merkle tree.

## Usage

```bash
scarb cairo-test          # 11 tests
scarb execute --arguments-file <args.json>
python3 scripts/bench.py  # full benchmark with proving
```

## Files

| File | Description |
|---|---|
| `src/lib.cairo` | Main circuit: FFT + Merkle |
| `src/fft.cairo` | Coset FFT (Cooley-Tukey butterflies) |
| `src/merkle.cairo` | Merkle tree construction |
| `src/blake_hash.cairo` | Blake2s hash wrapper |
| `src/constants.cairo` | Precomputed twiddle factors |
| `scripts/reference.py` | Python reference for verification |
| `scripts/bench.py` | Benchmark harness |
