# GPU Poseidon16

CUDA kernel for KoalaBear Poseidon16 (the hash used by leanVM's STARK prover).

## Architecture

- **Kernel** (`kernel/poseidon16.cu`): Each CUDA thread computes one `poseidon16_compress`. State lives in registers (16 `u32`s). Round constants and sparse decomposition matrices live in `__constant__` memory.
- **Rust harness** (`src/lib.rs`): Extracts precomputed Montgomery-form constants from the `backend` crate, uploads to GPU, launches kernels, validates against CPU reference.
- **Property tests** (`tests/proptest_gpu_vs_cpu.rs`): 200 random inputs per test, comparing GPU vs CPU for compress, permute, and batch compress.

## Algorithm

Poseidon16 on KoalaBear (p = 0x7F000001, Montgomery form):

1. **4 initial full rounds**: AddRC -> x^3 S-box (all 16 elements) -> circulant MDS
2. **20 partial rounds** (sparse decomposition): S-box on state[0] only -> scalar constant -> sparse matmul (dot product + rank-1 update)
3. **4 terminal full rounds**: same as initial

Compress: `output[0..8] = (perm(state) + state)[0..8]`

## Build

Requires CUDA toolkit (nvcc) and a CUDA-capable GPU.

```bash
export CUDA_HOME=/path/to/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# Default target: sm_86 (Ampere, RTX 3060). Override with:
export CUDA_ARCH=90  # for Hopper (H100)

cargo build --release
```

## Run

```bash
# Benchmark 1M compress calls, compare GPU vs CPU
cargo run --release -- --n 1000000 --bench-cpu

# Verify correctness on 10K random inputs
cargo run --release -- --n 10000 --verify

# Full benchmark + verification
cargo run --release -- --n 1000000 --bench-cpu --verify
```

## Test

```bash
# CPU reference tests (no GPU needed)
cargo test --release --lib

# GPU property-based tests (200 random cases per test)
cargo test --release --test proptest_gpu_vs_cpu
```

## Performance (RTX 3060, 12 GB)

| Batch size | GPU time | GPU throughput | CPU throughput | Speedup |
|------------|----------|----------------|----------------|---------|
| 1M         | 9.4 ms   | 106 M/s        | 0.34 M/s       | 311x    |
| 10M        | 95.9 ms  | 104 M/s        | 0.34 M/s       | 304x    |
