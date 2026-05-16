#!/bin/bash
# Cloud GPU setup: install deps, clone repo, build, run benchmarks.
# SCP this to the remote machine and run it, or pipe via SSH:
#   ssh ubuntu@IP 'bash -s' < bench/setup.sh
set -euo pipefail

echo "=== Installing Rust ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "=== Verifying CUDA ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NVCC=$(command -v nvcc 2>/dev/null || find /usr -name nvcc -type f 2>/dev/null | head -1)
if [ -z "$NVCC" ]; then
    echo "ERROR: nvcc not found"; exit 1
fi
export PATH="$(dirname $NVCC):$PATH"
nvcc --version | tail -1

# Auto-detect CUDA compute capability
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *A100*) export CUDA_ARCH=80 ;;
    *H100*) export CUDA_ARCH=90 ;;
    *L40*)  export CUDA_ARCH=89 ;;
    *4090*) export CUDA_ARCH=89 ;;
    *3090*|*3080*|*3070*|*3060*) export CUDA_ARCH=86 ;;
    *A6000*) export CUDA_ARCH=86 ;;
    *) export CUDA_ARCH=86 ;;
esac
echo "Detected GPU: $GPU_NAME → CUDA_ARCH=$CUDA_ARCH"

echo "=== Cloning repo ==="
cd /tmp
[ -d starkdal ] && rm -rf starkdal
git clone --depth 1 --recurse-submodules "${REPO_URL:-https://github.com/nicola/starkdal.git}"
cd starkdal

echo "=== Building (release, CPU + GPU) ==="
cd lean-da
cargo build --release -p lean-da 2>&1 | tail -3
cargo build --release -p lean-da --features gpu 2>&1 | tail -3
cargo build --release -p lean_prover --lib 2>&1 | tail -3
cargo build --release -p lean_prover --lib --features gpu 2>&1 | tail -3

echo ""
echo "========================================"
echo " LEAN-DA BENCHMARK (RS Data Availability)"
echo "========================================"

for N in 4 8 48; do
    echo "── $N blobs ──"
    echo "CPU:"
    cargo run --release -p lean-da -- --n-blobs $N 2>&1 | grep -E "Proving time|Proof size|Throughput"
    echo "GPU:"
    cargo run --release -p lean-da --features gpu -- --n-blobs $N 2>&1 | grep -E "Proving time|Proof size|Throughput"
    echo ""
done

echo "========================================"
echo " LEAN-VM UNIT PROOFS"
echo "========================================"

echo "CPU: all_precompiles"
cargo test -p lean_prover --lib --release -- test_zk_vm_all_precompiles --test-threads=1 --nocapture 2>&1 | grep "Proof time"
echo "GPU: all_precompiles"
cargo test -p lean_prover --lib --release --features gpu -- test_zk_vm_all_precompiles --test-threads=1 --nocapture 2>&1 | grep "Proof time"

echo "CPU: fibonacci"
cargo test -p lean_prover --lib --release -- test_prove_fibonacci --test-threads=1 --nocapture 2>&1 | grep "Proof time"
echo "GPU: fibonacci"
cargo test -p lean_prover --lib --release --features gpu -- test_prove_fibonacci --test-threads=1 --nocapture 2>&1 | grep "Proof time"

echo "CPU: small_memory"
cargo test -p lean_prover --lib --release -- test_small_memory --test-threads=1 --nocapture 2>&1 | grep "Proof time"
echo "GPU: small_memory"
cargo test -p lean_prover --lib --release --features gpu -- test_small_memory --test-threads=1 --nocapture 2>&1 | grep "Proof time"

echo ""
echo "=== Done ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
