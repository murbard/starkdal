#!/bin/bash
# Deploy starkdal to a remote GPU machine and run lean-da + unit proof benchmarks.
# Usage: ./bench/deploy_and_bench.sh <user@host> [ssh_key]
#
# Requires: the repo tarball at /tmp/starkdal.tar.gz (created by bench/pack.sh)
set -euo pipefail

HOST="${1:?Usage: deploy_and_bench.sh <user@host> [ssh_key]}"
KEY="${2:-$HOME/.ssh/bench_key}"
SSH="ssh -o StrictHostKeyChecking=no -i $KEY"
SCP="scp -o StrictHostKeyChecking=no -i $KEY"

echo "=== Deploying to $HOST ==="
$SCP /tmp/starkdal.tar.gz "$HOST:/tmp/"

$SSH "$HOST" bash -s << 'REMOTE'
set -euo pipefail
source "$HOME/.cargo/env" 2>/dev/null || {
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
    source "$HOME/.cargo/env"
}

NVCC=$(find /usr -name nvcc -type f 2>/dev/null | head -1)
[ -n "$NVCC" ] && export PATH="$(dirname $NVCC):$PATH"

# Auto-detect CUDA arch
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

cd /tmp && rm -rf starkdal && mkdir starkdal && cd starkdal
tar xzf /tmp/starkdal.tar.gz
sed -i "s|/home/coder/workspace/starkdal|/tmp/starkdal|g" lean-da/Cargo.toml
cd lean-da

echo "========================================"
echo " GPU: $GPU_NAME (CUDA_ARCH=$CUDA_ARCH)"
echo " $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo " $(nvcc --version 2>/dev/null | tail -1)"
echo "========================================"

echo "=== Building (release) ==="
cargo build --release -p lean-da -p lean_prover 2>&1 | tail -1
cargo build --release -p lean-da -p lean_prover --features gpu 2>&1 | tail -1

echo ""
echo "======== LEAN-DA (RS Data Availability) ========"
for N in 4 8 48 56; do
    echo "── $N blobs ──"
    echo "CPU:"
    cargo run --release -p lean-da -- --n-blobs $N 2>&1 | grep -E "Proving time|Proof size|Throughput" || echo "FAILED"
    echo "GPU:"
    cargo run --release -p lean-da --features gpu -- --n-blobs $N 2>&1 | grep -E "Proving time|Proof size|Throughput" || echo "FAILED"
done

echo ""
echo "======== LEANVM UNIT PROOFS ========"
for T in test_prove_fibonacci test_small_memory test_zk_vm_all_precompiles; do
    echo "── $T ──"
    echo "CPU:"
    cargo test -p lean_prover --lib --release -- $T --test-threads=1 --nocapture 2>&1 | grep "Proof time" || echo "FAILED"
    echo "GPU:"
    cargo test -p lean_prover --lib --release --features gpu -- $T --test-threads=1 --nocapture 2>&1 | grep "Proof time" || echo "FAILED"
done

echo ""
echo "=== DONE ==="
REMOTE
