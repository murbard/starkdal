#!/bin/bash
# Lean-DA benchmark: RS data availability proving.
# Measures proof time, proof size (KiB), throughput (KiB/s) for N blobs.
# Each blob = 2^13 ext-field elements ~ 155 KiB useful payload.
#
# Usage:
#   ./bench/bench_lean_da.sh          — CPU and GPU, default blob counts
#   ./bench/bench_lean_da.sh gpu      — GPU only
#   ./bench/bench_lean_da.sh cpu      — CPU only
#   N_BLOBS="4 8" ./bench/bench_lean_da.sh gpu  — custom blob counts
set -euo pipefail

MODE="${1:-both}"
BLOB_COUNTS="${N_BLOBS:-4 8 48}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../lean-da"

for p in "$HOME/.local/cuda/toolkit/bin" /usr/local/cuda/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done

echo "========================================"
echo " Lean-DA Benchmark (RS Data Availability)"
echo " Blob size: 2^13 ext elements ~ 155 KiB"
echo " Blob counts: $BLOB_COUNTS"
echo " Mode: $MODE"
echo " Date: $(date -u)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"
echo ""

for N in $BLOB_COUNTS; do
    echo "── $N blobs (~$(( N * 155 )) KiB payload) ──"

    if [ "$MODE" = "cpu" ] || [ "$MODE" = "both" ]; then
        echo "CPU:"
        cargo run --release -p lean-da -- --n-blobs $N 2>&1 \
            | grep -E "Proving time|Proof size|Throughput|Cycles|Poseidon"
    fi

    if [ "$MODE" = "gpu" ] || [ "$MODE" = "both" ]; then
        echo "GPU:"
        cargo run --release -p lean-da --features gpu -- --n-blobs $N 2>&1 \
            | grep -E "Proving time|Proof size|Throughput|Cycles|Poseidon|CPU witness|GPU proving"
    fi
    echo ""
done
