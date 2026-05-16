#!/bin/bash
# XMSS signature aggregation benchmark.
# Measures per-node proving time, proof size, XMSS/s throughput.
#
# Usage:
#   ./bench/bench_xmss.sh              — default signature counts
#   ./bench/bench_xmss.sh gpu          — GPU only
#   N_SIGS="4 16 64" ./bench/bench_xmss.sh   — custom counts
set -euo pipefail

MODE="${1:-both}"
SIG_COUNTS="${N_SIGS:-4 16 64 128}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../lean-da"

for p in "$HOME/.local/cuda/toolkit/bin" /usr/local/cuda/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done

echo "========================================"
echo " XMSS Aggregation Benchmark"
echo " Signature counts: $SIG_COUNTS"
echo " Mode: $MODE"
echo " Date: $(date -u)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU)"
echo ""

# The aggregation benchmark is in the test suite; we run it via cargo test
# with --nocapture to see the live tree output.
for N in $SIG_COUNTS; do
    echo "── $N XMSS signatures ──"

    if [ "$MODE" = "cpu" ] || [ "$MODE" = "both" ]; then
        echo "CPU:"
        # Run the test_aggregation test which iterates [1,2,4,...,128]
        # We can't parameterize it directly, but we can run the binary
        cargo test --release -p rec_aggregation -- test_aggregation_throughput_per_num_xmss --nocapture --ignored 2>&1 \
            | grep -E "$N," || echo "  (run full aggregation suite to get $N-sig data)"
    fi

    if [ "$MODE" = "gpu" ] || [ "$MODE" = "both" ]; then
        echo "GPU:"
        cargo test --release -p rec_aggregation --features gpu -- test_aggregation_throughput_per_num_xmss --nocapture --ignored 2>&1 \
            | grep -E "$N," || echo "  (run full aggregation suite to get $N-sig data)"
    fi
    echo ""
done

echo ""
echo "For full XMSS aggregation sweep, run:"
echo "  cargo test --release -p rec_aggregation -- test_aggregation_throughput_per_num_xmss --nocapture --ignored"
