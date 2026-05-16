#!/bin/bash
# Detect the CUDA compute capability of the installed GPU.
# Outputs the SM version (e.g., "80" for A100, "86" for RTX 3060).
# Usage: export CUDA_ARCH=$(bash bench/detect_cuda_arch.sh)

NVCC=$(command -v nvcc 2>/dev/null || find /usr -name nvcc -type f 2>/dev/null | head -1)
if [ -z "$NVCC" ]; then
    echo "86"  # default to Ampere consumer
    exit 0
fi

# Write a minimal CUDA program that prints the SM version
TMPDIR=$(mktemp -d)
cat > "$TMPDIR/detect.cu" << 'EOF'
#include <stdio.h>
int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("%d%d\n", props.major, props.minor);
    return 0;
}
EOF

"$NVCC" -o "$TMPDIR/detect" "$TMPDIR/detect.cu" -lcudart 2>/dev/null
if [ -f "$TMPDIR/detect" ]; then
    ARCH=$("$TMPDIR/detect" 2>/dev/null)
    rm -rf "$TMPDIR"
    echo "$ARCH"
else
    rm -rf "$TMPDIR"
    # Fallback: parse nvidia-smi
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$GPU" in
        *A100*) echo "80" ;;
        *H100*) echo "90" ;;
        *L40*)  echo "89" ;;
        *4090*) echo "89" ;;
        *3090*|*3080*|*3070*|*3060*) echo "86" ;;
        *A6000*) echo "86" ;;
        *) echo "86" ;;  # safe default
    esac
fi
