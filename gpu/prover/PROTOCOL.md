# GPU Prover Protocol Specification

## Overview

The GPU prover reimplements the STARK proving protocol from scratch using flat u32 GPU buffers. It does NOT use leanVM's type system (MleOwned, EFPacking, etc.). Only leanVM's VM execution (trace generation) and ProverState (Fiat-Shamir) are used.

## Data Types

All data on GPU is flat `u32` arrays:
- Base field element: 1 u32 (KoalaBear Montgomery form)
- Extension field element: 5 u32s (quintic, contiguous)
- Digest: 8 u32s (Poseidon16 output)

No SIMD packing. No packed types. Just flat arrays.

## Protocol Steps

### Input
- Trace columns: Vec<Vec<u32>> per table (base field, column-major)
- Memory: Vec<u32> (base field)
- Bytecode: Vec<u32> (base field)
- Public input: Vec<u32>

### Step 1: Upload (one-time, ~200MB)
Upload ALL trace data to GPU as flat u32 CudaSlice buffers.

### Step 2: Access counts (GPU)
For each lookup in each table, histogram accumulation using atomicAdd.
Output: memory_acc, bytecode_acc (flat u32 on GPU).

### Step 3: Polynomial stacking (GPU)
Concatenate [memory | memory_acc | bytecode_acc | table_cols...] into
one flat u32 buffer on GPU. Column copy with offsets.
Output: d_stacked (2^stacked_n_vars u32s on GPU).

### Step 4: WHIR commit (GPU)
4a. Reorder: prepare_evals_for_fft (GPU kernel)
4b. DFT: batched NTT on reordered matrix (GPU kernel)
4c. Merkle: leaf hash + binary reduction (GPU kernel)
4d. Download root (32 bytes) → CPU ProverState
4e. OOD evaluation: MLE eval at sampled points (GPU kernel)
4f. Download OOD answers (~80 bytes) → CPU ProverState
Output: DFT matrix on GPU (for Merkle path opening), Merkle tree on GPU.

### Step 5: Logup (GPU)
5a. Sample challenges from ProverState (CPU, tiny)
5b. Fingerprint computation (GPU kernel): build numerators/denominators
5c. Endianness reorder (GPU kernel)
5d. GKR quotient: iterative layer reduction using gkr_sum_quotients (GPU kernel)
5e. Download top layer (~2KB) → CPU ProverState
5f. Backward GKR layers: prove_gkr_layer (GPU sumcheck per layer)
5g. Column evaluations at GKR point (GPU MLE eval)
5h. Download evaluations → CPU ProverState
Output: GKR point, column evaluation values.

### Step 6: AIR sumcheck (GPU)
For each table (Execution, ExtensionOp, Poseidon16):
6a. Build shifted (down) columns on GPU (shift_down kernel)
6b. Build eq factor on GPU (eq_polynomial kernel)
6c. For each sumcheck round:
    - Evaluate constraints at z=0,2,3,...,degree (GPU AIR constraint kernel)
    - Multiply by alpha powers and eq factor (GPU)
    - Block-reduce to round polynomial coefficients (GPU)
    - Download coefficients (~60 bytes) → CPU ProverState
    - Upload challenge (~20 bytes) → GPU
    - Fold all columns + eq with challenge (GPU fold kernel)
Output: sumcheck evaluation point, final column values.

### Step 7: WHIR prove (GPU)
For each WHIR round:
7a. DFT on current polynomial (GPU NTT)
7b. Merkle commit (GPU kernel)
7c. Download root → CPU ProverState
7d. OOD evaluation (GPU MLE eval) → download
7e. PoW grinding (GPU kernel) → download nonce
7f. Query sampling (CPU ProverState)
7g. Merkle path opening (GPU kernel) → download paths
7h. Eq polynomial accumulation (GPU kernel)
7i. Product sumcheck rounds (GPU fold + sumcheck)
7j. Download round polynomial → CPU ProverState → upload challenge
Output: WHIR evaluation randomness.

### Step 8: Download proof
Download transcript + Merkle paths from ProverState.
Total download: ~300KB.

## PCIe Traffic Summary
- Step 1: ~200MB upload (one-time)
- Steps 2-7: ~200 bytes per sumcheck round (~100 rounds total = ~20KB)
- Step 8: ~300KB download
- Total: ~200MB upload + ~320KB download
