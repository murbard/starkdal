# GPU Prover Handover — May 15, 2026

## Goal

Build a **fully monolithic GPU STARK prover** for the generic leanVM. After
initial trace upload, the entire proving pipeline runs on GPU with **zero CPU
orchestration** between upload and final proof download.

```
CPU: execute VM -> build trace
     |  one-time upload
GPU: stack/commit -> logup/GKR -> AIR sumcheck -> WHIR prove
     |  one-time download
CPU: assemble proof struct
```

**Not acceptable** after proving starts: CPU phase sequencing, CPU Fiat-Shamir
sampling between phases, intermediate GPU-to-host reads, phase-specific uploads,
or silent CPU fallbacks.

## Current State

**All tests pass on GPU hardware.** Both the lean-da integration path (10 tests)
and the leanvm generic path (`test_zk_vm_all_precompiles` ~0.54s) verify
successfully.

The generic leanvm path (`leanvm/crates/lean_prover/src/gpu_prove_execution.rs`)
is the primary implementation. The lean-da submodule has an older, smaller copy
that also works but is not the focus of ongoing development.

### What works

- **All protocol sub-steps run on GPU**: logup fingerprints, GKR quotient (CUDA
  graph capture per layer), AIR sumcheck (CUDA graph capture for all rounds),
  WHIR prove (product sumcheck + fold + DFT + Merkle).
- **GPU Fiat-Shamir throughout**: Poseidon16 challenger on device. Constants
  uploaded once at GPU backend init and shared across all phases.
- **Device-resident data flow**: column evaluations, eq polynomials, challenges,
  and WHIR statement buffers stay on device between phases.
- **CUDA graph capture**: AIR sumcheck (single graph for all rounds), GKR
  quotient (per-layer graphs with top-claim graph), WHIR product sumcheck
  (initial + per-round captured), logup eval staging, initial commitment OOD,
  WHIR round OOD, WHIR round STIR queries, WHIR round constraints, WHIR
  initial constraints, WHIR initial device-statement accumulation.
- **Preallocated workspaces**: `GpuActiveProverWorkspaces` and
  `GpuWhirProverWorkspaces` allocate all buffers during initial upload. Active
  proving consumes them without allocation.
- **21 source-level residency audit tests** that verify the strict integrated
  path has no `memcpy_dtov`, no legacy GPU transcript finishing, no standalone
  host-boundary Fiat-Shamir, no mid-proof downloads, and that various
  operations use uploaded workspaces inside graph capture.

### Architecture assessment

The function `prove_uploaded_plan_cpu_orchestrated()` sequences phases from
Rust, but analysis shows **zero `synchronize`/`memcpy_dtov`/`memcpy_stod` calls**
in its body. The Rust code between phases only does buffer routing
(`take_gpu_workspace` moves `Option<CudaSlice>` into local variables). All GPU
work is launched asynchronously via `graph.launch()` on the same stream — the
GPU executes all graphs back-to-back without waiting for CPU.

```rust
fn prove_uploaded_plan_cpu_orchestrated(g, prover_state, uploaded_plan) {
    // STEP 4-5: stack + DFT + Merkle + OOD (graph.launch, async)
    // [Rust: buffer routing only, no sync]
    // STEP 6: logup + GKR (graph.launch x N, async)
    // [Rust: buffer routing only, no sync]
    // STEP 7: AIR sumcheck (graph.launch, async)
    // [Rust: buffer routing only, no sync]
    // STEP 8: WHIR prove (graph.launch x many, async)
    // WHIR materialization: single memcpy_dtov (final proof)
}
```

The phase functions (gpu_air, gpu_logup, gpu_gkr) also have zero
`synchronize`/`memcpy_dtov` calls. The only host synchronization in the
entire proving pipeline is WHIR's final proof materialization download.

Despite the function name suggesting "CPU orchestrated", the actual execution
is effectively device-resident — the GPU never waits for CPU between phases.
The remaining concern is the WHIR round loop in `gpu_open.rs` which iterates
from Rust but launches captured graphs asynchronously per round.

### Host download audit (integrated path)

| File | `memcpy_dtov` | Notes |
|------|---------------|-------|
| `gpu_prove_execution.rs` | 1 | Final proof materialization only |
| `gpu_air.rs` | 0 | Fully device-resident |
| `gpu_logup.rs` | 0 | Fully device-resident |
| `gpu_gkr.rs` | 10 | 1 conditional (disabled in integrated path), 9 test-only |
| `whir/gpu_open.rs` | 19 | Merkle opening materialization (proof data) + legacy paths |

## Code Organization

### Two codebases

| Path | Role | Size |
|------|------|------|
| `leanvm/` | Generic leanVM prover (primary, actively developed) | ~11K LOC GPU code |
| `lean-da/` | Integration workload submodule (older, smaller copy) | ~5K LOC GPU code |

The `lean-da` submodule is an integration test target. The generic implementation
lives in `leanvm/`. When `--features gpu` is enabled, `prove_execution()` dispatches
to `gpu_prove_execution()`.

### leanvm GPU prover files

| File | Lines | Role |
|------|-------|------|
| `gpu_prove_execution.rs` | 2,619 | Main entry, plan upload, phase orchestration, 21 residency audits |
| `gpu_logup.rs` | 1,577 | Logup fingerprint witness + GKR integration |
| `gpu_gkr.rs` | 1,279 | GKR quotient protocol (captured layers) |
| `gpu_air.rs` | 1,161 | AIR sumcheck (single CUDA graph) |
| `whir/gpu_open.rs` | 3,432 | WHIR prove (round loop, Merkle, materialization) |
| `whir/gpu_combine.rs` | 571 | Device-resident WHIR statement accumulation |
| `whir/gpu_backend.rs` | 225 | Shared GPU context singleton |

### GPU kernel crates (`gpu/`)

| Crate | Kernel LOC | Binding LOC | Purpose |
|-------|-----------|-------------|---------|
| `gpu/sumcheck` | 3,784 | 5,781 | AIR multi-z eval, protocol step, Fiat-Shamir, eq polynomial |
| `gpu/ntt` | 321 | 1,436 | Radix-2 NTT/DFT with async twiddle management |
| `gpu/poly_fold` | 332 | 804 | Polynomial folding (base->ext, ext->ext, evals-to-coeffs) |
| `gpu/merkle` | 314 | 779 | Leaf hashing, binary reduction, batch row gathering |
| `gpu/trace_ops` | 278 | 601 | Bit-reversal, base-to-ext, fill, copy, negate, shift |
| `gpu/pow_grind` | 199 | 447 | PoW nonce search |
| `gpu/poseidon16` | 194 | 280 | Poseidon16 permutation + compression |
| `gpu/field` | 321 (hdr) | -- | KoalaBear + quintic extension field arithmetic |

### Key architecture patterns

- **CUDA graph capture**: Protocol-step kernels (Lagrange interpolation,
  expand_bare_to_full, Fiat-Shamir) run inside captured graphs. The graph is
  built once and launched, executing all rounds without CPU intervention.

- **Preallocated workspaces**: `GpuActiveProverWorkspaces` (stacked poly, DFT
  output, Merkle layers, OOD buffers, NTT twiddles) and
  `GpuWhirProverWorkspaces` (per-round OOD/STIR/constraint buffers, DFT/Merkle
  layers) are allocated during `upload_initial_prover_plan()`. Active proving
  uses `take_gpu_workspace()` to consume them without allocation.

- **GpuFsPhase**: Device-resident Fiat-Shamir wrapper. Accumulates transcript
  chunks on device. `finish()` packs everything into one download.

- **Dual codebases**: lean-da has the working verified copy. leanvm has the
  actively-developed version with workspace preplanning and residency audits.

## Proving Pipeline Detail

### Entry point

`gpu_prove_execution()` (line 1746 of `gpu_prove_execution.rs`):
1. **STEP 1**: CPU VM execution (inherently sequential)
2. **STEP 2**: CPU ProverState setup (tiny Fiat-Shamir init)
3. **STEP 3**: `upload_initial_prover_plan()` — one-time upload of trace,
   columns, constants, and all preallocated workspace buffers
4. **Active proving**: `prove_uploaded_plan_cpu_orchestrated()` — the body
   that needs to become a single device-resident operation

### prove_uploaded_plan_cpu_orchestrated() flow

**STEP 4-5** (stack + commit): Build stacked polynomial on device, DFT with
uploaded twiddles into uploaded output buffer, Merkle tree into uploaded layers.
Initial OOD sampling/evaluation captured as graph.

**STEP 6** (logup + GKR): Logup challenge sampling captured as graph. Logup
witness construction on device. GKR quotient with captured top-claim graph +
captured per-layer graphs. Post-GKR logup eval staging captured as graph.
Logup table WHIR statement buffers filled in-place.

**STEP 7** (AIR sumcheck): AIR beta/alpha/eta challenge sampling inside captured
AIR graph. Multi-z constraint evaluation, protocol step (Lagrange + expand +
Fiat-Shamir), column folding — all in one CUDA graph. AIR table WHIR statement
buffers filled in-place inside the graph. Public-memory WHIR sampling/eval
inside AIR graph tail.

**STEP 8** (WHIR prove): Device-resident WHIR with captured graphs for initial
constraints, initial device-statement accumulation, per-round OOD, per-round
STIR queries, per-round constraints. DFT uses uploaded twiddles, Merkle uses
uploaded layers. Final materialization packs transcript + Merkle paths into
one device buffer for single download.

## Testing

```bash
# Full leanvm suite (31 tests, all pass with 16MB stack):
cd leanvm
PATH=/home/coder/.local/cuda/toolkit/bin:$PATH \
RUST_MIN_STACK=16777216 \
  cargo test -p lean_prover --lib --features gpu -- --test-threads=1

# lean-da integration (10 tests, all pass):
cd lean-da
PATH=/home/coder/.local/cuda/toolkit/bin:$PATH \
  cargo test -p lean_prover --lib --features gpu -- --test-threads=1
```

Note: nvcc is at `/home/coder/.local/cuda/toolkit/bin/nvcc` (CUDA 12.6).

## Known Issues

- `tests::compute_snark_domain_sep` needs 16MB stack (`RUST_MIN_STACK=16777216`)
  due to recursive bytecode compilation. Not GPU-related. With larger stack, all
  31 leanvm tests pass.
- WHIR round loop is still Rust-iterated (each round launches captured graphs
  but the loop is CPU).
- Merkle opening materialization in WHIR has per-query downloads (inherent —
  proof data must be downloaded).
- `prove_uploaded_plan_cpu_orchestrated` is explicitly named to indicate it is
  the noncompliant CPU scheduler that needs to be replaced.
- Logup witness assembly cannot be CUDA-graph-captured — CUDA rejects the
  capture at `end_capture` with `CUDA_ERROR_INVALID_VALUE`. The source audit
  guards against reintroducing this known-bad path.

## Key Correctness Fixes (History)

- **AIR padding sentinel**: `0` was ambiguous (meant both "no padding" and "all
  rows are padding" for empty tables). Changed to `u32::MAX` as sentinel. This
  fixed empty ExtensionOp/Poseidon16 tables in Fibonacci.
- **Protocol step ea_inv**: `qe_inv(ea, ea_inv)` inside the protocol step kernel
  caused stack corruption (pe[65] + bare[60] + qe_inv locals exceeded thread
  stack). Fixed by precomputing ea_inv on host and passing via `TS_EA_INV`.
- **AIR graph-side eval observe**: An attempted single batched AIR eval observe
  changed the challenger state and failed with `InvalidGrindingWitness`. The
  validated path keeps per-table challenger updates.
- **GpuFsPhase shared state**: The AIR graph must use the same device-resident
  challenger state as the bus_beta/air_alpha/air_eta sampling phase (via shared
  `GpuFsPhase`), not re-upload from CPU prover state.

## Legacy Code

`gpu/prover/` is a legacy prototype / kernel test harness. It contains
CPU-orchestrated callbacks, intermediate downloads, and host-side stacking.
It is NOT the target prover architecture. The target is `leanvm/crates/lean_prover/`.

## Performance Analysis (updated 2026-05-16)

**RTX 3060 (12GB, 3584 cores, 336 GB/s):**

| Blobs | CPU Time | GPU Time (CPU+GPU) | GPU KiB/s | Speedup |
|-------|----------|-------------------|-----------|---------|
| 4     | 2.21s    | 1.24s (0.21+1.02) | 500       | 1.78x   |
| 8     | 4.17s    | 1.86s (0.31+1.56) | 665       | 2.24x   |
| 10    | ~5.2s    | 1.96s             | 791       | ~2.7x   |
| 12    | ~6.3s    | 2.71s             | 686       | ~2.3x   |

Max blobs on 12GB: 12 (16 OOMs in GKR layer workspace allocation).
Unit proofs: 20-65x speedup (fibonacci 24x, small_memory 65x, all_precompiles 20x).

**A100 80GB PCIe (6912 cores, 2 TB/s, 28 CPU cores, CUDA 12.8):**

| Blobs | CPU Time | GPU Time (CPU+GPU) | GPU KiB/s | Speedup |
|-------|----------|-------------------|-----------|---------|
| 8     | 2.40s    | 1.64s (0.34+1.30) | 757       | 1.47x   |
| 48    | 9.19s    | 4.49s (1.90+2.59) | 1656      | 2.04x   |
| 56    | 14.50s   | 5.99s (2.25+3.71) | 1448      | 2.42x   |

Peak throughput: **1656 KiB/s (1.62 MiB/s) at 48 blobs**.
GPU-only proving is 3.55x faster than 28-core CPU at 48 blobs.

**Optimizations applied (May 16 session, 15 commits):**
- `alloc_zeros` instead of `memcpy_stod(vec![0])`: saved 150ms on 640MB zero-fill
- Async MLE fold with ping-pong buffers (base, ext, multi-point): eliminated per-fold sync
- Per-column trace upload: eliminated CPU-side 200MB+ concatenation (457ms → 86ms on A100)
- GPU-side AIR column slicing + shifted column computation
- Batched multi-column MLE evaluation for logup columns (13+ cols → single fold chain)
- Batched initial OOD and memory/memory_acc evaluations
- Async STIR query sampling, NTT twiddle preloading, sync-free DFT layers
- GKR layer memory optimization (eager buffer freeing)
- Shared memory caching for dense_eq_accumulate_from_points kernel
- Support for FUSED_NTT_LOG > 13 with cuFuncSetAttribute

**Where the time goes (lean-da 48 blobs, A100, sync mode):**
- Witness gen (CPU): 1.97s — VM execution 1.75s + trace build 231ms
- Trace staging: 86ms — per-column upload + access counts
- Stack+commit: 224ms — DFT 47ms + Merkle 153ms
- Logup+GKR: 986ms — witness 122ms, GKR 763ms, column evals ~100ms
- AIR sumcheck: 421ms — single CUDA graph
- WHIR prove: 874ms — initial 105ms, R0 ~250ms, R1+R2 ~200ms, mat 11ms

**FUSED_NTT_LOG finding:** FUSED_LOG=15 gives only 1.3% improvement on A100.
FUSED_LOG=13 (32KB) remains optimal — NTT is only 2% of proving time.

## XMSS Aggregation (RTX 3060, lean-da branch with GPU)

| XMSS Sigs | GPU (XMSS/s) | CPU (XMSS/s) | Speedup |
|-----------|-------------|-------------|---------|
| 100       | 169         | 120         | 1.41x   |
| 190       | 285         | 187         | 1.52x   |
| 400       | 356         | 123         | 2.89x   |
| 780       | 502 (peak)  | ~170        | ~2.95x  |

GPU peak: **502 XMSS/s at 780 sigs** on RTX 3060.
GPU advantage grows at power-of-2 boundaries (200, 400, 800) where CPU
throughput drops sharply but GPU stays higher.

## Next Steps

1. **NTT-based Poseidon16 MDS**: replace the 256-mul direct circulant multiply
   with a 128-mul NTT-based approach. Would speed up all Poseidon16 operations
   (Merkle tree hashing is 5-6% of proving).
2. **Higher folding factor**: increasing `ff` from 7 to 8 or 9 reduces WHIR
   round count but increases per-round work. Could reduce WHIR overhead.
3. **CPU witness parallelization**: at 48 blobs, CPU witness (1.97s) is 44%
   of total. Trace build (231ms) could potentially be parallelized.
4. **A100 XMSS benchmark**: run XMSS aggregation on A100 (was out of stock).
   Expected: 800-1200 XMSS/s based on 2-3x lean-da speedup factor.
