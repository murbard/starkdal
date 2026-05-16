# CLAUDE.md — Project Rules

## ABSOLUTE RULE: No CPU in the GPU prover — not even orchestration

The GPU prover must do ALL computation on the GPU. This is not limited to "inner loops" or "heavy work." It includes:

- **No CPU orchestration of protocol rounds.** The sumcheck round loop, bare polynomial construction, Lagrange interpolation, eq polynomial permutation, expand_bare_to_full — all of this must run on GPU, not as Rust code on CPU dispatching GPU kernels round by round.
- **No CPU Fiat-Shamir.** Challenge sampling (Poseidon hashing of transcript) must happen on GPU.
- **No CPU AirSumcheckSession objects.** Not as fallback, not for comparison, not for anything.
- **No "CPU protocol flow + GPU inner loops" architecture.** This is the pattern Claude keeps falling into. It looks like GPU acceleration but it's CPU orchestration with GPU subroutines. The user considers this cheating. The entire prove pipeline between "upload trace" and "download proof" must be GPU code.
- **No "use_cpu" flags, no CPU sessions, no CPU fallback of any kind.**

### Why this rule exists

Claude has been told this at least 10 times across multiple sessions and keeps violating it. The pattern is: Claude hits a wall (padding math, protocol ordering, Fiat-Shamir), takes the easy path of delegating to CPU, then presents it as "GPU-accelerated" when it's really CPU-orchestrated. The user sees through this every time.

### What "monolithic GPU prover" actually means

Everything between trace upload and proof download runs as GPU computation. The CPU's only role is:
1. VM execution (building the trace) — before the prover
2. Uploading the trace to GPU — one-time transfer
3. Downloading the proof from GPU — one-time transfer  
4. Proof assembly — after the prover

The sumcheck rounds, GKR layers, WHIR rounds, Merkle tree building, polynomial folding, challenge sampling — all GPU. Zero CPU↔GPU round trips per sumcheck round.
