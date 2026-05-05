//! Chunked parallel proving: split evaluations into 2^k chunks, prove each independently.
//!
//! Each chunk proves: "my evaluations commit to subtree_root and have partial syndrome
//! sum S_i for each Fiat-Shamir challenge." The Rust harness verifies all proofs and
//! checks Σ S_i = 0.
//!
//! Usage: cargo run --release --example bench_chunked -- --log-n 14 --log-chunk 10

use rayon::prelude::*;
use starkdal_leanvm::*;
use std::collections::HashMap;
use std::time::Instant;

/// Generate the syndrome-check program for a single chunk.
/// The chunk covers evaluations [chunk_offset .. chunk_offset + chunk_size).
/// Public input: [subtree_root(8), beta_0..3(4), expected_S_0..3(4)] = 16 FE
fn generate_chunk_program(cp: &CircuitParams, chunk_idx: usize, chunk_size: usize) -> String {
    assert!(cp.log_blowup == 1, "syndrome requires log_blowup == 1");
    let chunk_offset = chunk_idx * chunk_size;
    let chunk_fpl = cp.fpl;
    let chunk_n_leaves = chunk_size / chunk_fpl;
    let chunk_tree_depth = chunk_n_leaves.trailing_zeros() as usize;
    let chunk_n_chunks_per_leaf = chunk_fpl / 8;

    // Merkle layer offsets for this chunk's subtree
    let mut layer_offsets = vec![0usize];
    let mut acc_off = 0;
    for k in 0..chunk_tree_depth {
        acc_off += (chunk_n_leaves >> k) * DIGEST_LEN;
        layer_offsets.push(acc_off);
    }
    let total_tree_size = acc_off + DIGEST_LEN;

    // Precompute per-j constants for this chunk's index range
    let mut x = cp.g * cp.omega.exp_u64(chunk_offset as u64); // y_{chunk_offset}
    let mut w = cp.g_1mn * cp.omega_1mn.exp_u64(chunk_offset as u64);
    let signs: Vec<F> = (0..chunk_size)
        .map(|j| {
            if (chunk_offset + j) % 2 == 0 {
                F::ONE
            } else {
                F::ZERO - F::ONE
            }
        })
        .collect();
    let coset_points: Vec<F> = (0..chunk_size)
        .map(|_| {
            let v = x;
            x *= cp.omega;
            v
        })
        .collect();
    x = cp.g * cp.omega.exp_u64(chunk_offset as u64); // reset
    let weights: Vec<F> = (0..chunk_size)
        .map(|_| {
            let v = w;
            w *= cp.omega_1mn;
            v
        })
        .collect();

    let mut p = String::new();
    p.push_str("from snark_lib import *\n\n");
    p.push_str(&format!("DIGEST_LEN = {DIGEST_LEN}\n"));
    p.push_str(&format!("CHUNK_SIZE = {chunk_size}\n"));
    p.push_str(&format!("N_LEAVES = {chunk_n_leaves}\n"));
    p.push_str(&format!("FELTS_PER_LEAF = {chunk_fpl}\n"));
    p.push_str(&format!("N_CHUNKS_PER_LEAF = {chunk_n_chunks_per_leaf}\n"));
    p.push_str(&format!("TOTAL_TREE_SIZE = {total_tree_size}\n"));
    p.push_str(&format!("LOG_N = {}\n", cp.log_n));
    p.push_str(&format!("G_N = {}\n\n", fmt_f(cp.g_n)));

    p.push_str("def main():\n");

    // Load chunk evaluations
    p.push_str("    V = Array(CHUNK_SIZE)\n");
    p.push_str("    hint_witness(\"evals_chunk\", V)\n\n");

    // Merkle subtree (unrolled)
    let chain_size = chunk_n_leaves * chunk_n_chunks_per_leaf * DIGEST_LEN;
    p.push_str("    zero_vec = Array(DIGEST_LEN)\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        zero_vec[i] = 0\n\n");
    p.push_str(&format!("    chain = Array({chain_size})\n"));
    p.push_str(&format!("    tree = Array(TOTAL_TREE_SIZE)\n\n"));

    for leaf in 0..chunk_n_leaves {
        let ld = leaf * chunk_fpl;
        let ch = leaf * chunk_n_chunks_per_leaf * DIGEST_LEN;
        p.push_str(&format!(
            "    poseidon16_compress(zero_vec, V + {ld}, chain + {ch})\n"
        ));
        for c in 1..chunk_n_chunks_per_leaf {
            p.push_str(&format!(
                "    poseidon16_compress(chain + {}, V + {}, chain + {})\n",
                ch + (c - 1) * DIGEST_LEN,
                ld + c * 8,
                ch + c * DIGEST_LEN,
            ));
        }
        let final_ch = ch + (chunk_n_chunks_per_leaf - 1) * DIGEST_LEN;
        let tree_leaf = leaf * DIGEST_LEN;
        for k in 0..DIGEST_LEN {
            p.push_str(&format!(
                "    tree[{}] = chain[{}]\n",
                tree_leaf + k,
                final_ch + k
            ));
        }
    }
    for layer in 0..chunk_tree_depth {
        let n_pairs = chunk_n_leaves >> (layer + 1);
        let src = layer_offsets[layer];
        let dst = layer_offsets[layer + 1];
        for pair in 0..n_pairs {
            p.push_str(&format!(
                "    poseidon16_compress(tree + {}, tree + {}, tree + {})\n",
                src + 2 * pair * DIGEST_LEN,
                src + (2 * pair + 1) * DIGEST_LEN,
                dst + pair * DIGEST_LEN,
            ));
        }
    }

    // Assert subtree root == public_input[0:8]
    let root_offset = layer_offsets[chunk_tree_depth];
    p.push_str(&format!("\n    root_ptr = tree + {root_offset}\n"));
    p.push_str("    pub_ptr = 0\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        assert root_ptr[i] == pub_ptr[i]\n\n");

    // Read challenges from public_input[8:12]
    // Read expected partial sums from public_input[12:16]

    // 4 syndrome checks — unrolled accumulation over chunk indices
    for check in 0..NUM_SYNDROME_CHECKS {
        let beta_pi_idx = 8 + check;
        let sum_pi_idx = 12 + check;
        p.push_str(&format!("    beta_{check} = pub_ptr[{beta_pi_idx}]\n"));
        p.push_str(&format!("    bn_{check}: Mut = beta_{check}\n"));
        for _ in 0..cp.log_n {
            p.push_str(&format!("    bn_{check} = bn_{check} * bn_{check}\n"));
        }
        // Unrolled accumulation
        let yninv0 = F::ONE / (cp.g_n * signs[0]);
        p.push_str(&format!(
            "    s_{check}_0 = V[0] * {} * (bn_{check} * {} - 1) / (beta_{check} - {})\n",
            fmt_f(weights[0]),
            fmt_f(yninv0),
            fmt_f(coset_points[0])
        ));
        for j in 1..chunk_size {
            let yninv = F::ONE / (cp.g_n * signs[j]);
            p.push_str(&format!(
                "    s_{check}_{j} = s_{check}_{prev} + V[{j}] * {w} * (bn_{check} * {yi} - 1) / (beta_{check} - {xj})\n",
                prev = j - 1,
                w = fmt_f(weights[j]),
                yi = fmt_f(yninv),
                xj = fmt_f(coset_points[j]),
            ));
        }
        p.push_str(&format!(
            "    assert s_{check}_{} == pub_ptr[{sum_pi_idx}]\n\n",
            chunk_size - 1
        ));
    }

    p.push_str("    return\n");
    p
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_n: usize = args
        .iter()
        .position(|a| a == "--log-n")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(10);
    let log_chunk: usize = args
        .iter()
        .position(|a| a == "--log-chunk")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(9); // default: 512 evals per chunk
    let log_blowup = 1usize;
    let log_total = log_n + log_blowup;
    let lfpl = pick_log_felts_per_leaf_kb(log_total);
    let cp = CircuitParams::new(log_n, log_blowup, lfpl);

    let chunk_size = 1usize << log_chunk;
    let n_chunks = cp.n_eval / chunk_size;
    assert!(
        n_chunks * chunk_size == cp.n_eval,
        "n_eval must be divisible by chunk_size"
    );
    assert!(chunk_size >= cp.fpl, "chunk must hold at least one leaf");

    eprintln!("============================================================");
    eprintln!("Chunked parallel proving (syndrome check)");
    eprintln!("============================================================");
    eprintln!("  log_n       = {log_n}  ({} coefficients)", cp.n);
    eprintln!("  log_chunk   = {log_chunk}  ({chunk_size} evals/chunk)");
    eprintln!("  n_chunks    = {n_chunks}");
    eprintln!("  n_eval      = {}", cp.n_eval);
    eprintln!();

    // 1. Compute full evaluations, root, challenges, partial sums
    let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
    let t0 = Instant::now();
    let evals = reference_coset_fft(&cp, &coeffs);
    let root = reference_merkle_root(&cp, &evals);

    // Derive challenges from full root (same as the single-proof version)
    let challenges: Vec<F> = (0..NUM_SYNDROME_CHECKS)
        .map(|i| {
            let mut input = [F::ZERO; 16];
            input[..8].copy_from_slice(&root);
            input[8] = F::from_u32(i as u32);
            poseidon16_compress(input)[0]
        })
        .collect();

    // Compute subtree roots and partial sums per chunk
    let chunk_data: Vec<([F; 8], Vec<F>)> = (0..n_chunks)
        .map(|ci| {
            let chunk_evals = &evals[ci * chunk_size..(ci + 1) * chunk_size];
            let subtree_root = {
                let chunk_cp = CircuitParams::new(
                    log_chunk.saturating_sub(1).max(1),
                    1,
                    lfpl,
                );
                // Compute subtree root directly
                let n_leaves = chunk_size / cp.fpl;
                let mut layer: Vec<[F; 8]> = (0..n_leaves)
                    .map(|i| poseidon_hash_chain(&chunk_evals[i * cp.fpl..(i + 1) * cp.fpl]))
                    .collect();
                while layer.len() > 1 {
                    layer = (0..layer.len() / 2)
                        .map(|i| poseidon_hash_pair(&layer[2 * i], &layer[2 * i + 1]))
                        .collect();
                }
                layer[0]
            };
            let partial_sums: Vec<F> = challenges
                .iter()
                .map(|&beta| {
                    let beta_n = beta.exp_u64(cp.n as u64);
                    let offset = ci * chunk_size;
                    let mut x = cp.g * cp.omega.exp_u64(offset as u64);
                    let mut w = cp.g_1mn * cp.omega_1mn.exp_u64(offset as u64);
                    let mut sign = if offset % 2 == 0 { F::ONE } else { F::ZERO - F::ONE };
                    let mut s = F::ZERO;
                    for j in 0..chunk_size {
                        let yj_n_inv = F::ONE / (cp.g_n * sign);
                        let numer = beta_n * yj_n_inv - F::ONE;
                        let denom = beta - x;
                        s += chunk_evals[j] * w * numer / denom;
                        x *= cp.omega;
                        w *= cp.omega_1mn;
                        sign = F::ZERO - sign;
                    }
                    s
                })
                .collect();
            (subtree_root, partial_sums)
        })
        .collect();
    let ref_time = t0.elapsed();
    eprintln!("[1/4] Reference: {:.3}s", ref_time.as_secs_f64());

    // Verify partial sums add to zero
    for check in 0..NUM_SYNDROME_CHECKS {
        let total: F = chunk_data.iter().map(|(_, sums)| sums[check]).sum();
        assert_eq!(total, F::ZERO, "partial syndrome sums don't add to zero for check {check}");
    }

    // 2. Generate + compile one chunk program (all chunks use same structure)
    let t0 = Instant::now();
    let program = generate_chunk_program(&cp, 0, chunk_size);
    eprintln!(
        "[2/4] Program: {} lines, {:.1} KB",
        program.lines().count(),
        program.len() as f64 / 1024.0
    );

    // Actually each chunk has different constants (coset points differ), so compile per chunk
    let programs: Vec<String> = (0..n_chunks)
        .map(|ci| generate_chunk_program(&cp, ci, chunk_size))
        .collect();
    let bytecodes: Vec<_> = programs
        .iter()
        .map(|prog| compile_program(&ProgramSource::Raw(prog.clone())))
        .collect();
    let compile_time = t0.elapsed();
    eprintln!("  Compiled {n_chunks} chunks in {:.3}s", compile_time.as_secs_f64());

    // 3. Prove all chunks
    eprintln!("[3/4] Proving {n_chunks} chunks...");
    let t0 = Instant::now();
    let proofs: Vec<_> = (0..n_chunks)
        .map(|ci| {
            let chunk_evals = evals[ci * chunk_size..(ci + 1) * chunk_size].to_vec();
            let (subtree_root, partial_sums) = &chunk_data[ci];

            // Build public input: [subtree_root(8), betas(4), partial_sums(4)]
            let mut pi = subtree_root.to_vec();
            pi.extend_from_slice(&challenges);
            pi.extend_from_slice(partial_sums);
            pi.resize(pi.len().next_power_of_two(), F::ZERO);

            let mut hints = HashMap::new();
            hints.insert("evals_chunk".to_string(), vec![chunk_evals]);

            let witness = ExecutionWitness {
                preamble_memory_len: 0,
                hints,
            };

            prove_execution(
                &bytecodes[ci],
                &pi,
                &witness,
                &default_whir_config(1),
                false,
            )
            .expect(&format!("chunk {ci} should prove"))
        })
        .collect();
    let prove_time = t0.elapsed();

    let total_cycles: usize = proofs.iter().map(|p| p.metadata.cycles).sum();
    let total_poseidons: usize = proofs.iter().map(|p| p.metadata.n_poseidons).sum();

    eprintln!(
        "  Total: {:.3}s ({:.3}s/chunk), {} cycles, {} poseidons",
        prove_time.as_secs_f64(),
        prove_time.as_secs_f64() / n_chunks as f64,
        total_cycles,
        total_poseidons,
    );

    // 4. Verify all chunks
    eprintln!("[4/4] Verifying {n_chunks} chunks...");
    let t0 = Instant::now();
    for ci in 0..n_chunks {
        let (subtree_root, partial_sums) = &chunk_data[ci];
        let mut pi = subtree_root.to_vec();
        pi.extend_from_slice(&challenges);
        pi.extend_from_slice(partial_sums);
        pi.resize(pi.len().next_power_of_two(), F::ZERO);
        verify_execution(&bytecodes[ci], &pi, proofs[ci].proof.clone())
            .expect(&format!("chunk {ci} should verify"));
    }
    let verify_time = t0.elapsed();
    eprintln!("  Verified in {:.3}s", verify_time.as_secs_f64());

    let peak_rss = system_info::peak_rss_bytes();

    eprintln!();
    eprintln!("------------------------------------------------------------");
    eprintln!("RESULTS");
    eprintln!("------------------------------------------------------------");
    eprintln!("  Chunks          : {n_chunks}");
    eprintln!("  Compile time    : {:.3}s", compile_time.as_secs_f64());
    eprintln!("  Prove time      : {:.3}s (sequential)", prove_time.as_secs_f64());
    eprintln!("  Per-chunk prove : {:.3}s", prove_time.as_secs_f64() / n_chunks as f64);
    eprintln!("  Verify time     : {:.3}s", verify_time.as_secs_f64());
    eprintln!("  Total cycles    : {total_cycles}");
    eprintln!("  Peak RSS        : {:.2} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("------------------------------------------------------------");

    println!(
        "{}",
        serde_json::json!({
            "variant": "chunked",
            "log_n": log_n,
            "log_chunk": log_chunk,
            "n_chunks": n_chunks,
            "compile_s": (compile_time.as_secs_f64() * 1000.0).round() / 1000.0,
            "prove_s": (prove_time.as_secs_f64() * 1000.0).round() / 1000.0,
            "per_chunk_s": (prove_time.as_secs_f64() / n_chunks as f64 * 1000.0).round() / 1000.0,
            "verify_s": (verify_time.as_secs_f64() * 1000.0).round() / 1000.0,
            "total_cycles": total_cycles,
            "total_poseidons": total_poseidons,
            "peak_rss": peak_rss,
        })
    );
}
