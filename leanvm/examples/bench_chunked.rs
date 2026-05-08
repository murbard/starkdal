//! Approach 2c: Chunked parallel syndrome proving with shared bytecode.
//!
//! Splits one large codeword into chunks, proves each chunk independently
//! (shared bytecode, per-chunk coset offsets via public input), then
//! verifies partial syndrome sums add to zero.
//!
//! Usage: cargo run --release --example bench_chunked -- --log-n 22 --log-chunk 12 --concurrency 8

use rayon::prelude::*;
use starkdal_leanvm::*;
use std::collections::HashMap;
use std::time::Instant;

fn generate_shared_chunk_program(cp: &CircuitParams, chunk_size: usize) -> String {
    assert!(cp.log_blowup == 1);
    let chunk_fpl = cp.fpl;
    let chunk_n_leaves = chunk_size / chunk_fpl;
    let chunk_tree_depth = chunk_n_leaves.trailing_zeros() as usize;
    let chunk_n_chunks_per_leaf = chunk_fpl / 8;

    let mut layer_offsets = vec![0usize];
    let mut acc_off = 0;
    for k in 0..chunk_tree_depth {
        acc_off += (chunk_n_leaves >> k) * DIGEST_LEN;
        layer_offsets.push(acc_off);
    }
    let total_tree_size = acc_off + DIGEST_LEN;

    let mut p = String::new();
    p.push_str("from snark_lib import *\n\n");
    p.push_str(&format!("DIGEST_LEN = {DIGEST_LEN}\n"));
    p.push_str(&format!("CHUNK_SIZE = {chunk_size}\n"));
    p.push_str(&format!("N_LEAVES = {chunk_n_leaves}\n"));
    p.push_str(&format!("FELTS_PER_LEAF = {chunk_fpl}\n"));
    p.push_str(&format!("N_CHUNKS_PER_LEAF = {chunk_n_chunks_per_leaf}\n"));
    p.push_str(&format!("TOTAL_TREE_SIZE = {total_tree_size}\n"));
    p.push_str(&format!("LOG_N = {}\n", cp.log_n));
    p.push_str(&format!("G_N = {}\n", fmt_f(cp.g_n)));
    p.push_str(&format!("OMEGA = {}\n", fmt_f(cp.omega)));
    p.push_str(&format!("OMEGA_1MN = {}\n\n", fmt_f(cp.omega_1mn)));

    p.push_str("def main():\n");
    p.push_str("    V = Array(CHUNK_SIZE)\n");
    p.push_str("    hint_witness(\"evals_chunk\", V)\n\n");

    // Merkle subtree (unrolled — same structure for all chunks)
    let chain_size = chunk_n_leaves * chunk_n_chunks_per_leaf * DIGEST_LEN;
    p.push_str("    zero_vec = Array(DIGEST_LEN)\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        zero_vec[i] = 0\n\n");
    p.push_str(&format!("    chain = Array({chain_size})\n"));
    p.push_str(&format!("    tree = Array(TOTAL_TREE_SIZE)\n\n"));

    for leaf in 0..chunk_n_leaves {
        let ld = leaf * chunk_fpl;
        let ch = leaf * chunk_n_chunks_per_leaf * DIGEST_LEN;
        p.push_str(&format!("    poseidon16_compress(zero_vec, V + {ld}, chain + {ch})\n"));
        for c in 1..chunk_n_chunks_per_leaf {
            p.push_str(&format!(
                "    poseidon16_compress(chain + {}, V + {}, chain + {})\n",
                ch + (c - 1) * DIGEST_LEN, ld + c * 8, ch + c * DIGEST_LEN,
            ));
        }
        let final_ch = ch + (chunk_n_chunks_per_leaf - 1) * DIGEST_LEN;
        let tree_leaf = leaf * DIGEST_LEN;
        for k in 0..DIGEST_LEN {
            p.push_str(&format!("    tree[{}] = chain[{}]\n", tree_leaf + k, final_ch + k));
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

    let root_offset = layer_offsets[chunk_tree_depth];
    p.push_str(&format!("\n    root_ptr = tree + {root_offset}\n"));
    p.push_str("    pub_ptr = 0\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        assert root_ptr[i] == pub_ptr[i]\n\n");

    // PI layout: [root(8), betas(4), sums(4), x_start, w_start, sign_start]
    p.push_str("    x_start = pub_ptr[16]\n");
    p.push_str("    w_start = pub_ptr[17]\n");
    p.push_str("    sign_start = pub_ptr[18]\n\n");

    for i in 0..NUM_SYNDROME_CHECKS {
        let beta_idx = 8 + i;
        let sum_idx = 12 + i;
        p.push_str(&format!("    beta_{i} = pub_ptr[{beta_idx}]\n"));
        p.push_str(&format!("    bn_{i}: Mut = beta_{i}\n"));
        for _ in 0..cp.log_n {
            p.push_str(&format!("    bn_{i} = bn_{i} * bn_{i}\n"));
        }
        p.push_str(&format!("    x_{i}: Mut = x_start\n"));
        p.push_str(&format!("    w_{i}: Mut = w_start\n"));
        p.push_str(&format!("    sign_{i}: Mut = sign_start\n"));
        p.push_str(&format!("    s_{i}: Mut = 0\n"));
        p.push_str(&format!("    for j_{i} in range(0, CHUNK_SIZE):\n"));
        p.push_str(&format!("        yj_n_inv_{i} = 1 / (G_N * sign_{i})\n"));
        p.push_str(&format!("        numer_{i} = bn_{i} * yj_n_inv_{i} - 1\n"));
        p.push_str(&format!("        denom_{i} = beta_{i} - x_{i}\n"));
        p.push_str(&format!("        s_{i} = s_{i} + V[j_{i}] * w_{i} * numer_{i} / denom_{i}\n"));
        p.push_str(&format!("        x_{i} = x_{i} * OMEGA\n"));
        p.push_str(&format!("        w_{i} = w_{i} * OMEGA_1MN\n"));
        p.push_str(&format!("        sign_{i} = 0 - sign_{i}\n"));
        p.push_str(&format!("    assert s_{i} == pub_ptr[{sum_idx}]\n\n"));
    }

    p.push_str("    return\n");
    p
}

struct ChunkData {
    subtree_root: [F; 8],
    partial_sums: Vec<F>,
    x_start: F,
    w_start: F,
    sign_start: F,
    evals: Vec<F>,
}

fn make_chunk_pi(chunk: &ChunkData, challenges: &[F]) -> Vec<F> {
    let mut pi: Vec<F> = chunk.subtree_root.to_vec();
    pi.extend_from_slice(challenges);
    pi.extend_from_slice(&chunk.partial_sums);
    pi.push(chunk.x_start);
    pi.push(chunk.w_start);
    pi.push(chunk.sign_start);
    pi.resize(pi.len().next_power_of_two(), F::ZERO);
    pi
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_n: usize = args.iter().position(|a| a == "--log-n")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(22);
    let log_chunk: usize = args.iter().position(|a| a == "--log-chunk")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(12);
    let concurrency: usize = args.iter().position(|a| a == "--concurrency")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(0); // 0 = auto (rayon default)
    let log_blowup = 1usize;
    let log_total = log_n + log_blowup;
    let lfpl = pick_log_felts_per_leaf_kb(log_total);
    let cp = CircuitParams::new(log_n, log_blowup, lfpl);

    let chunk_size = 1usize << log_chunk;
    let n_chunks = cp.n_eval / chunk_size;
    assert!(n_chunks * chunk_size == cp.n_eval);
    assert!(chunk_size >= cp.fpl);
    let payload_mb = (cp.n * 4) as f64 / (1024.0 * 1024.0);
    let n_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    eprintln!("============================================================");
    eprintln!("Chunked parallel proving");
    eprintln!("============================================================");
    eprintln!("  payload     = {:.1} MB ({} coefficients)", payload_mb, cp.n);
    eprintln!("  n_eval      = {}", cp.n_eval);
    eprintln!("  chunk_size  = {chunk_size} (log_chunk={log_chunk})");
    eprintln!("  n_chunks    = {n_chunks}");
    eprintln!("  cores       = {n_cores}");
    eprintln!("  concurrency = {}", if concurrency == 0 { "auto".to_string() } else { concurrency.to_string() });
    eprintln!();

    // 1. Reference computation
    let t0 = Instant::now();
    let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
    let evals = reference_coset_fft(&cp, &coeffs);
    let full_root = reference_merkle_root(&cp, &evals);
    let challenges: Vec<F> = (0..NUM_SYNDROME_CHECKS)
        .map(|i| {
            let mut input = [F::ZERO; 16];
            input[..8].copy_from_slice(&full_root);
            input[8] = F::from_u32(i as u32);
            poseidon16_compress(input)[0]
        })
        .collect();

    let chunks: Vec<ChunkData> = (0..n_chunks)
        .into_par_iter()
        .map(|ci| {
            let offset = ci * chunk_size;
            let chunk_evals = evals[offset..offset + chunk_size].to_vec();
            let n_leaves = chunk_size / cp.fpl;
            let mut layer: Vec<[F; 8]> = (0..n_leaves)
                .map(|i| poseidon_hash_chain(&chunk_evals[i * cp.fpl..(i + 1) * cp.fpl]))
                .collect();
            while layer.len() > 1 {
                layer = (0..layer.len() / 2)
                    .map(|i| poseidon_hash_pair(&layer[2 * i], &layer[2 * i + 1]))
                    .collect();
            }
            let subtree_root = layer[0];
            let x_start = cp.g * cp.omega.exp_u64(offset as u64);
            let w_start = cp.g_1mn * cp.omega_1mn.exp_u64(offset as u64);
            let sign_start = if offset % 2 == 0 { F::ONE } else { F::ZERO - F::ONE };
            let partial_sums: Vec<F> = challenges
                .iter()
                .map(|&beta| {
                    let beta_n = beta.exp_u64(cp.n as u64);
                    let mut x = x_start;
                    let mut w = w_start;
                    let mut sign = sign_start;
                    let mut s = F::ZERO;
                    for j in 0..chunk_size {
                        let yj_n_inv = F::ONE / (cp.g_n * sign);
                        s += chunk_evals[j] * w * (beta_n * yj_n_inv - F::ONE) / (beta - x);
                        x *= cp.omega;
                        w *= cp.omega_1mn;
                        sign = F::ZERO - sign;
                    }
                    s
                })
                .collect();
            ChunkData { subtree_root, partial_sums, x_start, w_start, sign_start, evals: chunk_evals }
        })
        .collect();
    let ref_time = t0.elapsed();

    for check in 0..NUM_SYNDROME_CHECKS {
        let total: F = chunks.iter().map(|c| c.partial_sums[check]).sum();
        assert_eq!(total, F::ZERO, "partial sums don't add to zero for check {check}");
    }
    eprintln!("[1/4] Reference: {:.3}s", ref_time.as_secs_f64());

    // 2. Compile once
    let t0 = Instant::now();
    let program = generate_shared_chunk_program(&cp, chunk_size);
    let bytecode = compile_program(&ProgramSource::Raw(program.clone()));
    let compile_time = t0.elapsed();
    eprintln!("[2/4] Compiled: {} lines, {:.3}s", program.lines().count(), compile_time.as_secs_f64());

    // 3. Prove chunks with controlled concurrency
    let batch_size = if concurrency == 0 { n_chunks } else { concurrency };
    eprintln!("[3/4] Proving {n_chunks} chunks (batch_size={batch_size})...");
    let t0 = Instant::now();
    let mut proofs: Vec<ExecutionProof> = Vec::with_capacity(n_chunks);
    for batch_start in (0..n_chunks).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n_chunks);
        let batch_proofs: Vec<_> = (batch_start..batch_end)
            .into_par_iter()
            .map(|ci| {
                let pi = make_chunk_pi(&chunks[ci], &challenges);
                let mut hints = HashMap::new();
                hints.insert("evals_chunk".to_string(), vec![chunks[ci].evals.clone()]);
                prove_execution(
                    &bytecode, &pi,
                    &ExecutionWitness { preamble_memory_len: 0, hints },
                    &default_whir_config(1), false,
                ).unwrap_or_else(|e| panic!("chunk {ci} failed: {e}"))
            })
            .collect();
        proofs.extend(batch_proofs);
    }
    let prove_time = t0.elapsed();

    let total_cycles: usize = proofs.iter().map(|p| p.metadata.cycles).sum();
    let total_poseidons: usize = proofs.iter().map(|p| p.metadata.n_poseidons).sum();
    eprintln!("  {:.3}s wall, {} total cycles", prove_time.as_secs_f64(), total_cycles);

    // 4. Verify (sample a few to save time at large n)
    let n_verify = n_chunks.min(64);
    eprintln!("[4/4] Verifying {n_verify}/{n_chunks} chunks...");
    let t0 = Instant::now();
    for ci in 0..n_verify {
        let pi = make_chunk_pi(&chunks[ci], &challenges);
        verify_execution(&bytecode, &pi, proofs[ci].proof.clone())
            .unwrap_or_else(|e| panic!("chunk {ci} verify: {e}"));
    }
    let verify_time = t0.elapsed();
    eprintln!("  {:.3}s", verify_time.as_secs_f64());

    let peak_rss = system_info::peak_rss_bytes();

    eprintln!();
    eprintln!("------------------------------------------------------------");
    eprintln!("  Payload         : {:.1} MB", payload_mb);
    eprintln!("  Compile (once)  : {:.3}s", compile_time.as_secs_f64());
    eprintln!("  Prove (parallel): {:.3}s on {} cores", prove_time.as_secs_f64(), n_cores);
    eprintln!("  Peak RSS        : {:.1} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("  Total cycles    : {total_cycles}");
    eprintln!("------------------------------------------------------------");

    println!("{}", serde_json::json!({
        "payload_mb": (payload_mb * 10.0).round() / 10.0,
        "log_n": log_n, "log_chunk": log_chunk, "n_chunks": n_chunks,
        "cores": n_cores,
        "compile_s": (compile_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "prove_s": (prove_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "ref_s": (ref_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "total_cycles": total_cycles, "total_poseidons": total_poseidons,
        "peak_rss": peak_rss,
    }));
}
