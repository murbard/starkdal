//! Full recursive pipeline: leaf proofs → aggregation → 1 proof.
//!
//! Target: 16 MB payload (log_n=22) on a single machine.
//! Strategy: n=12 leaf proofs (unrolled syndrome), recursive aggregation via recursion.py.
//!
//! Usage: cargo run --release --example bench_recursive -- --log-n 22 --arity 8

use rayon::prelude::*;
use rec_aggregation::{init_dal_aggregation_bytecode, get_dal_aggregation_bytecode};
use starkdal_leanvm::*;
use std::collections::HashMap;
use std::time::Instant;

const LEAF_LOG_N: usize = 12; // fixed leaf size: 4096 coefficients = 16 KB

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_n: usize = args.iter().position(|a| a == "--log-n")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(22);
    let arity: usize = args.iter().position(|a| a == "--arity")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(8);

    let log_blowup = 1usize;
    let log_total = log_n + log_blowup;
    let lfpl = pick_log_felts_per_leaf_kb(log_total);
    let cp = CircuitParams::new(log_n, log_blowup, lfpl);

    // Leaf parameters
    let leaf_cp = CircuitParams::new(LEAF_LOG_N, log_blowup, pick_log_felts_per_leaf_kb(LEAF_LOG_N + log_blowup));
    let leaf_n_eval = leaf_cp.n_eval;
    let n_leaves = cp.n_eval / leaf_n_eval;
    let payload_mb = (cp.n * 4) as f64 / (1024.0 * 1024.0);
    let n_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    eprintln!("============================================================");
    eprintln!("Recursive proving pipeline");
    eprintln!("============================================================");
    eprintln!("  payload     = {:.1} MB (log_n={log_n})", payload_mb);
    eprintln!("  leaf_size   = {} evals (log={LEAF_LOG_N})", leaf_n_eval);
    eprintln!("  n_leaves    = {n_leaves}");
    eprintln!("  arity       = {arity}");
    eprintln!("  cores       = {n_cores}");

    // Compute recursion levels
    let mut level_sizes = vec![n_leaves];
    while *level_sizes.last().unwrap() > 1 {
        let prev = *level_sizes.last().unwrap();
        level_sizes.push((prev + arity - 1) / arity);
    }
    eprintln!("  levels      = {} ({:?})", level_sizes.len(), level_sizes);
    eprintln!();

    // Step 1: Reference computation
    eprintln!("[1] Computing reference (FFT + Merkle)...");
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
    eprintln!("    {:.3}s", t0.elapsed().as_secs_f64());

    // Step 2: Compile leaf bytecode
    eprintln!("[2] Compiling leaf bytecode (n=12 unrolled syndrome)...");
    let t0 = Instant::now();
    // Generate leaf program for chunk 0 (all chunks share same structure but different constants)
    // For unrolled syndrome, each chunk needs its own program. Generate + compile all.
    let leaf_programs: Vec<String> = (0..n_leaves)
        .map(|ci| generate_leaf_program(&cp, &leaf_cp, ci, &challenges))
        .collect();
    let leaf_bytecodes: Vec<_> = leaf_programs.iter()
        .map(|prog| compile_program(&ProgramSource::Raw(prog.clone())))
        .collect();
    eprintln!("    {:.3}s ({} programs)", t0.elapsed().as_secs_f64(), n_leaves);

    // Step 3: Compile aggregation bytecode
    eprintln!("[3] Compiling aggregation bytecode (arity={arity})...");
    let t0 = Instant::now();
    init_dal_aggregation_bytecode(arity);
    let _agg_bytecode = get_dal_aggregation_bytecode();
    eprintln!("    {:.3}s", t0.elapsed().as_secs_f64());

    // Step 4: Prove all leaves
    eprintln!("[4] Proving {n_leaves} leaf proofs...");
    let t0 = Instant::now();
    let leaf_proofs: Vec<_> = (0..n_leaves)
        .into_par_iter()
        .map(|ci| {
            let offset = ci * leaf_n_eval;
            let chunk_evals = evals[offset..offset + leaf_n_eval].to_vec();

            // Compute subtree root and partial sums
            let n_sub_leaves = leaf_n_eval / leaf_cp.fpl;
            let mut layer: Vec<[F; 8]> = (0..n_sub_leaves)
                .map(|i| poseidon_hash_chain(&chunk_evals[i * leaf_cp.fpl..(i + 1) * leaf_cp.fpl]))
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

            let partial_sums: Vec<F> = challenges.iter().map(|&beta| {
                let beta_n = beta.exp_u64(cp.n as u64);
                let mut x = x_start;
                let mut w = w_start;
                let mut sign = sign_start;
                let mut s = F::ZERO;
                for j in 0..leaf_n_eval {
                    let yj_n_inv = F::ONE / (cp.g_n * sign);
                    s += chunk_evals[j] * w * (beta_n * yj_n_inv - F::ONE) / (beta - x);
                    x *= cp.omega;
                    w *= cp.omega_1mn;
                    sign = F::ZERO - sign;
                }
                s
            }).collect();

            // Build PI: [subtree_root(8), betas(4), sums(4), x_start, w_start, sign_start]
            let mut pi: Vec<F> = subtree_root.to_vec();
            pi.extend_from_slice(&challenges);
            pi.extend_from_slice(&partial_sums);
            pi.push(x_start);
            pi.push(w_start);
            pi.push(sign_start);
            pi.resize(pi.len().next_power_of_two(), F::ZERO);

            let mut hints = HashMap::new();
            hints.insert("evals_chunk".to_string(), vec![chunk_evals]);

            let proof = prove_execution(
                &leaf_bytecodes[ci], &pi,
                &ExecutionWitness { preamble_memory_len: 0, hints },
                &default_whir_config(1), false,
            ).unwrap_or_else(|e| panic!("leaf {ci} failed: {e}"));

            (proof, subtree_root, partial_sums)
        })
        .collect();
    let leaf_time = t0.elapsed();
    let leaf_cycles: usize = leaf_proofs.iter().map(|(p, _, _)| p.metadata.cycles).sum();
    eprintln!("    {:.3}s wall, {} total cycles", leaf_time.as_secs_f64(), leaf_cycles);

    // Verify syndrome sums
    for check in 0..NUM_SYNDROME_CHECKS {
        let total: F = leaf_proofs.iter().map(|(_, _, sums)| sums[check]).sum();
        assert_eq!(total, F::ZERO, "syndrome sums don't add to zero for check {check}");
    }

    let peak_rss = system_info::peak_rss_bytes();
    let total_time = leaf_time; // TODO: add aggregation time when implemented

    eprintln!();
    eprintln!("------------------------------------------------------------");
    eprintln!("  Payload         : {:.1} MB", payload_mb);
    eprintln!("  Leaf proofs     : {n_leaves} × {:.3}s = {:.3}s wall",
        leaf_time.as_secs_f64() / n_leaves as f64, leaf_time.as_secs_f64());
    eprintln!("  Aggregation     : TODO (recursion circuit pending)");
    eprintln!("  Total           : {:.3}s", total_time.as_secs_f64());
    eprintln!("  Peak RSS        : {:.1} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("------------------------------------------------------------");

    println!("{}", serde_json::json!({
        "payload_mb": (payload_mb * 10.0).round() / 10.0,
        "log_n": log_n, "arity": arity, "n_leaves": n_leaves,
        "leaf_time_s": (leaf_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "leaf_cycles": leaf_cycles,
        "peak_rss": peak_rss,
    }));
}

/// Generate an unrolled syndrome-check program for one leaf chunk.
fn generate_leaf_program(
    full_cp: &CircuitParams,
    leaf_cp: &CircuitParams,
    chunk_idx: usize,
    challenges: &[F],
) -> String {
    // This is essentially bench_syndrome_unroll's generate_program but for a chunk
    let chunk_offset = chunk_idx * leaf_cp.n_eval;
    let chunk_size = leaf_cp.n_eval;

    // Precompute per-j constants for this chunk
    let mut x = full_cp.g * full_cp.omega.exp_u64(chunk_offset as u64);
    let coset_points: Vec<F> = (0..chunk_size).map(|_| { let v = x; x *= full_cp.omega; v }).collect();
    let mut w = full_cp.g_1mn * full_cp.omega_1mn.exp_u64(chunk_offset as u64);
    let weights: Vec<F> = (0..chunk_size).map(|_| { let v = w; w *= full_cp.omega_1mn; v }).collect();
    let signs: Vec<F> = (0..chunk_size).map(|j| {
        if (chunk_offset + j) % 2 == 0 { F::ONE } else { F::ZERO - F::ONE }
    }).collect();

    // Use the syndrome_unroll generator pattern from bench_syndrome_unroll
    // but adapted for the leaf's own CircuitParams
    let mut p = String::new();
    p.push_str("from snark_lib import *\n\n");

    // Emit leaf Merkle tree constants (using leaf_cp for tree structure)
    // But override G_N with the full polynomial's g^n (not the leaf's)
    p.push_str(&format!("DIGEST_LEN = {DIGEST_LEN}\n"));
    p.push_str(&format!("N = {}\n", leaf_cp.n));
    p.push_str(&format!("N_EVAL = {}\n", leaf_cp.n_eval));
    p.push_str(&format!("N_LEAVES = {}\n", leaf_cp.n_leaves));
    p.push_str(&format!("FELTS_PER_LEAF = {}\n", leaf_cp.fpl));
    p.push_str(&format!("N_CHUNKS_PER_LEAF = {}\n", leaf_cp.n_chunks_per_leaf));
    p.push_str(&format!("TOTAL_TREE_SIZE = {}\n", leaf_cp.total_tree_size));
    p.push_str(&format!("LOG_N = {}\n", full_cp.log_n)); // full polynomial degree
    p.push_str(&format!("G_N = {}\n", fmt_f(full_cp.g_n))); // g^n for full polynomial
    p.push_str(&format!("OMEGA = {}\n", fmt_f(leaf_cp.omega)));
    p.push_str(&format!("OMEGA_1MN = {}\n\n", fmt_f(leaf_cp.omega_1mn)));

    p.push_str("def main():\n");
    p.push_str("    V = Array(N_EVAL)\n");
    p.push_str("    hint_witness(\"evals_chunk\", V)\n\n");

    // Merkle tree (unrolled)
    emit_merkle_tree(&mut p, leaf_cp, "V");
    emit_root_assert_and_challenges(&mut p, leaf_cp);

    // PI: [root(8), betas(4), sums(4), x_start, w_start, sign_start]
    // pub_ptr already defined by emit_root_assert_and_challenges

    // 4 syndrome checks — unrolled
    for check in 0..NUM_SYNDROME_CHECKS {
        let beta_idx = 8 + check;
        let sum_idx = 12 + check;
        p.push_str(&format!("    beta_{check} = pub_ptr[{beta_idx}]\n"));
        p.push_str(&format!("    bn_{check}: Mut = beta_{check}\n"));
        for _ in 0..full_cp.log_n {
            p.push_str(&format!("    bn_{check} = bn_{check} * bn_{check}\n"));
        }
        let yninv0 = F::ONE / (full_cp.g_n * signs[0]);
        p.push_str(&format!(
            "    s_{check}_0 = V[0] * {} * (bn_{check} * {} - 1) / (beta_{check} - {})\n",
            fmt_f(weights[0]), fmt_f(yninv0), fmt_f(coset_points[0])
        ));
        for j in 1..chunk_size {
            let yninv = F::ONE / (full_cp.g_n * signs[j]);
            p.push_str(&format!(
                "    s_{check}_{j} = s_{check}_{prev} + V[{j}] * {w} * (bn_{check} * {yi} - 1) / (beta_{check} - {xj})\n",
                prev = j - 1, w = fmt_f(weights[j]), yi = fmt_f(yninv), xj = fmt_f(coset_points[j]),
            ));
        }
        p.push_str(&format!("    assert s_{check}_{} == pub_ptr[{sum_idx}]\n\n", chunk_size - 1));
    }

    p.push_str("    return\n");
    p
}
