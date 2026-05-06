//! Full recursive pipeline: leaf proofs → aggregation → 1 proof.
//!
//! Target: 16 MB payload (log_n=22) on a single machine.
//! Strategy: n=12 leaf proofs (unrolled syndrome), recursive aggregation via recursion.py.
//!
//! Usage: cargo run --release --example bench_recursive -- --log-n 22 --arity 8

use rayon::prelude::*;
use backend::{
    Evaluation, MleGroupOwned, MultilinearPoint, VerifierState,
    dot_product, eval_eq_packed_scaled, flatten_scalars_to_base,
    sumcheck_prove, sumcheck_verify, ProductComputation, RawProof,
};
use rec_aggregation::{
    init_dal_aggregation_bytecode, get_dal_aggregation_bytecode,
    extract_bytecode_claim_from_input_data, hash_bytecode_claims,
};
use starkdal_leanvm::*;
use lean_prover::SNARK_DOMAIN_SEP;
use lean_vm::{EF, DIMENSION, N_INSTRUCTION_COLUMNS, DIGEST_LEN as VM_DIGEST_LEN};
use utils::{build_prover_state, get_poseidon16, poseidon16_compress_pair};
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

    // Step 2: Compile ONE shared leaf bytecode (all chunks use same program)
    eprintln!("[2] Compiling shared leaf bytecode...");
    let t0 = Instant::now();
    let leaf_program = generate_shared_leaf_program(&cp, &leaf_cp);
    let leaf_bytecode = compile_program(&ProgramSource::Raw(leaf_program.clone()));
    eprintln!("    {:.3}s ({} lines)", t0.elapsed().as_secs_f64(), leaf_program.lines().count());

    // Step 3: Compile aggregation bytecode
    eprintln!("[3] Compiling aggregation bytecode (arity={arity})...");
    let t0 = Instant::now();
    init_dal_aggregation_bytecode(arity, leaf_bytecode.log_size());
    let _agg_bytecode = get_dal_aggregation_bytecode();
    eprintln!("    {:.3}s", t0.elapsed().as_secs_f64());

    // Step 4: Prove all leaves
    eprintln!("[4] Proving {n_leaves} leaf proofs...");
    let t0 = Instant::now();
    // Batch leaf proving with controlled concurrency — unbounded par_iter
    // thrashes cache when the WHIR prover's internal rayon competes with outer parallelism.
    let concurrency = 8; // optimal from bench_chunked sweep on G4
    let mut leaf_proofs = Vec::with_capacity(n_leaves);
    for batch_start in (0..n_leaves).step_by(concurrency) {
        let batch_end = (batch_start + concurrency).min(n_leaves);
        let batch: Vec<_> = (batch_start..batch_end).into_par_iter().map(|ci| {
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

            // Build leaf_data: [subtree_root(8), betas(4), sums(4), x_start, w_start, sign_start]
            let mut leaf_data: Vec<F> = subtree_root.to_vec();
            leaf_data.extend_from_slice(&challenges);
            leaf_data.extend_from_slice(&partial_sums);
            leaf_data.push(x_start);
            leaf_data.push(w_start);
            leaf_data.push(sign_start);
            leaf_data.resize(24, F::ZERO); // pad to 24 (3 chunks of 8)

            // PI = hash(leaf_data) = 8 FE
            let pi_hash = hash_leaf_data(&leaf_data);
            let mut pi = pi_hash.to_vec();
            pi.resize(pi.len().next_power_of_two(), F::ZERO);

            let mut hints = HashMap::new();
            hints.insert("leaf_data".to_string(), vec![leaf_data.clone()]);
            hints.insert("evals_chunk".to_string(), vec![chunk_evals]);

            let proof = prove_execution(
                &leaf_bytecode, &pi,
                &ExecutionWitness { preamble_memory_len: 0, hints },
                &default_whir_config(1), false,
            ).unwrap_or_else(|e| panic!("leaf {ci} failed: {e}"));

            (proof, subtree_root, partial_sums, leaf_data)
        }).collect();
        leaf_proofs.extend(batch);
    }
    let leaf_time = t0.elapsed();
    let leaf_cycles: usize = leaf_proofs.iter().map(|(p, _, _, _)| p.metadata.cycles).sum();
    eprintln!("    {:.3}s wall, {} total cycles", leaf_time.as_secs_f64(), leaf_cycles);

    // Verify syndrome sums
    for check in 0..NUM_SYNDROME_CHECKS {
        let total: F = leaf_proofs.iter().map(|(_, _, sums, _)| sums[check]).sum();
        assert_eq!(total, F::ZERO, "syndrome sums don't add to zero for check {check}");
    }

    // Step 5: Aggregation — verify leaf proofs and feed to aggregation circuit
    eprintln!("[5] Aggregating {} leaf proofs (arity={arity})...", n_leaves);
    let t0 = Instant::now();

    let agg_bytecode = get_dal_aggregation_bytecode();

    // Process leaf proofs in groups of `arity`
    let mut current_proofs: Vec<_> = leaf_proofs.iter().map(|(p, sr, ps, ld)| {
        (p.proof.clone(), *sr, ps.clone(), ld.clone())
    }).collect();

    let mut agg_level = 0;
    while current_proofs.len() > 1 {
        agg_level += 1;
        let n_groups = (current_proofs.len() + arity - 1) / arity;
        eprintln!("    Level {agg_level}: {} proofs → {} groups of {arity}", current_proofs.len(), n_groups);

        // Phase 1: Prepare all group hints (verify children, build hints) — sequential, fast
        let preamble_len = 61;
        let leaf_bytecode_point_n_vars = leaf_bytecode.log_size() + log2_ceil_usize(N_INSTRUCTION_COLUMNS);
        let bytecode_claim_size = (leaf_bytecode_point_n_vars + 1) * DIMENSION;
        let bytecode_claim_size_padded = bytecode_claim_size.next_multiple_of(VM_DIGEST_LEN);
        let bytecode_hash_domsep = poseidon16_compress_pair(&leaf_bytecode.hash, &SNARK_DOMAIN_SEP);

        struct AggJob {
            pi: Vec<F>,
            hints: HashMap<String, Vec<Vec<F>>>,
        }

        let mut jobs: Vec<AggJob> = Vec::with_capacity(n_groups);
        for group_idx in 0..n_groups {
            let start = group_idx * arity;
            let end = (start + arity).min(current_proofs.len());
            let group = &current_proofs[start..end];

            // Verify first child, extract RawProof
            let (proof, _, _, leaf_data) = &group[0];
            let pi_hash = hash_leaf_data(leaf_data);
            let mut child_pi = pi_hash.to_vec();
            child_pi.resize(child_pi.len().next_power_of_two(), F::ZERO);
            let (details, raw_proof) = verify_execution(&leaf_bytecode, &child_pi, proof.clone())
                .unwrap_or_else(|e| panic!("agg group {group_idx} verify: {e}"));

            let mut inner_claim = vec![F::ZERO; bytecode_claim_size_padded];
            inner_claim[leaf_bytecode_point_n_vars * DIMENSION] = leaf_bytecode.instructions_multilinear[0];

            // Build input_data
            let mut input_data: Vec<F> = full_root.to_vec();
            input_data.extend_from_slice(&challenges);
            input_data.push(F::from_u32(group.len() as u32));
            input_data.resize(DIGEST_LEN + 4 + 1 + bytecode_claim_size_padded, F::ZERO);
            // Write bytecode_claim_output (default for leaf level)
            input_data[DIGEST_LEN + 4 + 1 + leaf_bytecode_point_n_vars * DIMENSION] = leaf_bytecode.instructions_multilinear[0];
            input_data.extend_from_slice(&bytecode_hash_domsep);
            input_data.resize(input_data.len().next_multiple_of(VM_DIGEST_LEN), F::ZERO);

            let agg_pi_hash = {
                let mut state = [F::ZERO; 8];
                for chunk in 0..input_data.len() / 8 {
                    let mut inp = [F::ZERO; 16];
                    inp[..8].copy_from_slice(&state);
                    inp[8..].copy_from_slice(&input_data[chunk * 8..(chunk + 1) * 8]);
                    state = poseidon16_compress(inp);
                }
                state
            };
            let mut agg_pi = agg_pi_hash.to_vec();
            agg_pi.resize(agg_pi.len().next_power_of_two(), F::ZERO);

            let (merkle_leaf_blobs, merkle_path_blobs): (Vec<Vec<F>>, Vec<Vec<F>>) =
                raw_proof.merkle_openings.iter().map(|o| {
                    (o.leaf_data.clone(), o.path.iter().flat_map(|d| d.iter().copied()).collect())
                }).unzip();

            let mut hints: HashMap<String, Vec<Vec<F>>> = HashMap::new();
            hints.insert("input_data".to_string(), vec![input_data]);
            hints.insert("child_pi".to_string(), vec![leaf_data.clone()]);
            hints.insert("inner_bytecode_claim".to_string(), vec![inner_claim]);
            hints.insert("bytecode_value_hint".to_string(), vec![details.bytecode_evaluation.value.as_basis_coefficients_slice().to_vec()]);
            hints.insert("proof_transcript_size".to_string(), vec![vec![F::from_usize(raw_proof.transcript.len())]]);
            hints.insert("proof_transcript".to_string(), vec![raw_proof.transcript]);
            hints.insert("merkle_leaf".to_string(), merkle_leaf_blobs);
            hints.insert("merkle_path".to_string(), merkle_path_blobs);

            jobs.push(AggJob { pi: agg_pi, hints });
        }
        eprintln!("    Prepared {} aggregation jobs", jobs.len());

        // Phase 2: Prove all groups with batched concurrency
        let mut agg_proofs = Vec::with_capacity(n_groups);
        for batch_start in (0..n_groups).step_by(concurrency) {
            let batch_end = (batch_start + concurrency).min(n_groups);
            let batch: Vec<_> = (batch_start..batch_end).into_par_iter().map(|gi| {
                prove_execution(
                    agg_bytecode, &jobs[gi].pi,
                    &ExecutionWitness { preamble_memory_len: preamble_len, hints: jobs[gi].hints.clone() },
                    &default_whir_config(1), false,
                ).unwrap_or_else(|e| panic!("agg group {gi} prove: {e}"))
            }).collect();
            agg_proofs.extend(batch);
        }
        let agg_cycles: usize = agg_proofs.iter().map(|p| p.metadata.cycles).sum();
        eprintln!("    {} aggregation proofs, {} total cycles", agg_proofs.len(), agg_cycles);

        break; // one level for now
    }
    let agg_time = t0.elapsed();
    eprintln!("    Aggregation verification: {:.3}s", agg_time.as_secs_f64());

    let peak_rss = system_info::peak_rss_bytes();
    let total_time = leaf_time.as_secs_f64() + agg_time.as_secs_f64();

    eprintln!();
    eprintln!("------------------------------------------------------------");
    eprintln!("  Payload         : {:.1} MB", payload_mb);
    eprintln!("  Leaf proofs     : {n_leaves} × {:.3}s = {:.3}s wall",
        leaf_time.as_secs_f64() / n_leaves as f64, leaf_time.as_secs_f64());
    eprintln!("  Aggregation     : {:.3}s (verify only, proving WIP)", agg_time.as_secs_f64());
    eprintln!("  Total           : {:.3}s", total_time);
    eprintln!("  Peak RSS        : {:.1} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("------------------------------------------------------------");

    println!("{}", serde_json::json!({
        "payload_mb": (payload_mb * 10.0).round() / 10.0,
        "log_n": log_n, "arity": arity, "n_leaves": n_leaves,
        "leaf_time_s": (leaf_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "agg_time_s": (agg_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "leaf_cycles": leaf_cycles,
        "peak_rss": peak_rss,
    }));
}

/// Generate a SHARED unrolled syndrome-check leaf program.
///
/// ALL chunks use this same program. Per-chunk starting values (x_start, w_start,
/// sign_start) come from leaf_data (hinted). The syndrome loop uses Mut variables
/// in an unrolled loop — coset progression is computed by multiplying by compile-time
/// constants OMEGA/OMEGA_1MN each step.
///
/// Public input: hash(leaf_data) = 8 FE.
fn generate_shared_leaf_program(
    full_cp: &CircuitParams,
    leaf_cp: &CircuitParams,
) -> String {
    let chunk_size = leaf_cp.n_eval;

    let mut p = String::new();
    p.push_str("from snark_lib import *\n\n");

    p.push_str(&format!("DIGEST_LEN = {DIGEST_LEN}\n"));
    p.push_str(&format!("N_EVAL = {}\n", leaf_cp.n_eval));
    p.push_str(&format!("N_LEAVES = {}\n", leaf_cp.n_leaves));
    p.push_str(&format!("FELTS_PER_LEAF = {}\n", leaf_cp.fpl));
    p.push_str(&format!("N_CHUNKS_PER_LEAF = {}\n", leaf_cp.n_chunks_per_leaf));
    p.push_str(&format!("TOTAL_TREE_SIZE = {}\n", leaf_cp.total_tree_size));
    p.push_str(&format!("LOG_N = {}\n", full_cp.log_n));
    p.push_str(&format!("G_N = {}\n", fmt_f(full_cp.g_n)));
    p.push_str(&format!("OMEGA = {}\n", fmt_f(full_cp.omega)));
    p.push_str(&format!("OMEGA_1MN = {}\n", fmt_f(full_cp.omega_1mn)));
    p.push_str("LEAF_DATA_SIZE = 24\n\n");

    p.push_str("def main():\n");

    // Load + hash leaf_data, assert == PI
    p.push_str("    leaf_data = Array(LEAF_DATA_SIZE)\n");
    p.push_str("    hint_witness(\"leaf_data\", leaf_data)\n");
    p.push_str("    zero_iv = Array(DIGEST_LEN)\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        zero_iv[i] = 0\n");
    p.push_str("    h0 = Array(DIGEST_LEN)\n");
    p.push_str("    poseidon16_compress(zero_iv, leaf_data, h0)\n");
    p.push_str("    h1 = Array(DIGEST_LEN)\n");
    p.push_str("    poseidon16_compress(h0, leaf_data + DIGEST_LEN, h1)\n");
    p.push_str("    h2 = Array(DIGEST_LEN)\n");
    p.push_str("    poseidon16_compress(h1, leaf_data + 2 * DIGEST_LEN, h2)\n");
    p.push_str("    pub_ptr = 0\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        assert h2[i] == pub_ptr[i]\n\n");

    // Load evals + Merkle tree
    p.push_str("    V = Array(N_EVAL)\n");
    p.push_str("    hint_witness(\"evals_chunk\", V)\n\n");
    emit_merkle_tree(&mut p, leaf_cp, "V");
    let root_offset = leaf_cp.layer_offsets[leaf_cp.tree_depth];
    p.push_str(&format!("    mroot = tree + {root_offset}\n"));
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        assert mroot[i] == leaf_data[i]\n\n");

    // Compute beta^n for each check
    for check in 0..NUM_SYNDROME_CHECKS {
        let beta_idx = 8 + check;
        p.push_str(&format!("    beta_{check} = leaf_data[{beta_idx}]\n"));
        p.push_str(&format!("    bn_{check}: Mut = beta_{check}\n"));
        for _ in 0..full_cp.log_n {
            p.push_str(&format!("    bn_{check} = bn_{check} * bn_{check}\n"));
        }
    }
    p.push_str("\n");

    // Syndrome: shared Mut state (x, w, sign), 4 accumulators
    p.push_str("    x: Mut = leaf_data[16]\n");
    p.push_str("    w: Mut = leaf_data[17]\n");
    p.push_str("    sign: Mut = leaf_data[18]\n");
    p.push_str("    s_0: Mut = 0\n");
    p.push_str("    s_1: Mut = 0\n");
    p.push_str("    s_2: Mut = 0\n");
    p.push_str("    s_3: Mut = 0\n\n");

    // Unrolled syndrome loop: Mut variables updated each step
    for j in 0..chunk_size {
        p.push_str(&format!("    yninv_{j} = 1 / (G_N * sign)\n"));
        p.push_str(&format!("    wv_{j} = V[{j}] * w\n"));
        for check in 0..NUM_SYNDROME_CHECKS {
            p.push_str(&format!(
                "    s_{check} = s_{check} + wv_{j} * (bn_{check} * yninv_{j} - 1) / (beta_{check} - x)\n"
            ));
        }
        p.push_str("    x = x * OMEGA\n");
        p.push_str("    w = w * OMEGA_1MN\n");
        p.push_str("    sign = 0 - sign\n");
    }

    // Assert partial sums match leaf_data
    for check in 0..NUM_SYNDROME_CHECKS {
        let sum_idx = 12 + check;
        p.push_str(&format!("    assert s_{check} == leaf_data[{sum_idx}]\n"));
    }
    p.push_str("    return\n");
    p
}

/// Hash leaf data [subtree_root(8), betas(4), sums(4), x_start, w_start, sign_start]
/// padded to 24 FE, using Poseidon sponge → 8 FE digest.
fn hash_leaf_data(data: &[F]) -> [F; 8] {
    assert!(data.len() <= 24);
    let mut padded = vec![F::ZERO; 24];
    padded[..data.len()].copy_from_slice(data);
    // Sponge: hash 3 chunks of 8
    let mut state = [F::ZERO; 8];
    for chunk in 0..3 {
        let mut input = [F::ZERO; 16];
        input[..8].copy_from_slice(&state);
        input[8..].copy_from_slice(&padded[chunk * 8..(chunk + 1) * 8]);
        state = poseidon16_compress(input);
    }
    state
}
