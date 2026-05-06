//! RLC + fold benchmark: batch m codewords via random linear combination,
//! then fold with random challenges.
//!
//! Exercises extension-field precompiles:
//!   - `dot_product_be` for RLC  (m base×ext muls per position, m trace rows)
//!   - `dot_product_ee` + `add_ee` for fold (2 trace rows per butterfly)
//!
//! Single proof:
//!   cargo run --release --example bench_fri_fold -- --log-poly 11 --m 510
//!
//! Parallel batches:
//!   cargo run --release --example bench_fri_fold -- --log-poly 11 --m 510 --batches 4 --concurrency 2

use rayon::prelude::*;
use starkdal_leanvm::*;
use std::collections::HashMap;
use std::time::Instant;

const DIM: usize = 5;

// ── zkDSL program generation ─────────────────────────────────────────────

fn generate_program(n_eval: usize, m: usize) -> String {
    let log_n_eval = n_eval.trailing_zeros() as usize;

    let mut p = String::with_capacity(1 << 20);
    p.push_str("from snark_lib import *\n\n");

    p.push_str(&format!("DIM = {DIM}\n"));
    p.push_str(&format!("DIGEST_LEN = {DIGEST_LEN}\n"));
    p.push_str(&format!("N = {n_eval}\n"));
    p.push_str(&format!("M = {m}\n"));
    p.push_str(&format!("LOG_N = {log_n_eval}\n\n"));

    p.push_str("def main():\n");

    // ── zero vector (Poseidon IV) ──
    p.push_str("    zero_vec = Array(DIGEST_LEN)\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        zero_vec[i] = 0\n\n");

    // ── load witnesses ──
    p.push_str("    cw_col = Array(N * M)\n");
    p.push_str("    hint_witness(\"codewords_col\", cw_col)\n\n");
    p.push_str("    roots = Array(M * DIGEST_LEN)\n");
    p.push_str("    hint_witness(\"roots\", roots)\n\n");

    // ── chain-hash roots (write-once: fresh array per step) ──
    p.push_str("    c_0 = Array(DIGEST_LEN)\n");
    p.push_str("    poseidon16_compress(zero_vec, roots, c_0)\n");
    for i in 1..m {
        p.push_str(&format!("    c_{i} = Array(DIGEST_LEN)\n"));
        p.push_str(&format!(
            "    poseidon16_compress(c_{}, roots + {}, c_{i})\n",
            i - 1,
            i * DIGEST_LEN
        ));
    }
    let chain_name = format!("c_{}", m - 1);

    // ── assert chain == PI[0:8] ──
    p.push_str("\n    pub_ptr = 0\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str(&format!("        assert {chain_name}[i] == pub_ptr[i]\n\n"));

    // ── derive RLC challenges (one EF per codeword) ──
    p.push_str("    challenges = Array(M * DIM)\n");
    for i in 0..m {
        p.push_str(&format!("    ch_{i} = Array(DIGEST_LEN)\n"));
        p.push_str(&format!("    ds_{i} = Array(DIGEST_LEN)\n"));
        p.push_str(&format!("    ds_{i}[0] = {i}\n"));
        for k in 1..DIGEST_LEN {
            p.push_str(&format!("    ds_{i}[{k}] = 0\n"));
        }
        p.push_str(&format!(
            "    poseidon16_compress(roots + {}, ds_{i}, ch_{i})\n",
            i * DIGEST_LEN
        ));
        for k in 0..DIM {
            p.push_str(&format!(
                "    challenges[{}] = ch_{i}[{k}]\n",
                i * DIM + k
            ));
        }
    }
    p.push_str("\n");

    // ── RLC: c*[j] = Σ_i cw_col[j·M+i] · challenges[i]  (dot_product_be) ──
    p.push_str("    c_star = Array(N * DIM)\n");
    p.push_str("    for j in unroll(0, N):\n");
    p.push_str("        dot_product_be(cw_col + j * M, challenges, c_star + j * DIM, M)\n\n");

    // ── fold: log(N) rounds of butterfly ──
    let mut src = "c_star".to_string();
    let mut cur_len = n_eval;

    for r in 0..log_n_eval {
        let half = cur_len / 2;
        let ft = format!("ft{r}");
        let fr = format!("fr{r}");

        p.push_str(&format!("    {ft} = Array({})\n", half * DIM));
        p.push_str(&format!("    {fr} = Array({})\n", half * DIM));

        // derive beta_r
        p.push_str(&format!("    bb{r} = Array(DIGEST_LEN)\n"));
        p.push_str(&format!("    bd{r} = Array(DIGEST_LEN)\n"));
        p.push_str(&format!("    bd{r}[0] = {}\n", 1000 + r));
        for k in 1..DIGEST_LEN {
            p.push_str(&format!("    bd{r}[{k}] = 0\n"));
        }
        p.push_str(&format!("    poseidon16_compress({chain_name}, bd{r}, bb{r})\n"));
        p.push_str(&format!("    b{r} = Array(DIM)\n"));
        for k in 0..DIM {
            p.push_str(&format!("    b{r}[{k}] = bb{r}[{k}]\n"));
        }

        // butterfly: fr[j] = src[2j] + b_r · src[2j+1]
        p.push_str(&format!("    for j in unroll(0, {half}):\n"));
        p.push_str(&format!(
            "        dot_product_ee(b{r}, {src} + (2 * j + 1) * DIM, {ft} + j * DIM)\n"
        ));
        p.push_str(&format!(
            "        add_ee({src} + 2 * j * DIM, {ft} + j * DIM, {fr} + j * DIM)\n"
        ));
        p.push_str("\n");

        src = fr;
        cur_len = half;
    }

    // ── assert fold result == PI[8:13] ──
    p.push_str("    fold_pi = 8\n");
    p.push_str("    for k in unroll(0, DIM):\n");
    p.push_str(&format!("        assert {src}[k] == fold_pi[k]\n"));

    // ── execution padding ──
    let ext_op_rows = m * n_eval + 2 * n_eval;
    let ext_op_log = ((ext_op_rows as f64).log2().ceil() as usize).max(1);
    let target_cycles = (1usize << ext_op_log.saturating_sub(1)) + 1;
    if target_cycles > n_eval {
        let pad_iters = (target_cycles - n_eval) / 15 + 10;
        p.push_str(&format!("\n    pad: Mut = 0\n"));
        p.push_str(&format!("    for _pi in range(0, {pad_iters}):\n"));
        p.push_str("        pad = pad + 1\n");
    }

    p.push_str("\n    return\n");
    p
}

// ── reference computation for one batch ──────────────────────────────────

struct BatchRef {
    pi: Vec<F>,
    cw_col: Vec<F>,
    roots_flat: Vec<F>,
    data_bytes: usize,
}

fn compute_batch_ref(
    batch_idx: usize,
    m: usize,
    d: usize,
    n_eval: usize,
    log_n_eval: usize,
    cp: &CircuitParams,
) -> BatchRef {
    // Generate m codewords for this batch (deterministic from batch_idx)
    let codewords: Vec<Vec<F>> = (0..m)
        .map(|i| {
            let global_i = batch_idx * m + i;
            let coeffs: Vec<F> = (0..d)
                .map(|k| F::from_u32(((global_i * d + k) % (P as usize - 2) + 1) as u32))
                .collect();
            reference_coset_fft(cp, &coeffs)
        })
        .collect();

    // Hash each codeword → root
    let roots: Vec<[F; 8]> = codewords.iter().map(|cw| poseidon_hash_chain(cw)).collect();

    // Chain-hash roots
    let mut chain = {
        let mut inp = [F::ZERO; 16];
        inp[8..].copy_from_slice(&roots[0]);
        poseidon16_compress(inp)
    };
    for root in roots.iter().skip(1) {
        let mut inp = [F::ZERO; 16];
        inp[..8].copy_from_slice(&chain);
        inp[8..].copy_from_slice(root);
        chain = poseidon16_compress(inp);
    }

    // Derive RLC challenges
    let challenges: Vec<EF> = (0..m)
        .map(|i| {
            let mut inp = [F::ZERO; 16];
            inp[..8].copy_from_slice(&roots[i]);
            inp[8] = F::from_u32(i as u32);
            let h = poseidon16_compress(inp);
            EF::from_basis_coefficients_slice(&h[..DIM]).unwrap()
        })
        .collect();

    // Compute RLC
    let c_star: Vec<EF> = (0..n_eval)
        .map(|j| {
            let mut acc = EF::ZERO;
            for i in 0..m {
                acc += challenges[i] * codewords[i][j];
            }
            acc
        })
        .collect();

    // Derive fold challenges
    let fold_betas: Vec<EF> = (0..log_n_eval)
        .map(|r| {
            let mut inp = [F::ZERO; 16];
            inp[..8].copy_from_slice(&chain);
            inp[8] = F::from_u32((1000 + r) as u32);
            let h = poseidon16_compress(inp);
            EF::from_basis_coefficients_slice(&h[..DIM]).unwrap()
        })
        .collect();

    // Fold
    let mut current = c_star;
    for beta in &fold_betas {
        let half = current.len() / 2;
        current = (0..half)
            .map(|j| current[2 * j] + *beta * current[2 * j + 1])
            .collect();
    }
    let fold_result = current[0];

    // Public input
    let fold_comps = fold_result.as_basis_coefficients_slice();
    let mut pi = chain.to_vec();
    pi.extend_from_slice(fold_comps);
    pi.resize(16, F::ZERO);

    // Column-major codewords
    let mut cw_col = vec![F::ZERO; n_eval * m];
    for j in 0..n_eval {
        for i in 0..m {
            cw_col[j * m + i] = codewords[i][j];
        }
    }
    let roots_flat: Vec<F> = roots.iter().flat_map(|r| r.iter().copied()).collect();

    BatchRef {
        pi,
        cw_col,
        roots_flat,
        data_bytes: m * d * 4,
    }
}

// ── main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_poly: usize = args.iter().position(|a| a == "--log-poly")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(3);
    let m: usize = args.iter().position(|a| a == "--m")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(3);
    let batches: usize = args.iter().position(|a| a == "--batches")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(1);
    let concurrency: usize = args.iter().position(|a| a == "--concurrency")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(1);

    assert!(m >= 1, "m must be at least 1");
    assert!(batches >= 1, "batches must be at least 1");
    let log_blowup = 1usize;
    let log_n_eval = log_poly + log_blowup;
    let n_eval = 1usize << log_n_eval;
    let d = 1usize << log_poly;
    let n_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    eprintln!("============================================================");
    eprintln!("RLC + Fold benchmark");
    eprintln!("============================================================");
    eprintln!("  n_eval={n_eval}  d={d}  m={m}  batches={batches}  concurrency={concurrency}");
    eprintln!("  cores={n_cores}  total_codewords={}  data={:.1} MB",
        batches * m, (batches * m * d * 4) as f64 / (1024.0 * 1024.0));

    // ── compile shared bytecode ──
    let program = generate_program(n_eval, m);
    eprintln!("  program: {} lines, {:.1} KB",
        program.lines().count(), program.len() as f64 / 1024.0);
    let t0 = Instant::now();
    let bytecode = compile_program(&ProgramSource::Raw(program));
    eprintln!("  compiled in {:.3}s", t0.elapsed().as_secs_f64());

    // ── compute reference for each batch ──
    let cp = CircuitParams::new(log_poly, log_blowup, pick_log_felts_per_leaf_kb(log_n_eval));
    eprintln!("  computing {} batch references...", batches);
    let t0 = Instant::now();
    let batch_refs: Vec<BatchRef> = (0..batches)
        .map(|b| compute_batch_ref(b, m, d, n_eval, log_n_eval, &cp))
        .collect();
    eprintln!("  references in {:.3}s", t0.elapsed().as_secs_f64());

    // ── prove all batches with controlled concurrency ──
    eprintln!("  proving {} batches (concurrency={concurrency})...", batches);
    let t0 = Instant::now();
    let mut all_proofs: Vec<ExecutionProof> = Vec::with_capacity(batches);
    for batch_start in (0..batches).step_by(concurrency) {
        let batch_end = (batch_start + concurrency).min(batches);
        let round: Vec<ExecutionProof> = (batch_start..batch_end)
            .into_par_iter()
            .map(|bi| {
                let br = &batch_refs[bi];
                let mut hints: HashMap<String, Vec<Vec<F>>> = HashMap::new();
                hints.insert("codewords_col".to_string(), vec![br.cw_col.clone()]);
                hints.insert("roots".to_string(), vec![br.roots_flat.clone()]);
                prove_execution(
                    &bytecode, &br.pi,
                    &ExecutionWitness { preamble_memory_len: 0, hints },
                    &default_whir_config(1), false,
                ).unwrap_or_else(|e| panic!("batch {bi} failed: {e}"))
            })
            .collect();
        all_proofs.extend(round);
    }
    let prove_time = t0.elapsed();
    let total_cycles: usize = all_proofs.iter().map(|p| p.metadata.cycles).sum();
    let total_poseidons: usize = all_proofs.iter().map(|p| p.metadata.n_poseidons).sum();

    // ── verify all proofs ──
    eprintln!("  verifying {} proofs...", batches);
    let t0 = Instant::now();
    for (bi, proof) in all_proofs.iter().enumerate() {
        verify_execution(&bytecode, &batch_refs[bi].pi, proof.proof.clone())
            .unwrap_or_else(|e| panic!("batch {bi} verify failed: {e}"));
    }
    let verify_time = t0.elapsed();

    let peak_rss = system_info::peak_rss_bytes();
    let total_data: usize = batch_refs.iter().map(|b| b.data_bytes).sum();

    eprintln!();
    eprintln!("------------------------------------------------------------");
    eprintln!("  Prove wall   : {:.3}s ({} batches × {concurrency} parallel)",
        prove_time.as_secs_f64(), batches);
    eprintln!("  Verify wall  : {:.3}s", verify_time.as_secs_f64());
    eprintln!("  Total cycles : {total_cycles}  poseidons: {total_poseidons}");
    eprintln!("  Data         : {:.2} MB", total_data as f64 / (1024.0 * 1024.0));
    eprintln!("  Throughput   : {:.1} KB/s (prove only)",
        total_data as f64 / 1024.0 / prove_time.as_secs_f64());
    eprintln!("  Throughput   : {:.1} KB/s (prove + verify)",
        total_data as f64 / 1024.0 / (prove_time.as_secs_f64() + verify_time.as_secs_f64()));
    eprintln!("  Peak RSS     : {:.2} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("------------------------------------------------------------");

    println!(
        "{}",
        serde_json::json!({
            "variant": "rlc_fold",
            "log_poly": log_poly,
            "n_eval": n_eval,
            "m": m,
            "batches": batches,
            "concurrency": concurrency,
            "prove_s": (prove_time.as_secs_f64() * 1000.0).round() / 1000.0,
            "verify_s": (verify_time.as_secs_f64() * 1000.0).round() / 1000.0,
            "total_cycles": total_cycles,
            "peak_rss": peak_rss,
            "data_bytes": total_data,
            "throughput_kbs":
                (total_data as f64 / 1024.0 / prove_time.as_secs_f64() * 10.0).round() / 10.0,
        })
    );
}
