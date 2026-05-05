//! Benchmark: RS codeword commitment via syndrome check + Poseidon16 Merkle.
//!
//! The prover computes V = coset_FFT(coeffs) externally and provides V as a witness.
//! The circuit:
//!   1. Commits V into a Poseidon16 Merkle tree
//!   2. Asserts the root matches public input
//!   3. Derives 4 Fiat-Shamir challenges from the root
//!   4. For each challenge, verifies the syndrome sum is zero (proving V is a
//!      valid degree-<n codeword via Schwartz-Zippel, 4×31 = 124 bits)
//!
//! Usage: cargo run --release --example fft_merkle_bench -- --log-n 8

use std::collections::HashMap;
use std::time::Instant;

use backend::*;
use lean_compiler::*;
use lean_prover::prove_execution::prove_execution;
use lean_prover::verify_execution::verify_execution;
use lean_prover::*;
use lean_vm::*;
use utils::poseidon16_compress;

// ── Field helpers ──────────────────────────────────────────────────────────

const P: u64 = 0x7F000001;
const DIGEST_LEN: usize = 8;
const TWO_ADICITY: usize = 24;
const NUM_SYNDROME_CHECKS: usize = 4;

fn get_omega(log_order: usize) -> F {
    assert!(
        log_order <= TWO_ADICITY,
        "log_order {log_order} exceeds KoalaBear 2-adicity {TWO_ADICITY}"
    );
    F::from_u32(3).exp_u64((P - 1) >> log_order)
}

fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut r = 0;
    let mut v = x;
    for _ in 0..bits {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

fn pick_log_felts_per_leaf_kb(log_total: usize) -> usize {
    let mut k = 3usize;
    while (1usize << k.saturating_sub(3)) + k < log_total {
        k += 1;
    }
    k
}

// ── Circuit parameters ────────────────────────────────────────────────────

#[allow(dead_code)]
struct CircuitParams {
    log_n: usize,
    log_blowup: usize,
    log_total: usize,
    log_felts_per_leaf: usize,
    n: usize,
    n_eval: usize,
    fpl: usize,
    n_leaves: usize,
    tree_depth: usize,
    n_chunks_per_leaf: usize,
    omega: F,
    g: F,
    g_n: F,         // g^n
    g_1mn: F,       // g^(1-n)
    omega_1mn: F,   // omega^(1-n)
    layer_offsets: Vec<usize>,
    total_tree_size: usize,
}

impl CircuitParams {
    fn new(log_n: usize, log_blowup: usize, log_felts_per_leaf: usize) -> Self {
        assert!(log_blowup == 1, "syndrome check currently requires log_blowup == 1 (N = 2n)");
        let log_total = log_n + log_blowup;
        let n = 1usize << log_n;
        let n_eval = 1usize << log_total;
        let fpl = 1usize << log_felts_per_leaf;
        let n_leaves = n_eval / fpl;
        let tree_depth = n_leaves.trailing_zeros() as usize;
        let n_chunks_per_leaf = fpl / 8;
        let omega = get_omega(log_total);
        let g = F::from_u32(3);
        let g_n = g.exp_u64(n as u64);
        // g^(1-n) = g^(P-1-(n-1)) by Fermat's little theorem
        let g_1mn = g.exp_u64(P - 1 - (n as u64 - 1));
        let omega_1mn = omega.exp_u64(P - 1 - (n as u64 - 1));

        let mut layer_offsets = vec![0usize];
        let mut acc_off = 0;
        for k in 0..tree_depth {
            acc_off += (n_leaves >> k) * DIGEST_LEN;
            layer_offsets.push(acc_off);
        }
        let total_tree_size = acc_off + DIGEST_LEN;

        Self {
            log_n, log_blowup, log_total, log_felts_per_leaf,
            n, n_eval, fpl, n_leaves, tree_depth, n_chunks_per_leaf,
            omega, g, g_n, g_1mn, omega_1mn,
            layer_offsets, total_tree_size,
        }
    }
}

// ── Reference computations (Rust-side) ────────────────────────────────────

fn reference_coset_fft(cp: &CircuitParams, coeffs: &[F]) -> Vec<F> {
    let mut data = vec![F::ZERO; cp.n_eval];
    let mut g_pow = F::ONE;
    for i in 0..cp.n {
        data[i] = coeffs[i] * g_pow;
        g_pow *= cp.g;
    }
    for i in 0..cp.n_eval {
        let j = bit_reverse(i, cp.log_total);
        if i < j { data.swap(i, j); }
    }
    let mut twiddles = Vec::with_capacity(cp.n_eval / 2);
    let mut acc = F::ONE;
    for _ in 0..cp.n_eval / 2 { twiddles.push(acc); acc *= cp.omega; }
    for s in 1..=cp.log_total {
        let m = 1 << s;
        let half = m >> 1;
        let stride = cp.n_eval / m;
        let mut k = 0;
        while k < cp.n_eval {
            for j in 0..half {
                let u = data[k + j];
                let t = twiddles[j * stride] * data[k + j + half];
                data[k + j] = u + t;
                data[k + j + half] = u - t;
            }
            k += m;
        }
    }
    data
}

fn poseidon_hash_chain(chunks: &[F]) -> [F; 8] {
    assert!(!chunks.is_empty() && chunks.len() % 8 == 0);
    let mut state = [F::ZERO; 8];
    for b in 0..chunks.len() / 8 {
        let mut input = [F::ZERO; 16];
        input[..8].copy_from_slice(&state);
        input[8..].copy_from_slice(&chunks[b * 8..(b + 1) * 8]);
        state = poseidon16_compress(input);
    }
    state
}

fn poseidon_hash_pair(left: &[F; 8], right: &[F; 8]) -> [F; 8] {
    let mut input = [F::ZERO; 16];
    input[..8].copy_from_slice(left);
    input[8..].copy_from_slice(right);
    poseidon16_compress(input)
}

fn reference_merkle_root(cp: &CircuitParams, evals: &[F]) -> [F; 8] {
    let mut layer: Vec<[F; 8]> = (0..cp.n_leaves)
        .map(|i| poseidon_hash_chain(&evals[i * cp.fpl..(i + 1) * cp.fpl]))
        .collect();
    while layer.len() > 1 {
        layer = (0..layer.len() / 2)
            .map(|i| poseidon_hash_pair(&layer[2 * i], &layer[2 * i + 1]))
            .collect();
    }
    layer[0]
}

/// Compute syndrome sum for one challenge. Returns 0 iff V is a valid codeword.
fn reference_syndrome(cp: &CircuitParams, evals: &[F], beta: F) -> F {
    let beta_n = beta.exp_u64(cp.n as u64);
    let mut x = cp.g;           // yⱼ = g·ωʲ
    let mut w = cp.g_1mn;       // yⱼ^(1-n)
    let mut sign = F::ONE;      // (-1)^j
    let mut s = F::ZERO;

    for j in 0..cp.n_eval {
        let yj_n_inv = F::ONE / (cp.g_n * sign);
        let numer = beta_n * yj_n_inv - F::ONE;
        let denom = beta - x;
        s += evals[j] * w * numer / denom;
        x *= cp.omega;
        w *= cp.omega_1mn;
        sign = F::ZERO - sign;
    }
    s
}

// ── zkDSL program generation ──────────────────────────────────────────────

fn generate_program(cp: &CircuitParams) -> String {
    let fmt_f = |f: F| format!("{}", f.as_canonical_u32());

    let mut p = String::new();
    p.push_str("from snark_lib import *\n\n");

    // Constants
    p.push_str(&format!("DIGEST_LEN = {DIGEST_LEN}\n"));
    p.push_str(&format!("N_EVAL = {}\n", cp.n_eval));
    p.push_str(&format!("N_LEAVES = {}\n", cp.n_leaves));
    p.push_str(&format!("FELTS_PER_LEAF = {}\n", cp.fpl));
    p.push_str(&format!("N_CHUNKS_PER_LEAF = {}\n", cp.n_chunks_per_leaf));
    p.push_str(&format!("TOTAL_TREE_SIZE = {}\n", cp.total_tree_size));
    p.push_str(&format!("LOG_N = {}\n", cp.log_n));
    p.push_str(&format!("G = {}\n", fmt_f(cp.g)));
    p.push_str(&format!("G_N = {}\n", fmt_f(cp.g_n)));
    p.push_str(&format!("G_1MN = {}\n", fmt_f(cp.g_1mn)));
    p.push_str(&format!("OMEGA = {}\n", fmt_f(cp.omega)));
    p.push_str(&format!("OMEGA_1MN = {}\n\n", fmt_f(cp.omega_1mn)));

    p.push_str("def main():\n");

    // 1. Load evaluations
    p.push_str("    V = Array(N_EVAL)\n");
    p.push_str("    hint_witness(\"evals\", V)\n\n");

    // 2. Merkle tree — unrolled leaf hashing + internal nodes
    let chain_size = cp.n_leaves * cp.n_chunks_per_leaf * DIGEST_LEN;
    p.push_str("    zero_vec = Array(DIGEST_LEN)\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        zero_vec[i] = 0\n\n");
    p.push_str(&format!("    chain = Array({chain_size})\n"));
    p.push_str(&format!("    tree = Array(TOTAL_TREE_SIZE)\n\n"));

    // Leaf hashing (unrolled over leaves)
    for leaf in 0..cp.n_leaves {
        let ld = leaf * cp.fpl;
        let ch = leaf * cp.n_chunks_per_leaf * DIGEST_LEN;
        p.push_str(&format!(
            "    poseidon16_compress(zero_vec, V + {ld}, chain + {ch})\n"
        ));
        for c in 1..cp.n_chunks_per_leaf {
            p.push_str(&format!(
                "    poseidon16_compress(chain + {}, V + {}, chain + {})\n",
                ch + (c - 1) * DIGEST_LEN, ld + c * 8, ch + c * DIGEST_LEN,
            ));
        }
        let final_ch = ch + (cp.n_chunks_per_leaf - 1) * DIGEST_LEN;
        let tree_leaf = leaf * DIGEST_LEN;
        for k in 0..DIGEST_LEN {
            p.push_str(&format!("    tree[{}] = chain[{}]\n", tree_leaf + k, final_ch + k));
        }
    }

    // Internal Merkle nodes (unrolled)
    for layer in 0..cp.tree_depth {
        let n_pairs = cp.n_leaves >> (layer + 1);
        let src = cp.layer_offsets[layer];
        let dst = cp.layer_offsets[layer + 1];
        for pair in 0..n_pairs {
            p.push_str(&format!(
                "    poseidon16_compress(tree + {}, tree + {}, tree + {})\n",
                src + 2 * pair * DIGEST_LEN,
                src + (2 * pair + 1) * DIGEST_LEN,
                dst + pair * DIGEST_LEN,
            ));
        }
    }

    // 3. Assert root == public_input[0:8]
    let root_offset = cp.layer_offsets[cp.tree_depth];
    p.push_str(&format!("\n    root_ptr = tree + {root_offset}\n"));
    p.push_str("    pub_ptr = 0\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        assert root_ptr[i] == pub_ptr[i]\n\n");

    // 4. Derive 4 challenges from root (Poseidon Fiat-Shamir)
    p.push_str("    # Fiat-Shamir challenges\n");
    for i in 0..NUM_SYNDROME_CHECKS {
        p.push_str(&format!("    chal_out_{i} = Array(DIGEST_LEN)\n"));
        // Copy root to chal_input (reuse for each challenge via domain sep in right half)
        p.push_str(&format!("    ds_{i} = Array(DIGEST_LEN)\n"));
        p.push_str(&format!("    ds_{i}[0] = {i}\n"));
        for k in 1..DIGEST_LEN {
            p.push_str(&format!("    ds_{i}[{k}] = 0\n"));
        }
        p.push_str(&format!(
            "    poseidon16_compress(root_ptr, ds_{i}, chal_out_{i})\n"
        ));
    }
    p.push_str("\n");

    // 5. Syndrome checks (4 checks, each a range loop)
    p.push_str("    # Syndrome checks\n");
    for i in 0..NUM_SYNDROME_CHECKS {
        p.push_str(&format!("    beta_{i} = chal_out_{i}[0]\n"));
        // Compute beta^n via repeated squaring
        p.push_str(&format!("    bn_{i}: Mut = beta_{i}\n"));
        for _ in 0..cp.log_n {
            p.push_str(&format!("    bn_{i} = bn_{i} * bn_{i}\n"));
        }
        // Syndrome accumulation loop
        p.push_str(&format!("    x_{i}: Mut = G\n"));
        p.push_str(&format!("    w_{i}: Mut = G_1MN\n"));
        p.push_str(&format!("    sign_{i}: Mut = 1\n"));
        p.push_str(&format!("    s_{i}: Mut = 0\n"));
        p.push_str(&format!("    for j_{i} in range(0, N_EVAL):\n"));
        p.push_str(&format!("        yj_n_inv_{i} = 1 / (G_N * sign_{i})\n"));
        p.push_str(&format!("        numer_{i} = bn_{i} * yj_n_inv_{i} - 1\n"));
        p.push_str(&format!("        denom_{i} = beta_{i} - x_{i}\n"));
        p.push_str(&format!("        s_{i} = s_{i} + V[j_{i}] * w_{i} * numer_{i} / denom_{i}\n"));
        p.push_str(&format!("        x_{i} = x_{i} * OMEGA\n"));
        p.push_str(&format!("        w_{i} = w_{i} * OMEGA_1MN\n"));
        p.push_str(&format!("        sign_{i} = 0 - sign_{i}\n"));
        p.push_str(&format!("    assert s_{i} == 0\n\n"));
    }

    p.push_str("    return\n");
    p
}

// ── Witness builder ───────────────────────────────────────────────────────

fn build_witness(cp: &CircuitParams, coeffs: &[F]) -> (Vec<F>, HashMap<String, Vec<Vec<F>>>) {
    let evals = reference_coset_fft(cp, coeffs);
    let mut hints = HashMap::new();
    hints.insert("evals".to_string(), vec![evals.clone()]);
    (evals, hints)
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_n: usize = args.iter().position(|a| a == "--log-n")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(8);
    let log_blowup: usize = args.iter().position(|a| a == "--log-blowup")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(1);
    let log_inv_rate: usize = args.iter().position(|a| a == "--log-inv-rate")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(1);
    let log_total = log_n + log_blowup;
    let lfpl = pick_log_felts_per_leaf_kb(log_total);
    let cp = CircuitParams::new(log_n, log_blowup, lfpl);

    eprintln!("============================================================");
    eprintln!("leanVM RS codeword commitment (syndrome check)");
    eprintln!("============================================================");
    eprintln!("  log_n              = {log_n}");
    eprintln!("  log_blowup         = {log_blowup}");
    eprintln!("  log_felts_per_leaf = {lfpl}");
    eprintln!("  n_coeffs           = {}", cp.n);
    eprintln!("  n_eval             = {}", cp.n_eval);
    eprintln!("  n_leaves           = {}", cp.n_leaves);
    eprintln!("  syndrome checks    = {NUM_SYNDROME_CHECKS}");
    eprintln!();

    let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();

    eprintln!("[1/4] Computing reference (FFT + Merkle) ...");
    let t0 = Instant::now();
    let (evals, hints) = build_witness(&cp, &coeffs);
    let root = reference_merkle_root(&cp, &evals);
    eprintln!("  reference: {:.3}s", t0.elapsed().as_secs_f64());

    eprintln!("[2/4] Generating program ...");
    let t0 = Instant::now();
    let program_str = generate_program(&cp);
    eprintln!(
        "  program: {} lines, {:.1} KB, generated in {:.3}s",
        program_str.lines().count(),
        program_str.len() as f64 / 1024.0,
        t0.elapsed().as_secs_f64()
    );

    eprintln!("[3/4] Compiling ...");
    let t0 = Instant::now();
    let bytecode = compile_program(&ProgramSource::Raw(program_str));
    let compile_time = t0.elapsed();
    eprintln!("  compiled in {:.3}s", compile_time.as_secs_f64());

    let mut public_input = root.to_vec();
    public_input.resize(public_input.len().next_power_of_two(), F::ZERO);

    let witness = ExecutionWitness { preamble_memory_len: 0, hints };

    eprintln!("[4/4] Proving ...");
    let t0 = Instant::now();
    let proof = prove_execution(
        &bytecode, &public_input, &witness,
        &default_whir_config(log_inv_rate), false,
    ).unwrap();
    let prove_time = t0.elapsed();

    let metadata = proof.metadata;
    let t0 = Instant::now();
    verify_execution(&bytecode, &public_input, proof.proof).unwrap();
    let verify_time = t0.elapsed();

    let peak_rss = system_info::peak_rss_bytes();

    eprintln!();
    eprintln!("{}", metadata.display());
    eprintln!("------------------------------------------------------------");
    eprintln!("BENCHMARK RESULTS");
    eprintln!("------------------------------------------------------------");
    eprintln!("  Compile time       : {:.3}s", compile_time.as_secs_f64());
    eprintln!("  Prove time (wall)  : {:.3}s", prove_time.as_secs_f64());
    eprintln!("  Prove peak RSS     : {:.2} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("  Verify time        : {:.3}s", verify_time.as_secs_f64());
    eprintln!("  Cycles             : {}", metadata.cycles);
    eprintln!("  Poseidon16 calls   : {}", metadata.n_poseidons);
    eprintln!("------------------------------------------------------------");

    let result = serde_json::json!({
        "log_n": log_n, "log_blowup": log_blowup, "log_felts_per_leaf": lfpl,
        "n_coeffs": cp.n, "n_eval": cp.n_eval, "n_leaves": cp.n_leaves,
        "prove_time_s": (prove_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "verify_time_s": (verify_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "peak_rss_bytes": peak_rss,
        "cycles": metadata.cycles, "n_poseidons": metadata.n_poseidons,
        "memory": metadata.memory,
    });
    println!("{result}");
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cp() -> CircuitParams {
        CircuitParams::new(4, 1, pick_log_felts_per_leaf_kb(5))
    }

    fn small_coeffs(cp: &CircuitParams) -> Vec<F> {
        (1..=cp.n as u32).map(F::from_u32).collect()
    }

    // ── Reference syndrome check ────────────────────────────────────

    #[test]
    fn test_syndrome_zero_for_valid_codeword() {
        let cp = small_cp();
        let coeffs = small_coeffs(&cp);
        let evals = reference_coset_fft(&cp, &coeffs);
        // Any challenge should give syndrome == 0 for a valid codeword
        for beta_val in [7u32, 42, 1000, 999999] {
            let beta = F::from_u32(beta_val);
            let s = reference_syndrome(&cp, &evals, beta);
            assert_eq!(s, F::ZERO, "syndrome != 0 for valid codeword at beta={beta_val}");
        }
    }

    #[test]
    fn test_syndrome_nonzero_for_invalid_codeword() {
        let cp = small_cp();
        // Random vector (not a valid RS codeword)
        let bad_evals: Vec<F> = (0..cp.n_eval as u32).map(|i| F::from_u32(i * 7 + 13)).collect();
        let beta = F::from_u32(42);
        let s = reference_syndrome(&cp, &bad_evals, beta);
        assert_ne!(s, F::ZERO, "syndrome == 0 for random non-codeword");
    }

    #[test]
    fn test_syndrome_detects_single_corruption() {
        let cp = small_cp();
        let coeffs = small_coeffs(&cp);
        let mut evals = reference_coset_fft(&cp, &coeffs);
        evals[cp.n_eval / 2] += F::ONE; // corrupt one evaluation
        let beta = F::from_u32(42);
        let s = reference_syndrome(&cp, &evals, beta);
        assert_ne!(s, F::ZERO, "syndrome missed single-eval corruption");
    }

    // ── End-to-end prove + verify ───────────────────────────────────

    #[test]
    fn test_correct_evals_accepted() {
        let cp = small_cp();
        let coeffs = small_coeffs(&cp);
        let (evals, hints) = build_witness(&cp, &coeffs);
        let root = reference_merkle_root(&cp, &evals);

        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);

        let proof = prove_execution(
            &bytecode, &pi,
            &ExecutionWitness { preamble_memory_len: 0, hints },
            &default_whir_config(1), false,
        ).expect("Valid codeword should be accepted");

        verify_execution(&bytecode, &pi, proof.proof).expect("Verification should pass");
    }

    #[test]
    fn test_wrong_root_rejected() {
        let cp = small_cp();
        let coeffs = small_coeffs(&cp);
        let (evals, hints) = build_witness(&cp, &coeffs);
        let mut wrong_root = reference_merkle_root(&cp, &evals);
        wrong_root[0] += F::ONE;

        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = wrong_root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);

        let result = prove_execution(
            &bytecode, &pi,
            &ExecutionWitness { preamble_memory_len: 0, hints },
            &default_whir_config(1), false,
        );
        assert!(result.is_err(), "Wrong root should be rejected");
    }

    #[test]
    fn test_invalid_codeword_rejected() {
        let cp = small_cp();
        // Provide a random (non-codeword) vector as evals
        let bad_evals: Vec<F> = (0..cp.n_eval as u32).map(|i| F::from_u32(i * 7 + 13)).collect();
        let root = reference_merkle_root(&cp, &bad_evals);

        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);

        let mut hints = HashMap::new();
        hints.insert("evals".to_string(), vec![bad_evals]);

        let result = prove_execution(
            &bytecode, &pi,
            &ExecutionWitness { preamble_memory_len: 0, hints },
            &default_whir_config(1), false,
        );
        assert!(result.is_err(), "Invalid codeword should be rejected by syndrome check");
    }

    // ── Reference FFT correctness ───────────────────────────────────

    #[test]
    fn test_fft_evaluates_at_coset_points() {
        let cp = CircuitParams::new(3, 1, 3);
        let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let evals = reference_coset_fft(&cp, &coeffs);
        for k in 0..cp.n_eval {
            let point = cp.g * cp.omega.exp_u64(k as u64);
            let mut val = F::ZERO;
            for i in (0..cp.n).rev() { val = val * point + coeffs[i]; }
            assert_eq!(evals[k], val, "FFT[{k}] != P(g·ω^{k})");
        }
    }

    #[test]
    fn test_fft_linearity() {
        let cp = CircuitParams::new(4, 1, 3);
        let a: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let b: Vec<F> = (100..100 + cp.n as u32).map(F::from_u32).collect();
        let ab: Vec<F> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
        let fft_a = reference_coset_fft(&cp, &a);
        let fft_b = reference_coset_fft(&cp, &b);
        let fft_ab = reference_coset_fft(&cp, &ab);
        for i in 0..cp.n_eval {
            assert_eq!(fft_ab[i], fft_a[i] + fft_b[i]);
        }
    }

    // ── Leaf packing ────────────────────────────────────────────────

    #[test]
    fn test_leaf_size_exceeds_proof_path() {
        for log_total in 5..=25 {
            let k = pick_log_felts_per_leaf_kb(log_total);
            let fpl = 1usize << k;
            let n_leaves = (1usize << log_total) / fpl;
            let depth = n_leaves.trailing_zeros() as usize;
            assert!(fpl * 4 >= depth * DIGEST_LEN * 4,
                "log_total={log_total}: leaf < path");
        }
    }

    // ── Field bounds ────────────────────────────────────────────────

    #[test]
    fn test_omega_is_primitive_root() {
        for log_order in 1..=20 {
            let omega = get_omega(log_order);
            assert_eq!(omega.exp_u64(1u64 << log_order), F::ONE);
            assert_ne!(omega.exp_u64(1u64 << (log_order - 1)), F::ONE);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds KoalaBear 2-adicity")]
    fn test_omega_rejects_too_large_order() {
        get_omega(25);
    }

    // ── Merkle binding ──────────────────────────────────────────────

    #[test]
    fn test_different_data_different_roots() {
        let cp = small_cp();
        let a: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let mut b = a.clone();
        b[0] += F::ONE;
        let root_a = reference_merkle_root(&cp, &reference_coset_fft(&cp, &a));
        let root_b = reference_merkle_root(&cp, &reference_coset_fft(&cp, &b));
        assert_ne!(root_a, root_b);
    }

    // ── End-to-end at different sizes ───────────────────────────────

    #[test]
    fn test_end_to_end_n6() {
        let cp = CircuitParams::new(6, 1, pick_log_felts_per_leaf_kb(7));
        let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let (evals, hints) = build_witness(&cp, &coeffs);
        let root = reference_merkle_root(&cp, &evals);
        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);
        let proof = prove_execution(
            &bytecode, &pi,
            &ExecutionWitness { preamble_memory_len: 0, hints },
            &default_whir_config(1), false,
        ).expect("n=6 should prove");
        verify_execution(&bytecode, &pi, proof.proof).expect("n=6 should verify");
    }
}
