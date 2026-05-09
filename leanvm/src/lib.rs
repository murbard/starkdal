//! Shared infrastructure for RS codeword commitment benchmarks.
//!
//! Provides field helpers, circuit parameter computation, reference implementations
//! (coset FFT, Merkle trees, syndrome checks), zkDSL code generation utilities,
//! and a benchmark harness for proving and verifying leanVM circuits.
//!
//! Used by all `examples/bench_*.rs` binaries.

use std::collections::HashMap;
use std::time::Instant;

pub use backend::*;
pub use lean_compiler::*;
pub use lean_prover::prove_execution::{prove_execution, ExecutionProof};
pub use lean_prover::verify_execution::verify_execution;
pub use backend::Proof;
pub use lean_prover::*;
pub use lean_vm::*;
pub use utils::poseidon16_compress;

pub const P: u64 = 0x7F000001;
pub const DIGEST_LEN: usize = 8;
pub const TWO_ADICITY: usize = 24;
pub const NUM_SYNDROME_CHECKS: usize = 4;

// ── Field helpers ──────────────────────────────────────────────────────────

pub fn get_omega(log_order: usize) -> F {
    assert!(log_order <= TWO_ADICITY, "log_order {log_order} exceeds KoalaBear 2-adicity {TWO_ADICITY}");
    F::from_u32(3).exp_u64((P - 1) >> log_order)
}

pub fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut r = 0;
    let mut v = x;
    for _ in 0..bits { r = (r << 1) | (v & 1); v >>= 1; }
    r
}

pub fn pick_log_felts_per_leaf_kb(log_total: usize) -> usize {
    let mut k = 3usize;
    while (1usize << k.saturating_sub(3)) + k < log_total { k += 1; }
    k
}

pub fn fmt_f(f: F) -> String { format!("{}", f.as_canonical_u32()) }

// ── Circuit parameters ────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct CircuitParams {
    pub log_n: usize,
    pub log_blowup: usize,
    pub log_total: usize,
    pub log_felts_per_leaf: usize,
    pub n: usize,
    pub n_eval: usize,
    pub fpl: usize,
    pub n_leaves: usize,
    pub tree_depth: usize,
    pub n_chunks_per_leaf: usize,
    pub omega: F,
    pub g: F,
    pub g_n: F,
    pub g_1mn: F,
    pub omega_1mn: F,
    pub twiddles: Vec<F>,
    pub bit_rev: Vec<usize>,
    pub g_powers: Vec<F>,
    pub layer_offsets: Vec<usize>,
    pub total_tree_size: usize,
}

impl CircuitParams {
    pub fn new(log_n: usize, log_blowup: usize, log_felts_per_leaf: usize) -> Self {
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
        let g_1mn = g.exp_u64(P - 1 - (n as u64 - 1));
        let omega_1mn = omega.exp_u64(P - 1 - (n as u64 - 1));

        let mut twiddles = Vec::with_capacity(n_eval / 2);
        let mut acc = F::ONE;
        for _ in 0..n_eval / 2 { twiddles.push(acc); acc *= omega; }

        let bit_rev_table: Vec<usize> = (0..n_eval).map(|i| bit_reverse(i, log_total)).collect();

        let mut g_powers = Vec::with_capacity(n);
        let mut gp = F::ONE;
        for _ in 0..n { g_powers.push(gp); gp *= g; }

        let mut layer_offsets = vec![0usize];
        let mut acc_off = 0;
        for k in 0..tree_depth { acc_off += (n_leaves >> k) * DIGEST_LEN; layer_offsets.push(acc_off); }
        let total_tree_size = acc_off + DIGEST_LEN;

        Self {
            log_n, log_blowup, log_total, log_felts_per_leaf,
            n, n_eval, fpl, n_leaves, tree_depth, n_chunks_per_leaf,
            omega, g, g_n, g_1mn, omega_1mn,
            twiddles, bit_rev: bit_rev_table, g_powers,
            layer_offsets, total_tree_size,
        }
    }
}

// ── Reference computations ────────────────────────────────────────────────

pub fn reference_coset_fft(cp: &CircuitParams, coeffs: &[F]) -> Vec<F> {
    let mut data = vec![F::ZERO; cp.n_eval];
    let mut g_pow = F::ONE;
    for i in 0..cp.n { data[i] = coeffs[i] * g_pow; g_pow *= cp.g; }
    for i in 0..cp.n_eval {
        let j = bit_reverse(i, cp.log_total);
        if i < j { data.swap(i, j); }
    }
    for s in 1..=cp.log_total {
        let m = 1 << s;
        let half = m >> 1;
        let stride = cp.n_eval / m;
        let mut k = 0;
        while k < cp.n_eval {
            for j in 0..half {
                let u = data[k + j];
                let t = cp.twiddles[j * stride] * data[k + j + half];
                data[k + j] = u + t;
                data[k + j + half] = u - t;
            }
            k += m;
        }
    }
    data
}

pub fn poseidon_hash_chain(chunks: &[F]) -> [F; 8] {
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

pub fn poseidon_hash_pair(left: &[F; 8], right: &[F; 8]) -> [F; 8] {
    let mut input = [F::ZERO; 16];
    input[..8].copy_from_slice(left);
    input[8..].copy_from_slice(right);
    poseidon16_compress(input)
}

pub fn reference_merkle_root(cp: &CircuitParams, evals: &[F]) -> [F; 8] {
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

pub fn reference_syndrome(cp: &CircuitParams, evals: &[F], beta: F) -> F {
    assert!(cp.log_blowup == 1, "syndrome requires log_blowup == 1");
    let beta_n = beta.exp_u64(cp.n as u64);
    let mut x = cp.g;
    let mut w = cp.g_1mn;
    let mut sign = F::ONE;
    let mut s = F::ZERO;
    for _j in 0..cp.n_eval {
        let yj_n_inv = F::ONE / (cp.g_n * sign);
        let numer = beta_n * yj_n_inv - F::ONE;
        let denom = beta - x;
        s += evals[_j] * w * numer / denom;
        x *= cp.omega;
        w *= cp.omega_1mn;
        sign = F::ZERO - sign;
    }
    s
}

// ── Shared Merkle tree zkDSL generation ───────────────────────────────────

/// Emit leaf hashing + internal Merkle nodes. Leaves read from `data_src_name`.
/// Fully unrolled (Poseidon calls as precompile are fast regardless).
pub fn emit_merkle_tree(p: &mut String, cp: &CircuitParams, data_src_name: &str) {
    let chain_size = cp.n_leaves * cp.n_chunks_per_leaf * DIGEST_LEN;
    p.push_str("    zero_vec = Array(DIGEST_LEN)\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        zero_vec[i] = 0\n\n");
    p.push_str(&format!("    chain = Array({chain_size})\n"));
    p.push_str(&format!("    tree = Array(TOTAL_TREE_SIZE)\n\n"));

    for leaf in 0..cp.n_leaves {
        let ld = leaf * cp.fpl;
        let ch = leaf * cp.n_chunks_per_leaf * DIGEST_LEN;
        p.push_str(&format!("    poseidon16_compress(zero_vec, {data_src_name} + {ld}, chain + {ch})\n"));
        for c in 1..cp.n_chunks_per_leaf {
            p.push_str(&format!(
                "    poseidon16_compress(chain + {}, {data_src_name} + {}, chain + {})\n",
                ch + (c - 1) * DIGEST_LEN, ld + c * 8, ch + c * DIGEST_LEN,
            ));
        }
        let final_ch = ch + (cp.n_chunks_per_leaf - 1) * DIGEST_LEN;
        let tree_leaf = leaf * DIGEST_LEN;
        for k in 0..DIGEST_LEN {
            p.push_str(&format!("    tree[{}] = chain[{}]\n", tree_leaf + k, final_ch + k));
        }
    }

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
}

/// Emit root assertion + Fiat-Shamir challenge derivation for syndrome checks.
pub fn emit_root_assert_and_challenges(p: &mut String, cp: &CircuitParams) {
    let root_offset = cp.layer_offsets[cp.tree_depth];
    p.push_str(&format!("\n    root_ptr = tree + {root_offset}\n"));
    p.push_str("    pub_ptr = 0\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        assert root_ptr[i] == pub_ptr[i]\n\n");
}

/// Emit constants block for all variants.
pub fn emit_constants(p: &mut String, cp: &CircuitParams) {
    p.push_str(&format!("DIGEST_LEN = {DIGEST_LEN}\n"));
    p.push_str(&format!("N = {}\n", cp.n));
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
    p.push_str(&format!("OMEGA_1MN = {}\n", fmt_f(cp.omega_1mn)));
}

// ── Bench harness ─────────────────────────────────────────────────────────

pub struct BenchResult {
    pub prove_time: std::time::Duration,
    pub verify_time: std::time::Duration,
    pub metadata: ExecutionMetadata,
    pub peak_rss: u64,
}

pub fn run_bench(
    label: &str,
    cp: &CircuitParams,
    program_str: String,
    public_input: &[F],
    hints: HashMap<String, Vec<Vec<F>>>,
    log_inv_rate: usize,
) -> BenchResult {
    eprintln!("============================================================");
    eprintln!("leanVM bench: {label}");
    eprintln!("============================================================");
    eprintln!("  log_n={} n_eval={} n_leaves={}", cp.log_n, cp.n_eval, cp.n_leaves);
    eprintln!("  program: {} lines, {:.1} KB",
        program_str.lines().count(), program_str.len() as f64 / 1024.0);

    let t0 = Instant::now();
    let bytecode = compile_program(&ProgramSource::Raw(program_str));
    eprintln!("  compiled in {:.3}s", t0.elapsed().as_secs_f64());

    let witness = ExecutionWitness { preamble_memory_len: 0, hints };

    eprintln!("  proving...");
    let t0 = Instant::now();
    let proof = prove_execution(&bytecode, public_input, &witness, &default_whir_config(log_inv_rate), false).unwrap();
    let prove_time = t0.elapsed();

    let metadata = proof.metadata;
    let t0 = Instant::now();
    verify_execution(&bytecode, public_input, proof.proof).unwrap();
    let verify_time = t0.elapsed();

    let peak_rss = system_info::peak_rss_bytes();

    eprintln!("  prove: {:.3}s | verify: {:.3}s | cycles: {} | poseidons: {} | RSS: {:.2} GB",
        prove_time.as_secs_f64(), verify_time.as_secs_f64(),
        metadata.cycles, metadata.n_poseidons,
        peak_rss as f64 / (1u64 << 30) as f64);

    BenchResult { prove_time, verify_time, metadata, peak_rss }
}

pub fn make_public_input(root: &[F; 8]) -> Vec<F> {
    let mut pi = root.to_vec();
    pi.resize(pi.len().next_power_of_two(), F::ZERO);
    pi
}
