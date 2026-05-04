//! Benchmark: coset FFT + Poseidon16 Merkle commitment over KoalaBear.
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

const P: u64 = 0x7F000001; // KoalaBear prime
const DIGEST_LEN: usize = 8;
const TWO_ADICITY: usize = 24;

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

/// Pick smallest k such that leaf_bytes >= path_bytes for KoalaBear.
/// leaf = 2^k elements × 4 bytes; path = (log_total - k) hops × 32 bytes/hop.
/// Constraint: 4·2^k ≥ 32·(log_total − k)  ⟹  2^(k−3) + k ≥ log_total.
fn pick_log_felts_per_leaf_kb(log_total: usize) -> usize {
    let mut k = 3usize; // minimum: 8 felts/leaf (one Poseidon block)
    while (1usize << k.saturating_sub(3)) + k < log_total {
        k += 1;
    }
    k
}

// ── Precomputed tables (single source of truth) ───────────────────────────

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
    twiddles: Vec<F>,
    bit_rev: Vec<usize>,
    g_powers: Vec<F>,
    layer_offsets: Vec<usize>,
    total_tree_size: usize,
}

impl CircuitParams {
    fn new(log_n: usize, log_blowup: usize, log_felts_per_leaf: usize) -> Self {
        let log_total = log_n + log_blowup;
        let n = 1usize << log_n;
        let n_eval = 1usize << log_total;
        let fpl = 1usize << log_felts_per_leaf;
        let n_leaves = n_eval / fpl;
        let tree_depth = n_leaves.trailing_zeros() as usize;
        let n_chunks_per_leaf = fpl / 8;
        let omega = get_omega(log_total);

        let mut twiddles = Vec::with_capacity(n_eval / 2);
        let mut acc = F::ONE;
        for _ in 0..n_eval / 2 {
            twiddles.push(acc);
            acc *= omega;
        }

        let bit_rev: Vec<usize> = (0..n_eval).map(|i| bit_reverse(i, log_total)).collect();

        let g = F::from_u32(3);
        let mut g_powers = Vec::with_capacity(n);
        let mut gp = F::ONE;
        for _ in 0..n {
            g_powers.push(gp);
            gp *= g;
        }

        let mut layer_offsets = vec![0usize];
        let mut acc_off = 0;
        for k in 0..tree_depth {
            acc_off += (n_leaves >> k) * DIGEST_LEN;
            layer_offsets.push(acc_off);
        }
        let total_tree_size = acc_off + DIGEST_LEN;

        Self {
            log_n,
            log_blowup,
            log_total,
            log_felts_per_leaf,
            n,
            n_eval,
            fpl,
            n_leaves,
            tree_depth,
            n_chunks_per_leaf,
            omega,
            twiddles,
            bit_rev,
            g_powers,
            layer_offsets,
            total_tree_size,
        }
    }
}

// ── Reference FFT + Merkle (uses CircuitParams) ───────────────────────────

fn reference_coset_fft(cp: &CircuitParams, coeffs: &[F]) -> Vec<F> {
    let mut data = vec![F::ZERO; cp.n_eval];
    for i in 0..cp.n {
        data[i] = coeffs[i] * cp.g_powers[i];
    }
    for i in 0..cp.n_eval {
        let j = bit_reverse(i, cp.log_total);
        if i < j {
            data.swap(i, j);
        }
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

// ── zkDSL program generation (fully unrolled) ─────────────────────────────

fn generate_program(cp: &CircuitParams) -> String {
    let fmt_f = |f: F| format!("{}", f.as_canonical_u32());
    let fmt_vec_f = |v: &[F]| v.iter().map(|x| fmt_f(*x)).collect::<Vec<_>>().join(", ");
    let fmt_vec_usize =
        |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");

    let mut p = String::new();
    p.push_str("from snark_lib import *\n\n");

    // Compile-time constants
    p.push_str(&format!("DIGEST_LEN = {DIGEST_LEN}\n"));
    p.push_str(&format!("N = {}\n", cp.n));
    p.push_str(&format!("N_EVAL = {}\n", cp.n_eval));
    p.push_str(&format!("N_LEAVES = {}\n", cp.n_leaves));
    p.push_str(&format!("TOTAL_TREE_SIZE = {}\n", cp.total_tree_size));
    p.push_str(&format!("N_CHUNKS_PER_LEAF = {}\n", cp.n_chunks_per_leaf));
    p.push_str(&format!("FELTS_PER_LEAF = {}\n", cp.fpl));
    p.push_str(&format!("OMEGA = {}\n\n", fmt_f(cp.omega)));

    p.push_str(&format!("BIT_REV = [{}]\n", fmt_vec_usize(&cp.bit_rev)));
    p.push_str(&format!("G_POWERS = [{}]\n", fmt_vec_f(&cp.g_powers)));
    p.push_str(&format!("TWIDDLES = [{}]\n\n", fmt_vec_f(&cp.twiddles)));

    p.push_str("def main():\n");

    // Load coefficients from witness
    p.push_str("    coeffs = Array(N)\n");
    p.push_str("    hint_witness(\"coeffs\", coeffs)\n\n");

    // Layered FFT storage
    let data_size = cp.n_eval * (cp.log_total + 1);
    p.push_str(&format!("    data = Array({data_size})\n\n"));

    // Layer 0: unrolled bit-reversed twisted input
    p.push_str("    for i in unroll(0, N):\n");
    p.push_str("        data[BIT_REV[i]] = coeffs[i] * G_POWERS[i]\n");
    if cp.n < cp.n_eval {
        p.push_str("    for i in unroll(N, N_EVAL):\n");
        p.push_str("        data[BIT_REV[i]] = 0\n");
    }
    p.push_str("\n");

    // Butterfly layers — fully unrolled
    for s in 1..=cp.log_total {
        let m = 1usize << s;
        let half = m >> 1;
        let stride = cp.n_eval >> s;
        let n_groups = cp.n_eval / m;
        let prev = (s - 1) * cp.n_eval;
        let curr = s * cp.n_eval;

        p.push_str(&format!("    # Layer {s}\n"));
        for gi in 0..n_groups {
            let gs = gi * m;
            for j in 0..half {
                let tw = fmt_f(cp.twiddles[j * stride]);
                let u_idx = prev + gs + j;
                let v_idx = prev + gs + j + half;
                let out_a = curr + gs + j;
                let out_b = curr + gs + j + half;
                p.push_str(&format!("    u_{s}_{gi}_{j} = data[{u_idx}]\n"));
                p.push_str(&format!(
                    "    t_{s}_{gi}_{j} = {tw} * data[{v_idx}]\n"
                ));
                p.push_str(&format!(
                    "    data[{out_a}] = u_{s}_{gi}_{j} + t_{s}_{gi}_{j}\n"
                ));
                p.push_str(&format!(
                    "    data[{out_b}] = u_{s}_{gi}_{j} - t_{s}_{gi}_{j}\n"
                ));
            }
        }
    }
    p.push_str("\n");

    // Evals pointer
    let evals_offset = cp.log_total * cp.n_eval;
    p.push_str(&format!("    evals_ptr = data + {evals_offset}\n\n"));

    // Zero vector
    p.push_str("    zero_vec = Array(DIGEST_LEN)\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        zero_vec[i] = 0\n\n");

    // Leaf hashing — unrolled
    let chain_size = cp.n_leaves * cp.n_chunks_per_leaf * DIGEST_LEN;
    p.push_str(&format!("    chain = Array({chain_size})\n"));
    p.push_str(&format!("    tree = Array(TOTAL_TREE_SIZE)\n\n"));

    for leaf in 0..cp.n_leaves {
        let ld = evals_offset + leaf * cp.fpl;
        let ch = leaf * cp.n_chunks_per_leaf * DIGEST_LEN;
        p.push_str(&format!(
            "    poseidon16_compress(zero_vec, data + {ld}, chain + {ch})\n"
        ));
        for c in 1..cp.n_chunks_per_leaf {
            p.push_str(&format!(
                "    poseidon16_compress(chain + {}, data + {}, chain + {})\n",
                ch + (c - 1) * DIGEST_LEN,
                ld + c * 8,
                ch + c * DIGEST_LEN,
            ));
        }
        let final_ch = ch + (cp.n_chunks_per_leaf - 1) * DIGEST_LEN;
        let tree_leaf = leaf * DIGEST_LEN;
        for k in 0..DIGEST_LEN {
            p.push_str(&format!(
                "    tree[{}] = chain[{}]\n",
                tree_leaf + k,
                final_ch + k
            ));
        }
    }
    p.push_str("\n");

    // Internal Merkle nodes — unrolled
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

    // Twiddle recurrence constraint: tw[0]==1 and tw[i]*omega==tw[i+1]
    p.push_str("\n    # Verify twiddle hint\n");
    p.push_str("    tw = Array(N_EVAL / 2)\n");
    p.push_str("    hint_witness(\"twiddles\", tw)\n");
    p.push_str("    assert tw[0] == 1\n");
    p.push_str("    for i in unroll(0, N_EVAL / 2 - 1):\n");
    p.push_str("        assert tw[i] * OMEGA == tw[i + 1]\n");

    // Root assertion
    let root_offset = cp.layer_offsets[cp.tree_depth];
    p.push_str(&format!("\n    root_ptr = tree + {root_offset}\n"));
    p.push_str("    pub_ptr = 0\n");
    p.push_str("    for i in unroll(0, DIGEST_LEN):\n");
    p.push_str("        assert root_ptr[i] == pub_ptr[i]\n");
    p.push_str("    return\n");

    p
}

// ── Witness builder ────────────────────────────────────────────────────────

fn build_witness(cp: &CircuitParams, coeffs: &[F]) -> HashMap<String, Vec<Vec<F>>> {
    let mut hints = HashMap::new();
    hints.insert("coeffs".to_string(), vec![coeffs.to_vec()]);
    hints.insert("twiddles".to_string(), vec![cp.twiddles.clone()]);
    hints
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_n: usize = args
        .iter()
        .position(|a| a == "--log-n")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(8);
    let log_blowup: usize = args
        .iter()
        .position(|a| a == "--log-blowup")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(1);
    let log_inv_rate: usize = args
        .iter()
        .position(|a| a == "--log-inv-rate")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(1);
    let log_total = log_n + log_blowup;
    let lfpl = pick_log_felts_per_leaf_kb(log_total);
    let cp = CircuitParams::new(log_n, log_blowup, lfpl);

    eprintln!("============================================================");
    eprintln!("leanVM fft_merkle benchmark");
    eprintln!("============================================================");
    eprintln!("  log_n              = {log_n}");
    eprintln!("  log_blowup         = {log_blowup}");
    eprintln!("  log_felts_per_leaf = {lfpl}");
    eprintln!("  n_coeffs           = {}", cp.n);
    eprintln!("  n_eval             = {}", cp.n_eval);
    eprintln!("  n_leaves           = {}", cp.n_leaves);
    eprintln!("  log_inv_rate       = {log_inv_rate}");
    eprintln!();

    let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();

    eprintln!("[1/4] Computing reference ...");
    let t0 = Instant::now();
    let evals = reference_coset_fft(&cp, &coeffs);
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

    let witness = ExecutionWitness {
        preamble_memory_len: 0,
        hints: build_witness(&cp, &coeffs),
    };

    eprintln!("[4/4] Proving ...");
    let t0 = Instant::now();
    let proof = prove_execution(
        &bytecode,
        &public_input,
        &witness,
        &default_whir_config(log_inv_rate),
        false,
    )
    .unwrap();
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
    eprintln!(
        "  Prove peak RSS     : {:.2} GB",
        peak_rss as f64 / (1u64 << 30) as f64
    );
    eprintln!("  Verify time        : {:.3}s", verify_time.as_secs_f64());
    eprintln!("  Cycles             : {}", metadata.cycles);
    eprintln!("  Poseidon16 calls   : {}", metadata.n_poseidons);
    eprintln!("------------------------------------------------------------");

    let result = serde_json::json!({
        "log_n": log_n,
        "log_blowup": log_blowup,
        "log_felts_per_leaf": lfpl,
        "n_coeffs": cp.n,
        "n_eval": cp.n_eval,
        "n_leaves": cp.n_leaves,
        "prove_time_s": (prove_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "verify_time_s": (verify_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "peak_rss_bytes": peak_rss,
        "cycles": metadata.cycles,
        "n_poseidons": metadata.n_poseidons,
        "memory": metadata.memory,
    });
    println!("{result}");
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cp() -> CircuitParams {
        let log_total = 5;
        CircuitParams::new(4, 1, pick_log_felts_per_leaf_kb(log_total))
    }

    fn small_coeffs(cp: &CircuitParams) -> Vec<F> {
        (1..=cp.n as u32).map(F::from_u32).collect()
    }

    // ── Correctness: root assertion ──────────────────────────────────

    #[test]
    fn test_wrong_root_is_rejected() {
        let cp = small_cp();
        let coeffs = small_coeffs(&cp);
        let evals = reference_coset_fft(&cp, &coeffs);
        let mut wrong_root = reference_merkle_root(&cp, &evals);
        wrong_root[0] += F::ONE;

        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = wrong_root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);

        let result = prove_execution(
            &bytecode,
            &pi,
            &ExecutionWitness {
                preamble_memory_len: 0,
                hints: build_witness(&cp, &coeffs),
            },
            &default_whir_config(1),
            false,
        );
        assert!(result.is_err(), "Wrong root was accepted");
    }

    #[test]
    fn test_correct_root_is_accepted() {
        let cp = small_cp();
        let coeffs = small_coeffs(&cp);
        let evals = reference_coset_fft(&cp, &coeffs);
        let root = reference_merkle_root(&cp, &evals);

        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);

        let proof = prove_execution(
            &bytecode,
            &pi,
            &ExecutionWitness {
                preamble_memory_len: 0,
                hints: build_witness(&cp, &coeffs),
            },
            &default_whir_config(1),
            false,
        )
        .expect("Correct root should be accepted");

        verify_execution(&bytecode, &pi, proof.proof).expect("Verification should pass");
    }

    // ── Correctness: reference FFT known vector ─────────────────────

    #[test]
    fn test_reference_fft_constant_polynomial() {
        // P(x) = 5 for all x. Coset evals should all be 5.
        let cp = CircuitParams::new(1, 1, 3);
        let coeffs = vec![F::from_u32(5), F::ZERO];
        let evals = reference_coset_fft(&cp, &coeffs);
        assert_eq!(evals.len(), 4);
        for (i, &e) in evals.iter().enumerate() {
            assert_eq!(e, F::from_u32(5), "eval[{i}] should be 5");
        }
    }

    #[test]
    fn test_reference_fft_spot_check() {
        // P(x) = 1+2x+3x^2+4x^3. P(g) where g=3: 1+6+27+108 = 142.
        let cp = CircuitParams::new(2, 1, 3);
        let coeffs = vec![
            F::from_u32(1),
            F::from_u32(2),
            F::from_u32(3),
            F::from_u32(4),
        ];
        let evals = reference_coset_fft(&cp, &coeffs);
        assert_eq!(evals[0], F::from_u32(142));
    }

    // ── Soundness: twiddle constraint ───────────────────────────────

    #[test]
    fn test_wrong_twiddles_rejected() {
        let cp = small_cp();
        let coeffs = small_coeffs(&cp);
        let evals = reference_coset_fft(&cp, &coeffs);
        let root = reference_merkle_root(&cp, &evals);

        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);

        // Corrupt one twiddle
        let mut bad_hints = build_witness(&cp, &coeffs);
        bad_hints.get_mut("twiddles").unwrap()[0][1] += F::ONE;

        let result = prove_execution(
            &bytecode,
            &pi,
            &ExecutionWitness {
                preamble_memory_len: 0,
                hints: bad_hints,
            },
            &default_whir_config(1),
            false,
        );
        assert!(result.is_err(), "Corrupted twiddles should be rejected");
    }

    // ── Leaf packing invariant ──────────────────────────────────────

    #[test]
    fn test_leaf_size_exceeds_proof_path() {
        for log_total in 5..=25 {
            let k = pick_log_felts_per_leaf_kb(log_total);
            let fpl = 1usize << k;
            let n_leaves = (1usize << log_total) / fpl;
            let depth = n_leaves.trailing_zeros() as usize;
            let leaf_bytes = fpl * 4;
            let path_bytes = depth * DIGEST_LEN * 4;
            assert!(
                leaf_bytes >= path_bytes,
                "log_total={log_total}, k={k}: leaf={leaf_bytes}B < path={path_bytes}B"
            );
        }
    }

    #[test]
    fn test_leaf_packing_is_minimal() {
        for log_total in 5..=25 {
            let k = pick_log_felts_per_leaf_kb(log_total);
            if k > 3 {
                let prev_fpl = 1usize << (k - 1);
                let prev_leaves = (1usize << log_total) / prev_fpl;
                let prev_depth = prev_leaves.trailing_zeros() as usize;
                assert!(
                    prev_fpl * 4 < prev_depth * DIGEST_LEN * 4,
                    "log_total={log_total}: k-1={} also works, k={k} not minimal",
                    k - 1
                );
            }
        }
    }

    // ── Field arithmetic properties ─────────────────────────────────

    #[test]
    #[should_panic(expected = "exceeds KoalaBear 2-adicity")]
    fn test_omega_rejects_too_large_order() {
        get_omega(25);
    }

    #[test]
    fn test_omega_is_primitive_root() {
        // omega^(2^log_order) == 1 AND omega^(2^(log_order-1)) != 1
        for log_order in 1..=20 {
            let omega = get_omega(log_order);
            let order = 1u64 << log_order;
            assert_eq!(
                omega.exp_u64(order),
                F::ONE,
                "omega^order != 1 at log_order={log_order}"
            );
            assert_ne!(
                omega.exp_u64(order / 2),
                F::ONE,
                "omega is not primitive at log_order={log_order}"
            );
        }
    }

    // ── bit_reverse properties ──────────────────────────────────────

    #[test]
    fn test_bit_reverse_is_involution() {
        for bits in 1..=16 {
            for x in 0..(1usize << bits) {
                assert_eq!(
                    bit_reverse(bit_reverse(x, bits), bits),
                    x,
                    "bit_reverse is not involution: bits={bits}, x={x}"
                );
            }
        }
    }

    #[test]
    fn test_bit_reverse_is_permutation() {
        for bits in 1..=12 {
            let n = 1usize << bits;
            let mut seen = vec![false; n];
            for x in 0..n {
                let r = bit_reverse(x, bits);
                assert!(r < n, "out of range: bits={bits}, x={x}, rev={r}");
                assert!(!seen[r], "collision: bits={bits}, x={x}, rev={r}");
                seen[r] = true;
            }
        }
    }

    // ── CircuitParams consistency ───────────────────────────────────

    #[test]
    fn test_circuit_params_consistency() {
        for (log_n, log_blowup) in [(4, 1), (8, 1), (8, 2), (10, 1)] {
            let log_total = log_n + log_blowup;
            let lfpl = pick_log_felts_per_leaf_kb(log_total);
            let cp = CircuitParams::new(log_n, log_blowup, lfpl);

            assert_eq!(cp.n, 1 << log_n);
            assert_eq!(cp.n_eval, 1 << log_total);
            assert_eq!(cp.n_eval, cp.n << log_blowup);
            assert_eq!(cp.n_leaves * cp.fpl, cp.n_eval);
            assert_eq!(cp.fpl, cp.n_chunks_per_leaf * 8);
            assert!(cp.n_leaves.is_power_of_two());
            assert_eq!(cp.bit_rev.len(), cp.n_eval);
            assert_eq!(cp.g_powers.len(), cp.n);
            assert_eq!(cp.twiddles.len(), cp.n_eval / 2);
            assert_eq!(cp.layer_offsets.len(), cp.tree_depth + 1);
        }
    }

    #[test]
    fn test_twiddles_are_powers_of_omega() {
        let cp = CircuitParams::new(4, 1, 3);
        for i in 0..cp.twiddles.len() {
            assert_eq!(
                cp.twiddles[i],
                cp.omega.exp_u64(i as u64),
                "twiddle[{i}] != omega^{i}"
            );
        }
    }

    #[test]
    fn test_g_powers_are_geometric() {
        let cp = CircuitParams::new(6, 1, 3);
        let g = F::from_u32(3);
        for i in 0..cp.g_powers.len() {
            assert_eq!(
                cp.g_powers[i],
                g.exp_u64(i as u64),
                "g_powers[{i}] != 3^{i}"
            );
        }
    }

    // ── FFT algebraic properties ────────────────────────────────────

    #[test]
    fn test_fft_zero_polynomial() {
        let cp = CircuitParams::new(4, 1, 3);
        let coeffs = vec![F::ZERO; cp.n];
        let evals = reference_coset_fft(&cp, &coeffs);
        for (i, &e) in evals.iter().enumerate() {
            assert_eq!(e, F::ZERO, "FFT(0)[{i}] != 0");
        }
    }

    #[test]
    fn test_fft_linearity() {
        // FFT(a + b) == FFT(a) + FFT(b)
        let cp = CircuitParams::new(4, 1, 3);
        let a: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let b: Vec<F> = (100..100 + cp.n as u32).map(F::from_u32).collect();
        let ab: Vec<F> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();

        let fft_a = reference_coset_fft(&cp, &a);
        let fft_b = reference_coset_fft(&cp, &b);
        let fft_ab = reference_coset_fft(&cp, &ab);

        for i in 0..cp.n_eval {
            assert_eq!(
                fft_ab[i],
                fft_a[i] + fft_b[i],
                "FFT(a+b)[{i}] != FFT(a)[{i}]+FFT(b)[{i}]"
            );
        }
    }

    #[test]
    fn test_fft_scalar_homogeneity() {
        // FFT(c * a) == c * FFT(a)
        let cp = CircuitParams::new(4, 1, 3);
        let a: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let c = F::from_u32(7);
        let ca: Vec<F> = a.iter().map(|&x| c * x).collect();

        let fft_a = reference_coset_fft(&cp, &a);
        let fft_ca = reference_coset_fft(&cp, &ca);

        for i in 0..cp.n_eval {
            assert_eq!(
                fft_ca[i],
                c * fft_a[i],
                "FFT(c*a)[{i}] != c*FFT(a)[{i}]"
            );
        }
    }

    #[test]
    fn test_fft_evaluates_at_coset_points() {
        // Directly evaluate P(g * omega^k) via Horner and compare to FFT output.
        let cp = CircuitParams::new(3, 1, 3);
        let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let evals = reference_coset_fft(&cp, &coeffs);
        let g = F::from_u32(3);

        for k in 0..cp.n_eval {
            let point = g * cp.omega.exp_u64(k as u64);
            // Horner evaluation
            let mut val = F::ZERO;
            for i in (0..cp.n).rev() {
                val = val * point + coeffs[i];
            }
            assert_eq!(
                evals[k], val,
                "FFT[{k}] != P(g*omega^{k})"
            );
        }
    }

    // ── Merkle binding ──────────────────────────────────────────────

    #[test]
    fn test_different_data_different_roots() {
        let cp = CircuitParams::new(4, 1, pick_log_felts_per_leaf_kb(5));
        let a: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let mut b = a.clone();
        b[0] += F::ONE; // flip one coefficient

        let evals_a = reference_coset_fft(&cp, &a);
        let evals_b = reference_coset_fft(&cp, &b);
        let root_a = reference_merkle_root(&cp, &evals_a);
        let root_b = reference_merkle_root(&cp, &evals_b);

        assert_ne!(root_a, root_b, "Different polynomials must yield different roots");
    }

    #[test]
    fn test_merkle_single_eval_change_changes_root() {
        let cp = CircuitParams::new(4, 1, pick_log_felts_per_leaf_kb(5));
        let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let evals = reference_coset_fft(&cp, &coeffs);
        let root = reference_merkle_root(&cp, &evals);

        // Flip one eval in the middle
        let mut evals_mod = evals.clone();
        evals_mod[cp.n_eval / 2] += F::ONE;
        let root_mod = reference_merkle_root(&cp, &evals_mod);

        assert_ne!(root, root_mod, "Changing one eval must change the root");
    }

    #[test]
    fn test_hash_chain_order_dependent() {
        let a: Vec<F> = (1..=8).map(|i| F::from_u32(i)).collect();
        let mut b = a.clone();
        b.swap(0, 1);
        assert_ne!(
            poseidon_hash_chain(&a),
            poseidon_hash_chain(&b),
            "hash_chain must be order-dependent"
        );
    }

    // ── End-to-end across parameter sizes ───────────────────────────

    #[test]
    fn test_end_to_end_n6() {
        // Prove and verify at a different size than the n=4 tests.
        let cp = CircuitParams::new(6, 1, pick_log_felts_per_leaf_kb(7));
        let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let evals = reference_coset_fft(&cp, &coeffs);
        let root = reference_merkle_root(&cp, &evals);

        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);

        let proof = prove_execution(
            &bytecode,
            &pi,
            &ExecutionWitness {
                preamble_memory_len: 0,
                hints: build_witness(&cp, &coeffs),
            },
            &default_whir_config(1),
            false,
        )
        .expect("n=6 should prove");

        verify_execution(&bytecode, &pi, proof.proof).expect("n=6 should verify");
    }

    #[test]
    fn test_end_to_end_with_blowup_2() {
        // k=2 blowup instead of k=1 — different eval domain size.
        let cp = CircuitParams::new(4, 2, pick_log_felts_per_leaf_kb(6));
        let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
        let evals = reference_coset_fft(&cp, &coeffs);
        let root = reference_merkle_root(&cp, &evals);

        let bytecode = compile_program(&ProgramSource::Raw(generate_program(&cp)));
        let mut pi = root.to_vec();
        pi.resize(pi.len().next_power_of_two(), F::ZERO);

        let proof = prove_execution(
            &bytecode,
            &pi,
            &ExecutionWitness {
                preamble_memory_len: 0,
                hints: build_witness(&cp, &coeffs),
            },
            &default_whir_config(1),
            false,
        )
        .expect("n=4 k=2 should prove");

        verify_execution(&bytecode, &pi, proof.proof).expect("n=4 k=2 should verify");
    }
}
