//! Syndrome check, fully unrolled accumulation.
//! Fast cycles, compact-ish program (O(n_eval) lines, not O(n_eval × log)).
use starkdal_leanvm::*;
use std::collections::HashMap;

fn generate_program(cp: &CircuitParams) -> String {
    assert!(cp.log_blowup == 1, "syndrome requires log_blowup == 1");
    let mut p = String::new();
    p.push_str("from snark_lib import *\n\n");
    emit_constants(&mut p, cp);
    p.push_str("\n");

    p.push_str("def main():\n");
    p.push_str("    V = Array(N_EVAL)\n");
    p.push_str("    hint_witness(\"evals\", V)\n\n");
    emit_merkle_tree(&mut p, cp, "V");
    emit_root_assert_and_challenges(&mut p, cp);

    // Fiat-Shamir challenges
    let root_offset = cp.layer_offsets[cp.tree_depth];
    for i in 0..NUM_SYNDROME_CHECKS {
        p.push_str(&format!("    chal_out_{i} = Array(DIGEST_LEN)\n"));
        p.push_str(&format!("    ds_{i} = Array(DIGEST_LEN)\n"));
        p.push_str(&format!("    ds_{i}[0] = {i}\n"));
        for k in 1..DIGEST_LEN { p.push_str(&format!("    ds_{i}[{k}] = 0\n")); }
        p.push_str(&format!("    poseidon16_compress(tree + {root_offset}, ds_{i}, chal_out_{i})\n"));
    }
    p.push_str("\n");

    // Syndrome checks — fully unrolled
    // Precompute per-j constants: x_j = g·ω^j, w_j = g^(1-n)·ω^(j(1-n)), sign_j = (-1)^j
    let mut x = cp.g;
    let mut w = cp.g_1mn;
    let mut sign = F::ONE;
    let coset_points: Vec<F> = (0..cp.n_eval).map(|_| { let v = x; x *= cp.omega; v }).collect();
    x = cp.g; // reset
    let weights: Vec<F> = (0..cp.n_eval).map(|_| { let v = w; w *= cp.omega_1mn; v }).collect();
    let signs: Vec<F> = (0..cp.n_eval).map(|j| { if j % 2 == 0 { F::ONE } else { F::ZERO - F::ONE } }).collect();

    for i in 0..NUM_SYNDROME_CHECKS {
        p.push_str(&format!("    beta_{i} = chal_out_{i}[0]\n"));
        // beta^n via repeated squaring
        p.push_str(&format!("    bn_{i}: Mut = beta_{i}\n"));
        for _ in 0..cp.log_n { p.push_str(&format!("    bn_{i} = bn_{i} * bn_{i}\n")); }
        // Unrolled accumulation
        p.push_str(&format!("    s_{i}_0 = V[0] * {} * (bn_{i} * {} - 1) / (beta_{i} - {})\n",
            fmt_f(weights[0]), fmt_f(F::ONE / (cp.g_n * signs[0])), fmt_f(coset_points[0])));
        for j in 1..cp.n_eval {
            p.push_str(&format!(
                "    s_{i}_{j} = s_{i}_{prev} + V[{j}] * {w} * (bn_{i} * {yninv} - 1) / (beta_{i} - {xj})\n",
                prev = j - 1,
                w = fmt_f(weights[j]),
                yninv = fmt_f(F::ONE / (cp.g_n * signs[j])),
                xj = fmt_f(coset_points[j]),
            ));
        }
        p.push_str(&format!("    assert s_{i}_{} == 0\n\n", cp.n_eval - 1));
    }

    p.push_str("    return\n");
    p
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_n: usize = args.iter().position(|a| a == "--log-n").map(|i| args[i+1].parse().unwrap()).unwrap_or(8);
    let log_blowup = 1usize;
    let cp = CircuitParams::new(log_n, log_blowup, pick_log_felts_per_leaf_kb(log_n + log_blowup));
    let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
    let evals = reference_coset_fft(&cp, &coeffs);
    let root = reference_merkle_root(&cp, &evals);
    let pi = make_public_input(&root);
    let mut hints = HashMap::new();
    hints.insert("evals".to_string(), vec![evals]);
    let program = generate_program(&cp);
    let r = run_bench("Syndrome unrolled", &cp, program, &pi, hints, 1);
    println!("{}", serde_json::json!({
        "variant": "syndrome_unroll", "log_n": log_n,
        "prove_s": (r.prove_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "cycles": r.metadata.cycles, "poseidons": r.metadata.n_poseidons,
        "memory": r.metadata.memory, "peak_rss": r.peak_rss,
    }));
}
