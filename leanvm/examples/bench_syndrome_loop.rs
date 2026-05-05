//! Syndrome check, range() loops for accumulation.
//! Compact program for any n, but range() loop overhead.
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

    // Syndrome checks — range loops
    for i in 0..NUM_SYNDROME_CHECKS {
        p.push_str(&format!("    beta_{i} = chal_out_{i}[0]\n"));
        p.push_str(&format!("    bn_{i}: Mut = beta_{i}\n"));
        for _ in 0..cp.log_n { p.push_str(&format!("    bn_{i} = bn_{i} * bn_{i}\n")); }
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
    let r = run_bench("Syndrome range loops", &cp, program, &pi, hints, 1);
    println!("{}", serde_json::json!({
        "variant": "syndrome_loop", "log_n": log_n,
        "prove_s": (r.prove_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "cycles": r.metadata.cycles, "poseidons": r.metadata.n_poseidons,
        "memory": r.metadata.memory, "peak_rss": r.peak_rss,
    }));
}
