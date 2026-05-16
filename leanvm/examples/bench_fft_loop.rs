//! Approach 1a: FFT + Merkle in circuit, range() loops for butterflies.
//! Compact program, but range() loop overhead (~10x slower than unroll).
//! Superseded by bench_syndrome_unroll (syndrome check avoids FFT entirely).
use starkdal_leanvm::*;
use std::collections::HashMap;

fn generate_program(cp: &CircuitParams) -> String {
    let mut p = String::new();
    p.push_str("from snark_lib import *\n\n");
    emit_constants(&mut p, cp);
    p.push_str("\n");

    // Module-level const arrays (must be outside def for parser to handle large arrays)
    let fmt_vec_f = |v: &[F]| v.iter().map(|x| fmt_f(*x)).collect::<Vec<_>>().join(", ");
    let fmt_vec_usize = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    p.push_str(&format!("BIT_REV = [{}]\n", fmt_vec_usize(&cp.bit_rev)));
    p.push_str(&format!("G_POWERS = [{}]\n\n", fmt_vec_f(&cp.g_powers)));

    p.push_str("def main():\n");
    p.push_str("    coeffs = Array(N)\n");
    p.push_str("    hint_witness(\"coeffs\", coeffs)\n");
    p.push_str("    tw = Array(N_EVAL / 2)\n");
    p.push_str("    hint_witness(\"twiddles\", tw)\n\n");

    let data_size = cp.n_eval * (cp.log_total + 1);
    p.push_str(&format!("    data = Array({data_size})\n"));
    p.push_str("    for i in unroll(0, N):\n");
    p.push_str("        data[BIT_REV[i]] = coeffs[i] * G_POWERS[i]\n");
    if cp.n < cp.n_eval {
        p.push_str("    for i in unroll(N, N_EVAL):\n");
        p.push_str("        data[BIT_REV[i]] = 0\n");
    }
    // Butterfly layers: range() for group + j, tw[] runtime array
    for s in 1..=cp.log_total {
        let m = 1usize << s;
        let half = m >> 1;
        let stride = cp.n_eval >> s;
        let n_groups = cp.n_eval / m;
        let prev = (s - 1) * cp.n_eval;
        let curr = s * cp.n_eval;
        p.push_str(&format!("    for g{s} in range(0, {n_groups}):\n"));
        p.push_str(&format!("        gs{s} = g{s} * {m}\n"));
        p.push_str(&format!("        for j{s} in range(0, {half}):\n"));
        p.push_str(&format!("            u{s} = data[{prev} + gs{s} + j{s}]\n"));
        p.push_str(&format!(
            "            t{s} = tw[j{s} * {stride}] * data[{prev} + gs{s} + j{s} + {half}]\n"
        ));
        p.push_str(&format!("            data[{curr} + gs{s} + j{s}] = u{s} + t{s}\n"));
        p.push_str(&format!(
            "            data[{curr} + gs{s} + j{s} + {half}] = u{s} - t{s}\n"
        ));
    }
    let evals_offset = cp.log_total * cp.n_eval;
    p.push_str(&format!("\n    evals_ptr = data + {evals_offset}\n\n"));
    emit_merkle_tree(&mut p, cp, "evals_ptr");
    emit_root_assert_and_challenges(&mut p, cp);
    // Twiddle constraint
    p.push_str("    assert tw[0] == 1\n");
    p.push_str("    for i in unroll(0, N_EVAL / 2 - 1):\n");
    p.push_str("        assert tw[i] * OMEGA == tw[i + 1]\n");
    p.push_str("    return\n");
    p
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_n: usize = args
        .iter()
        .position(|a| a == "--log-n")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(8);
    let log_blowup = 1usize;
    let cp = CircuitParams::new(log_n, log_blowup, pick_log_felts_per_leaf_kb(log_n + log_blowup));
    let coeffs: Vec<F> = (1..=cp.n as u32).map(F::from_u32).collect();
    let evals = reference_coset_fft(&cp, &coeffs);
    let root = reference_merkle_root(&cp, &evals);
    let pi = make_public_input(&root);
    let mut hints = HashMap::new();
    hints.insert("coeffs".to_string(), vec![coeffs]);
    hints.insert("twiddles".to_string(), vec![cp.twiddles.clone()]);
    let program = generate_program(&cp);
    let r = run_bench("FFT range loops", &cp, program, &pi, hints, 1);
    println!(
        "{}",
        serde_json::json!({
            "variant": "fft_loop", "log_n": log_n,
            "prove_s": (r.prove_time.as_secs_f64() * 1000.0).round() / 1000.0,
            "cycles": r.metadata.cycles, "poseidons": r.metadata.n_poseidons,
            "memory": r.metadata.memory, "peak_rss": r.peak_rss,
        })
    );
}
