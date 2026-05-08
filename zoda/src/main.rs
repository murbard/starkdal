//! ZODA Tensor Variation (Appendix E) — full encode → open → verify → decode pipeline.
//!
//! Z = G·X̃·G'^T entirely over base field (4× expansion). Proof vectors
//! z_r, z'_{r'} in extension field (negligible network cost).
//!
//! Usage:
//!   cargo run --release -- --n 4096             # 64 MB encode
//!   cargo run --release -- --n 4096 --decode    # + decode roundtrip
//!   cargo run --release -- --n 8192 --samples 50
use std::time::Instant;
use zoda_bench::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(64);
    let num_samples: usize = args.iter().position(|a| a == "--samples")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(17);
    let do_decode: bool = args.iter().any(|a| a == "--decode");

    assert!(n.is_power_of_two() && n >= 32);
    let n_prime = n;
    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let data_bytes = n * n_prime * 4;
    let z_bytes = m * m_prime * 4;

    eprintln!("============================================================");
    eprintln!("ZODA Tensor Variation (Appendix E) — all base-field NTTs");
    eprintln!("============================================================");
    eprintln!("  n={n}  m={m}  data={:.2} MB  Z={:.2} MB (4× expansion)",
        data_bytes as f64 / (1024.0 * 1024.0), z_bytes as f64 / (1024.0 * 1024.0));

    let data: Vec<F> = (0..n * n_prime)
        .map(|i| F::from_u32((i % (P as usize - 2) + 1) as u32))
        .collect();

    let encoding = encode(&data, n, n_prime);
    let t = &encoding.timing;
    eprintln!("  encode breakdown:");
    eprintln!("    row NTT (X̃·G'^T)  : {:>8.3}s", t.row_fft.as_secs_f64());
    eprintln!("    col NTT (G·W')     : {:>8.3}s", t.col_fft.as_secs_f64());
    eprintln!("    commit (2 trees)   : {:>8.3}s", t.commit.as_secs_f64());
    eprintln!("    proof vectors      : {:>8.3}s", t.proof_vecs.as_secs_f64());
    eprintln!("    TOTAL              : {:>8.3}s  ({:.1} MB/s data, {:.1} MB/s NTT)",
        t.total.as_secs_f64(),
        data_bytes as f64 / (1024.0 * 1024.0) / t.total.as_secs_f64(),
        data_bytes as f64 / (1024.0 * 1024.0) / (t.row_fft + t.col_fft).as_secs_f64());

    // Sample + open
    let row_root = encoding.tree_rows.root();
    let col_root = encoding.tree_cols.root();
    let row_indices = sample_indices(&row_root, 0, num_samples, m);
    let col_indices = sample_indices(&col_root, 1, num_samples, m_prime);

    let t0 = Instant::now();
    let row_opens: Vec<RowOpening> = row_indices.iter().map(|&i| encoding.open_row(i)).collect();
    let col_opens: Vec<ColOpening> = col_indices.iter().map(|&j| encoding.open_col(j)).collect();
    let open_time = t0.elapsed();
    eprintln!("  open: {:.3}s", open_time.as_secs_f64());

    // Verify (using only the commitment, not the full encoding)
    let commitment = ZodaCommitment::from(&encoding);
    let t0 = Instant::now();
    let result = verify(&commitment, &row_opens, &col_opens);
    let verify_time = t0.elapsed();
    eprintln!("  verify: {:.3}s  {}  (rows {}/{}, cols {}/{}, cross {})",
        verify_time.as_secs_f64(),
        if result.accept() { "ACCEPT" } else { "REJECT" },
        result.row_passed, result.row_checks,
        result.col_passed, result.col_checks,
        result.cross_check);
    assert!(result.accept(), "verification failed");

    // Decode
    if do_decode {
        let z_rows: Vec<Vec<F>> = (0..m)
            .map(|row| (0..m_prime).map(|col| encoding.z_col_vecs[col][row]).collect())
            .collect();
        let t0 = Instant::now();
        let decoded = decode_from_rows(&z_rows, m, m_prime, n, n_prime);
        let dt = t0.elapsed();
        assert_eq!(decoded, data, "decode mismatch");
        eprintln!("  decode: {:.3}s CORRECT", dt.as_secs_f64());
    }

    let peak_rss = std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok()))
        .unwrap_or(0) * 1024;

    eprintln!();
    eprintln!("------------------------------------------------------------");
    eprintln!("  Data       : {:.2} MB (4× → {:.2} MB encoded)",
        data_bytes as f64 / (1024.0*1024.0), z_bytes as f64 / (1024.0*1024.0));
    eprintln!("  NTT only   : {:.3}s ({:.0} MB/s per core on {} data)",
        (t.row_fft + t.col_fft).as_secs_f64(),
        data_bytes as f64 / (1024.0*1024.0) / (t.row_fft + t.col_fft).as_secs_f64() / num_cpus() as f64,
        "user");
    eprintln!("  Peak RSS   : {:.2} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("------------------------------------------------------------");

    println!("{}", serde_json::json!({
        "n": n, "m": m, "data_bytes": data_bytes, "z_bytes": z_bytes,
        "row_fft_s": (t.row_fft.as_secs_f64() * 1000.0).round() / 1000.0,
        "col_fft_s": (t.col_fft.as_secs_f64() * 1000.0).round() / 1000.0,
        "commit_s": (t.commit.as_secs_f64() * 1000.0).round() / 1000.0,
        "proof_vecs_s": (t.proof_vecs.as_secs_f64() * 1000.0).round() / 1000.0,
        "total_s": (t.total.as_secs_f64() * 1000.0).round() / 1000.0,
        "verify_s": (verify_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "data_throughput_mbps": (data_bytes as f64 / (1024.0*1024.0) / t.total.as_secs_f64() * 10.0).round() / 10.0,
        "ntt_throughput_mbps": (data_bytes as f64 / (1024.0*1024.0) / (t.row_fft + t.col_fft).as_secs_f64() * 10.0).round() / 10.0,
        "peak_rss": peak_rss,
    }));
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
