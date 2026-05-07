//! ZODA benchmark — full encode → open → verify → decode pipeline.
//!
//! Usage:
//!   cargo run --release -- --n 2048
//!   cargo run --release -- --n 2048 --samples 193 --decode

use std::time::Instant;
use zoda_bench::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(64);
    let num_samples: usize = args.iter().position(|a| a == "--samples")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(17);
    let do_decode: bool = args.iter().any(|a| a == "--decode");

    assert!(n.is_power_of_two(), "n must be a power of 2");
    assert!(n >= 8, "n must be at least 8");

    let n_prime = n;
    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let data_bytes = n * n_prime * 4;

    eprintln!("============================================================");
    eprintln!("ZODA (field extension variant, BLAKE3 Merkle)");
    eprintln!("============================================================");
    eprintln!("  n={n}  m={m}  data={:.2} MB  samples={num_samples}",
        data_bytes as f64 / (1024.0 * 1024.0));

    // Generate data
    let data: Vec<F> = (0..n * n_prime)
        .map(|i| F::from_u32((i % (P as usize - 2) + 1) as u32))
        .collect();

    // Encode
    let encoding = encode(&data, n, n_prime);
    let t = &encoding.timing;
    eprintln!("  encode breakdown:");
    eprintln!("    col FFT (X=G·X̃)   : {:>8.3}s", t.col_fft.as_secs_f64());
    eprintln!("    commit X (Merkle)  : {:>8.3}s", t.commit_x.as_secs_f64());
    eprintln!("    diag scale (X̃·D)  : {:>8.3}s", t.diag_scale.as_secs_f64());
    eprintln!("    row FFT (Y)        : {:>8.3}s", t.row_fft.as_secs_f64());
    eprintln!("    commit Y (Merkle)  : {:>8.3}s", t.commit_y.as_secs_f64());
    eprintln!("    TOTAL              : {:>8.3}s  ({:.1} MB/s)",
        t.total.as_secs_f64(),
        data_bytes as f64 / (1024.0 * 1024.0) / t.total.as_secs_f64());

    // Commitment (what verifiers receive: two BLAKE3 roots)
    let commitment = ZodaCommitment::from(&encoding);

    // Sample + open
    let row_indices = sample_indices(&encoding.root_x, 0, num_samples, m);
    let col_indices = sample_indices(&encoding.root_x, 1, num_samples, m_prime);

    let t0 = Instant::now();
    let x_openings: Vec<XRowOpening> = row_indices.iter()
        .map(|&i| encoding.open_x_row(i)).collect();
    let y_openings: Vec<YColOpening> = col_indices.iter()
        .map(|&j| encoding.open_y_col(j)).collect();
    let open_time = t0.elapsed();

    let x_bytes: usize = x_openings.iter()
        .map(|o| o.data.len() * 4 + o.proof.byte_size()).sum();
    let y_bytes: usize = y_openings.iter()
        .map(|o| o.data.len() * DIM * 4 + o.proof.byte_size()).sum();
    eprintln!("  open: {:.3}s  X={:.1} KB  Y={:.1} KB  total={:.1} KB",
        open_time.as_secs_f64(),
        x_bytes as f64 / 1024.0, y_bytes as f64 / 1024.0,
        (x_bytes + y_bytes) as f64 / 1024.0);

    // Verify
    let t0 = Instant::now();
    let result = verify(&commitment, &x_openings, &y_openings);
    let verify_time = t0.elapsed();
    eprintln!("  verify: {:.3}s  {}  ({}/{})",
        verify_time.as_secs_f64(),
        if result.accept() { "ACCEPT" } else { "REJECT" },
        result.consistency_passed, result.consistency_checks);
    assert!(result.accept(), "verification failed");

    // Decode
    if do_decode {
        let t0 = Instant::now();
        let decoded = decode_from_x_cols(&encoding.x_cols, n, n_prime);
        let dt = t0.elapsed();
        assert_eq!(decoded, data, "X-decode mismatch");
        eprintln!("  decode (X rows): {:.3}s CORRECT", dt.as_secs_f64());

        let t0 = Instant::now();
        let decoded_y = decode_from_y_rows(&encoding.y, &encoding.diag, n, n_prime);
        let dt = t0.elapsed();
        assert_eq!(decoded_y, data, "Y-decode mismatch");
        eprintln!("  decode (Y rows): {:.3}s CORRECT", dt.as_secs_f64());
    }

    // Summary
    let peak_rss = std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok()))
        .unwrap_or(0) * 1024;

    eprintln!();
    eprintln!("------------------------------------------------------------");
    eprintln!("  Data       : {:.2} MB", data_bytes as f64 / (1024.0*1024.0));
    eprintln!("  Encode     : {:.3}s  ({:.1} MB/s)",
        t.total.as_secs_f64(),
        data_bytes as f64 / (1024.0*1024.0) / t.total.as_secs_f64());
    eprintln!("  Verify     : {:.3}s  ({} samples)",
        verify_time.as_secs_f64(), num_samples);
    eprintln!("  Downloaded : {:.1} KB", (x_bytes + y_bytes) as f64 / 1024.0);
    eprintln!("  Peak RSS   : {:.2} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("------------------------------------------------------------");

    println!("{}", serde_json::json!({
        "n": n, "m": m,
        "data_bytes": data_bytes, "samples": num_samples,
        "encode_s": (t.total.as_secs_f64() * 1000.0).round() / 1000.0,
        "col_fft_s": (t.col_fft.as_secs_f64() * 1000.0).round() / 1000.0,
        "row_fft_s": (t.row_fft.as_secs_f64() * 1000.0).round() / 1000.0,
        "commit_x_s": (t.commit_x.as_secs_f64() * 1000.0).round() / 1000.0,
        "commit_y_s": (t.commit_y.as_secs_f64() * 1000.0).round() / 1000.0,
        "verify_s": (verify_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "open_bytes": x_bytes + y_bytes,
        "encode_throughput_mbps": (data_bytes as f64 / (1024.0*1024.0) / t.total.as_secs_f64() * 10.0).round() / 10.0,
        "peak_rss": peak_rss,
    }));
}
