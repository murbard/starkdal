//! ZODA benchmark — exercises the full encode → open → verify → decode pipeline.
//!
//! Usage:
//!   cargo run --release -- --n 2048
//!   cargo run --release -- --n 2048 --samples 193

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

    let n_prime = n; // square data matrix
    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let data_bytes = n * n_prime * 4;

    eprintln!("============================================================");
    eprintln!("ZODA full pipeline (field extension variant)");
    eprintln!("============================================================");
    eprintln!("  n={n}  n'={n_prime}  m={m}  m'={m_prime}");
    eprintln!("  data: {:.2} MB  |  samples: {num_samples}  |  decode: {do_decode}",
        data_bytes as f64 / (1024.0 * 1024.0));

    // ── Generate data ──
    let t0 = Instant::now();
    let data: Vec<F> = (0..n * n_prime)
        .map(|i| F::from_u32((i % (P as usize - 2) + 1) as u32))
        .collect();
    eprintln!("  data generation      : {:.3}s", t0.elapsed().as_secs_f64());

    // ── Encode ──
    let t0 = Instant::now();
    let encoding = encode(&data, n, n_prime);
    let encode_time = t0.elapsed();
    eprintln!("  encode (full ZODA)   : {:.3}s  ({:.1} MB/s)",
        encode_time.as_secs_f64(),
        data_bytes as f64 / (1024.0 * 1024.0) / encode_time.as_secs_f64());

    // ── Extract commitment (what verifiers receive) ──
    let commitment = ZodaCommitment::from(&encoding);

    // ── Sample indices ──
    let row_indices = sample_indices(&encoding.root_x, 0, num_samples, m);
    let col_indices = sample_indices(&encoding.root_x, 1, num_samples, m_prime);

    // ── Open (encoder sends openings to sampler) ──
    let t0 = Instant::now();
    let x_openings: Vec<XRowOpening> = row_indices.iter()
        .map(|&i| encoding.open_x_row(i))
        .collect();
    let y_openings: Vec<YColOpening> = col_indices.iter()
        .map(|&j| encoding.open_y_col(j))
        .collect();
    // Open Z entries at each (i, j) intersection
    let z_openings: Vec<ZEntryOpening> = row_indices.iter()
        .flat_map(|&i| col_indices.iter().map(move |&j| (i, j)))
        .map(|(i, j)| encoding.open_z_entry(i, j))
        .collect();
    let open_time = t0.elapsed();

    let x_open_bytes: usize = x_openings.iter()
        .map(|o| o.data.len() * 4 + o.proof.byte_size())
        .sum();
    let y_open_bytes: usize = y_openings.iter()
        .map(|o| o.data.len() * DIM * 4 + o.proof.byte_size())
        .sum();
    let z_open_bytes: usize = z_openings.iter()
        .map(|o| o.row_data.len() * DIM * 4 + o.proof.byte_size())
        .sum();
    let total_open_bytes = x_open_bytes + y_open_bytes + z_open_bytes;

    eprintln!("  open ({} X rows, {} Y cols, {} Z entries): {:.3}s",
        x_openings.len(), y_openings.len(), z_openings.len(),
        open_time.as_secs_f64());
    eprintln!("    opening sizes: X={:.1} KB  Y={:.1} KB  Z={:.1} KB  total={:.1} KB",
        x_open_bytes as f64 / 1024.0, y_open_bytes as f64 / 1024.0,
        z_open_bytes as f64 / 1024.0, total_open_bytes as f64 / 1024.0);

    // ── Verify (sampler checks consistency) ──
    let t0 = Instant::now();
    let result = verify(&commitment, &x_openings, &y_openings, &z_openings);
    let verify_time = t0.elapsed();
    eprintln!("  verify               : {:.3}s  {}",
        verify_time.as_secs_f64(),
        if result.accept() { "ACCEPT" } else { "REJECT" });
    eprintln!("    consistency: {}/{}  z-entries: {}/{}  merkle: {}",
        result.consistency_passed, result.consistency_checks,
        result.z_passed, result.z_checks,
        if result.merkle_ok { "ok" } else { "FAIL" });
    assert!(result.accept(), "verification failed");

    // ── Decode (optional: reconstruct X̃ from X rows) ──
    if do_decode {
        let t0 = Instant::now();
        let decoded = decode_from_x_rows(&encoding.x, n, n_prime);
        let decode_time = t0.elapsed();
        let correct = decoded == data;
        eprintln!("  decode (from X rows) : {:.3}s  {}",
            decode_time.as_secs_f64(), if correct { "CORRECT" } else { "MISMATCH" });
        assert!(correct, "decoded data does not match original");

        // Also test Y-based decoding
        let t0 = Instant::now();
        let decoded_y = decode_from_y_rows(&encoding.y, &encoding.diag, n, n_prime);
        let decode_y_time = t0.elapsed();
        let correct_y = decoded_y == data;
        eprintln!("  decode (from Y rows) : {:.3}s  {}",
            decode_y_time.as_secs_f64(), if correct_y { "CORRECT" } else { "MISMATCH" });
        assert!(correct_y, "Y-decoded data does not match original");
    }

    // ── Summary ──
    let peak_rss = {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
            })
            .unwrap_or(0) * 1024
    };

    eprintln!();
    eprintln!("------------------------------------------------------------");
    eprintln!("  Data         : {:.2} MB ({n} x {n_prime} x 4B)", data_bytes as f64 / (1024.0*1024.0));
    eprintln!("  Encode       : {:.3}s  ({:.1} MB/s)", encode_time.as_secs_f64(),
        data_bytes as f64 / (1024.0*1024.0) / encode_time.as_secs_f64());
    eprintln!("  Open+Verify  : {:.3}s  ({:.1} KB downloaded)",
        (open_time + verify_time).as_secs_f64(), total_open_bytes as f64 / 1024.0);
    eprintln!("  Peak RSS     : {:.2} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("------------------------------------------------------------");

    println!("{}", serde_json::json!({
        "n": n, "n_prime": n_prime, "m": m, "m_prime": m_prime,
        "data_bytes": data_bytes, "samples": num_samples,
        "encode_s": (encode_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "open_s": (open_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "verify_s": (verify_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "open_bytes": total_open_bytes,
        "consistency_checks": result.consistency_checks,
        "z_checks": result.z_checks,
        "encode_throughput_mbps": (data_bytes as f64 / (1024.0*1024.0) / encode_time.as_secs_f64() * 10.0).round() / 10.0,
        "peak_rss": peak_rss,
    }));
}
