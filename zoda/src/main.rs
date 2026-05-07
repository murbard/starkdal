//! ZODA (Zero-Overhead Data Availability) benchmark.
//!
//! Implements the field-extension variant of the ZODA construction from
//! Evans, Mohnblatt, Angeris (ePrint 2025/034).
//!
//! Data X̃ ∈ F^{n×n'} is column-encoded via RS into X = G·X̃ ∈ F^{m×n'}.
//! A random diagonal D (in extension field E) is derived via Fiat-Shamir from
//! the Merkle commitment of X's rows. Then Y = X̃·D·G'^T ∈ E^{n×m'} is the
//! randomized row encoding. Z = G·Y ∈ E^{m×m'} is the full tensor encoding.
//!
//! Verification: sample |S| rows of X and |S'| cols of Y, check consistency.
//!
//! Usage:
//!   cargo run --release -- --n 2048
//!   cargo run --release -- --n 2048 --samples 193

use backend::*;
use std::time::Instant;
use utils::poseidon16_compress;

// KoalaBear types from the backend
type F = KoalaBear;
type EF = QuinticExtensionFieldKB;

const P: u64 = 0x7F000001;
const DIGEST_LEN: usize = 8;
const DIM: usize = 5; // extension degree

// ── NTT / RS encoding ────────────────────────────────────────────────────

/// Primitive nth root of unity in KoalaBear.
fn get_omega(log_order: usize) -> F {
    F::from_u32(3).exp_u64((P - 1) >> log_order)
}

/// Coset FFT: evaluate polynomial (given by `coeffs`) on coset g·<ω>.
/// g = 3 (generator of multiplicative group), ω = primitive nth root.
fn coset_fft(coeffs: &[F], log_n: usize) -> Vec<F> {
    let n = 1 << log_n;
    let g = F::from_u32(3);
    let omega = get_omega(log_n);

    // Scale coefficients by g^i
    let mut data = vec![F::ZERO; n];
    let mut g_pow = F::ONE;
    for i in 0..coeffs.len().min(n) {
        data[i] = coeffs[i] * g_pow;
        g_pow *= g;
    }

    // Bit-reverse permutation
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    for s in 1..=log_n {
        let m = 1 << s;
        let half = m >> 1;
        let stride = n / m;
        let mut k = 0;
        while k < n {
            let mut w = F::ONE;
            for j in 0..half {
                let u = data[k + j];
                let t = w * data[k + j + half];
                data[k + j] = u + t;
                data[k + j + half] = u - t;
                w *= omega.exp_u64(stride as u64); // twiddle
            }
            k += m;
        }
    }
    data
}

/// Precompute twiddle factors for NTT.
fn precompute_twiddles(log_n: usize) -> Vec<F> {
    let n = 1 << log_n;
    let omega = get_omega(log_n);
    let mut tw = Vec::with_capacity(n / 2);
    let mut acc = F::ONE;
    for _ in 0..n / 2 {
        tw.push(acc);
        acc *= omega;
    }
    tw
}

/// Fast coset FFT using precomputed twiddles.
fn coset_fft_fast(coeffs: &[F], log_n: usize, twiddles: &[F]) -> Vec<F> {
    let n = 1 << log_n;
    let g = F::from_u32(3);
    let mut data = vec![F::ZERO; n];
    let mut g_pow = F::ONE;
    for i in 0..coeffs.len().min(n) {
        data[i] = coeffs[i] * g_pow;
        g_pow *= g;
    }
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }
    for s in 1..=log_n {
        let m = 1 << s;
        let half = m >> 1;
        let stride = n / m;
        let mut k = 0;
        while k < n {
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

/// Coset FFT over extension field (column of E-valued matrix).
fn coset_fft_ef(coeffs: &[EF], log_n: usize, twiddles: &[F]) -> Vec<EF> {
    let n = 1 << log_n;
    let g = F::from_u32(3);
    let mut data = vec![EF::ZERO; n];
    let mut g_pow = F::ONE;
    for i in 0..coeffs.len().min(n) {
        data[i] = coeffs[i] * g_pow;
        g_pow *= g;
    }
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }
    for s in 1..=log_n {
        let m = 1 << s;
        let half = m >> 1;
        let stride = n / m;
        let mut k = 0;
        while k < n {
            for j in 0..half {
                let u = data[k + j];
                let t = data[k + j + half] * twiddles[j * stride]; // EF * F
                data[k + j] = u + t;
                data[k + j + half] = u - t;
            }
            k += m;
        }
    }
    data
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

// ── Poseidon Merkle ──────────────────────────────────────────────────────

fn poseidon_hash(left: &[F; 8], right: &[F; 8]) -> [F; 8] {
    let mut input = [F::ZERO; 16];
    input[..8].copy_from_slice(left);
    input[8..].copy_from_slice(right);
    poseidon16_compress(input)
}

fn poseidon_chain(data: &[F]) -> [F; 8] {
    assert!(!data.is_empty() && data.len() % 8 == 0);
    let mut state = [F::ZERO; 8];
    for chunk in data.chunks_exact(8) {
        let mut input = [F::ZERO; 16];
        input[..8].copy_from_slice(&state);
        input[8..].copy_from_slice(chunk);
        state = poseidon16_compress(input);
    }
    state
}

/// Merkle root over rows. Each row is hashed via poseidon_chain, then tree.
fn merkle_root_of_rows(rows: &[Vec<F>]) -> [F; 8] {
    // Hash each row (parallel)
    let mut leaves: Vec<[F; 8]> = rows.par_iter().map(|row| poseidon_chain(row)).collect();
    // Build tree
    while leaves.len() > 1 {
        if leaves.len() % 2 != 0 {
            leaves.push([F::ZERO; 8]); // pad
        }
        leaves = (0..leaves.len() / 2)
            .map(|i| poseidon_hash(&leaves[2 * i], &leaves[2 * i + 1]))
            .collect();
    }
    leaves[0]
}

// ── ZODA construction ────────────────────────────────────────────────────

/// Derive |n'| extension field challenges from a Merkle root via Fiat-Shamir.
fn derive_diagonal(root: &[F; 8], n_prime: usize) -> Vec<EF> {
    let mut diag = Vec::with_capacity(n_prime);
    for i in 0..n_prime {
        let mut inp = [F::ZERO; 16];
        inp[..8].copy_from_slice(root);
        inp[8] = F::from_u32(i as u32);
        inp[9] = F::from_u32(0x5A4F4441); // "ZODA" domain separator
        let h = poseidon16_compress(inp);
        diag.push(EF::from_basis_coefficients_slice(&h[..DIM]).unwrap());
    }
    diag
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(64);
    let num_samples: usize = args.iter().position(|a| a == "--samples")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(17);

    // Parameters
    // X̃ ∈ F^{n × n'} where n' = n (square data matrix)
    // Rate 1/2: m = 2n, m' = 2n'
    let n_prime = n;
    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let log_m = m.trailing_zeros() as usize;
    let log_m_prime = m_prime.trailing_zeros() as usize;
    let data_bytes = n * n_prime * 4; // data = n² field elements × 4 bytes

    eprintln!("============================================================");
    eprintln!("ZODA benchmark (field extension variant)");
    eprintln!("============================================================");
    eprintln!("  n={n}  n'={n_prime}  m={m}  m'={m_prime}");
    eprintln!("  data: {:.2} MB  |  full encoding: {:.2} MB (base) + {:.2} MB (ext)",
        data_bytes as f64 / (1024.0 * 1024.0),
        (m * n_prime * 4) as f64 / (1024.0 * 1024.0),
        (n * m_prime * DIM * 4 + m * m_prime * DIM * 4) as f64 / (1024.0 * 1024.0));
    eprintln!("  samples: {num_samples} rows + {num_samples} cols");

    // ── Step 1: Generate data X̃ ──
    let t0 = Instant::now();
    let x_tilde: Vec<Vec<F>> = (0..n)
        .map(|row| {
            (0..n_prime)
                .map(|col| F::from_u32(((row * n_prime + col) % (P as usize - 2) + 1) as u32))
                .collect()
        })
        .collect();
    eprintln!("  data generation: {:.3}s", t0.elapsed().as_secs_f64());

    // ── Step 2: Column encode X = G · X̃ ──
    // Each column of X̃ is RS-encoded via coset FFT from n coefficients to m evaluations.
    let t0 = Instant::now();
    let twiddles_m = precompute_twiddles(log_m);
    // X is m × n', stored as n' columns of length m
    let x_cols: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let coeffs: Vec<F> = (0..n).map(|row| x_tilde[row][col]).collect();
            coset_fft_fast(&coeffs, log_m, &twiddles_m)
        })
        .collect();
    // Convert to row-major for Merkle: X_rows[i] = row i of X
    let x_rows: Vec<Vec<F>> = (0..m)
        .map(|i| (0..n_prime).map(|col| x_cols[col][i]).collect())
        .collect();
    let col_encode_time = t0.elapsed();
    eprintln!("  column encode (X=G·X̃): {:.3}s", col_encode_time.as_secs_f64());

    // ── Step 3: Commit to rows of X ──
    let t0 = Instant::now();
    let x_root = merkle_root_of_rows(&x_rows);
    let commit_time = t0.elapsed();
    eprintln!("  commit (Merkle root): {:.3}s", commit_time.as_secs_f64());

    // ── Step 4: Derive random diagonal D ∈ E^{n'} from commitment ──
    let t0 = Instant::now();
    let diag = derive_diagonal(&x_root, n_prime);
    let diag_time = t0.elapsed();
    eprintln!("  derive diagonal: {:.3}s", diag_time.as_secs_f64());

    // ── Step 5: Compute X̃·D (scale columns of X̃ by diagonal) ──
    // Result: n × n' matrix in E
    let t0 = Instant::now();
    let x_tilde_d: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| {
            (0..n_prime)
                .map(|col| EF::from(x_tilde[row][col]) * diag[col])
                .collect()
        })
        .collect();
    let scale_time = t0.elapsed();
    eprintln!("  diagonal scale (X̃·D): {:.3}s", scale_time.as_secs_f64());

    // ── Step 6: Row encode Y = (X̃·D) · G'^T ──
    // Each row of X̃·D is RS-encoded from n' coefficients to m' evaluations.
    // Y is n × m', stored as rows.
    let t0 = Instant::now();
    let twiddles_mp = precompute_twiddles(log_m_prime);
    let y_rows: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| coset_fft_ef(&x_tilde_d[row], log_m_prime, &twiddles_mp))
        .collect();
    let row_encode_time = t0.elapsed();
    eprintln!("  row encode (Y=(X̃·D)·G'^T): {:.3}s", row_encode_time.as_secs_f64());

    // ── Step 7: Full encode Z = G · Y ──
    // Each column of Y is RS-encoded from n rows to m rows.
    // Z is m × m', stored column-major.
    let t0 = Instant::now();
    let z_cols: Vec<Vec<EF>> = (0..m_prime)
        .into_par_iter()
        .map(|col| {
            let y_col: Vec<EF> = (0..n).map(|row| y_rows[row][col]).collect();
            coset_fft_ef(&y_col, log_m, &twiddles_m)
        })
        .collect();
    let full_encode_time = t0.elapsed();
    eprintln!("  full encode (Z=G·Y): {:.3}s", full_encode_time.as_secs_f64());

    let total_encode = col_encode_time + commit_time + diag_time + scale_time + row_encode_time + full_encode_time;
    eprintln!("  TOTAL ENCODE: {:.3}s", total_encode.as_secs_f64());

    // ── Verification: sample |S| rows of X, |S'| cols of Y ──
    // Check: X_S · D · g'_j == (G · y_j)_S for each j ∈ S'
    let t0 = Instant::now();

    // Sample row indices (deterministic for reproducibility)
    let sample_rows: Vec<usize> = (0..num_samples).map(|i| (i * 7 + 3) % m).collect();
    let sample_cols: Vec<usize> = (0..num_samples).map(|i| (i * 11 + 5) % m_prime).collect();

    // For each sampled column j, check consistency
    let omega_mp = get_omega(log_m_prime);
    let g = F::from_u32(3);
    let mut checks_passed = 0usize;
    let mut checks_total = 0usize;

    for &j in &sample_cols {
        // g'_j is the jth column of G' (RS matrix).
        // For coset RS: g'_j[k] = (g · ω^j)^k for k = 0..n'-1
        // But actually G' maps coefficients → evaluations, and g'_j is an evaluation vector.
        // The consistency check is: X_S · D · g'_j == (G · y_j)_S
        //
        // LHS: for each row i ∈ S, compute sum_k X[i][k] · D[k] · (eval_point_j)^k
        // where eval_point_j = g · ω^j (the jth evaluation point of the coset)
        let eval_point = g * omega_mp.exp_u64(j as u64);

        for &i in &sample_rows {
            // LHS: X[i] · D · g'_j = sum_k x_rows[i][k] · diag[k] · eval_point^k
            let mut lhs = EF::ZERO;
            let mut ep_pow = EF::ONE; // eval_point^k as EF
            let ep_ef = EF::from(eval_point);
            for k in 0..n_prime {
                lhs += EF::from(x_rows[i][k]) * diag[k] * ep_pow;
                ep_pow *= ep_ef;
            }

            // RHS: Z[i][j] = z_cols[j][i]
            let rhs = z_cols[j][i];

            checks_total += 1;
            if lhs == rhs {
                checks_passed += 1;
            }
        }
    }
    let verify_time = t0.elapsed();
    eprintln!("  verification ({checks_total} checks): {:.3}s — {checks_passed}/{checks_total} passed",
        verify_time.as_secs_f64());

    // ── Summary ──
    let peak_rss = {
        // Rough estimate from /proc/self/status
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
    eprintln!("  Data payload   : {:.2} MB ({} × {} × 4B)", data_bytes as f64 / (1024.0*1024.0), n, n_prime);
    eprintln!("  Total encode   : {:.3}s", total_encode.as_secs_f64());
    eprintln!("  Encode tput    : {:.1} MB/s", data_bytes as f64 / (1024.0*1024.0) / total_encode.as_secs_f64());
    eprintln!("  Verify ({num_samples}s) : {:.3}s", verify_time.as_secs_f64());
    eprintln!("  Peak RSS       : {:.2} GB", peak_rss as f64 / (1u64 << 30) as f64);
    eprintln!("------------------------------------------------------------");

    println!("{}", serde_json::json!({
        "n": n,
        "n_prime": n_prime,
        "m": m,
        "m_prime": m_prime,
        "data_bytes": data_bytes,
        "encode_s": (total_encode.as_secs_f64() * 1000.0).round() / 1000.0,
        "col_encode_s": (col_encode_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "row_encode_s": (row_encode_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "full_encode_s": (full_encode_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "verify_s": (verify_time.as_secs_f64() * 1000.0).round() / 1000.0,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "encode_throughput_mbps": (data_bytes as f64 / (1024.0*1024.0) / total_encode.as_secs_f64() * 10.0).round() / 10.0,
        "peak_rss": peak_rss,
    }));
}
