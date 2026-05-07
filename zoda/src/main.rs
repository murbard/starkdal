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
use rayon::prelude::*;
use std::time::Instant;
use utils::poseidon16_compress;

type F = KoalaBear;
type EF = QuinticExtensionFieldKB;

const P: u64 = 0x7F000001;
const DIM: usize = 5;

// ── NTT / RS encoding ────────────────────────────────────────────────────

fn get_omega(log_order: usize) -> F {
    F::from_u32(3).exp_u64((P - 1) >> log_order)
}

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

/// Coset FFT over base field using precomputed twiddles.
/// Evaluates polynomial `coeffs` on coset g·<ω> where g=3.
fn coset_fft_f(coeffs: &[F], log_n: usize, twiddles: &[F]) -> Vec<F> {
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

/// Coset FFT over extension field. Twiddles are base-field elements.
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
                let t = data[k + j + half] * twiddles[j * stride];
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
    let mut leaves: Vec<[F; 8]> = rows.par_iter().map(|row| poseidon_chain(row)).collect();
    while leaves.len() > 1 {
        if leaves.len() % 2 != 0 {
            leaves.push([F::ZERO; 8]);
        }
        leaves = (0..leaves.len() / 2)
            .map(|i| poseidon_hash(&leaves[2 * i], &leaves[2 * i + 1]))
            .collect();
    }
    leaves[0]
}

// ── Flat matrix ──────────────────────────────────────────────────────────

/// Row-major flat matrix for cache-friendly access.
struct FlatMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Copy + Default + Send + Sync> FlatMatrix<T> {
    fn new(rows: usize, cols: usize) -> Self {
        Self { data: vec![T::default(); rows * cols], rows, cols }
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, val: T) {
        self.data[row * self.cols + col] = val;
    }

    fn row_slice(&self, row: usize) -> &[T] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }
}

// ── ZODA construction ────────────────────────────────────────────────────

/// Derive n' extension field challenges from a Merkle root via Fiat-Shamir.
fn derive_diagonal(root: &[F; 8], n_prime: usize) -> Vec<EF> {
    // Chain-hash to produce enough randomness without per-element independent hashing.
    // Squeeze DIM base-field elements per EF from a running Poseidon sponge.
    let mut diag = Vec::with_capacity(n_prime);
    let mut state = *root;
    for i in 0..n_prime {
        let mut inp = [F::ZERO; 16];
        inp[..8].copy_from_slice(&state);
        inp[8] = F::from_u32(i as u32);
        inp[9] = F::from_u32(0x5A4F4441); // "ZODA" ascii
        inp[10] = F::from_u32(n_prime as u32);
        state = poseidon16_compress(inp);
        diag.push(EF::from_basis_coefficients_slice(&state[..DIM]).unwrap());
    }
    diag
}

/// Generate deterministic but unique sample indices via a simple LCG seeded from root.
fn sample_indices(root: &[F; 8], domain_sep: u32, count: usize, modulus: usize) -> Vec<usize> {
    let seed = root[0].as_canonical_u32() as u64
        ^ (root[1].as_canonical_u32() as u64) << 16
        ^ domain_sep as u64;
    let mut indices = Vec::with_capacity(count);
    let mut seen = vec![false; modulus];
    let mut state = seed;
    while indices.len() < count {
        // LCG step
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (state >> 16) as usize % modulus;
        if !seen[idx] {
            seen[idx] = true;
            indices.push(idx);
        }
    }
    indices
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.iter().position(|a| a == "--n")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(64);
    let num_samples: usize = args.iter().position(|a| a == "--samples")
        .map(|i| args[i + 1].parse().unwrap()).unwrap_or(17);

    assert!(n.is_power_of_two(), "n must be a power of 2");
    assert!(n >= 8, "n must be at least 8");

    let n_prime = n;
    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let log_m = m.trailing_zeros() as usize;
    let log_m_prime = m_prime.trailing_zeros() as usize;
    let data_bytes = n * n_prime * 4;

    eprintln!("============================================================");
    eprintln!("ZODA benchmark (field extension variant)");
    eprintln!("============================================================");
    eprintln!("  n={n}  n'={n_prime}  m={m}  m'={m_prime}");
    eprintln!("  data: {:.2} MB  |  full encoding: {:.2} MB (base) + {:.2} MB (ext)",
        data_bytes as f64 / (1024.0 * 1024.0),
        (m * n_prime * 4) as f64 / (1024.0 * 1024.0),
        (n * m_prime * DIM * 4 + m * m_prime * DIM * 4) as f64 / (1024.0 * 1024.0));
    eprintln!("  samples: {num_samples} rows + {num_samples} cols");

    // ── Step 1: Generate data X̃ (flat matrix, row-major) ──
    let t0 = Instant::now();
    let mut x_tilde = FlatMatrix::<F>::new(n, n_prime);
    for row in 0..n {
        for col in 0..n_prime {
            x_tilde.set(row, col, F::from_u32(((row * n_prime + col) % (P as usize - 2) + 1) as u32));
        }
    }
    eprintln!("  data generation: {:.3}s", t0.elapsed().as_secs_f64());

    // ── Step 2: Column encode X = G · X̃ ──
    let t0 = Instant::now();
    let twiddles_m = precompute_twiddles(log_m);

    // Encode each column: X_col[c] = FFT(X̃[:,c])
    let x_col_vecs: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let coeffs: Vec<F> = (0..n).map(|row| x_tilde.get(row, col)).collect();
            coset_fft_f(&coeffs, log_m, &twiddles_m)
        })
        .collect();

    // Transpose to row-major flat matrix X (m × n')
    let mut x_mat = FlatMatrix::<F>::new(m, n_prime);
    for col in 0..n_prime {
        for row in 0..m {
            x_mat.set(row, col, x_col_vecs[col][row]);
        }
    }
    drop(x_col_vecs);
    let col_encode_time = t0.elapsed();
    eprintln!("  column encode (X=G·X̃): {:.3}s", col_encode_time.as_secs_f64());

    // ── Step 3: Commit to rows of X ──
    let t0 = Instant::now();
    let x_rows_for_merkle: Vec<Vec<F>> = (0..m)
        .map(|i| x_mat.row_slice(i).to_vec())
        .collect();
    let x_root = merkle_root_of_rows(&x_rows_for_merkle);
    drop(x_rows_for_merkle);
    let commit_time = t0.elapsed();
    eprintln!("  commit (Merkle root): {:.3}s", commit_time.as_secs_f64());

    // ── Step 4: Derive random diagonal D ∈ E^{n'} ──
    let t0 = Instant::now();
    let diag = derive_diagonal(&x_root, n_prime);
    let diag_time = t0.elapsed();
    eprintln!("  derive diagonal: {:.3}s", diag_time.as_secs_f64());

    // ── Step 5: Compute X̃·D (scale columns of X̃ by diagonal) ──
    let t0 = Instant::now();
    let x_tilde_d: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| {
            (0..n_prime)
                .map(|col| EF::from(x_tilde.get(row, col)) * diag[col])
                .collect()
        })
        .collect();
    let scale_time = t0.elapsed();
    eprintln!("  diagonal scale (X̃·D): {:.3}s", scale_time.as_secs_f64());

    // ── Step 6: Row encode Y = (X̃·D) · G'^T ──
    let t0 = Instant::now();
    let twiddles_mp = precompute_twiddles(log_m_prime);
    let y_rows: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| coset_fft_ef(&x_tilde_d[row], log_m_prime, &twiddles_mp))
        .collect();
    let row_encode_time = t0.elapsed();
    eprintln!("  row encode (Y=(X̃·D)·G'^T): {:.3}s", row_encode_time.as_secs_f64());

    // ── Step 7: Full encode Z = G · Y ──
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

    // ── Verification ──
    // Paper §3.1.2 check 5: X_S · D · g'_j = (G · y_j)_S for each j ∈ S'
    // LHS: for row i ∈ S, sum_k X[i][k] · D[k] · eval_point_j^k
    // RHS: Z[i][j] (since Z = G·Y and column j of Z is G · y_j)
    let t0 = Instant::now();

    let sample_rows = sample_indices(&x_root, 0, num_samples, m);
    let sample_cols = sample_indices(&x_root, 1, num_samples, m_prime);

    let omega_mp = get_omega(log_m_prime);
    let g = F::from_u32(3);
    let mut checks_passed = 0usize;
    let mut checks_total = 0usize;

    for &j in &sample_cols {
        let eval_point = g * omega_mp.exp_u64(j as u64);
        let ep_ef = EF::from(eval_point);

        // Precompute D[k] · eval_point^k for all k — shared across rows
        let mut d_ep = Vec::with_capacity(n_prime);
        let mut ep_pow = EF::ONE;
        for k in 0..n_prime {
            d_ep.push(diag[k] * ep_pow);
            ep_pow *= ep_ef;
        }

        for &i in &sample_rows {
            // LHS: sum_k X[i][k] · (D[k] · eval_point^k)
            let mut lhs = EF::ZERO;
            for k in 0..n_prime {
                lhs += EF::from(x_mat.get(i, k)) * d_ep[k];
            }

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

    assert_eq!(checks_passed, checks_total, "verification failed: {checks_passed}/{checks_total}");

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
    eprintln!("  Data payload   : {:.2} MB ({n} x {n_prime} x 4B)", data_bytes as f64 / (1024.0*1024.0));
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
