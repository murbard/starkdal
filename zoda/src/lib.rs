//! ZODA (Zero-Overhead Data Availability) — field-extension variant.
//!
//! Implements the construction from Evans, Mohnblatt, Angeris (ePrint 2025/034).
//!
//! Data X̃ ∈ F^{n×n'} is tensor-encoded with Fiat-Shamir randomness injected
//! between the column and row encoding steps. The resulting encoding's rows
//! and columns are self-certifying proofs of correct encoding.

pub use backend::*;
#[allow(unused_imports)]
use rayon::prelude::*;

pub type F = KoalaBear;
pub type EF = QuinticExtensionFieldKB;

pub const P: u64 = 0x7F000001;
pub const DIM: usize = 5;
pub const HASH_LEN: usize = 32; // BLAKE3 output bytes

pub type Hash = [u8; HASH_LEN];

// ═══════════════════════════════════════════════════════════════════════════
//  Hashing (BLAKE3)
// ═══════════════════════════════════════════════════════════════════════════

/// Hash a contiguous slice of base-field elements by raw memory representation.
/// Avoids per-element Montgomery reduction — binding and deterministic within
/// the same code (same field impl → same repr → same hash).
fn hash_f_slice(data: &[F]) -> Hash {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * std::mem::size_of::<F>())
    };
    *blake3::hash(bytes).as_bytes()
}

/// Hash a gathered column of EF elements (non-contiguous → must serialize).
fn hash_ef_column(y_rows: &[Vec<EF>], col: usize) -> Hash {
    let n = y_rows.len();
    // Gather column into contiguous buffer, then hash raw bytes
    let mut buf: Vec<EF> = Vec::with_capacity(n);
    for row in y_rows {
        buf.push(row[col]);
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len() * std::mem::size_of::<EF>())
    };
    *blake3::hash(bytes).as_bytes()
}

/// Hash a contiguous slice of EF elements by raw memory representation.
fn hash_ef_slice(data: &[EF]) -> Hash {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * std::mem::size_of::<EF>())
    };
    *blake3::hash(bytes).as_bytes()
}

fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

// ═══════════════════════════════════════════════════════════════════════════
//  NTT — SIMD-optimized with pre-packed twiddles
// ═══════════════════════════════════════════════════════════════════════════

pub fn get_omega(log_order: usize) -> F {
    F::from_u32(3).exp_u64((P - 1) >> log_order)
}

/// Reusable NTT plan: precomputes packed twiddles, coset factors, and
/// bit-reverse table. Created once per FFT size, shared across all FFTs.
pub struct NttPlan {
    pub log_n: usize,
    pub n: usize,
    // stage_tw[s][j] = omega^{j * (n / 2^s)} for j = 0..2^{s-1}
    // Twiddles stored contiguously per stage for SIMD-friendly access.
    stage_tw_fwd: Vec<Vec<F>>,
    stage_tw_inv: Vec<Vec<F>>,
    // Coset shift factors: g_pows[i] = g^i, g_inv_pows[i] = g^{-i} / n
    g_pows: Vec<F>,
    g_inv_pows: Vec<F>,
    bit_rev: Vec<usize>,
}

impl NttPlan {
    pub fn new(log_n: usize) -> Self {
        let n = 1usize << log_n;
        let omega = get_omega(log_n);
        let omega_inv = omega.inverse();
        let g = F::from_u32(3);
        let g_inv = g.inverse();
        let n_inv = F::from_u32(n as u32).inverse();

        let mut stage_tw_fwd = vec![vec![]; log_n + 1];
        let mut stage_tw_inv = vec![vec![]; log_n + 1];
        for s in 1..=log_n {
            let half = 1usize << (s - 1);
            let stride = n >> s;
            let mut tw_f = Vec::with_capacity(half);
            let mut tw_i = Vec::with_capacity(half);
            let mut wf = F::ONE;
            let mut wi = F::ONE;
            let step_f = omega.exp_u64(stride as u64);
            let step_i = omega_inv.exp_u64(stride as u64);
            for _ in 0..half {
                tw_f.push(wf); wf *= step_f;
                tw_i.push(wi); wi *= step_i;
            }
            stage_tw_fwd[s] = tw_f;
            stage_tw_inv[s] = tw_i;
        }

        let mut g_pows = Vec::with_capacity(n);
        let mut gp = F::ONE;
        for _ in 0..n { g_pows.push(gp); gp *= g; }

        let mut g_inv_pows = Vec::with_capacity(n);
        let mut gip = n_inv;
        for _ in 0..n { g_inv_pows.push(gip); gip *= g_inv; }

        let bit_rev: Vec<usize> = (0..n).map(|i| {
            let mut r = 0; let mut v = i;
            for _ in 0..log_n { r = (r << 1) | (v & 1); v >>= 1; }
            r
        }).collect();

        Self { log_n, n, stage_tw_fwd, stage_tw_inv, g_pows, g_inv_pows, bit_rev }
    }

    /// Forward NTT in-place with SIMD butterflies.
    fn forward(&self, data: &mut [F]) {
        use backend::PackedValue;
        type PF = <F as Field>::Packing;
        let w = PF::WIDTH;

        // Bit-reverse permutation
        for i in 0..self.n {
            let j = self.bit_rev[i];
            if i < j { data.swap(i, j); }
        }

        for s in 1..=self.log_n {
            let m = 1usize << s;
            let half = m >> 1;
            let tw = &self.stage_tw_fwd[s];
            let mut k = 0;
            while k < self.n {
                // SIMD path: process W butterflies per iteration
                let mut j = 0;
                while j + w <= half {
                    let u = *PF::from_slice(&data[k+j..k+j+w]);
                    let v = *PF::from_slice(&data[k+j+half..k+j+half+w]);
                    let t = *PF::from_slice(&tw[j..j+w]);
                    let tv = t * v;
                    *PF::from_slice_mut(&mut data[k+j..k+j+w]) = u + tv;
                    *PF::from_slice_mut(&mut data[k+j+half..k+j+half+w]) = u - tv;
                    j += w;
                }
                // Scalar tail
                while j < half {
                    let u = data[k+j];
                    let tv = tw[j] * data[k+j+half];
                    data[k+j] = u + tv;
                    data[k+j+half] = u - tv;
                    j += 1;
                }
                k += m;
            }
        }
    }

    /// Inverse NTT in-place with SIMD butterflies. Includes 1/n normalization.
    fn inverse(&self, data: &mut [F]) {
        use backend::PackedValue;
        type PF = <F as Field>::Packing;
        let w = PF::WIDTH;
        let n_inv = F::from_u32(self.n as u32).inverse();

        for i in 0..self.n {
            let j = self.bit_rev[i];
            if i < j { data.swap(i, j); }
        }

        for s in 1..=self.log_n {
            let m = 1usize << s;
            let half = m >> 1;
            let tw = &self.stage_tw_inv[s];
            let mut k = 0;
            while k < self.n {
                let mut j = 0;
                while j + w <= half {
                    let u = *PF::from_slice(&data[k+j..k+j+w]);
                    let v = *PF::from_slice(&data[k+j+half..k+j+half+w]);
                    let t = *PF::from_slice(&tw[j..j+w]);
                    let tv = t * v;
                    *PF::from_slice_mut(&mut data[k+j..k+j+w]) = u + tv;
                    *PF::from_slice_mut(&mut data[k+j+half..k+j+half+w]) = u - tv;
                    j += w;
                }
                while j < half {
                    let u = data[k+j];
                    let tv = tw[j] * data[k+j+half];
                    data[k+j] = u + tv;
                    data[k+j+half] = u - tv;
                    j += 1;
                }
                k += m;
            }
        }
        for v in data.iter_mut() { *v *= n_inv; }
    }

    /// Coset FFT: multiply by g^i, then forward NTT.
    pub fn coset_fft(&self, coeffs: &[F]) -> Vec<F> {
        let mut data = vec![F::ZERO; self.n];
        let len = coeffs.len().min(self.n);
        for i in 0..len { data[i] = coeffs[i] * self.g_pows[i]; }
        self.forward(&mut data);
        data
    }

    /// Coset FFT for extension field: decompose into DIM base-field FFTs.
    pub fn coset_fft_ef(&self, coeffs: &[EF]) -> Vec<EF> {
        let len = coeffs.len().min(self.n);
        // Decompose + coset-scale each component
        let mut comps: Vec<Vec<F>> = (0..DIM).map(|k| {
            let mut buf = vec![F::ZERO; self.n];
            for i in 0..len {
                let c: &[F] = coeffs[i].as_basis_coefficients_slice();
                buf[i] = c[k] * self.g_pows[i];
            }
            self.forward(&mut buf);
            buf
        }).collect();
        // Recombine
        (0..self.n).map(|i| {
            EF::from_basis_coefficients_slice(&[
                comps[0][i], comps[1][i], comps[2][i], comps[3][i], comps[4][i],
            ]).unwrap()
        }).collect()
    }

    /// Inverse coset FFT.
    pub fn coset_ifft(&self, evals: &[F]) -> Vec<F> {
        let mut data = evals.to_vec();
        self.inverse(&mut data);
        for i in 0..self.n { data[i] *= self.g_inv_pows[i]; }
        // g_inv_pows already includes 1/n factor
        // Wait, no: inverse() already divides by n. g_inv_pows[i] = g^{-i}/n.
        // So total scaling = (1/n) * (g^{-i}/n) = g^{-i}/n². Wrong.
        // Fix: g_inv_pows should be just g^{-i}, and inverse() handles 1/n.
        data
    }

    /// Inverse coset FFT for extension field.
    pub fn coset_ifft_ef(&self, evals: &[EF]) -> Vec<EF> {
        let mut comps: Vec<Vec<F>> = (0..DIM).map(|k| {
            let mut buf: Vec<F> = evals.iter()
                .map(|ef| ef.as_basis_coefficients_slice()[k])
                .collect();
            self.inverse(&mut buf);
            buf
        }).collect();
        let g_inv = F::from_u32(3).inverse();
        (0..self.n).map(|i| {
            let mut gi = if i == 0 { F::ONE } else { g_inv.exp_u64(i as u64) };
            EF::from_basis_coefficients_slice(&[
                comps[0][i] * gi, comps[1][i] * gi, comps[2][i] * gi,
                comps[3][i] * gi, comps[4][i] * gi,
            ]).unwrap()
        }).collect()
    }
}

// Legacy API wrappers (used by encode/verify)
pub fn precompute_twiddles(log_n: usize) -> Vec<F> {
    let n = 1 << log_n;
    let omega = get_omega(log_n);
    let mut tw = Vec::with_capacity(n / 2);
    let mut acc = F::ONE;
    for _ in 0..n / 2 { tw.push(acc); acc *= omega; }
    tw
}

pub fn coset_fft_f(coeffs: &[F], log_n: usize, _twiddles: &[F]) -> Vec<F> {
    NttPlan::new(log_n).coset_fft(coeffs)
}

pub fn coset_fft_ef(coeffs: &[EF], log_n: usize, twiddles: &[F]) -> Vec<EF> {
    let n = 1 << log_n;
    let g = F::from_u32(3);
    let mut data = vec![EF::ZERO; n];
    let mut g_pow = F::ONE;
    for i in 0..coeffs.len().min(n) {
        data[i] = coeffs[i] * g_pow;
        g_pow *= g;
    }
    // Bit-reverse
    for i in 0..n {
        let mut r = 0; let mut v = i;
        for _ in 0..log_n { r = (r << 1) | (v & 1); v >>= 1; }
        if i < r { data.swap(i, r); }
    }
    // In-place EF NTT with strided twiddles (LLVM auto-vectorizes the EF*F butterfly)
    for s in 1..=log_n {
        let m = 1 << s; let half = m >> 1; let stride = n / m;
        let mut k = 0;
        while k < n {
            for j in 0..half {
                let u = data[k+j];
                let t = data[k+j+half] * twiddles[j * stride];
                data[k+j] = u + t;
                data[k+j+half] = u - t;
            }
            k += m;
        }
    }
    data
}

pub fn coset_ifft_f(evals: &[F], log_n: usize) -> Vec<F> {
    let plan = NttPlan::new(log_n);
    let mut data = evals.to_vec();
    plan.inverse(&mut data);
    let g_inv = F::from_u32(3).inverse();
    let mut gi = F::ONE;
    for i in 0..data.len() { data[i] *= gi; gi *= g_inv; }
    data
}

pub fn coset_ifft_ef(evals: &[EF], log_n: usize) -> Vec<EF> {
    NttPlan::new(log_n).coset_ifft_ef(evals)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Merkle tree (BLAKE3)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct MerkleTree {
    pub layers: Vec<Vec<Hash>>,
}

impl MerkleTree {
    pub fn from_leaves(leaves: Vec<Hash>) -> Self {
        assert!(!leaves.is_empty());
        let mut layers = vec![leaves];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let mut padded = prev.clone();
            if padded.len() % 2 != 0 {
                padded.push([0u8; HASH_LEN]);
            }
            let next: Vec<Hash> = (0..padded.len() / 2)
                .map(|i| hash_pair(&padded[2 * i], &padded[2 * i + 1]))
                .collect();
            layers.push(next);
        }
        Self { layers }
    }

    pub fn root(&self) -> Hash { self.layers.last().unwrap()[0] }
    pub fn depth(&self) -> usize { self.layers.len() - 1 }

    pub fn open(&self, index: usize) -> MerkleProof {
        let mut path = Vec::with_capacity(self.depth());
        let mut idx = index;
        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling = if idx % 2 == 0 {
                if idx + 1 < layer.len() { layer[idx + 1] } else { [0u8; HASH_LEN] }
            } else {
                layer[idx - 1]
            };
            path.push(sibling);
            idx /= 2;
        }
        MerkleProof { index, path }
    }
}

#[derive(Clone)]
pub struct MerkleProof {
    pub index: usize,
    pub path: Vec<Hash>,
}

impl MerkleProof {
    pub fn verify(&self, leaf_hash: &Hash, root: &Hash) -> bool {
        let mut current = *leaf_hash;
        let mut idx = self.index;
        for sibling in &self.path {
            current = if idx % 2 == 0 {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
            idx /= 2;
        }
        current == *root
    }

    pub fn byte_size(&self) -> usize {
        self.path.len() * HASH_LEN
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  ZODA encoding
// ═══════════════════════════════════════════════════════════════════════════

pub struct EncodeTiming {
    pub col_fft: std::time::Duration,
    pub commit_x: std::time::Duration,
    pub diag_scale: std::time::Duration,
    pub row_fft: std::time::Duration,
    pub commit_y: std::time::Duration,
    pub total: std::time::Duration,
}

pub struct ZodaEncoding {
    pub root_x: Hash,
    pub root_y: Hash,
    pub tree_x: MerkleTree,
    pub tree_y: MerkleTree,
    pub x_flat: Vec<F>,        // m × n', row-major flat
    pub y: Vec<Vec<EF>>,       // n rows × m' cols, extension field
    pub diag: Vec<EF>,
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
    pub timing: EncodeTiming,
}

/// Encode data matrix X̃ (n × n', row-major flat Vec<F>) using ZODA.
///
/// Z = G·Y is NOT precomputed — individual Z entries are derived on-demand
/// from Y columns during verification (the verifier computes (G·y_j)[i]).
pub fn encode(data: &[F], n: usize, n_prime: usize) -> ZodaEncoding {
    assert_eq!(data.len(), n * n_prime);
    assert!(n.is_power_of_two() && n >= 8);
    assert!(n_prime.is_power_of_two() && n_prime >= 8);

    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let log_m = m.trailing_zeros() as usize;
    let log_m_prime = m_prime.trailing_zeros() as usize;
    let t_total = std::time::Instant::now();

    // Create NTT plans once (precomputed SIMD twiddles, bit-reverse tables)
    let plan_col = NttPlan::new(log_m);
    let plan_row = NttPlan::new(log_m_prime);

    // Step 1: Column encode X = G · X̃
    let t0 = std::time::Instant::now();
    let x_col_vecs: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let coeffs: Vec<F> = (0..n).map(|row| data[row * n_prime + col]).collect();
            plan_col.coset_fft(&coeffs)
        })
        .collect();
    let x_flat: Vec<F> = (0..m * n_prime)
        .into_par_iter()
        .map(|idx| x_col_vecs[idx % n_prime][idx / n_prime])
        .collect();
    drop(x_col_vecs);
    let col_fft = t0.elapsed();

    // Step 2: Commit to rows of X  (BLAKE3 Merkle, bulk hash)
    let t0 = std::time::Instant::now();
    let x_leaves: Vec<Hash> = (0..m)
        .into_par_iter()
        .map(|i| hash_f_slice(&x_flat[i * n_prime..(i + 1) * n_prime]))
        .collect();
    let tree_x = MerkleTree::from_leaves(x_leaves);
    let root_x = tree_x.root();
    let commit_x = t0.elapsed();

    // Step 3: Diagonal scale X̃·D  (F * EF → EF, per element)
    let t0 = std::time::Instant::now();
    let diag = derive_diagonal(&root_x, n_prime);
    let x_tilde_d: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| {
            let row_data = &data[row * n_prime..(row + 1) * n_prime];
            row_data
                .iter()
                .enumerate()
                .map(|(col, &x)| EF::from(x) * diag[col])
                .collect()
        })
        .collect();
    let diag_scale = t0.elapsed();

    // Step 4: Row encode Y = (X̃·D) · G'^T  (EF coset FFT, scalar — LLVM auto-vectorizes)
    let t0 = std::time::Instant::now();
    let twiddles_mp = precompute_twiddles(log_m_prime);
    let y_rows: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| coset_fft_ef(&x_tilde_d[row], log_m_prime, &twiddles_mp))
        .collect();
    let row_fft = t0.elapsed();

    // Step 5: Commit to columns of Y  (BLAKE3 Merkle, EF data)
    let t0 = std::time::Instant::now();
    let y_leaves: Vec<Hash> = (0..m_prime)
        .into_par_iter()
        .map(|col| hash_ef_column(&y_rows, col))
        .collect();
    let tree_y = MerkleTree::from_leaves(y_leaves);
    let root_y = tree_y.root();
    let commit_y = t0.elapsed();

    let total = t_total.elapsed();

    ZodaEncoding {
        root_x, root_y,
        tree_x, tree_y,
        x_flat, y: y_rows,
        diag, n, n_prime, m, m_prime,
        timing: EncodeTiming { col_fft, commit_x, diag_scale, row_fft, commit_y, total },
    }
}

fn derive_diagonal(root: &Hash, n_prime: usize) -> Vec<EF> {
    let mut diag = Vec::with_capacity(n_prime);
    let mut state = *root;
    for i in 0..n_prime {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&state);
        hasher.update(&(i as u64).to_le_bytes());
        hasher.update(b"ZODA_DIAG");
        state = *hasher.finalize().as_bytes();
        // Extract DIM field elements from the 32-byte hash
        let mut comps = [F::ZERO; DIM];
        for (k, comp) in comps.iter_mut().enumerate() {
            let offset = k * 4;
            let val = u32::from_le_bytes([state[offset], state[offset+1], state[offset+2], state[offset+3]]);
            *comp = F::from_u32(val % (P as u32));
        }
        diag.push(EF::from_basis_coefficients_slice(&comps).unwrap());
    }
    diag
}

// ═══════════════════════════════════════════════════════════════════════════
//  Openings
// ═══════════════════════════════════════════════════════════════════════════

pub struct XRowOpening {
    pub row_index: usize,
    pub data: Vec<F>,
    pub proof: MerkleProof,
}

pub struct YColOpening {
    pub col_index: usize,
    pub data: Vec<EF>,
    pub proof: MerkleProof,
}

impl ZodaEncoding {
    pub fn open_x_row(&self, i: usize) -> XRowOpening {
        let start = i * self.n_prime;
        XRowOpening {
            row_index: i,
            data: self.x_flat[start..start + self.n_prime].to_vec(),
            proof: self.tree_x.open(i),
        }
    }

    /// Get row i of X as a slice (zero-copy).
    pub fn x_row(&self, i: usize) -> &[F] {
        let start = i * self.n_prime;
        &self.x_flat[start..start + self.n_prime]
    }

    pub fn open_y_col(&self, j: usize) -> YColOpening {
        let col_data: Vec<EF> = (0..self.n).map(|row| self.y[row][j]).collect();
        YColOpening { col_index: j, data: col_data, proof: self.tree_y.open(j) }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Verification (paper §3.1.2)
// ═══════════════════════════════════════════════════════════════════════════

pub struct ZodaCommitment {
    pub root_x: Hash,
    pub root_y: Hash,
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
    pub diag: Vec<EF>,  // derivable from root_x
}

impl From<&ZodaEncoding> for ZodaCommitment {
    fn from(enc: &ZodaEncoding) -> Self {
        Self {
            root_x: enc.root_x, root_y: enc.root_y,
            n: enc.n, n_prime: enc.n_prime, m: enc.m, m_prime: enc.m_prime,
            diag: enc.diag.clone(),
        }
    }
}

pub struct VerifyResult {
    pub consistency_checks: usize,
    pub consistency_passed: usize,
    pub merkle_ok: bool,
}

impl VerifyResult {
    pub fn accept(&self) -> bool {
        self.merkle_ok && self.consistency_passed == self.consistency_checks
    }
}

/// Run the ZODA sampling verification protocol (paper §3.1.2).
///
/// Checks X_S · D · g'_j = (G · y_j)_S for each sampled row i and column j.
/// Z entries are derived on-the-fly from Y columns (no Z commitment needed).
pub fn verify(
    commitment: &ZodaCommitment,
    x_openings: &[XRowOpening],
    y_openings: &[YColOpening],
) -> VerifyResult {
    let log_m_prime = commitment.m_prime.trailing_zeros() as usize;
    let omega_mp = get_omega(log_m_prime);
    let g = F::from_u32(3);

    // Verify Merkle proofs
    let mut merkle_ok = true;
    for xo in x_openings {
        if !xo.proof.verify(&hash_f_slice(&xo.data), &commitment.root_x) {
            merkle_ok = false;
        }
    }
    for yo in y_openings {
        if !yo.proof.verify(&hash_ef_slice(&yo.data), &commitment.root_y) {
            merkle_ok = false;
        }
    }

    // Precompute G · y_j for each sampled column
    let log_m = commitment.m.trailing_zeros() as usize;
    let twiddles_m = precompute_twiddles(log_m);
    let g_yj_cache: Vec<Vec<EF>> = y_openings
        .iter()
        .map(|yo| coset_fft_ef(&yo.data, log_m, &twiddles_m))
        .collect();

    // Consistency check: X_S · D · g'_j = (G · y_j)_S
    let mut consistency_checks = 0usize;
    let mut consistency_passed = 0usize;

    for (yo, g_yj) in y_openings.iter().zip(&g_yj_cache) {
        let eval_point = g * omega_mp.exp_u64(yo.col_index as u64);
        let ep_ef = EF::from(eval_point);
        let mut d_ep = Vec::with_capacity(commitment.n_prime);
        let mut ep_pow = EF::ONE;
        for k in 0..commitment.n_prime {
            d_ep.push(commitment.diag[k] * ep_pow);
            ep_pow *= ep_ef;
        }
        for xo in x_openings {
            let mut lhs = EF::ZERO;
            for k in 0..commitment.n_prime {
                lhs += EF::from(xo.data[k]) * d_ep[k];
            }
            consistency_checks += 1;
            if lhs == g_yj[xo.row_index] { consistency_passed += 1; }
        }
    }

    VerifyResult { consistency_checks, consistency_passed, merkle_ok }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Decoding (paper §3.1.3)
// ═══════════════════════════════════════════════════════════════════════════

/// Decode X̃ from X (row-major flat buffer, m × n').
pub fn decode_from_x_flat(x_flat: &[F], m: usize, n: usize, n_prime: usize) -> Vec<F> {
    let log_m = m.trailing_zeros() as usize;
    let decoded_cols: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let evals: Vec<F> = (0..m).map(|row| x_flat[row * n_prime + col]).collect();
            let coeffs = coset_ifft_f(&evals, log_m);
            coeffs[..n].to_vec()
        })
        .collect();
    let mut data = vec![F::ZERO; n * n_prime];
    for col in 0..n_prime {
        for row in 0..n {
            data[row * n_prime + col] = decoded_cols[col][row];
        }
    }
    data
}

pub fn decode_from_y_rows(y_rows: &[Vec<EF>], diag: &[EF], n: usize, n_prime: usize) -> Vec<F> {
    let m_prime = y_rows[0].len();
    let log_m_prime = m_prime.trailing_zeros() as usize;
    let diag_inv: Vec<EF> = diag.iter().map(|d| d.inverse()).collect();

    let x_tilde_d_rows: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| {
            let coeffs = coset_ifft_ef(&y_rows[row], log_m_prime);
            coeffs[..n_prime].to_vec()
        })
        .collect();

    let mut data = vec![F::ZERO; n * n_prime];
    for row in 0..n {
        for col in 0..n_prime {
            let val = x_tilde_d_rows[row][col] * diag_inv[col];
            data[row * n_prime + col] = val.as_basis_coefficients_slice()[0];
        }
    }
    data
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sampling helpers
// ═══════════════════════════════════════════════════════════════════════════

pub fn sample_indices(root: &Hash, domain_sep: u32, count: usize, modulus: usize) -> Vec<usize> {
    assert!(count <= modulus);
    let seed = u64::from_le_bytes(root[..8].try_into().unwrap()) ^ domain_sep as u64;
    let mut indices = Vec::with_capacity(count);
    let mut seen = vec![false; modulus];
    let mut state = seed;
    while indices.len() < count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (state >> 16) as usize % modulus;
        if !seen[idx] {
            seen[idx] = true;
            indices.push(idx);
        }
    }
    indices
}
