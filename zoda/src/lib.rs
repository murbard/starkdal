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
use utils::poseidon16_compress;

pub type F = KoalaBear;
pub type EF = QuinticExtensionFieldKB;

pub const P: u64 = 0x7F000001;
pub const DIM: usize = 5;
pub const DIGEST: usize = 8;

// ═══════════════════════════════════════════════════════════════════════════
//  NTT
// ═══════════════════════════════════════════════════════════════════════════

pub fn get_omega(log_order: usize) -> F {
    F::from_u32(3).exp_u64((P - 1) >> log_order)
}

pub fn precompute_twiddles(log_n: usize) -> Vec<F> {
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

pub fn coset_fft_f(coeffs: &[F], log_n: usize, twiddles: &[F]) -> Vec<F> {
    let n = 1 << log_n;
    let g = F::from_u32(3);
    let mut data = vec![F::ZERO; n];
    let mut g_pow = F::ONE;
    for i in 0..coeffs.len().min(n) {
        data[i] = coeffs[i] * g_pow;
        g_pow *= g;
    }
    bit_reverse_permute(&mut data, log_n);
    ntt_in_place(&mut data, log_n, twiddles);
    data
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
    bit_reverse_permute_ef(&mut data, log_n);
    ntt_in_place_ef(&mut data, log_n, twiddles);
    data
}

/// Inverse coset FFT: recover coefficients from evaluations on coset g·<ω>.
pub fn coset_ifft_f(evals: &[F], log_n: usize, _twiddles: &[F]) -> Vec<F> {
    let n = 1 << log_n;
    let g = F::from_u32(3);
    let n_inv = F::from_u32(n as u32).inverse();

    // Inverse NTT: same butterfly with inverse twiddles, then scale by 1/n
    let omega_inv = get_omega(log_n).inverse();
    let mut inv_tw = Vec::with_capacity(n / 2);
    let mut acc = F::ONE;
    for _ in 0..n / 2 {
        inv_tw.push(acc);
        acc *= omega_inv;
    }

    let mut data = evals.to_vec();
    bit_reverse_permute(&mut data, log_n);
    ntt_in_place(&mut data, log_n, &inv_tw);

    // Scale by 1/n and undo coset shift: divide by g^i
    let g_inv = g.inverse();
    let mut g_inv_pow = n_inv;
    for v in &mut data {
        *v *= g_inv_pow;
        g_inv_pow *= g_inv;
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

fn bit_reverse_permute(data: &mut [F], log_n: usize) {
    let n = data.len();
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

fn bit_reverse_permute_ef(data: &mut [EF], log_n: usize) {
    let n = data.len();
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

fn ntt_in_place(data: &mut [F], log_n: usize, twiddles: &[F]) {
    let n = data.len();
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
}

fn ntt_in_place_ef(data: &mut [EF], log_n: usize, twiddles: &[F]) {
    let n = data.len();
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
}

// ═══════════════════════════════════════════════════════════════════════════
//  Poseidon Merkle tree
// ═══════════════════════════════════════════════════════════════════════════

fn poseidon_compress(left: &[F; DIGEST], right: &[F; DIGEST]) -> [F; DIGEST] {
    let mut input = [F::ZERO; 16];
    input[..DIGEST].copy_from_slice(left);
    input[DIGEST..].copy_from_slice(right);
    poseidon16_compress(input)
}

/// Sponge-hash a slice of field elements (length must be a multiple of DIGEST).
pub fn poseidon_chain(data: &[F]) -> [F; DIGEST] {
    assert!(!data.is_empty() && data.len() % DIGEST == 0);
    let mut state = [F::ZERO; DIGEST];
    for chunk in data.chunks_exact(DIGEST) {
        let mut input = [F::ZERO; 16];
        input[..DIGEST].copy_from_slice(&state);
        input[DIGEST..].copy_from_slice(chunk);
        state = poseidon16_compress(input);
    }
    state
}

/// Hash EF elements: flatten to base-field, pad to multiple of DIGEST, chain-hash.
pub fn poseidon_chain_ef(data: &[EF]) -> [F; DIGEST] {
    let flat: Vec<F> = data
        .iter()
        .flat_map(|ef| ef.as_basis_coefficients_slice().iter().copied())
        .collect();
    let padded_len = ((flat.len() + DIGEST - 1) / DIGEST) * DIGEST;
    let mut padded = flat;
    padded.resize(padded_len, F::ZERO);
    poseidon_chain(&padded)
}

/// Merkle tree storing all layers for opening proofs.
#[derive(Clone)]
pub struct MerkleTree {
    /// layers[0] = leaf hashes, layers[depth] = [root]
    pub layers: Vec<Vec<[F; DIGEST]>>,
}

impl MerkleTree {
    /// Build from pre-hashed leaves.
    pub fn from_leaves(leaves: Vec<[F; DIGEST]>) -> Self {
        assert!(!leaves.is_empty());
        let mut layers = vec![leaves];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let mut padded = prev.clone();
            if padded.len() % 2 != 0 {
                padded.push([F::ZERO; DIGEST]);
            }
            let next: Vec<[F; DIGEST]> = (0..padded.len() / 2)
                .map(|i| poseidon_compress(&padded[2 * i], &padded[2 * i + 1]))
                .collect();
            layers.push(next);
        }
        Self { layers }
    }

    pub fn root(&self) -> [F; DIGEST] {
        self.layers.last().unwrap()[0]
    }

    pub fn depth(&self) -> usize {
        self.layers.len() - 1
    }

    /// Generate an opening proof for leaf at `index`.
    pub fn open(&self, index: usize) -> MerkleProof {
        let mut path = Vec::with_capacity(self.depth());
        let mut idx = index;
        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling = if idx % 2 == 0 {
                if idx + 1 < layer.len() { layer[idx + 1] } else { [F::ZERO; DIGEST] }
            } else {
                layer[idx - 1]
            };
            path.push(sibling);
            idx /= 2;
        }
        MerkleProof { index, path }
    }
}

/// Merkle authentication path.
#[derive(Clone)]
pub struct MerkleProof {
    pub index: usize,
    pub path: Vec<[F; DIGEST]>,
}

impl MerkleProof {
    /// Verify that `leaf_hash` at `self.index` is consistent with `root`.
    pub fn verify(&self, leaf_hash: &[F; DIGEST], root: &[F; DIGEST]) -> bool {
        let mut current = *leaf_hash;
        let mut idx = self.index;
        for sibling in &self.path {
            current = if idx % 2 == 0 {
                poseidon_compress(&current, sibling)
            } else {
                poseidon_compress(sibling, &current)
            };
            idx /= 2;
        }
        current == *root
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  ZODA encoding
// ═══════════════════════════════════════════════════════════════════════════

/// The full ZODA encoding output.
pub struct ZodaEncoding {
    // Public commitments
    pub root_x: [F; DIGEST],
    pub root_y: [F; DIGEST],
    pub root_z: [F; DIGEST],

    // Merkle trees (for opening proofs)
    pub tree_x: MerkleTree,
    pub tree_y: MerkleTree,
    pub tree_z: MerkleTree,

    // Encoded data
    pub x: Vec<Vec<F>>,    // m rows, each n' base-field elements
    pub y: Vec<Vec<EF>>,   // n rows, each m' extension-field elements
    pub z: Vec<Vec<EF>>,   // m rows, each m' extension-field elements

    // Fiat-Shamir diagonal
    pub diag: Vec<EF>,

    // Parameters
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
}

/// Encode data matrix X̃ (n × n', row-major flat Vec<F>) using ZODA.
pub fn encode(data: &[F], n: usize, n_prime: usize) -> ZodaEncoding {
    assert_eq!(data.len(), n * n_prime);
    assert!(n.is_power_of_two() && n >= 8);
    assert!(n_prime.is_power_of_two() && n_prime >= 8);

    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let log_m = m.trailing_zeros() as usize;
    let log_m_prime = m_prime.trailing_zeros() as usize;

    let twiddles_m = precompute_twiddles(log_m);
    let twiddles_mp = precompute_twiddles(log_m_prime);

    // Step 1: Column encode X = G · X̃
    let x_col_vecs: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let coeffs: Vec<F> = (0..n).map(|row| data[row * n_prime + col]).collect();
            coset_fft_f(&coeffs, log_m, &twiddles_m)
        })
        .collect();

    // Transpose to row-major: x_rows[i][k] = X[i][k]
    let x_rows: Vec<Vec<F>> = (0..m)
        .into_par_iter()
        .map(|i| (0..n_prime).map(|col| x_col_vecs[col][i]).collect())
        .collect();

    // Step 2: Commit to rows of X
    let x_leaves: Vec<[F; DIGEST]> = x_rows.par_iter().map(|row| poseidon_chain(row)).collect();
    let tree_x = MerkleTree::from_leaves(x_leaves);
    let root_x = tree_x.root();

    // Step 3: Derive random diagonal D ∈ E^{n'}
    let diag = derive_diagonal(&root_x, n_prime);

    // Step 4: Scale X̃ by D
    let x_tilde_d: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| {
            (0..n_prime)
                .map(|col| EF::from(data[row * n_prime + col]) * diag[col])
                .collect()
        })
        .collect();

    // Step 5: Row encode Y = (X̃·D) · G'^T
    let y_rows: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| coset_fft_ef(&x_tilde_d[row], log_m_prime, &twiddles_mp))
        .collect();

    // Commit to columns of Y
    let y_leaves: Vec<[F; DIGEST]> = (0..m_prime)
        .into_par_iter()
        .map(|col| {
            let col_data: Vec<EF> = (0..n).map(|row| y_rows[row][col]).collect();
            poseidon_chain_ef(&col_data)
        })
        .collect();
    let tree_y = MerkleTree::from_leaves(y_leaves);
    let root_y = tree_y.root();

    // Step 6: Full encode Z = G · Y
    let z_col_vecs: Vec<Vec<EF>> = (0..m_prime)
        .into_par_iter()
        .map(|col| {
            let y_col: Vec<EF> = (0..n).map(|row| y_rows[row][col]).collect();
            coset_fft_ef(&y_col, log_m, &twiddles_m)
        })
        .collect();

    // Z in row-major
    let z_rows: Vec<Vec<EF>> = (0..m)
        .into_par_iter()
        .map(|i| (0..m_prime).map(|col| z_col_vecs[col][i]).collect())
        .collect();

    // Commit to rows of Z
    let z_leaves: Vec<[F; DIGEST]> = z_rows.par_iter().map(|row| poseidon_chain_ef(row)).collect();
    let tree_z = MerkleTree::from_leaves(z_leaves);
    let root_z = tree_z.root();

    ZodaEncoding {
        root_x, root_y, root_z,
        tree_x, tree_y, tree_z,
        x: x_rows, y: y_rows, z: z_rows,
        diag, n, n_prime, m, m_prime,
    }
}

fn derive_diagonal(root: &[F; DIGEST], n_prime: usize) -> Vec<EF> {
    let mut diag = Vec::with_capacity(n_prime);
    let mut state = *root;
    for i in 0..n_prime {
        let mut inp = [F::ZERO; 16];
        inp[..DIGEST].copy_from_slice(&state);
        inp[DIGEST] = F::from_u32(i as u32);
        inp[DIGEST + 1] = F::from_u32(0x5A4F4441);
        inp[DIGEST + 2] = F::from_u32(n_prime as u32);
        state = poseidon16_compress(inp);
        diag.push(EF::from_basis_coefficients_slice(&state[..DIM]).unwrap());
    }
    diag
}

// ═══════════════════════════════════════════════════════════════════════════
//  Opening: what the encoder sends to a sampler
// ═══════════════════════════════════════════════════════════════════════════

/// A row of X with its Merkle proof.
pub struct XRowOpening {
    pub row_index: usize,
    pub data: Vec<F>,         // n' base-field elements
    pub proof: MerkleProof,
}

/// A column of Y with its Merkle proof.
pub struct YColOpening {
    pub col_index: usize,
    pub data: Vec<EF>,        // n extension-field elements
    pub proof: MerkleProof,
}

/// An entry of Z with a Merkle proof for its row.
pub struct ZEntryOpening {
    pub row: usize,
    pub col: usize,
    pub value: EF,
    pub row_data: Vec<EF>,    // full row for Merkle verification
    pub proof: MerkleProof,
}

impl ZodaEncoding {
    /// Open row `i` of X.
    pub fn open_x_row(&self, i: usize) -> XRowOpening {
        XRowOpening {
            row_index: i,
            data: self.x[i].clone(),
            proof: self.tree_x.open(i),
        }
    }

    /// Open column `j` of Y.
    pub fn open_y_col(&self, j: usize) -> YColOpening {
        let col_data: Vec<EF> = (0..self.n).map(|row| self.y[row][j]).collect();
        YColOpening {
            col_index: j,
            data: col_data,
            proof: self.tree_y.open(j),
        }
    }

    /// Open entry Z[i][j] (includes full row for Merkle).
    pub fn open_z_entry(&self, i: usize, j: usize) -> ZEntryOpening {
        ZEntryOpening {
            row: i,
            col: j,
            value: self.z[i][j],
            row_data: self.z[i].clone(),
            proof: self.tree_z.open(i),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Verification (paper §3.1.2)
// ═══════════════════════════════════════════════════════════════════════════

/// Everything the verifier needs (public commitments + parameters).
pub struct ZodaCommitment {
    pub root_x: [F; DIGEST],
    pub root_y: [F; DIGEST],
    pub root_z: [F; DIGEST],
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
    pub diag: Vec<EF>,  // derivable from root_x, included for convenience
}

impl From<&ZodaEncoding> for ZodaCommitment {
    fn from(enc: &ZodaEncoding) -> Self {
        Self {
            root_x: enc.root_x, root_y: enc.root_y, root_z: enc.root_z,
            n: enc.n, n_prime: enc.n_prime, m: enc.m, m_prime: enc.m_prime,
            diag: enc.diag.clone(),
        }
    }
}

/// Result of the sampling verification protocol.
pub struct VerifyResult {
    pub consistency_checks: usize,
    pub consistency_passed: usize,
    pub z_checks: usize,
    pub z_passed: usize,
    pub merkle_ok: bool,
}

impl VerifyResult {
    pub fn accept(&self) -> bool {
        self.merkle_ok
            && self.consistency_passed == self.consistency_checks
            && self.z_passed == self.z_checks
    }
}

/// Run the ZODA sampling verification protocol (paper §3.1.2).
///
/// Given sampled rows of X, columns of Y, and entries of Z:
///   Step 5: check X_S · D · g'_j = (G · y_j)_S for each j ∈ S'
///   Step 6: check Z_{ij} = g^T_i · y_j for each i ∈ S, j ∈ S'
pub fn verify(
    commitment: &ZodaCommitment,
    x_openings: &[XRowOpening],
    y_openings: &[YColOpening],
    z_openings: &[ZEntryOpening],
) -> VerifyResult {
    let log_m_prime = commitment.m_prime.trailing_zeros() as usize;
    let omega_mp = get_omega(log_m_prime);
    let g = F::from_u32(3);

    // Verify all Merkle proofs
    let mut merkle_ok = true;
    for xo in x_openings {
        let leaf_hash = poseidon_chain(&xo.data);
        if !xo.proof.verify(&leaf_hash, &commitment.root_x) {
            merkle_ok = false;
        }
    }
    for yo in y_openings {
        let leaf_hash = poseidon_chain_ef(&yo.data);
        if !yo.proof.verify(&leaf_hash, &commitment.root_y) {
            merkle_ok = false;
        }
    }
    for zo in z_openings {
        let row_hash = poseidon_chain_ef(&zo.row_data);
        if !zo.proof.verify(&row_hash, &commitment.root_z) {
            merkle_ok = false;
        }
        if zo.row_data[zo.col] != zo.value {
            merkle_ok = false;
        }
    }

    // Precompute G · y_j for each sampled column (used in both step 5 and step 6)
    let log_m = commitment.m.trailing_zeros() as usize;
    let twiddles_m = precompute_twiddles(log_m);
    let g_yj_cache: Vec<(usize, Vec<EF>)> = y_openings
        .iter()
        .map(|yo| (yo.col_index, coset_fft_ef(&yo.data, log_m, &twiddles_m)))
        .collect();

    // Step 5: consistency check  X_S · D · g'_j = (G · y_j)_S
    let mut consistency_checks = 0usize;
    let mut consistency_passed = 0usize;

    for (yo, (_j, g_yj)) in y_openings.iter().zip(&g_yj_cache) {
        let eval_point = g * omega_mp.exp_u64(yo.col_index as u64);
        let ep_ef = EF::from(eval_point);

        // Precompute D[k] · eval_point^k
        let mut d_ep = Vec::with_capacity(commitment.n_prime);
        let mut ep_pow = EF::ONE;
        for k in 0..commitment.n_prime {
            d_ep.push(commitment.diag[k] * ep_pow);
            ep_pow *= ep_ef;
        }

        for xo in x_openings {
            let i = xo.row_index;
            let mut lhs = EF::ZERO;
            for k in 0..commitment.n_prime {
                lhs += EF::from(xo.data[k]) * d_ep[k];
            }
            let rhs = g_yj[i];

            consistency_checks += 1;
            if lhs == rhs {
                consistency_passed += 1;
            }
        }
    }

    // Step 6: Z entry checks  Z_{ij} = (G · y_j)[i]
    let mut z_checks = 0usize;
    let mut z_passed = 0usize;

    for zo in z_openings {
        if let Some((_, g_yj)) = g_yj_cache.iter().find(|(col, _)| *col == zo.col) {
            z_checks += 1;
            if zo.value == g_yj[zo.row] {
                z_passed += 1;
            }
        }
    }

    VerifyResult { consistency_checks, consistency_passed, z_checks, z_passed, merkle_ok }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Decoding (paper §3.1.3)
// ═══════════════════════════════════════════════════════════════════════════

/// Decode X̃ from ALL m rows of X (simple case: inverse coset FFT each column).
/// Returns the n × n' data matrix as a flat Vec<F>.
pub fn decode_from_x_rows(x_rows: &[Vec<F>], n: usize, n_prime: usize) -> Vec<F> {
    let m = x_rows.len();
    let log_m = m.trailing_zeros() as usize;
    let twiddles_m = precompute_twiddles(log_m);

    // Inverse FFT each column to recover X̃ columns
    let decoded_cols: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let evals: Vec<F> = (0..m).map(|row| x_rows[row][col]).collect();
            let coeffs = coset_ifft_f(&evals, log_m, &twiddles_m);
            coeffs[..n].to_vec() // first n coefficients = data
        })
        .collect();

    // Flatten to row-major
    let mut data = vec![F::ZERO; n * n_prime];
    for col in 0..n_prime {
        for row in 0..n {
            data[row * n_prime + col] = decoded_cols[col][row];
        }
    }
    data
}

/// Decode X̃ from ALL n rows of Y (inverse coset FFT each column of Y,
/// then undo the diagonal scaling).
pub fn decode_from_y_rows(y_rows: &[Vec<EF>], diag: &[EF], n: usize, n_prime: usize) -> Vec<F> {
    let m_prime = y_rows[0].len();
    let log_m_prime = m_prime.trailing_zeros() as usize;

    let omega_inv = get_omega(log_m_prime).inverse();
    let mut inv_tw = Vec::with_capacity(m_prime / 2);
    let mut acc = F::ONE;
    for _ in 0..m_prime / 2 {
        inv_tw.push(acc);
        acc *= omega_inv;
    }

    let n_inv = F::from_u32(m_prime as u32).inverse();
    let g = F::from_u32(3);
    let g_inv = g.inverse();

    // Inverse FFT each row of Y to get X̃ · D rows
    let x_tilde_d_rows: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| {
            let mut data = y_rows[row].clone();
            bit_reverse_permute_ef(&mut data, log_m_prime);
            ntt_in_place_ef(&mut data, log_m_prime, &inv_tw);
            let mut g_inv_pow = n_inv;
            for v in &mut data {
                *v *= g_inv_pow;
                g_inv_pow *= g_inv;
            }
            data[..n_prime].to_vec()
        })
        .collect();

    // Undo diagonal: X̃[row][col] = (X̃·D)[row][col] / D[col]
    let diag_inv: Vec<EF> = diag.iter().map(|d| d.inverse()).collect();
    let mut data = vec![F::ZERO; n * n_prime];
    for row in 0..n {
        for col in 0..n_prime {
            let val = x_tilde_d_rows[row][col] * diag_inv[col];
            // Result should be a base-field element (embedded in EF)
            let comps = val.as_basis_coefficients_slice();
            data[row * n_prime + col] = comps[0];
        }
    }
    data
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sampling helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate unique sample indices via LCG seeded from root + domain separator.
pub fn sample_indices(root: &[F; DIGEST], domain_sep: u32, count: usize, modulus: usize) -> Vec<usize> {
    assert!(count <= modulus, "cannot sample {count} unique indices from {modulus}");
    let seed = root[0].as_canonical_u32() as u64
        ^ (root[1].as_canonical_u32() as u64) << 16
        ^ domain_sep as u64;
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
