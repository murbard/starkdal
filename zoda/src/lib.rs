//! ZODA (Zero-Overhead Data Availability) — field-extension variant.
//!
//! Uses concrete-ntt's SIMD-optimized negacyclic NTT for RS encoding.
//! The negacyclic NTT evaluates at ψ^{2·bitrev(k)+1} where ψ^{2n}=1.
//! These are n distinct points → valid MDS RS code for ZODA.

pub use backend::*;
#[allow(unused_imports)]
use rayon::prelude::*;

pub type F = KoalaBear;
pub type EF = QuinticExtensionFieldKB;

pub const P: u64 = 0x7F000001;
pub const DIM: usize = 5;
pub const HASH_LEN: usize = 32;
pub type Hash = [u8; HASH_LEN];

// ═══════════════════════════════════════════════════════════════════════════
//  Hashing (BLAKE3, raw Montgomery bytes)
// ═══════════════════════════════════════════════════════════════════════════

fn hash_f_slice(data: &[F]) -> Hash {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * std::mem::size_of::<F>())
    };
    *blake3::hash(bytes).as_bytes()
}

fn hash_ef_column(y_rows: &[Vec<EF>], col: usize) -> Hash {
    let mut buf: Vec<EF> = Vec::with_capacity(y_rows.len());
    for row in y_rows { buf.push(row[col]); }
    let bytes = unsafe {
        std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len() * std::mem::size_of::<EF>())
    };
    *blake3::hash(bytes).as_bytes()
}

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
//  NTT via concrete-ntt (negacyclic, SIMD-optimized)
// ═══════════════════════════════════════════════════════════════════════════

fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut r = 0; let mut v = x;
    for _ in 0..bits { r = (r << 1) | (v & 1); v >>= 1; }
    r
}

/// Reusable NTT plan backed by concrete-ntt.
/// Output is in bit-reversed order (concrete-ntt's native convention).
pub struct NttPlan {
    pub log_n: usize,
    pub n: usize,
    plan: concrete_ntt::prime32::Plan,
    /// ψ = primitive 2n-th root of unity used by concrete-ntt.
    /// eval_point(k) = ψ^{2·bitrev(k)+1}.
    psi: F,
    /// Precomputed eval points: eval_pts[k] = ψ^{2·bitrev(k)+1}
    pub eval_pts: Vec<F>,
}

impl NttPlan {
    pub fn new(log_n: usize) -> Self {
        let n = 1usize << log_n;
        let plan = concrete_ntt::prime32::Plan::try_new(n, P as u32)
            .unwrap_or_else(|| panic!("concrete-ntt doesn't support size {n} for p={P:#x}"));

        // Extract ψ from NTT of [0, monty(1), 0, ...]:
        // output[0] = monty(1) · ψ^{2·bitrev(0)+1} = monty(1) · ψ
        // → interpreted as MontyField31 → canonical value ψ
        let one_raw: u32 = unsafe { *(&F::from_u32(1) as *const F as *const u32) };
        let mut probe = vec![0u32; n];
        probe[1] = one_raw;
        plan.fwd(&mut probe);
        let psi: F = unsafe { *(&probe[0] as *const u32 as *const F) };

        // Precompute eval points
        let eval_pts: Vec<F> = (0..n)
            .map(|k| psi.exp_u64((2 * bit_reverse(k, log_n) + 1) as u64))
            .collect();

        Self { log_n, n, plan, psi, eval_pts }
    }

    /// Forward negacyclic NTT in-place. Output in bit-reversed order.
    /// MontyField31 is repr(transparent) over u32 — safe to transmute.
    pub fn fwd(&self, data: &mut [F]) {
        assert_eq!(data.len(), self.n);
        let data_u32: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u32>(), self.n)
        };
        self.plan.fwd(data_u32);
    }

    /// Inverse negacyclic NTT in-place. Includes 1/n normalization.
    pub fn inv(&self, data: &mut [F]) {
        assert_eq!(data.len(), self.n);
        let data_u32: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u32>(), self.n)
        };
        self.plan.inv(data_u32);
        let n_inv = F::from_u32(self.n as u32).inverse();
        for v in data.iter_mut() { *v *= n_inv; }
    }

    /// Forward NTT of a polynomial (pad with zeros).
    pub fn fwd_poly(&self, coeffs: &[F]) -> Vec<F> {
        let mut data = vec![F::ZERO; self.n];
        data[..coeffs.len().min(self.n)].copy_from_slice(&coeffs[..coeffs.len().min(self.n)]);
        self.fwd(&mut data);
        data
    }

    /// Forward NTT of an EF polynomial: decompose into DIM base-field NTTs.
    pub fn fwd_poly_ef(&self, coeffs: &[EF]) -> Vec<EF> {
        let len = coeffs.len().min(self.n);
        let comps: Vec<Vec<F>> = (0..DIM).map(|k| {
            let mut buf = vec![F::ZERO; self.n];
            for i in 0..len {
                let c: &[F] = coeffs[i].as_basis_coefficients_slice();
                buf[i] = c[k];
            }
            self.fwd(&mut buf);
            buf
        }).collect();
        (0..self.n).map(|i| {
            EF::from_basis_coefficients_slice(&[
                comps[0][i], comps[1][i], comps[2][i], comps[3][i], comps[4][i],
            ]).unwrap()
        }).collect()
    }

    /// Inverse NTT to recover polynomial coefficients from evaluations.
    pub fn inv_evals(&self, evals: &[F]) -> Vec<F> {
        let mut data = evals.to_vec();
        self.inv(&mut data);
        data
    }

    /// Inverse NTT for EF evaluations: decompose into DIM base-field INTTs.
    pub fn inv_evals_ef(&self, evals: &[EF]) -> Vec<EF> {
        let comps: Vec<Vec<F>> = (0..DIM).map(|k| {
            let mut buf: Vec<F> = evals.iter()
                .map(|ef| ef.as_basis_coefficients_slice()[k])
                .collect();
            self.inv(&mut buf);
            buf
        }).collect();
        (0..self.n).map(|i| {
            EF::from_basis_coefficients_slice(&[
                comps[0][i], comps[1][i], comps[2][i], comps[3][i], comps[4][i],
            ]).unwrap()
        }).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Merkle tree (BLAKE3)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct MerkleTree { pub layers: Vec<Vec<Hash>> }

impl MerkleTree {
    pub fn from_leaves(leaves: Vec<Hash>) -> Self {
        assert!(!leaves.is_empty());
        let mut layers = vec![leaves];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let mut padded = prev.clone();
            if padded.len() % 2 != 0 { padded.push([0u8; HASH_LEN]); }
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
            } else { layer[idx - 1] };
            path.push(sibling);
            idx /= 2;
        }
        MerkleProof { index, path }
    }
}

#[derive(Clone)]
pub struct MerkleProof { pub index: usize, pub path: Vec<Hash> }

impl MerkleProof {
    pub fn verify(&self, leaf_hash: &Hash, root: &Hash) -> bool {
        let mut current = *leaf_hash;
        let mut idx = self.index;
        for sibling in &self.path {
            current = if idx % 2 == 0 { hash_pair(&current, sibling) }
                      else { hash_pair(sibling, &current) };
            idx /= 2;
        }
        current == *root
    }
    pub fn byte_size(&self) -> usize { self.path.len() * HASH_LEN }
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
    pub x_flat: Vec<F>,       // m × n', row-major (bit-reversed eval order)
    pub y: Vec<Vec<EF>>,      // n rows × m' cols (bit-reversed eval order)
    pub diag: Vec<EF>,
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
    pub timing: EncodeTiming,
}

pub fn encode(data: &[F], n: usize, n_prime: usize) -> ZodaEncoding {
    assert_eq!(data.len(), n * n_prime);
    assert!(n.is_power_of_two() && n >= 32);
    assert!(n_prime.is_power_of_two() && n_prime >= 32);

    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let t_total = std::time::Instant::now();

    let plan_col = NttPlan::new(m.trailing_zeros() as usize);
    let plan_row = NttPlan::new(m_prime.trailing_zeros() as usize);

    // Step 1: Column encode X = G · X̃ (per-FFT allocation, parallel)
    let t0 = std::time::Instant::now();
    let x_col_vecs: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let coeffs: Vec<F> = (0..n).map(|row| data[row * n_prime + col]).collect();
            plan_col.fwd_poly(&coeffs)
        })
        .collect();
    let x_flat: Vec<F> = (0..m * n_prime)
        .into_par_iter()
        .map(|idx| x_col_vecs[idx % n_prime][idx / n_prime])
        .collect();
    drop(x_col_vecs);
    let col_fft = t0.elapsed();

    // Step 2: Commit to rows of X
    let t0 = std::time::Instant::now();
    let x_leaves: Vec<Hash> = (0..m)
        .into_par_iter()
        .map(|i| hash_f_slice(&x_flat[i * n_prime..(i + 1) * n_prime]))
        .collect();
    let tree_x = MerkleTree::from_leaves(x_leaves);
    let root_x = tree_x.root();
    let commit_x = t0.elapsed();

    // Step 3: Diagonal scale X̃·D
    let t0 = std::time::Instant::now();
    let diag = derive_diagonal(&root_x, n_prime);
    let x_tilde_d: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| {
            let rd = &data[row * n_prime..(row + 1) * n_prime];
            rd.iter().enumerate()
                .map(|(col, &x)| EF::from(x) * diag[col])
                .collect()
        })
        .collect();
    let diag_scale = t0.elapsed();

    // Step 4: Row encode Y = (X̃·D) · G'^T (EF via per-FFT 5-way decompose)
    let t0 = std::time::Instant::now();
    let y_rows: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| plan_row.fwd_poly_ef(&x_tilde_d[row]))
        .collect();
    let row_fft = t0.elapsed();

    // Step 5: Commit to columns of Y
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
        root_x, root_y, tree_x, tree_y,
        x_flat, y: y_rows, diag,
        n, n_prime, m, m_prime,
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

pub struct XRowOpening { pub row_index: usize, pub data: Vec<F>, pub proof: MerkleProof }
pub struct YColOpening { pub col_index: usize, pub data: Vec<EF>, pub proof: MerkleProof }

impl ZodaEncoding {
    pub fn open_x_row(&self, i: usize) -> XRowOpening {
        let start = i * self.n_prime;
        XRowOpening {
            row_index: i,
            data: self.x_flat[start..start + self.n_prime].to_vec(),
            proof: self.tree_x.open(i),
        }
    }
    pub fn x_row(&self, i: usize) -> &[F] {
        &self.x_flat[i * self.n_prime..(i + 1) * self.n_prime]
    }
    pub fn open_y_col(&self, j: usize) -> YColOpening {
        let col_data: Vec<EF> = (0..self.n).map(|row| self.y[row][j]).collect();
        YColOpening { col_index: j, data: col_data, proof: self.tree_y.open(j) }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Verification
// ═══════════════════════════════════════════════════════════════════════════

pub struct ZodaCommitment {
    pub root_x: Hash,
    pub root_y: Hash,
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
    pub diag: Vec<EF>,
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

pub fn verify(
    commitment: &ZodaCommitment,
    x_openings: &[XRowOpening],
    y_openings: &[YColOpening],
) -> VerifyResult {
    let plan_row = NttPlan::new(commitment.m_prime.trailing_zeros() as usize);
    let plan_col = NttPlan::new(commitment.m.trailing_zeros() as usize);

    // Verify Merkle proofs
    let mut merkle_ok = true;
    for xo in x_openings {
        if !xo.proof.verify(&hash_f_slice(&xo.data), &commitment.root_x) { merkle_ok = false; }
    }
    for yo in y_openings {
        if !yo.proof.verify(&hash_ef_slice(&yo.data), &commitment.root_y) { merkle_ok = false; }
    }

    // Precompute G · y_j (column-code NTT of each opened Y column)
    let g_yj_cache: Vec<Vec<EF>> = y_openings
        .iter()
        .map(|yo| plan_col.fwd_poly_ef(&yo.data))
        .collect();

    // Consistency check: Σ_k X[i][k] · D[k] · e_j^k = (G · y_j)[i]
    // where e_j = plan_row.eval_pts[j] is the j-th eval point of the row code
    let mut consistency_checks = 0usize;
    let mut consistency_passed = 0usize;

    for (yo, g_yj) in y_openings.iter().zip(&g_yj_cache) {
        let eval_pt = plan_row.eval_pts[yo.col_index];
        let ep_ef = EF::from(eval_pt);
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
//  Decoding
// ═══════════════════════════════════════════════════════════════════════════

pub fn decode_from_x_flat(x_flat: &[F], m: usize, n: usize, n_prime: usize) -> Vec<F> {
    let plan = NttPlan::new(m.trailing_zeros() as usize);
    let decoded_cols: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let evals: Vec<F> = (0..m).map(|row| x_flat[row * n_prime + col]).collect();
            let coeffs = plan.inv_evals(&evals);
            coeffs[..n].to_vec()
        })
        .collect();
    let mut data = vec![F::ZERO; n * n_prime];
    for col in 0..n_prime { for row in 0..n { data[row * n_prime + col] = decoded_cols[col][row]; } }
    data
}

pub fn decode_from_y_rows(y_rows: &[Vec<EF>], diag: &[EF], n: usize, n_prime: usize) -> Vec<F> {
    let m_prime = y_rows[0].len();
    let plan = NttPlan::new(m_prime.trailing_zeros() as usize);
    let diag_inv: Vec<EF> = diag.iter().map(|d| d.inverse()).collect();
    let x_tilde_d_rows: Vec<Vec<EF>> = (0..n)
        .into_par_iter()
        .map(|row| {
            let coeffs = plan.inv_evals_ef(&y_rows[row]);
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
        if !seen[idx] { seen[idx] = true; indices.push(idx); }
    }
    indices
}
