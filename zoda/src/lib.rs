//! ZODA Tensor Variation (Appendix E of Evans, Mohnblatt, Angeris 2025/034).
//!
//! Standard tensor code Z = G·X̃·G'^T entirely over base field F (4× expansion).
//! Proof of correct encoding via two short random-linear-combination vectors
//! z_r and z'_{r'} in extension field E. All NTTs are base-field.

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

/// Domain-separated leaf hash. `tag` distinguishes row vs column Merkle trees.
fn hash_f_slice_tagged(tag: u8, data: &[F]) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[tag]);
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * std::mem::size_of::<F>())
    };
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

const TAG_ROW: u8 = 0x00;
const TAG_COL: u8 = 0x01;

const TAG_INTERNAL: u8 = 0x02;

fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&[TAG_INTERNAL]);
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
/// Output is in bit-reversed order (negacyclic convention).
/// eval_pts[k] = ψ^{2·bitrev(k)+1} is the evaluation point at output position k.
pub struct NttPlan {
    pub log_n: usize,
    pub n: usize,
    plan: concrete_ntt::prime32::Plan,
    n_inv: F,
    pub psi: F,
    pub eval_pts: Vec<F>,
}

impl NttPlan {
    pub fn new(log_n: usize) -> Self {
        let n = 1usize << log_n;
        let plan = concrete_ntt::prime32::Plan::try_new(n, P as u32)
            .unwrap_or_else(|| panic!("concrete-ntt: size {n} unsupported for p={P:#x}"));

        // Extract ψ from NTT of [0, monty(1), 0, ..., 0].
        let one_raw: u32 = unsafe { *(&F::from_u32(1) as *const F as *const u32) };
        let mut probe = vec![0u32; n];
        probe[1] = one_raw;
        plan.fwd(&mut probe);
        let psi: F = unsafe { *(&probe[0] as *const u32 as *const F) };

        let eval_pts: Vec<F> = (0..n)
            .map(|k| psi.exp_u64((2 * bit_reverse(k, log_n) + 1) as u64))
            .collect();

        let n_inv = F::from_u32(n as u32).inverse();

        Self { log_n, n, plan, n_inv, psi, eval_pts }
    }

    /// Forward negacyclic NTT in-place.
    pub fn fwd(&self, data: &mut [F]) {
        assert_eq!(data.len(), self.n);
        let data_u32: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u32>(), self.n)
        };
        self.plan.fwd(data_u32);
    }

    /// Inverse negacyclic NTT in-place, with 1/n normalization.
    pub fn inv(&self, data: &mut [F]) {
        assert_eq!(data.len(), self.n);
        let data_u32: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u32>(), self.n)
        };
        self.plan.inv(data_u32);
        for v in data.iter_mut() { *v *= self.n_inv; }
    }

    /// Forward NTT of a polynomial (zero-padded to size n).
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

    /// Inverse NTT recovering polynomial coefficients.
    pub fn inv_evals(&self, evals: &[F]) -> Vec<F> {
        let mut data = evals.to_vec();
        self.inv(&mut data);
        data
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Merkle tree (BLAKE3)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct MerkleTree { pub layers: Vec<Vec<Hash>> }

/// Sentinel for padding odd-length layers (distinct from any real hash).
const MERKLE_PAD: Hash = [0xFF; HASH_LEN];

impl MerkleTree {
    pub fn from_leaves(leaves: Vec<Hash>) -> Self {
        assert!(!leaves.is_empty());
        let mut layers = vec![leaves];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let n = prev.len();
            let pairs = n / 2;
            let mut next = Vec::with_capacity(pairs + (n & 1));
            for i in 0..pairs {
                next.push(hash_pair(&prev[2*i], &prev[2*i+1]));
            }
            if n % 2 != 0 {
                next.push(hash_pair(&prev[n - 1], &MERKLE_PAD));
            }
            layers.push(next);
        }
        Self { layers }
    }
    pub fn root(&self) -> Hash { self.layers.last().unwrap()[0] }
    pub fn open(&self, index: usize) -> MerkleProof {
        let mut path = Vec::with_capacity(self.layers.len() - 1);
        let mut idx = index;
        for layer in &self.layers[..self.layers.len()-1] {
            let sibling = if idx % 2 == 0 {
                if idx+1 < layer.len() { layer[idx+1] } else { MERKLE_PAD }
            } else { layer[idx-1] };
            path.push(sibling); idx /= 2;
        }
        MerkleProof { index, path }
    }
}

#[derive(Clone)]
pub struct MerkleProof { pub index: usize, pub path: Vec<Hash> }

impl MerkleProof {
    pub fn verify(&self, leaf_hash: &Hash, root: &Hash) -> bool {
        let mut current = *leaf_hash; let mut idx = self.index;
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
//  ZODA Tensor Variation (Appendix E)
// ═══════════════════════════════════════════════════════════════════════════

pub struct EncodeTiming {
    pub row_fft: std::time::Duration,
    pub col_fft: std::time::Duration,
    pub commit: std::time::Duration,
    pub proof_vecs: std::time::Duration,
    pub total: std::time::Duration,
}

pub struct ZodaEncoding {
    pub tree_rows: MerkleTree,   // commit Z by rows
    pub tree_cols: MerkleTree,   // commit Z by columns
    pub z_col_vecs: Vec<Vec<F>>, // z_col_vecs[col][row], m' columns of length m
    pub z_r: Vec<EF>,            // proof vector (length n)
    pub z_r_prime: Vec<EF>,      // proof vector (length n')
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
    pub timing: EncodeTiming,
}

/// What the verifier receives: Merkle roots + proof vectors + parameters.
/// Random vectors ḡ_r and ḡ'_{r'} are NOT stored — the verifier re-derives
/// them from the roots via Fiat-Shamir (prevents a malicious prover from
/// substituting different random vectors).
pub struct ZodaCommitment {
    pub root_rows: Hash,
    pub root_cols: Hash,
    pub z_r: Vec<EF>,        // proof vector (length n)
    pub z_r_prime: Vec<EF>,  // proof vector (length n')
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
}

impl From<&ZodaEncoding> for ZodaCommitment {
    fn from(enc: &ZodaEncoding) -> Self {
        Self {
            root_rows: enc.tree_rows.root(),
            root_cols: enc.tree_cols.root(),
            z_r: enc.z_r.clone(),
            z_r_prime: enc.z_r_prime.clone(),
            n: enc.n, n_prime: enc.n_prime, m: enc.m, m_prime: enc.m_prime,
        }
    }
}

/// Derive the Fiat-Shamir seed from both Merkle roots.
fn fs_seed(root_rows: &Hash, root_cols: &Hash) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(root_rows);
    hasher.update(root_cols);
    *hasher.finalize().as_bytes()
}

/// Encode data matrix X̃ (n × n', row-major) using the tensor variation.
/// Z = G·X̃·G'^T is entirely base-field. Proof vectors z_r, z'_{r'} are in EF.
pub fn encode(data: &[F], n: usize, n_prime: usize) -> ZodaEncoding {
    assert_eq!(data.len(), n * n_prime);
    assert!(n.is_power_of_two() && n >= 32);
    assert!(n_prime.is_power_of_two() && n_prime >= 32);

    let m = 2 * n;
    let m_prime = 2 * n_prime;
    let plan_row = NttPlan::new(m_prime.trailing_zeros() as usize);
    let plan_col = NttPlan::new(m.trailing_zeros() as usize);
    let t_total = std::time::Instant::now();

    // Step 1: Row encode W' = X̃·G'^T (n NTTs of size m', base field)
    let t0 = std::time::Instant::now();
    let w_prime_rows: Vec<Vec<F>> = (0..n)
        .into_par_iter()
        .map(|row| plan_row.fwd_poly(&data[row * n_prime..(row + 1) * n_prime]))
        .collect();
    let row_fft = t0.elapsed();

    // Step 2: Column encode Z = G·W' (m' NTTs of size m, base field)
    let t0 = std::time::Instant::now();
    let z_col_vecs: Vec<Vec<F>> = (0..m_prime)
        .into_par_iter()
        .map(|col| {
            let column: Vec<F> = (0..n).map(|row| w_prime_rows[row][col]).collect();
            plan_col.fwd_poly(&column)
        })
        .collect();
    let col_fft = t0.elapsed();

    // Step 3: Commit Z by rows AND by columns (domain-separated hashes)
    let t0 = std::time::Instant::now();
    let col_leaves: Vec<Hash> = z_col_vecs.par_iter()
        .map(|col| hash_f_slice_tagged(TAG_COL, col))
        .collect();
    let row_leaves: Vec<Hash> = (0..m)
        .into_par_iter()
        .map(|row| {
            let row_data: Vec<F> = (0..m_prime).map(|col| z_col_vecs[col][row]).collect();
            hash_f_slice_tagged(TAG_ROW, &row_data)
        })
        .collect();
    let tree_rows = MerkleTree::from_leaves(row_leaves);
    let tree_cols = MerkleTree::from_leaves(col_leaves);
    let commit = t0.elapsed();

    // Step 4: Proof vectors via Fiat-Shamir (randomness binds BOTH roots)
    let t0 = std::time::Instant::now();
    let seed = fs_seed(&tree_rows.root(), &tree_cols.root());
    let g_bar: Vec<EF> = derive_ef_vector(&seed, m_prime, b"ZODA_G_BAR");
    let g_bar_prime: Vec<EF> = derive_ef_vector(&seed, m, b"ZODA_G_BAR_PRIME");

    // z_r = W' · ḡ_r ∈ EF^n where W' = X̃·G'^T (from step 1)
    let z_r: Vec<EF> = (0..n)
        .into_par_iter()
        .map(|row| {
            let mut acc = EF::ZERO;
            for j in 0..m_prime { acc += EF::from(w_prime_rows[row][j]) * g_bar[j]; }
            acc
        })
        .collect();

    // z'_{r'} = X^T · ḡ'_{r'} ∈ EF^{n'} where X = G·X̃
    let x_col_vecs: Vec<Vec<F>> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let coeffs: Vec<F> = (0..n).map(|row| data[row * n_prime + col]).collect();
            plan_col.fwd_poly(&coeffs)
        })
        .collect();
    let z_r_prime: Vec<EF> = (0..n_prime)
        .into_par_iter()
        .map(|col| {
            let mut acc = EF::ZERO;
            for eval in 0..m { acc += EF::from(x_col_vecs[col][eval]) * g_bar_prime[eval]; }
            acc
        })
        .collect();

    let proof_vecs = t0.elapsed();
    let total = t_total.elapsed();

    ZodaEncoding {
        tree_rows, tree_cols, z_col_vecs,
        z_r, z_r_prime,
        n, n_prime, m, m_prime,
        timing: EncodeTiming { row_fft, col_fft, commit, proof_vecs, total },
    }
}

/// Derive EF random vector via BLAKE3 XOF with rejection sampling.
fn derive_ef_vector(seed: &Hash, len: usize, domain: &[u8]) -> Vec<EF> {
    let mut base_hasher = blake3::Hasher::new();
    base_hasher.update(seed);
    base_hasher.update(domain);
    let mut reader = base_hasher.finalize_xof();

    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        let mut comps = [F::ZERO; DIM];
        for comp in &mut comps {
            loop {
                let mut buf = [0u8; 4];
                reader.fill(&mut buf);
                let val = u32::from_le_bytes(buf);
                if val < P as u32 {
                    *comp = F::from_u32(val);
                    break;
                }
                // Rejection: val >= P, XOF automatically advances to next bytes
            }
        }
        vec.push(EF::from_basis_coefficients_slice(&comps).unwrap());
    }
    vec
}

// ═══════════════════════════════════════════════════════════════════════════
//  Openings
// ═══════════════════════════════════════════════════════════════════════════

pub struct RowOpening { pub index: usize, pub data: Vec<F>, pub proof: MerkleProof }
pub struct ColOpening { pub index: usize, pub data: Vec<F>, pub proof: MerkleProof }

impl ZodaEncoding {
    pub fn open_row(&self, i: usize) -> RowOpening {
        let data: Vec<F> = (0..self.m_prime).map(|col| self.z_col_vecs[col][i]).collect();
        RowOpening { index: i, data, proof: self.tree_rows.open(i) }
    }
    pub fn open_col(&self, j: usize) -> ColOpening {
        ColOpening {
            index: j,
            data: self.z_col_vecs[j].clone(),
            proof: self.tree_cols.open(j),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Verification (Appendix E sampling algorithm)
// ═══════════════════════════════════════════════════════════════════════════

pub struct VerifyResult {
    pub row_checks: usize,
    pub row_passed: usize,
    pub col_checks: usize,
    pub col_passed: usize,
    pub cross_check: bool,
    pub merkle_ok: bool,
}

impl VerifyResult {
    pub fn accept(&self) -> bool {
        self.merkle_ok
            && self.row_passed == self.row_checks
            && self.col_passed == self.col_checks
            && self.cross_check
    }
}

/// Verify ZODA proof from commitment + opened rows/columns.
/// Re-derives random vectors from the roots (doesn't trust the prover).
pub fn verify(
    comm: &ZodaCommitment,
    row_openings: &[RowOpening],
    col_openings: &[ColOpening],
) -> VerifyResult {
    let plan_col = NttPlan::new(comm.m.trailing_zeros() as usize);
    let plan_row = NttPlan::new(comm.m_prime.trailing_zeros() as usize);

    // Re-derive random vectors from roots (Fiat-Shamir)
    let seed = fs_seed(&comm.root_rows, &comm.root_cols);
    let g_bar: Vec<EF> = derive_ef_vector(&seed, comm.m_prime, b"ZODA_G_BAR");
    let g_bar_prime: Vec<EF> = derive_ef_vector(&seed, comm.m, b"ZODA_G_BAR_PRIME");

    // Verify Merkle proofs (domain-separated hashes)
    let mut merkle_ok = true;
    for ro in row_openings {
        if !ro.proof.verify(&hash_f_slice_tagged(TAG_ROW, &ro.data), &comm.root_rows) { merkle_ok = false; }
    }
    for co in col_openings {
        if !co.proof.verify(&hash_f_slice_tagged(TAG_COL, &co.data), &comm.root_cols) { merkle_ok = false; }
    }

    // Check 6: W_S · ḡ_r = G_S · z_r
    let g_zr = plan_col.fwd_poly_ef(&comm.z_r);

    let mut row_checks = 0usize;
    let mut row_passed = 0usize;
    for ro in row_openings {
        let mut lhs = EF::ZERO;
        for j in 0..comm.m_prime { lhs += EF::from(ro.data[j]) * g_bar[j]; }
        row_checks += 1;
        if lhs == g_zr[ro.index] { row_passed += 1; }
    }

    // Check 7: (Y^T)_{S'} · ḡ'_{r'} = G'_{S'} · z'_{r'}
    let g_prime_zrp = plan_row.fwd_poly_ef(&comm.z_r_prime);

    let mut col_checks = 0usize;
    let mut col_passed = 0usize;
    for co in col_openings {
        let mut lhs = EF::ZERO;
        for i in 0..comm.m { lhs += EF::from(co.data[i]) * g_bar_prime[i]; }
        col_checks += 1;
        if lhs == g_prime_zrp[co.index] { col_passed += 1; }
    }

    // Check 8: ḡ'^T_{r'} · G · z_r = ḡ^T_r · G' · z'_{r'}
    let mut cross_lhs = EF::ZERO;
    for i in 0..comm.m { cross_lhs += g_bar_prime[i] * g_zr[i]; }
    let mut cross_rhs = EF::ZERO;
    for j in 0..comm.m_prime { cross_rhs += g_bar[j] * g_prime_zrp[j]; }

    VerifyResult {
        row_checks, row_passed,
        col_checks, col_passed,
        cross_check: cross_lhs == cross_rhs,
        merkle_ok,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Decoding
// ═══════════════════════════════════════════════════════════════════════════

/// Decode X̃ from all rows of Z by inverting both dimensions of the tensor code.
/// Step 1: inverse column NTT on each column of Z → columns of W' = X̃·G'^T.
/// Step 2: inverse row NTT on each row of W' → rows of X̃.
pub fn decode_from_rows(z_rows: &[Vec<F>], m: usize, m_prime: usize, n: usize, n_prime: usize) -> Vec<F> {
    let plan_row = NttPlan::new(m_prime.trailing_zeros() as usize);
    let plan_col = NttPlan::new(m.trailing_zeros() as usize);

    let w_prime_cols: Vec<Vec<F>> = (0..m_prime)
        .into_par_iter()
        .map(|col| {
            let z_col: Vec<F> = (0..m).map(|row| z_rows[row][col]).collect();
            let coeffs = plan_col.inv_evals(&z_col);
            coeffs[..n].to_vec()
        })
        .collect();

    let data: Vec<F> = (0..n)
        .into_par_iter()
        .flat_map_iter(|row| {
            let w_row: Vec<F> = (0..m_prime).map(|col| w_prime_cols[col][row]).collect();
            let coeffs = plan_row.inv_evals(&w_row);
            coeffs[..n_prime].to_vec().into_iter()
        })
        .collect();
    data
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sampling helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate unique sample indices via BLAKE3 XOF (cryptographic PRNG).
pub fn sample_indices(root: &Hash, domain_sep: u32, count: usize, modulus: usize) -> Vec<usize> {
    assert!(count <= modulus);
    let mut hasher = blake3::Hasher::new_keyed(root);
    hasher.update(&domain_sep.to_le_bytes());
    let mut reader = hasher.finalize_xof();

    let mut indices = Vec::with_capacity(count);
    let mut seen = vec![false; modulus];
    while indices.len() < count {
        let mut buf = [0u8; 8];
        reader.fill(&mut buf);
        let val = u64::from_le_bytes(buf);
        let idx = (val as usize) % modulus;
        if !seen[idx] { seen[idx] = true; indices.push(idx); }
    }
    indices
}
