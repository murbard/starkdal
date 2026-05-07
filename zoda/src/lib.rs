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

fn hash_f_slice(data: &[F]) -> Hash {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * std::mem::size_of::<F>())
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

pub struct NttPlan {
    pub log_n: usize,
    pub n: usize,
    plan: concrete_ntt::prime32::Plan,
    pub psi: F,
    pub eval_pts: Vec<F>,
}

impl NttPlan {
    pub fn new(log_n: usize) -> Self {
        let n = 1usize << log_n;
        let plan = concrete_ntt::prime32::Plan::try_new(n, P as u32)
            .unwrap_or_else(|| panic!("concrete-ntt: size {n} unsupported for p={P:#x}"));
        let one_raw: u32 = unsafe { *(&F::from_u32(1) as *const F as *const u32) };
        let mut probe = vec![0u32; n];
        probe[1] = one_raw;
        plan.fwd(&mut probe);
        let psi: F = unsafe { *(&probe[0] as *const u32 as *const F) };
        let eval_pts: Vec<F> = (0..n)
            .map(|k| psi.exp_u64((2 * bit_reverse(k, log_n) + 1) as u64))
            .collect();
        Self { log_n, n, plan, psi, eval_pts }
    }

    pub fn fwd(&self, data: &mut [F]) {
        let data_u32: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u32>(), self.n)
        };
        self.plan.fwd(data_u32);
    }

    pub fn inv(&self, data: &mut [F]) {
        let data_u32: &mut [u32] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u32>(), self.n)
        };
        self.plan.inv(data_u32);
        let n_inv = F::from_u32(self.n as u32).inverse();
        for v in data.iter_mut() { *v *= n_inv; }
    }

    pub fn fwd_poly(&self, coeffs: &[F]) -> Vec<F> {
        let mut data = vec![F::ZERO; self.n];
        data[..coeffs.len().min(self.n)].copy_from_slice(&coeffs[..coeffs.len().min(self.n)]);
        self.fwd(&mut data);
        data
    }

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

impl MerkleTree {
    pub fn from_leaves(leaves: Vec<Hash>) -> Self {
        assert!(!leaves.is_empty());
        let mut layers = vec![leaves];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let mut padded = prev.clone();
            if padded.len() % 2 != 0 { padded.push([0u8; HASH_LEN]); }
            let next: Vec<Hash> = (0..padded.len() / 2)
                .map(|i| hash_pair(&padded[2*i], &padded[2*i+1])).collect();
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
                if idx+1 < layer.len() { layer[idx+1] } else { [0u8; HASH_LEN] }
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
    pub z_row_major: Vec<F>,     // z_row_major[row * m' + col], m × m' base field
    pub z_col_vecs: Vec<Vec<F>>, // z_col_vecs[col][row], m' columns of length m
    pub z_r: Vec<EF>,            // proof vector z_r = X̃·G'^T·ḡ_r (length m)
    pub z_r_prime: Vec<EF>,      // proof vector z'_{r'} = X̃^T·G^T·ḡ'_{r'} (length m')
    pub g_bar: Vec<EF>,          // random vector ḡ_r (length n')
    pub g_bar_prime: Vec<EF>,    // random vector ḡ'_{r'} (length n)
    pub n: usize,
    pub n_prime: usize,
    pub m: usize,
    pub m_prime: usize,
    pub timing: EncodeTiming,
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

    // Build row-major Z for row commitments
    let z_row_major: Vec<F> = (0..m * m_prime)
        .into_par_iter()
        .map(|idx| z_col_vecs[idx % m_prime][idx / m_prime])
        .collect();

    // Step 3: Commit Z by rows AND by columns
    let t0 = std::time::Instant::now();
    let row_leaves: Vec<Hash> = (0..m)
        .into_par_iter()
        .map(|i| hash_f_slice(&z_row_major[i * m_prime..(i + 1) * m_prime]))
        .collect();
    let col_leaves: Vec<Hash> = z_col_vecs.par_iter()
        .map(|col| hash_f_slice(col))
        .collect();
    let tree_rows = MerkleTree::from_leaves(row_leaves);
    let tree_cols = MerkleTree::from_leaves(col_leaves);
    let commit = t0.elapsed();

    // Step 4: Derive random vectors and compute proof vectors
    let t0 = std::time::Instant::now();
    let root_rows = tree_rows.root();
    let root_cols = tree_cols.root();

    // Dimensions (from verification checks):
    // ḡ_r ∈ EF^{m'}, z_r ∈ EF^n: z_r = (X̃·G'^T) · ḡ_r
    // ḡ'_{r'} ∈ EF^m, z'_{r'} ∈ EF^{n'}: z'_{r'} = (G·X̃)^T · ḡ'_{r'}
    let g_bar: Vec<EF> = derive_ef_vector(&root_rows, m_prime, b"ZODA_G_BAR");
    let g_bar_prime: Vec<EF> = derive_ef_vector(&root_cols, m, b"ZODA_G_BAR_PRIME");

    // z_r[row] = sum_j Z[row][j] * ḡ_r[j]  (Z row = X·G'^T row, already computed)
    // Actually z_r = (X̃·G'^T)·ḡ_r, but Z = G·(X̃·G'^T), so Z[eval] = G-encoded row.
    // We need W' = X̃·G'^T, not Z. But we computed Z = X·G'^T where X = G·X̃.
    // Z[eval][j] = row NTT of X[eval] at j. X[eval] is already column-encoded.
    // So Z[eval] ≠ W'[row]. We need W'[row] = row NTT of X̃[row].
    // W'[row][j] = z_rows_vecs... no, z_rows_vecs[eval][j] = Z[eval][j] = NTT_row(X[eval]).
    // We don't have W' directly. But z_r = W' · ḡ_r where W'[row] = NTT_row(X̃[row]).
    // z_r[row] = sum_j NTT_row(X̃[row])[j] * ḡ_r[j].
    // We can compute this without storing W': for each row, NTT X̃[row] then dot with ḡ_r.
    // But that's n extra NTTs. Instead: use Z and X.
    // z_r = W' · ḡ_r ∈ EF^n (reuse W' = X̃·G'^T from step 1)
    let z_r: Vec<EF> = (0..n)
        .into_par_iter()
        .map(|row| {
            let mut acc = EF::ZERO;
            for j in 0..m_prime { acc += EF::from(w_prime_rows[row][j]) * g_bar[j]; }
            acc
        })
        .collect();

    // z'_{r'} = X^T · ḡ'_{r'} where X = G·X̃. Compute X columns (n' NTTs, reusing data).
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
        tree_rows, tree_cols,
        z_row_major, z_col_vecs,
        z_r, z_r_prime, g_bar, g_bar_prime,
        n, n_prime, m, m_prime,
        timing: EncodeTiming { row_fft, col_fft, commit, proof_vecs, total },
    }
}

fn derive_ef_vector(root: &Hash, len: usize, domain: &[u8]) -> Vec<EF> {
    let mut vec = Vec::with_capacity(len);
    let mut state = *root;
    for i in 0..len {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&state);
        hasher.update(&(i as u64).to_le_bytes());
        hasher.update(domain);
        state = *hasher.finalize().as_bytes();
        let mut comps = [F::ZERO; DIM];
        for (k, comp) in comps.iter_mut().enumerate() {
            let off = k * 4;
            let val = u32::from_le_bytes([state[off], state[off+1], state[off+2], state[off+3]]);
            *comp = F::from_u32(val % (P as u32));
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
        RowOpening {
            index: i,
            data: self.z_row_major[i * self.m_prime..(i+1) * self.m_prime].to_vec(),
            proof: self.tree_rows.open(i),
        }
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

pub fn verify(
    enc: &ZodaEncoding,
    row_openings: &[RowOpening],
    col_openings: &[ColOpening],
) -> VerifyResult {
    let plan_col = NttPlan::new(enc.m.trailing_zeros() as usize);

    // Verify Merkle proofs
    let mut merkle_ok = true;
    for ro in row_openings {
        if !ro.proof.verify(&hash_f_slice(&ro.data), &enc.tree_rows.root()) { merkle_ok = false; }
    }
    for co in col_openings {
        if !co.proof.verify(&hash_f_slice(&co.data), &enc.tree_cols.root()) { merkle_ok = false; }
    }

    // Check 6: W_S · ḡ_r = G_S · z_r
    // W_S[s] is row s of Z. W_S[s] · ḡ_r = sum_j Z[s][j] * ḡ_r[j].
    // G_S · z_r: G_S[s] is row s of G. (G · z_r)[s] = NTT(z_r) at position s.
    // But z_r is EF and G is base-field NTT. We need G·z_r = column-code NTT of z_r.
    // Decompose z_r into 5 base-field NTTs.
    let g_zr: Vec<EF> = {
        let comps: Vec<Vec<F>> = (0..DIM).map(|k| {
            let coeffs: Vec<F> = enc.z_r.iter().map(|ef| ef.as_basis_coefficients_slice()[k]).collect();
            plan_col.fwd_poly(&coeffs)
        }).collect();
        (0..enc.m).map(|i| {
            EF::from_basis_coefficients_slice(&[
                comps[0][i], comps[1][i], comps[2][i], comps[3][i], comps[4][i],
            ]).unwrap()
        }).collect()
    };

    let mut row_checks = 0usize;
    let mut row_passed = 0usize;
    for ro in row_openings {
        // LHS: sum_j Z[i][j] * ḡ_r[j]
        let mut lhs = EF::ZERO;
        for j in 0..enc.m_prime { lhs += EF::from(ro.data[j]) * enc.g_bar[j]; }
        // RHS: (G · z_r)[i]
        let rhs = g_zr[ro.index];
        row_checks += 1;
        if lhs == rhs { row_passed += 1; }
    }

    // Check 7: (Y^T)_{S'} · ḡ'_{r'} = G'_{S'} · z'_{r'}
    // (Y^T)_{S'}[s] is column s of Z. (Y^T)[s] · ḡ'_{r'} = sum_i Z[i][s] * ḡ'_{r'}[i].
    // G'_{S'} · z'_{r'}: (G' · z'_{r'})[s] = row-code NTT of z'_{r'} at position s.
    let plan_row = NttPlan::new(enc.m_prime.trailing_zeros() as usize);
    let g_prime_zrp: Vec<EF> = {
        let comps: Vec<Vec<F>> = (0..DIM).map(|k| {
            let coeffs: Vec<F> = enc.z_r_prime.iter().map(|ef| ef.as_basis_coefficients_slice()[k]).collect();
            plan_row.fwd_poly(&coeffs)
        }).collect();
        (0..enc.m_prime).map(|i| {
            EF::from_basis_coefficients_slice(&[
                comps[0][i], comps[1][i], comps[2][i], comps[3][i], comps[4][i],
            ]).unwrap()
        }).collect()
    };

    let mut col_checks = 0usize;
    let mut col_passed = 0usize;
    for co in col_openings {
        let mut lhs = EF::ZERO;
        for i in 0..enc.m { lhs += EF::from(co.data[i]) * enc.g_bar_prime[i]; }
        let rhs = g_prime_zrp[co.index];
        col_checks += 1;
        if lhs == rhs { col_passed += 1; }
    }

    // Check 8: ḡ'^T_{r'} · G · z_r = ḡ^T_r · G' · z'_{r'}
    // LHS: sum_i ḡ'_{r'}[i] * (G·z_r)[i] = sum_i ḡ'_{r'}[i] * g_zr[i]
    let mut cross_lhs = EF::ZERO;
    for i in 0..enc.m { cross_lhs += enc.g_bar_prime[i] * g_zr[i]; }
    // RHS: sum_j ḡ_r[j] * (G'·z'_{r'})[j] = sum_j ḡ_r[j] * g_prime_zrp[j]
    let mut cross_rhs = EF::ZERO;
    for j in 0..enc.m_prime { cross_rhs += enc.g_bar[j] * g_prime_zrp[j]; }

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

/// Decode X̃ from rows of Z. Each row of Z = row of G·X̃·G'^T.
/// Inverse row-NTT gives a row of G·X̃, then inverse col-NTT gives X̃.
pub fn decode_from_rows(z_rows: &[Vec<F>], m: usize, m_prime: usize, n: usize, n_prime: usize) -> Vec<F> {
    let plan_row = NttPlan::new(m_prime.trailing_zeros() as usize);
    let plan_col = NttPlan::new(m.trailing_zeros() as usize);
    // Inverse row NTT: Z rows → W' rows (= X̃·G'^T rows → X̃ rows after inv row NTT)
    // Actually Z = G·W', so inv col NTT of Z columns gives W' columns.
    // Then inv row NTT of W' rows gives X̃ rows.
    // Simpler: inv row NTT each Z row → gives (G·X̃)[:,col] at row positions → still encoded by G.
    // Need to process columnwise. Let me just inv both dimensions.

    // Step 1: inv row NTT each row of Z → W' rows
    // Wait, Z[i][j] = (G·W')[i][j]. Inv row NTT on Z rows doesn't undo G (column encoding).
    // Z = G·(X̃·G'^T). Columns of Z: Z[:,j] = G · (X̃·G'^T)[:,j].
    // Inv col NTT of Z[:,j] → (X̃·G'^T)[:,j] = W'[:,j].
    // Then W' rows are X̃·G'^T rows. Inv row NTT of W'[row] → X̃[row].

    // Step 1: Gather columns, inv col NTT → W' columns
    let w_prime_cols: Vec<Vec<F>> = (0..m_prime)
        .into_par_iter()
        .map(|col| {
            let z_col: Vec<F> = (0..m).map(|row| z_rows[row][col]).collect();
            let coeffs = plan_col.inv_evals(&z_col);
            coeffs[..n].to_vec()
        })
        .collect();
    // Step 2: Inv row NTT each row of W' → X̃ rows
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
