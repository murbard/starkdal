use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream};
use field::PrimeCharacteristicRing;
use koala_bear::{
    KoalaBear, poseidon1_round_constants, poseidon1_sparse_first_round_constants, poseidon1_sparse_first_row,
    poseidon1_sparse_m_i, poseidon1_sparse_scalar_round_constants, poseidon1_sparse_v,
};
use std::sync::{Arc, OnceLock};
use symetric::DIGEST_ELEMS;

pub(crate) struct GpuBackend {
    pub merkle: gpu_merkle::GpuMerkle,
    pub ntt: gpu_ntt::GpuNtt,
    pub pow: gpu_pow_grind::GpuPowGrinder,
    pub graph_pow: gpu_pow_grind::GpuPowGrinder,
    pub sumcheck: gpu_sumcheck::GpuSumcheck,
    pub graph_sumcheck: gpu_sumcheck::GpuSumcheck,
    pub fold: gpu_poly_fold::GpuPolyFold,
    pub graph_fold: gpu_poly_fold::GpuPolyFold,
    pub stream: Arc<CudaStream>,
    pub graph_stream: Arc<CudaStream>,
    pub p16: GpuPoseidon16Constants,
    pub d_ext_one: CudaSlice<u32>,
}

pub(crate) struct GpuPoseidon16Constants {
    pub d_rc: CudaSlice<u32>,
    pub d_mds: CudaSlice<u32>,
    pub d_sparse: CudaSlice<u32>,
}

impl GpuPoseidon16Constants {
    fn new(stream: &Arc<CudaStream>) -> Self {
        let rc = poseidon1_round_constants();
        let rf: Vec<u32> = rc.iter().flat_map(|r| r.iter().map(|v| kb_u32(*v))).collect();
        let d_rc = stream.memcpy_stod(&rf).expect("upload poseidon round constants");

        let mds: [u32; 16] =
            [1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1].map(|v| kb_u32(KoalaBear::from_u32(v)));
        let d_mds = stream.memcpy_stod(&mds).expect("upload poseidon mds");

        let mi = poseidon1_sparse_m_i();
        let fr = poseidon1_sparse_first_row();
        let v = poseidon1_sparse_v();
        let sr = poseidon1_sparse_scalar_round_constants();
        let mut sparse: Vec<u32> = Vec::with_capacity(912);
        for row in mi.iter() {
            for val in row.iter() {
                sparse.push(kb_u32(*val));
            }
        }
        for row in fr.iter() {
            for val in row.iter() {
                sparse.push(kb_u32(*val));
            }
        }
        for row in v.iter() {
            for i in 0..15 {
                sparse.push(kb_u32(row[i]));
            }
        }
        for val in sr.iter() {
            sparse.push(kb_u32(*val));
        }
        while sparse.len() < 896 {
            sparse.push(0);
        }
        for val in poseidon1_sparse_first_round_constants().iter() {
            sparse.push(kb_u32(*val));
        }
        let d_sparse = stream.memcpy_stod(&sparse).expect("upload poseidon sparse constants");
        Self { d_rc, d_mds, d_sparse }
    }

    pub(crate) fn as_slices(&self) -> (&CudaSlice<u32>, &CudaSlice<u32>, &CudaSlice<u32>) {
        (&self.d_rc, &self.d_mds, &self.d_sparse)
    }
}

fn kb_u32(v: KoalaBear) -> u32 {
    unsafe { std::mem::transmute(v) }
}

static GPU: OnceLock<Option<GpuBackend>> = OnceLock::new();

fn build_backend(stream: Arc<CudaStream>, graph_stream: Arc<CudaStream>) -> GpuBackend {
    tracing::info!("GPU prover backend initialized");
    let p16 = GpuPoseidon16Constants::new(&stream);
    let ext_one = [
        kb_u32(KoalaBear::ONE),
        kb_u32(KoalaBear::ZERO),
        kb_u32(KoalaBear::ZERO),
        kb_u32(KoalaBear::ZERO),
        kb_u32(KoalaBear::ZERO),
    ];
    let d_ext_one = stream.memcpy_stod(&ext_one).expect("upload extension one");
    GpuBackend {
        merkle: gpu_merkle::GpuMerkle::new(stream.clone()),
        ntt: gpu_ntt::GpuNtt::new(stream.clone()),
        pow: gpu_pow_grind::GpuPowGrinder::new(stream.clone()),
        graph_pow: gpu_pow_grind::GpuPowGrinder::new(graph_stream.clone()),
        sumcheck: gpu_sumcheck::GpuSumcheck::new(stream.clone()),
        graph_sumcheck: gpu_sumcheck::GpuSumcheck::new(graph_stream.clone()),
        fold: gpu_poly_fold::GpuPolyFold::new(stream.clone()),
        graph_fold: gpu_poly_fold::GpuPolyFold::new(graph_stream.clone()),
        stream,
        graph_stream,
        p16,
        d_ext_one,
    }
}

pub(crate) fn gpu_on_stream(stream: Arc<CudaStream>) -> Option<&'static GpuBackend> {
    let requested_stream = stream.clone();
    let backend = GPU
        .get_or_init(|| {
            let graph_stream = stream.context().new_stream().ok()?;
            Some(build_backend(stream, graph_stream))
        })
        .as_ref()?;
    (backend.stream.as_ref() == requested_stream.as_ref()).then_some(backend)
}

pub(crate) fn gpu() -> Option<&'static GpuBackend> {
    GPU.get_or_init(|| match CudaContext::new(0) {
        Ok(ctx) => {
            // Disable event tracking so device_ptr() calls don't insert
            // stream.wait(event), which breaks CUDA graph capture.
            unsafe {
                ctx.disable_event_tracking();
            }
            let stream = ctx.new_stream().expect("failed to create CUDA stream");
            let graph_stream = ctx.new_stream().expect("failed to create CUDA graph stream");
            Some(build_backend(stream, graph_stream))
        }
        Err(e) => {
            tracing::warn!("GPU not available: {e:?}");
            None
        }
    })
    .as_ref()
}

pub(crate) fn gpu_build_merkle_digests(
    base_values: &[KoalaBear],
    height: usize,
    row_width: usize,
    dft_width: usize,
) -> Option<Vec<Vec<[KoalaBear; DIGEST_ELEMS]>>> {
    let g = gpu()?;
    let base_u32: &[u32] = unsafe { std::slice::from_raw_parts(base_values.as_ptr().cast::<u32>(), base_values.len()) };
    let d_matrix = g.stream.memcpy_stod(base_u32).ok()?;
    let (_root_flat, layers_flat) =
        g.merkle
            .build_tree_from_device(&d_matrix, height as u32, row_width as u32, dft_width as u32);
    Some(
        layers_flat
            .iter()
            .map(|layer| {
                let n = layer.len() / DIGEST_ELEMS;
                (0..n)
                    .map(|i| {
                        let mut d = [KoalaBear::ZERO; DIGEST_ELEMS];
                        for j in 0..DIGEST_ELEMS {
                            d[j] = unsafe { std::mem::transmute(layer[i * DIGEST_ELEMS + j]) };
                        }
                        d
                    })
                    .collect()
            })
            .collect(),
    )
}

/// GPU-chained: reorder → DFT → Merkle, all on device.
/// Returns (digest_layers, dft_output_as_cpu_vec).
/// The DFT output is also downloaded for CPU-side operations (OOD eval, Merkle opening).
pub(crate) fn gpu_reorder_dft_merkle(
    base_values: &[KoalaBear],
    n_evals: usize,
    folding_factor: usize,
    log_inv_rate: usize,
    merkle_row_width: usize,
) -> Option<(Vec<Vec<[KoalaBear; DIGEST_ELEMS]>>, Vec<KoalaBear>)> {
    let g = gpu()?;

    let base_u32: &[u32] = unsafe { std::slice::from_raw_parts(base_values.as_ptr().cast::<u32>(), base_values.len()) };

    // Upload polynomial to GPU once.
    let d_evals = g.stream.memcpy_stod(base_u32).ok()?;

    // Reorder + DFT on device.
    let d_dft = g
        .ntt
        .reorder_and_dft_device(&d_evals, n_evals as u32, folding_factor, log_inv_rate);

    // Merkle on DFT output (stays on device — no re-upload!).
    let n_cols = 1u32 << folding_factor;
    let full_len = (n_evals as u64) << log_inv_rate;
    let height = (full_len / n_cols as u64) as u32;
    let (_root_flat, layers_flat) = g
        .merkle
        .build_tree_from_device(&d_dft, height, merkle_row_width as u32, n_cols);

    // Download DFT output for CPU-side operations.
    let dft_u32 = g.stream.memcpy_dtov(&d_dft).ok()?;
    let dft_kb: Vec<KoalaBear> = unsafe { std::mem::transmute::<Vec<u32>, Vec<KoalaBear>>(dft_u32) };

    let digest_layers = layers_flat
        .iter()
        .map(|layer| {
            let n = layer.len() / DIGEST_ELEMS;
            (0..n)
                .map(|i| {
                    let mut d = [KoalaBear::ZERO; DIGEST_ELEMS];
                    for j in 0..DIGEST_ELEMS {
                        d[j] = unsafe { std::mem::transmute(layer[i * DIGEST_ELEMS + j]) };
                    }
                    d
                })
                .collect()
        })
        .collect();

    Some((digest_layers, dft_kb))
}
