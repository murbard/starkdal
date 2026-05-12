use std::sync::{Arc, OnceLock};
use cudarc::driver::safe::{CudaContext, CudaStream};
use field::PrimeCharacteristicRing;
use koala_bear::KoalaBear;
use symetric::DIGEST_ELEMS;

pub(crate) struct GpuBackend {
    pub merkle: gpu_merkle::GpuMerkle,
    pub ntt: gpu_ntt::GpuNtt,
    pub sumcheck: gpu_sumcheck::GpuSumcheck,
    pub fold: gpu_poly_fold::GpuPolyFold,
    pub stream: Arc<CudaStream>,
}

static GPU: OnceLock<Option<GpuBackend>> = OnceLock::new();

pub(crate) fn gpu() -> Option<&'static GpuBackend> {
    GPU.get_or_init(|| {
        match CudaContext::new(0) {
            Ok(ctx) => {
                let stream = ctx.default_stream();
                tracing::info!("GPU prover backend initialized");
                Some(GpuBackend {
                    merkle: gpu_merkle::GpuMerkle::new(stream.clone()),
                    ntt: gpu_ntt::GpuNtt::new(stream.clone()),
                    sumcheck: gpu_sumcheck::GpuSumcheck::new(stream.clone()),
                    fold: gpu_poly_fold::GpuPolyFold::new(stream.clone()),
                    stream,
                })
            }
            Err(e) => {
                tracing::warn!("GPU not available: {e:?}");
                None
            }
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
    let base_u32: &[u32] = unsafe {
        std::slice::from_raw_parts(base_values.as_ptr().cast::<u32>(), base_values.len())
    };
    let d_matrix = g.stream.memcpy_stod(base_u32).ok()?;
    let (_root_flat, layers_flat) = g.merkle.build_tree_from_device(
        &d_matrix, height as u32, row_width as u32, dft_width as u32,
    );
    Some(layers_flat.iter().map(|layer| {
        let n = layer.len() / DIGEST_ELEMS;
        (0..n).map(|i| {
            let mut d = [KoalaBear::ZERO; DIGEST_ELEMS];
            for j in 0..DIGEST_ELEMS {
                d[j] = unsafe { std::mem::transmute(layer[i * DIGEST_ELEMS + j]) };
            }
            d
        }).collect()
    }).collect())
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

    let base_u32: &[u32] = unsafe {
        std::slice::from_raw_parts(base_values.as_ptr().cast::<u32>(), base_values.len())
    };

    // Upload polynomial to GPU once.
    let d_evals = g.stream.memcpy_stod(base_u32).ok()?;

    // Reorder + DFT on device.
    let d_dft = g.ntt.reorder_and_dft_device(
        &d_evals, n_evals as u32, folding_factor, log_inv_rate,
    );

    // Merkle on DFT output (stays on device — no re-upload!).
    let n_cols = 1u32 << folding_factor;
    let full_len = (n_evals as u64) << log_inv_rate;
    let height = (full_len / n_cols as u64) as u32;
    let (_root_flat, layers_flat) = g.merkle.build_tree_from_device(
        &d_dft, height, merkle_row_width as u32, n_cols,
    );

    // Download DFT output for CPU-side operations.
    let dft_u32 = g.stream.memcpy_dtov(&d_dft).ok()?;
    let dft_kb: Vec<KoalaBear> = unsafe {
        std::mem::transmute::<Vec<u32>, Vec<KoalaBear>>(dft_u32)
    };

    let digest_layers = layers_flat.iter().map(|layer| {
        let n = layer.len() / DIGEST_ELEMS;
        (0..n).map(|i| {
            let mut d = [KoalaBear::ZERO; DIGEST_ELEMS];
            for j in 0..DIGEST_ELEMS {
                d[j] = unsafe { std::mem::transmute(layer[i * DIGEST_ELEMS + j]) };
            }
            d
        }).collect()
    }).collect();

    Some((digest_layers, dft_kb))
}
