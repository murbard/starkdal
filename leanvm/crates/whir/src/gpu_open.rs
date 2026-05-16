//! GPU-accelerated WHIR prove — complete round loop on GPU.
//!
//! Keeps polynomial evaluations (`d_evals`) and weights (`d_weights`) on GPU.
//! Per WHIR round: GPU product sumcheck + fold + DFT + Merkle construction.
//! Remaining host interaction is Fiat-Shamir plus Merkle opening material.

use std::sync::Arc;

use cudarc::driver::{
    safe::{CudaSlice, CudaStream, DevicePtr},
    sys,
};
use fiat_shamir::{FSProver, MerklePath};
use field::{ExtensionField, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use poly::*;
use tracing::{info_span, instrument};
use utils::{log2_ceil_usize, log2_strict_usize};

use crate::commit::{MerkleData, Witness};
use crate::config::WhirConfig;
use crate::gpu_backend;
use crate::gpu_combine::{
    GpuDeviceStatementAccumulationGuard, GpuDeviceStatementAccumulationWorkspaces,
    gpu_accumulate_device_statements_with_device_gamma, gpu_accumulate_device_statements_with_device_gamma_into_async,
    gpu_accumulate_statement_with_device_gamma, gpu_combine_statement,
};
use crate::gpu_prove::{ef_from_u32, ef_to_u32};
use crate::{DIGEST_ELEMS, GpuDeviceSlice, GpuSparseStatement, SparseStatement};

#[derive(Debug, Clone, Copy)]
enum GpuMerkleLeafKind {
    Base,
    Extension,
}

#[derive(Debug)]
pub struct GpuMerkleProverData {
    d_leaf_matrix: CudaSlice<u32>,
    d_digest_layers: Vec<Arc<CudaSlice<u32>>>,
    _dft_guards: Vec<CudaSlice<u32>>,
    height: usize,
    full_leaf_base_width: usize,
    leaf_kind: GpuMerkleLeafKind,
}

#[derive(Debug)]
pub struct GpuInitialOodData {
    pub d_points: CudaSlice<u32>,
    pub d_answers: GpuDeviceSlice,
    pub n_samples: usize,
    pub _d_univariate_points: CudaSlice<u32>,
    pub _answer_intermediates: Vec<CudaSlice<u32>>,
}

#[derive(Debug)]
pub struct GpuTranscriptChunk {
    pub d_words: GpuDeviceSlice,
    pub n_words: usize,
}

#[derive(Debug)]
pub struct GpuTranscriptSeed {
    pub d_challenger_state: CudaSlice<u32>,
    pub d_empty_observe: CudaSlice<u32>,
    pub transcript_chunks: Vec<GpuTranscriptChunk>,
}

pub struct GpuWhirProverWorkspaces {
    d_weights: Option<CudaSlice<u32>>,
    d_initial_gamma: Option<CudaSlice<u32>>,
    d_initial_scalars: Option<CudaSlice<u32>>,
    d_zero_sum: Option<CudaSlice<u32>>,
    d_initial_ood_sum: Option<CudaSlice<u32>>,
    initial_device_statement_accumulation: Option<GpuDeviceStatementAccumulationWorkspaces>,
    d_randomness_words: Option<CudaSlice<u32>>,
    initial_sumcheck: Option<GpuWhirCapturedSumcheckWorkspaces>,
    round_workspaces: Vec<GpuWhirRoundWorkspaces>,
    final_queries: Option<GpuWhirFinalQueryWorkspaces>,
    final_sumcheck: Option<GpuWhirCapturedSumcheckWorkspaces>,
    d_final_materialization: Option<CudaSlice<u32>>,
    final_openings: Option<Vec<GpuWhirOpeningWorkspaces>>,
}

struct GpuWhirRoundWorkspaces {
    d_ood_points: Option<CudaSlice<u32>>,
    d_ood_challenges: Option<CudaSlice<u32>>,
    d_ood_answer_states: Vec<Option<CudaSlice<u32>>>,
    d_query_pow_witness: Option<CudaSlice<u32>>,
    d_query_pow_flag: Option<CudaSlice<u32>>,
    d_stir_sample_words: Option<CudaSlice<u32>>,
    d_stir_challenges: Option<CudaSlice<u32>>,
    d_stir_indices: Option<CudaSlice<u32>>,
    d_stir_evaluations: Option<CudaSlice<u32>>,
    d_gamma_comb: Option<CudaSlice<u32>>,
    d_constraint_scalars: Option<CudaSlice<u32>>,
    d_ood_constraint_sum: Option<CudaSlice<u32>>,
    d_stir_constraint_sum: Option<CudaSlice<u32>>,
    dft_twiddles: Option<gpu_ntt::GpuNttTwiddles>,
    d_dft_output: Option<CudaSlice<u32>>,
    merkle_layers: Option<Vec<CudaSlice<u32>>>,
    sumcheck: Option<GpuWhirCapturedSumcheckWorkspaces>,
}

struct GpuWhirFinalQueryWorkspaces {
    d_query_pow_witness: CudaSlice<u32>,
    d_query_pow_flag: CudaSlice<u32>,
    d_sample_words: CudaSlice<u32>,
    d_query_points: CudaSlice<u32>,
    d_indices: CudaSlice<u32>,
}

struct GpuWhirOpeningWorkspaces {
    d_row_words: Option<CudaSlice<u32>>,
    d_sibling_layers: Vec<Option<CudaSlice<u32>>>,
}

#[derive(Clone, Copy)]
struct GpuWhirOpeningWorkspaceSpec {
    n_samples: usize,
    height: usize,
    row_width: usize,
}

struct GpuWhirCapturedSumcheckWorkspaces {
    eval_states: Vec<CudaSlice<u32>>,
    weight_states: Vec<CudaSlice<u32>>,
    d_round_challenges: CudaSlice<u32>,
    d_round_rs: Vec<CudaSlice<u32>>,
    d_round_tails: Vec<CudaSlice<u32>>,
    d_pow_witnesses: Vec<CudaSlice<u32>>,
    d_pow_flags: Vec<CudaSlice<u32>>,
    d_c0_partials: CudaSlice<u32>,
    d_c2_partials: CudaSlice<u32>,
    d_c0_out: CudaSlice<u32>,
    d_c2_out: CudaSlice<u32>,
    d_poly: CudaSlice<u32>,
    d_empty_observe: CudaSlice<u32>,
}

fn take_whir_workspace<T>(slot: &mut Option<T>, name: &str) -> T {
    slot.take()
        .unwrap_or_else(|| panic!("WHIR workspace already consumed: {name}"))
}

impl GpuWhirCapturedSumcheckWorkspaces {
    fn allocate(stream: &Arc<CudaStream>, n_elements: usize, n_rounds: usize, pow_bits: usize, dim: usize) -> Self {
        let mut element_lens = Vec::with_capacity(n_rounds + 1);
        let mut half_lens = Vec::with_capacity(n_rounds);
        let mut current_len = n_elements;
        element_lens.push(current_len);
        for _ in 0..n_rounds {
            let half = current_len / 2;
            half_lens.push(half as u32);
            current_len = half;
            element_lens.push(current_len);
        }
        let max_blocks = if n_rounds == 0 {
            0
        } else {
            half_lens
                .iter()
                .map(|&half| (half as usize).div_ceil(256).max(1))
                .max()
                .unwrap_or(1)
        };
        let scratch_ext_words = if n_rounds == 0 { 0 } else { 5 };
        let poly_words = if n_rounds == 0 { 0 } else { 15 };
        let empty_observe_words = if n_rounds == 0 { 0 } else { 1 };
        Self {
            eval_states: element_lens[1..]
                .iter()
                .map(|&next_len| {
                    stream
                        .alloc_zeros::<u32>(next_len * dim)
                        .expect("alloc WHIR captured sumcheck eval state")
                })
                .collect(),
            weight_states: element_lens[1..]
                .iter()
                .map(|&next_len| {
                    stream
                        .alloc_zeros::<u32>(next_len * dim)
                        .expect("alloc WHIR captured sumcheck weight state")
                })
                .collect(),
            d_round_challenges: stream
                .alloc_zeros::<u32>(n_rounds * dim)
                .expect("alloc WHIR captured sumcheck challenges"),
            d_round_rs: (0..n_rounds)
                .map(|_| {
                    stream
                        .alloc_zeros::<u32>(dim)
                        .expect("alloc WHIR captured sumcheck round challenge")
                })
                .collect(),
            d_round_tails: (0..n_rounds)
                .map(|_| {
                    stream
                        .alloc_zeros::<u32>(10)
                        .expect("alloc WHIR captured sumcheck transcript tail")
                })
                .collect(),
            d_pow_witnesses: if pow_bits > 0 {
                (0..n_rounds)
                    .map(|_| {
                        stream
                            .alloc_zeros::<u32>(1)
                            .expect("alloc WHIR captured sumcheck pow witness")
                    })
                    .collect()
            } else {
                Vec::new()
            },
            d_pow_flags: if pow_bits > 0 {
                (0..n_rounds)
                    .map(|_| {
                        stream
                            .alloc_zeros::<u32>(1)
                            .expect("alloc WHIR captured sumcheck pow flag")
                    })
                    .collect()
            } else {
                Vec::new()
            },
            d_c0_partials: stream
                .alloc_zeros::<u32>(max_blocks * 5)
                .expect("alloc WHIR captured sumcheck c0 partials"),
            d_c2_partials: stream
                .alloc_zeros::<u32>(max_blocks * 5)
                .expect("alloc WHIR captured sumcheck c2 partials"),
            d_c0_out: stream
                .alloc_zeros::<u32>(scratch_ext_words)
                .expect("alloc WHIR captured sumcheck c0 output"),
            d_c2_out: stream
                .alloc_zeros::<u32>(scratch_ext_words)
                .expect("alloc WHIR captured sumcheck c2 output"),
            d_poly: stream
                .alloc_zeros::<u32>(poly_words)
                .expect("alloc WHIR captured sumcheck round polynomial"),
            d_empty_observe: stream
                .alloc_zeros::<u32>(empty_observe_words)
                .expect("alloc WHIR captured sumcheck empty observe"),
        }
    }
}

impl GpuWhirOpeningWorkspaces {
    fn allocate(stream: &Arc<CudaStream>, spec: GpuWhirOpeningWorkspaceSpec) -> Self {
        Self {
            d_row_words: Some(
                stream
                    .alloc_zeros::<u32>(spec.n_samples * spec.row_width)
                    .expect("alloc WHIR final opening row workspace"),
            ),
            d_sibling_layers: (0..log2_ceil_usize(spec.height))
                .map(|_| {
                    Some(
                        stream
                            .alloc_zeros::<u32>(spec.n_samples * DIGEST_ELEMS)
                            .expect("alloc WHIR final opening sibling workspace"),
                    )
                })
                .collect(),
        }
    }
}

impl<EF> WhirConfig<EF>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: PrimeField64 + TwoAdicField,
{
    pub fn gpu_allocate_prover_workspaces(
        &self,
        stream: &Arc<CudaStream>,
        device_statement_prefix: &[GpuSparseStatement],
        final_materialization_word_capacity: usize,
    ) -> GpuWhirProverWorkspaces {
        let dim = EF::DIMENSION;
        let n_total = 1usize << self.num_variables;
        let initial_device_statement_values = device_statement_prefix
            .iter()
            .map(|statement| statement.values.len())
            .sum::<usize>();
        let initial_device_statement_polys = device_statement_prefix
            .iter()
            .map(|statement| {
                if statement.is_next || statement.point_len > 0 {
                    Some(
                        stream
                            .alloc_zeros::<u32>((1usize << statement.point_len) * dim)
                            .expect("alloc WHIR initial device-statement poly workspace"),
                    )
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let initial_device_statement_accumulation =
            (initial_device_statement_values > 0).then(|| GpuDeviceStatementAccumulationWorkspaces {
                d_gamma_powers: stream
                    .alloc_zeros::<u32>((self.commitment_ood_samples + initial_device_statement_values) * dim)
                    .expect("alloc WHIR initial device-statement gamma powers"),
                d_statement_values: stream
                    .alloc_zeros::<u32>(initial_device_statement_values * dim)
                    .expect("alloc WHIR initial device-statement values"),
                d_sum: stream
                    .alloc_zeros::<u32>(dim)
                    .expect("alloc WHIR initial device-statement sum"),
                d_polys: initial_device_statement_polys,
            });
        let ff0 = self.folding_factor.at_round(0);
        let mut opening_workspace_specs = Vec::with_capacity(self.n_rounds() + 1);
        let mut current_opening_workspace_spec = GpuWhirOpeningWorkspaceSpec {
            n_samples: 0,
            height: self.starting_domain_size() >> ff0,
            row_width: 1usize << ff0,
        };
        let mut workspace_domain_size = self.starting_domain_size();
        let round_workspaces = (0..self.n_rounds())
            .map(|round_index| {
                let round_params = &self.round_parameters[round_index];
                opening_workspace_specs.push(GpuWhirOpeningWorkspaceSpec {
                    n_samples: round_params.num_queries,
                    ..current_opening_workspace_spec
                });
                let num_variables = self.num_variables - self.folding_factor.total_number(round_index);
                let ff_next = self.folding_factor.at_round(round_index + 1);
                let domain_reduction = 1 << self.rs_reduction_factor(round_index);
                let new_domain_size = workspace_domain_size / domain_reduction;
                let inv_rate = new_domain_size >> num_variables;
                let log_inv_rate = log2_strict_usize(inv_rate);
                let dft_n_cols = 1usize << ff_next;
                let dft_full_len = (1usize << num_variables) << log_inv_rate;
                let dft_height = dft_full_len / dft_n_cols;
                let dft_twiddles =
                    gpu_ntt::GpuNttTwiddles::upload(stream, log2_strict_usize(dft_height), dft_n_cols * dim);
                workspace_domain_size = new_domain_size;
                current_opening_workspace_spec = GpuWhirOpeningWorkspaceSpec {
                    n_samples: 0,
                    height: dft_height,
                    row_width: dft_n_cols * dim,
                };
                let n_constraint_scalars = round_params.ood_samples + round_params.num_queries;
                let ood_answer_state_count = num_variables.max(1);
                let mut ood_answer_len = 1usize << num_variables;
                let d_ood_answer_states = (0..ood_answer_state_count)
                    .map(|state_idx| {
                        let state_words = if round_params.ood_samples == 0 {
                            0
                        } else if num_variables == 0 {
                            round_params.ood_samples * dim
                        } else {
                            ood_answer_len /= 2;
                            round_params.ood_samples * ood_answer_len * dim
                        };
                        Some(
                            stream
                                .alloc_zeros::<u32>(state_words)
                                .unwrap_or_else(|_| panic!("alloc WHIR round OOD answer state {state_idx}")),
                        )
                    })
                    .collect();
                GpuWhirRoundWorkspaces {
                    d_ood_points: Some(
                        stream
                            .alloc_zeros::<u32>(round_params.ood_samples * dim)
                            .expect("alloc WHIR OOD point workspace"),
                    ),
                    d_ood_challenges: Some(
                        stream
                            .alloc_zeros::<u32>(round_params.ood_samples * num_variables * dim)
                            .expect("alloc WHIR OOD challenge workspace"),
                    ),
                    d_ood_answer_states,
                    d_query_pow_witness: Some(
                        stream
                            .alloc_zeros::<u32>(1)
                            .expect("alloc WHIR query PoW witness workspace"),
                    ),
                    d_query_pow_flag: Some(
                        stream
                            .alloc_zeros::<u32>(1)
                            .expect("alloc WHIR query PoW flag workspace"),
                    ),
                    d_stir_sample_words: Some(
                        stream
                            .alloc_zeros::<u32>(round_params.num_queries)
                            .expect("alloc WHIR STIR sample word workspace"),
                    ),
                    d_stir_challenges: Some(
                        stream
                            .alloc_zeros::<u32>(round_params.num_queries * num_variables * dim)
                            .expect("alloc WHIR STIR challenge workspace"),
                    ),
                    d_stir_indices: Some(
                        stream
                            .alloc_zeros::<u32>(round_params.num_queries)
                            .expect("alloc WHIR STIR index workspace"),
                    ),
                    d_stir_evaluations: Some(
                        stream
                            .alloc_zeros::<u32>(round_params.num_queries * dim)
                            .expect("alloc WHIR STIR evaluation workspace"),
                    ),
                    d_gamma_comb: Some(
                        stream
                            .alloc_zeros::<u32>(dim)
                            .expect("alloc WHIR gamma-combination workspace"),
                    ),
                    d_constraint_scalars: Some(
                        stream
                            .alloc_zeros::<u32>(n_constraint_scalars * dim)
                            .expect("alloc WHIR constraint scalar workspace"),
                    ),
                    d_ood_constraint_sum: Some(
                        stream
                            .alloc_zeros::<u32>(dim)
                            .expect("alloc WHIR OOD constraint sum workspace"),
                    ),
                    d_stir_constraint_sum: Some(
                        stream
                            .alloc_zeros::<u32>(dim)
                            .expect("alloc WHIR STIR constraint sum workspace"),
                    ),
                    dft_twiddles: Some(dft_twiddles),
                    d_dft_output: Some(
                        stream
                            .alloc_zeros::<u32>(dft_full_len * dim)
                            .expect("alloc WHIR round DFT output workspace"),
                    ),
                    merkle_layers: Some(gpu_merkle::GpuMerkle::allocate_tree_layers(stream, dft_height as u32)),
                    sumcheck: Some(GpuWhirCapturedSumcheckWorkspaces::allocate(
                        stream,
                        1usize << num_variables,
                        self.folding_factor.at_round(round_index + 1),
                        round_params.folding_pow_bits,
                        dim,
                    )),
                }
            })
            .collect();
        opening_workspace_specs.push(GpuWhirOpeningWorkspaceSpec {
            n_samples: self.final_queries,
            ..current_opening_workspace_spec
        });
        let final_openings = opening_workspace_specs
            .into_iter()
            .map(|spec| GpuWhirOpeningWorkspaces::allocate(stream, spec))
            .collect();
        let final_num_variables = self.num_variables - self.folding_factor.total_number(self.n_rounds());
        GpuWhirProverWorkspaces {
            d_weights: Some(
                stream
                    .alloc_zeros::<u32>(n_total * dim)
                    .expect("alloc WHIR weights workspace"),
            ),
            d_initial_gamma: Some(
                stream
                    .alloc_zeros::<u32>(dim)
                    .expect("alloc WHIR initial gamma workspace"),
            ),
            d_initial_scalars: Some(
                stream
                    .alloc_zeros::<u32>(self.commitment_ood_samples * dim)
                    .expect("alloc WHIR initial scalar workspace"),
            ),
            d_zero_sum: Some(stream.alloc_zeros::<u32>(dim).expect("alloc WHIR zero-sum workspace")),
            d_initial_ood_sum: Some(
                stream
                    .alloc_zeros::<u32>(dim)
                    .expect("alloc WHIR initial OOD sum workspace"),
            ),
            initial_device_statement_accumulation,
            d_randomness_words: Some(
                stream
                    .alloc_zeros::<u32>(self.num_variables * dim)
                    .expect("alloc WHIR randomness workspace"),
            ),
            initial_sumcheck: Some(GpuWhirCapturedSumcheckWorkspaces::allocate(
                stream,
                n_total,
                self.folding_factor.at_round(0),
                self.starting_folding_pow_bits,
                dim,
            )),
            round_workspaces,
            final_queries: Some(GpuWhirFinalQueryWorkspaces {
                d_query_pow_witness: stream
                    .alloc_zeros::<u32>(1)
                    .expect("alloc WHIR final query PoW witness workspace"),
                d_query_pow_flag: stream
                    .alloc_zeros::<u32>(1)
                    .expect("alloc WHIR final query PoW flag workspace"),
                d_sample_words: stream
                    .alloc_zeros::<u32>(self.final_queries)
                    .expect("alloc WHIR final query sample workspace"),
                d_query_points: stream
                    .alloc_zeros::<u32>(0)
                    .expect("alloc WHIR final query point workspace"),
                d_indices: stream
                    .alloc_zeros::<u32>(self.final_queries)
                    .expect("alloc WHIR final query index workspace"),
            }),
            final_sumcheck: (self.final_sumcheck_rounds > 0).then(|| {
                GpuWhirCapturedSumcheckWorkspaces::allocate(
                    stream,
                    1usize << final_num_variables,
                    self.final_sumcheck_rounds,
                    0,
                    dim,
                )
            }),
            d_final_materialization: Some(
                stream
                    .alloc_zeros::<u32>(final_materialization_word_capacity)
                    .expect("alloc WHIR final materialization workspace"),
            ),
            final_openings: Some(final_openings),
        }
    }
}

impl GpuMerkleProverData {
    pub fn new_base(
        d_leaf_matrix: CudaSlice<u32>,
        tree: gpu_merkle::DeviceMerkleTree,
        height: usize,
        full_leaf_base_width: usize,
    ) -> Self {
        Self::new_base_with_guards(d_leaf_matrix, tree, height, full_leaf_base_width, Vec::new())
    }

    pub fn new_base_with_guards(
        d_leaf_matrix: CudaSlice<u32>,
        tree: gpu_merkle::DeviceMerkleTree,
        height: usize,
        full_leaf_base_width: usize,
        dft_guards: Vec<CudaSlice<u32>>,
    ) -> Self {
        Self {
            d_leaf_matrix,
            d_digest_layers: tree.layers,
            _dft_guards: dft_guards,
            height,
            full_leaf_base_width,
            leaf_kind: GpuMerkleLeafKind::Base,
        }
    }

    fn new_extension_with_guards(
        d_leaf_matrix: CudaSlice<u32>,
        tree: gpu_merkle::DeviceMerkleTree,
        height: usize,
        full_leaf_base_width: usize,
        dft_guards: Vec<CudaSlice<u32>>,
    ) -> Self {
        Self {
            d_leaf_matrix,
            d_digest_layers: tree.layers,
            _dft_guards: dft_guards,
            height,
            full_leaf_base_width,
            leaf_kind: GpuMerkleLeafKind::Extension,
        }
    }

    fn read_answer<EF: ExtensionField<PF<EF>>>(&self, g: &gpu_backend::GpuBackend, index: usize) -> MleOwned<EF> {
        let row_start = index * self.full_leaf_base_width;
        let row_end = row_start + self.full_leaf_base_width;
        let row_words = g
            .stream
            .memcpy_dtov(&self.d_leaf_matrix.slice(row_start..row_end))
            .expect("download merkle leaf row");

        match self.leaf_kind {
            GpuMerkleLeafKind::Base => {
                let leaf: Vec<PF<EF>> = unsafe { std::mem::transmute::<Vec<u32>, Vec<PF<EF>>>(row_words) };
                MleOwned::Base(leaf)
            }
            GpuMerkleLeafKind::Extension => MleOwned::Extension(reconstitute_ext::<EF>(&row_words)),
        }
    }

    fn read_sibling_hashes<EF: ExtensionField<PF<EF>>>(
        &self,
        g: &gpu_backend::GpuBackend,
        index: usize,
    ) -> Vec<[PF<EF>; DIGEST_ELEMS]> {
        let log_height = log2_ceil_usize(self.height);
        (0..log_height)
            .map(|level| {
                let sibling_index = (index >> level) ^ 1;
                let layer = &self.d_digest_layers[level];
                let start = sibling_index * DIGEST_ELEMS;
                let end = start + DIGEST_ELEMS;
                let words = g
                    .stream
                    .memcpy_dtov(&layer.slice(start..end))
                    .expect("download merkle sibling digest");
                u32_slice_to_root::<EF>(&words)
            })
            .collect()
    }

    fn open<EF: ExtensionField<PF<EF>>>(
        &self,
        g: &gpu_backend::GpuBackend,
        index: usize,
    ) -> (MleOwned<EF>, Vec<[PF<EF>; DIGEST_ELEMS]>) {
        (
            self.read_answer::<EF>(g, index),
            self.read_sibling_hashes::<EF>(g, index),
        )
    }

    fn open_batch<EF: ExtensionField<PF<EF>>>(
        &self,
        g: &gpu_backend::GpuBackend,
        d_indices: &CudaSlice<u32>,
        n_samples: usize,
    ) -> Vec<(usize, MleOwned<EF>, Vec<[PF<EF>; DIGEST_ELEMS]>)> {
        if n_samples == 0 {
            return Vec::new();
        }

        let row_words_len = n_samples * self.full_leaf_base_width;
        let sibling_words_per_level = n_samples * DIGEST_ELEMS;
        let log_height = log2_ceil_usize(self.height);
        let total_words = n_samples + row_words_len + log_height * sibling_words_per_level;
        let mut d_opening_words = g
            .stream
            .alloc_zeros::<u32>(total_words)
            .expect("allocate merkle opening download buffer");
        g.sumcheck
            .memcpy_d2d_async(d_indices, 0, &mut d_opening_words, 0, n_samples);

        let d_row_words = g.merkle.gather_rows_device_async(
            &self.d_leaf_matrix,
            d_indices,
            n_samples as u32,
            self.full_leaf_base_width as u32,
        );
        g.sumcheck
            .memcpy_d2d_async(&d_row_words, 0, &mut d_opening_words, n_samples, row_words_len);

        let mut d_sibling_layers = Vec::with_capacity(log_height);
        for level in 0..log_height {
            let d_siblings = g.merkle.gather_sibling_hashes_device_async(
                &self.d_digest_layers[level],
                d_indices,
                n_samples as u32,
                level as u32,
            );
            let dst_offset = n_samples + row_words_len + level * sibling_words_per_level;
            g.sumcheck.memcpy_d2d_async(
                &d_siblings,
                0,
                &mut d_opening_words,
                dst_offset,
                sibling_words_per_level,
            );
            d_sibling_layers.push(d_siblings);
        }

        let opening_words = g
            .stream
            .memcpy_dtov(&d_opening_words)
            .expect("download batched merkle opening data");
        let (index_words, rest) = opening_words.split_at(n_samples);
        let (row_words, sibling_words) = rest.split_at(row_words_len);

        let mut sibling_by_sample = vec![Vec::<[PF<EF>; DIGEST_ELEMS]>::with_capacity(log_height); n_samples];
        for level in 0..log_height {
            let level_start = level * sibling_words_per_level;
            let level_end = level_start + sibling_words_per_level;
            for (sample_idx, chunk) in sibling_words[level_start..level_end]
                .chunks_exact(DIGEST_ELEMS)
                .enumerate()
            {
                sibling_by_sample[sample_idx].push(u32_slice_to_root::<EF>(chunk));
            }
        }

        let _guards = (d_row_words, d_sibling_layers);
        index_words
            .iter()
            .copied()
            .map(|idx| idx as usize)
            .zip(row_words.chunks_exact(self.full_leaf_base_width).zip(sibling_by_sample))
            .map(|(index, (row, sibling_hashes))| {
                let answer = match self.leaf_kind {
                    GpuMerkleLeafKind::Base => {
                        let leaf: Vec<PF<EF>> = unsafe { std::mem::transmute::<Vec<u32>, Vec<PF<EF>>>(row.to_vec()) };
                        MleOwned::Base(leaf)
                    }
                    GpuMerkleLeafKind::Extension => MleOwned::Extension(reconstitute_ext::<EF>(row)),
                };
                (index, answer, sibling_hashes)
            })
            .collect()
    }

    fn eval_at_randomness<EF: ExtensionField<PF<EF>>>(
        &self,
        g: &gpu_backend::GpuBackend,
        index: usize,
        point: &MultilinearPoint<EF>,
    ) -> EF {
        let row_start = index * self.full_leaf_base_width;
        let row_end = row_start + self.full_leaf_base_width;
        match self.leaf_kind {
            GpuMerkleLeafKind::Base => {
                assert_eq!(self.full_leaf_base_width, 1usize << point.0.len());
                if point.0.is_empty() {
                    let row = g
                        .stream
                        .memcpy_dtov(&self.d_leaf_matrix.slice(row_start..row_start + 1))
                        .expect("download constant base leaf");
                    return EF::from(unsafe { std::mem::transmute_copy::<u32, PF<EF>>(&row[0]) });
                }
                let mut current = g
                    .stream
                    .clone_dtod(&self.d_leaf_matrix.slice(row_start..row_end))
                    .expect("clone base leaf row");
                let mut current_len = self.full_leaf_base_width;
                current =
                    g.fold
                        .fold_base_to_ext_device(&current, (current_len / 2) as u32, &ef_to_u32::<EF>(&point.0[0]));
                current_len /= 2;
                for coord in &point.0[1..] {
                    current = g
                        .fold
                        .fold_ext_device(&current, (current_len / 2) as u32, &ef_to_u32::<EF>(coord));
                    current_len /= 2;
                }
                let out = g.stream.memcpy_dtov(&current).expect("download folded base leaf eval");
                ef_from_u32::<EF>(&out[..5].try_into().unwrap())
            }
            GpuMerkleLeafKind::Extension => {
                let n_ext = self.full_leaf_base_width / EF::DIMENSION;
                assert_eq!(n_ext, 1usize << point.0.len());
                if point.0.is_empty() {
                    let row = g
                        .stream
                        .memcpy_dtov(&self.d_leaf_matrix.slice(row_start..row_start + EF::DIMENSION))
                        .expect("download constant ext leaf");
                    return ef_from_u32::<EF>(&row[..5].try_into().unwrap());
                }
                let mut current = g
                    .stream
                    .clone_dtod(&self.d_leaf_matrix.slice(row_start..row_end))
                    .expect("clone ext leaf row");
                let mut current_len = n_ext;
                for coord in &point.0 {
                    current = g
                        .fold
                        .fold_ext_device(&current, (current_len / 2) as u32, &ef_to_u32::<EF>(coord));
                    current_len /= 2;
                }
                let out = g.stream.memcpy_dtov(&current).expect("download folded ext leaf eval");
                ef_from_u32::<EF>(&out[..5].try_into().unwrap())
            }
        }
    }

    fn eval_at_randomness_device<EF: ExtensionField<PF<EF>>>(
        &self,
        g: &gpu_backend::GpuBackend,
        index: usize,
        d_point_words: &CudaSlice<u32>,
        n_coords: usize,
    ) -> EF {
        let row_start = index * self.full_leaf_base_width;
        let row_end = row_start + self.full_leaf_base_width;
        match self.leaf_kind {
            GpuMerkleLeafKind::Base => {
                assert_eq!(self.full_leaf_base_width, 1usize << n_coords);
                if n_coords == 0 {
                    let row = g
                        .stream
                        .memcpy_dtov(&self.d_leaf_matrix.slice(row_start..row_start + 1))
                        .expect("download constant base leaf");
                    return EF::from(unsafe { std::mem::transmute_copy::<u32, PF<EF>>(&row[0]) });
                }
                let row = self.d_leaf_matrix.slice(row_start..row_end);
                let mut current_len = self.full_leaf_base_width;
                let d_first_coord = d_point_words.slice(0..EF::DIMENSION);
                let mut current =
                    g.fold
                        .fold_base_to_ext_device_with_challenge(&row, (current_len / 2) as u32, &d_first_coord);
                current_len /= 2;
                for coord_idx in 1..n_coords {
                    let start = coord_idx * EF::DIMENSION;
                    let end = start + EF::DIMENSION;
                    let d_coord = d_point_words.slice(start..end);
                    current = g
                        .fold
                        .fold_ext_device_with_challenge(&current, (current_len / 2) as u32, &d_coord);
                    current_len /= 2;
                }
                let out = g.stream.memcpy_dtov(&current).expect("download folded base leaf eval");
                ef_from_u32::<EF>(&out[..5].try_into().unwrap())
            }
            GpuMerkleLeafKind::Extension => {
                let n_ext = self.full_leaf_base_width / EF::DIMENSION;
                assert_eq!(n_ext, 1usize << n_coords);
                if n_coords == 0 {
                    let row = g
                        .stream
                        .memcpy_dtov(&self.d_leaf_matrix.slice(row_start..row_start + EF::DIMENSION))
                        .expect("download constant ext leaf");
                    return ef_from_u32::<EF>(&row[..5].try_into().unwrap());
                }
                let row = self.d_leaf_matrix.slice(row_start..row_end);
                let mut current_len = n_ext;
                let first_coord = d_point_words.slice(0..EF::DIMENSION);
                let mut current = g
                    .fold
                    .fold_ext_device_with_challenge(&row, (current_len / 2) as u32, &first_coord);
                current_len /= 2;
                for coord_idx in 1..n_coords {
                    let start = coord_idx * EF::DIMENSION;
                    let end = start + EF::DIMENSION;
                    let d_coord = d_point_words.slice(start..end);
                    current = g
                        .fold
                        .fold_ext_device_with_challenge(&current, (current_len / 2) as u32, &d_coord);
                    current_len /= 2;
                }
                let out = g.stream.memcpy_dtov(&current).expect("download folded ext leaf eval");
                ef_from_u32::<EF>(&out[..5].try_into().unwrap())
            }
        }
    }

    fn eval_rows_at_randomness_device<P>(
        &self,
        g: &gpu_backend::GpuBackend,
        d_indices: &CudaSlice<u32>,
        n_samples: usize,
        d_point_words: &P,
        n_coords: usize,
    ) -> CudaSlice<u32>
    where
        P: DevicePtr<u32>,
    {
        g.merkle.eval_rows_at_randomness_device_async(
            &self.d_leaf_matrix,
            d_indices,
            n_samples as u32,
            self.full_leaf_base_width as u32,
            d_point_words,
            n_coords as u32,
            matches!(self.leaf_kind, GpuMerkleLeafKind::Extension),
        )
    }

    fn eval_rows_at_randomness_device_into<P>(
        &self,
        g: &gpu_backend::GpuBackend,
        d_indices: &CudaSlice<u32>,
        n_samples: usize,
        d_point_words: &P,
        n_coords: usize,
        d_out: &mut CudaSlice<u32>,
    ) where
        P: DevicePtr<u32>,
    {
        g.merkle.eval_rows_at_randomness_device_into_async(
            &self.d_leaf_matrix,
            d_indices,
            n_samples as u32,
            self.full_leaf_base_width as u32,
            d_point_words,
            n_coords as u32,
            matches!(self.leaf_kind, GpuMerkleLeafKind::Extension),
            d_out,
        );
    }
}

enum InitialCommitmentOod<EF: ExtensionField<PF<EF>>> {
    Host { points: Vec<EF>, answers: Vec<EF> },
    Device(GpuInitialOodData),
}

enum RoundMerkleProverData<EF: ExtensionField<PF<EF>>> {
    Host(MerkleData<EF>),
    Device(GpuMerkleProverData),
}

impl<EF: ExtensionField<PF<EF>>> RoundMerkleProverData<EF> {
    fn eval_at_randomness(&self, g: &gpu_backend::GpuBackend, index: usize, point: &MultilinearPoint<EF>) -> EF {
        match self {
            Self::Host(data) => data.open(index).0.evaluate(point),
            Self::Device(data) => data.eval_at_randomness::<EF>(g, index, point),
        }
    }

    fn open(&self, g: &gpu_backend::GpuBackend, index: usize) -> (MleOwned<EF>, Vec<[PF<EF>; DIGEST_ELEMS]>) {
        match self {
            Self::Host(data) => data.open(index),
            Self::Device(data) => data.open::<EF>(g, index),
        }
    }

    fn open_batch(
        &self,
        g: &gpu_backend::GpuBackend,
        indices: PendingQueryIndices,
    ) -> Vec<(usize, MleOwned<EF>, Vec<[PF<EF>; DIGEST_ELEMS]>)> {
        match (self, indices) {
            (Self::Host(data), PendingQueryIndices::Host(indices)) => indices
                .into_iter()
                .map(|idx| {
                    let (answer, sibling_hashes) = data.open(idx);
                    (idx, answer, sibling_hashes)
                })
                .collect(),
            (
                Self::Host(data),
                PendingQueryIndices::RawDeviceIndices {
                    d_indices,
                    _d_sample_words: _,
                },
            ) => {
                let indices = g
                    .stream
                    .memcpy_dtov(&d_indices)
                    .expect("download host merkle query indices");
                indices
                    .into_iter()
                    .map(|idx| {
                        let idx = idx as usize;
                        let (answer, sibling_hashes) = data.open(idx);
                        (idx, answer, sibling_hashes)
                    })
                    .collect()
            }
            (Self::Device(data), PendingQueryIndices::Host(indices)) => indices
                .into_iter()
                .map(|idx| {
                    let (answer, sibling_hashes) = data.open::<EF>(g, idx);
                    (idx, answer, sibling_hashes)
                })
                .collect(),
            (
                Self::Device(data),
                PendingQueryIndices::RawDeviceIndices {
                    d_indices,
                    _d_sample_words: _,
                },
            ) => data.open_batch::<EF>(g, &d_indices, d_indices.len()),
        }
    }

    fn eval_at_randomness_device(
        &self,
        g: &gpu_backend::GpuBackend,
        index: usize,
        d_point_words: &CudaSlice<u32>,
        n_coords: usize,
    ) -> EF {
        match self {
            Self::Host(data) => {
                let point_words = g
                    .stream
                    .memcpy_dtov(&d_point_words.slice(0..n_coords * EF::DIMENSION))
                    .expect("download host merkle randomness");
                let point = MultilinearPoint(
                    point_words
                        .chunks_exact(EF::DIMENSION)
                        .map(|chunk| ef_from_u32::<EF>(chunk.try_into().unwrap()))
                        .collect(),
                );
                data.open(index).0.evaluate(&point)
            }
            Self::Device(data) => data.eval_at_randomness_device::<EF>(g, index, d_point_words, n_coords),
        }
    }

    fn eval_rows_at_randomness_device(
        &self,
        g: &gpu_backend::GpuBackend,
        d_indices: &CudaSlice<u32>,
        n_samples: usize,
        d_point_words: &CudaSlice<u32>,
        n_coords: usize,
    ) -> CudaSlice<u32> {
        match self {
            Self::Host(data) => {
                let indices = g
                    .stream
                    .memcpy_dtov(d_indices)
                    .expect("download host merkle query indices");
                let mut out = Vec::<u32>::with_capacity(n_samples * EF::DIMENSION);
                for idx in indices.into_iter().take(n_samples) {
                    let value = self.eval_at_randomness_device(g, idx as usize, d_point_words, n_coords);
                    out.extend_from_slice(&ef_to_u32::<EF>(&value));
                }
                g.stream.memcpy_stod(&out).expect("upload host merkle batch evals")
            }
            Self::Device(data) => data.eval_rows_at_randomness_device(g, d_indices, n_samples, d_point_words, n_coords),
        }
    }
}

struct PendingMerkleQueries<EF: ExtensionField<PF<EF>>> {
    merkle_data: RoundMerkleProverData<EF>,
    indices: PendingQueryIndices,
}

struct GpuFinalQueryOutput {
    indices: PendingQueryIndices,
    guards: GpuFinalQueryGuards,
}

struct GpuFinalQueryGuards {
    _d_query_points: Option<CudaSlice<u32>>,
    _d_query_pow_flag: Option<CudaSlice<u32>>,
}

struct DeviceMerkleOpeningPlan {
    merkle_data: GpuMerkleProverData,
    d_indices: CudaSlice<u32>,
    _d_sample_words: Option<CudaSlice<u32>>,
    n_samples: usize,
    offset: usize,
    total_words: usize,
    row_words_len: usize,
    sibling_words_per_level: usize,
    log_height: usize,
    _d_row_words: Option<CudaSlice<u32>>,
    _d_sibling_layers: Vec<CudaSlice<u32>>,
}

enum PendingQueryIndices {
    Host(Vec<usize>),
    RawDeviceIndices {
        d_indices: CudaSlice<u32>,
        _d_sample_words: Option<CudaSlice<u32>>,
    },
}

struct GpuMleEvalExtManyOutput {
    d_values: CudaSlice<u32>,
    _intermediates: Vec<CudaSlice<u32>>,
}

struct GpuRoundOodData {
    d_challenges: CudaSlice<u32>,
    d_answers: GpuDeviceSlice,
    _answer_intermediates: Vec<CudaSlice<u32>>,
    _d_univariate_points: CudaSlice<u32>,
}

struct GpuInitialConstraintGuard {
    _initial_ood: GpuInitialOodData,
    _d_initial_scalars: CudaSlice<u32>,
    _d_zero_sum: CudaSlice<u32>,
}

struct GpuExtDotAccumulationGuard {
    _d_base_sum: CudaSlice<u32>,
    _d_values: Option<CudaSlice<u32>>,
}

struct GpuRoundConstraintGuard {
    _d_gamma_comb: CudaSlice<u32>,
    _d_constraint_scalars: CudaSlice<u32>,
    _round_ood: Option<GpuRoundOodData>,
    _d_stir_challenges: CudaSlice<u32>,
    _d_query_pow_flag: Option<CudaSlice<u32>>,
    _d_folding_randomness: Option<CudaSlice<u32>>,
    _d_stir_evaluations: Option<CudaSlice<u32>>,
    _dot_guards: Vec<GpuExtDotAccumulationGuard>,
}

fn kb_u32<F: PrimeCharacteristicRing>(v: F) -> u32 {
    unsafe { std::mem::transmute_copy(&v) }
}

fn kb_from_u32<F: PrimeCharacteristicRing>(v: u32) -> F {
    unsafe { std::mem::transmute_copy(&v) }
}

struct GpuTranscript<'a, EF: ExtensionField<PF<EF>>> {
    d_challenger_state: CudaSlice<u32>,
    d_p16_rc: &'a CudaSlice<u32>,
    d_p16_mds: &'a CudaSlice<u32>,
    d_p16_sparse: &'a CudaSlice<u32>,
    d_empty_observe: CudaSlice<u32>,
    transcript_chunks: Vec<GpuTranscriptChunk>,
    _marker: std::marker::PhantomData<EF>,
}

impl<'a, EF> GpuTranscript<'a, EF>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: PrimeField64,
{
    fn new(g: &'a gpu_backend::GpuBackend, prover_state: &impl FSProver<EF>) -> Self {
        let (d_p16_rc, d_p16_mds, d_p16_sparse) = g.p16.as_slices();
        let state_words = prover_state.gpu_challenger_state().map(kb_u32);
        let d_challenger_state = g.stream.memcpy_stod(&state_words).expect("upload challenger state");
        let d_empty_observe = g
            .stream
            .alloc_zeros::<u32>(1)
            .expect("allocate empty transcript observe");
        Self {
            d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            d_empty_observe,
            transcript_chunks: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    fn from_seed(g: &'a gpu_backend::GpuBackend, seed: GpuTranscriptSeed) -> Self {
        let (d_p16_rc, d_p16_mds, d_p16_sparse) = g.p16.as_slices();
        Self {
            d_challenger_state: seed.d_challenger_state,
            d_p16_rc,
            d_p16_mds,
            d_p16_sparse,
            d_empty_observe: seed.d_empty_observe,
            transcript_chunks: seed.transcript_chunks,
            _marker: std::marker::PhantomData,
        }
    }

    fn observe_base_scalars_device<D>(&mut self, g: &gpu_backend::GpuBackend, d_scalars: D, n_words: usize)
    where
        D: Into<GpuDeviceSlice>,
    {
        if n_words == 0 {
            return;
        }
        let d_scalars = d_scalars.into();
        g.sumcheck.challenger_observe_device_scalars_async(
            &mut self.d_challenger_state,
            &self.d_p16_rc,
            &self.d_p16_mds,
            &self.d_p16_sparse,
            &d_scalars,
            n_words as u32,
        );
        self.transcript_chunks.push(GpuTranscriptChunk {
            d_words: d_scalars,
            n_words,
        });
    }

    fn observe_extension_scalars_device<D>(&mut self, g: &gpu_backend::GpuBackend, d_scalars: D, n_ext_scalars: usize)
    where
        D: Into<GpuDeviceSlice>,
    {
        self.observe_base_scalars_device(g, d_scalars, n_ext_scalars * EF::DIMENSION);
    }

    fn sample_ext(&mut self, g: &gpu_backend::GpuBackend) -> EF {
        ef_from_u32::<EF>(
            &g.sumcheck.challenger_observe_and_sample_exts(
                &mut self.d_challenger_state,
                &self.d_p16_rc,
                &self.d_p16_mds,
                &self.d_p16_sparse,
                &[],
                1,
            )[0],
        )
    }

    fn sample_ext_device_into(&mut self, g: &gpu_backend::GpuBackend, d_sample: &mut CudaSlice<u32>) {
        self.sample_ext_vec_device_into(g, 1, d_sample);
    }

    fn sample_ext_vec_device_into(&mut self, g: &gpu_backend::GpuBackend, len: usize, d_samples: &mut CudaSlice<u32>) {
        assert_eq!(
            d_samples.len(),
            len * EF::DIMENSION,
            "WHIR transcript sample workspace has wrong length"
        );
        g.sumcheck.challenger_sample_exts_device_into_async(
            &mut self.d_challenger_state,
            self.d_p16_rc,
            self.d_p16_mds,
            self.d_p16_sparse,
            len as u32,
            &self.d_empty_observe,
            d_samples,
        );
    }

    fn sample_in_range_device(&mut self, g: &gpu_backend::GpuBackend, n_samples: usize) -> CudaSlice<u32> {
        g.sumcheck.challenger_sample_base_scalars_device_async(
            &mut self.d_challenger_state,
            &self.d_p16_rc,
            &self.d_p16_mds,
            &self.d_p16_sparse,
            n_samples as u32,
        )
    }

    fn sample_in_range_device_into(
        &mut self,
        g: &gpu_backend::GpuBackend,
        n_samples: usize,
        d_samples: &mut CudaSlice<u32>,
    ) {
        g.sumcheck.challenger_sample_base_scalars_device_into_async(
            &mut self.d_challenger_state,
            &self.d_p16_rc,
            &self.d_p16_mds,
            &self.d_p16_sparse,
            n_samples as u32,
            d_samples,
        );
    }

    fn pow_grinding(&mut self, g: &gpu_backend::GpuBackend, bits: usize) {
        assert!(bits < PF::<EF>::bits());
        if bits == 0 {
            return;
        }

        let d_witness = g
            .pow
            .grind_from_device_state_device(&self.d_challenger_state, 8, 8, bits as u32, 1 << 28)
            .expect("gpu pow grind");
        g.sumcheck.challenger_observe_device_scalars_async(
            &mut self.d_challenger_state,
            &self.d_p16_rc,
            &self.d_p16_mds,
            &self.d_p16_sparse,
            &d_witness,
            1,
        );
        self.transcript_chunks.push(GpuTranscriptChunk {
            d_words: d_witness.into(),
            n_words: 1,
        });
    }

    fn pow_grinding_device_into(
        &mut self,
        g: &gpu_backend::GpuBackend,
        bits: usize,
        mut d_witness: CudaSlice<u32>,
        mut d_flag: CudaSlice<u32>,
    ) -> Option<CudaSlice<u32>> {
        assert!(bits < PF::<EF>::bits());
        if bits == 0 {
            return Some(d_flag);
        }
        assert!(d_witness.len() >= 1);
        assert!(d_flag.len() >= 1);
        g.pow.grind_from_device_state_device_async(
            &self.d_challenger_state,
            8,
            8,
            bits as u32,
            1 << 28,
            &mut d_witness,
            &mut d_flag,
        );
        g.sumcheck.challenger_observe_device_scalars_async(
            &mut self.d_challenger_state,
            &self.d_p16_rc,
            &self.d_p16_mds,
            &self.d_p16_sparse,
            &d_witness,
            1,
        );
        self.transcript_chunks.push(GpuTranscriptChunk {
            d_words: d_witness.into(),
            n_words: 1,
        });
        Some(d_flag)
    }

    fn finish(self, g: &gpu_backend::GpuBackend, prover_state: &mut impl FSProver<EF>) {
        let total_words: usize = self.transcript_chunks.iter().map(|chunk| chunk.n_words).sum();
        let mut d_final = g
            .stream
            .alloc_zeros::<u32>(total_words + 8)
            .expect("allocate final transcript buffer");
        let mut offset = 0usize;
        for chunk in &self.transcript_chunks {
            g.sumcheck
                .memcpy_d2d_async(&chunk.d_words, 0, &mut d_final, offset, chunk.n_words);
            offset += chunk.n_words;
        }
        g.sumcheck
            .memcpy_d2d_async(&self.d_challenger_state, 0, &mut d_final, total_words, 8);
        let final_words = g.stream.memcpy_dtov(&d_final).expect("download final transcript state");
        let transcript_scalars = final_words[..total_words]
            .iter()
            .copied()
            .map(kb_from_u32::<PF<EF>>)
            .collect::<Vec<_>>();
        let challenger_state_words = &final_words[total_words..total_words + 8];
        let final_challenger_state = [
            kb_from_u32::<PF<EF>>(challenger_state_words[0]),
            kb_from_u32::<PF<EF>>(challenger_state_words[1]),
            kb_from_u32::<PF<EF>>(challenger_state_words[2]),
            kb_from_u32::<PF<EF>>(challenger_state_words[3]),
            kb_from_u32::<PF<EF>>(challenger_state_words[4]),
            kb_from_u32::<PF<EF>>(challenger_state_words[5]),
            kb_from_u32::<PF<EF>>(challenger_state_words[6]),
            kb_from_u32::<PF<EF>>(challenger_state_words[7]),
        ];
        prover_state.inject_gpu_transcript_state(&transcript_scalars, final_challenger_state);
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_sumcheck_round_challenge<EF>(
    g: &gpu_backend::GpuBackend,
    gpu_fs: &mut GpuTranscript<'_, EF>,
    d_sum: &mut CudaSlice<u32>,
    round: usize,
    pow_bits: usize,
    d_c0_out: &CudaSlice<u32>,
    d_c2_out: &CudaSlice<u32>,
    d_poly: &mut CudaSlice<u32>,
    d_round_tail: &mut CudaSlice<u32>,
    d_round_r: &mut CudaSlice<u32>,
    d_round_challenges: &mut CudaSlice<u32>,
    d_pow_witnesses: &mut [CudaSlice<u32>],
    d_pow_flags: &mut [CudaSlice<u32>],
    d_empty_observe: &CudaSlice<u32>,
) where
    EF: ExtensionField<PF<EF>>,
{
    let dim = EF::DIMENSION;
    g.graph_sumcheck.product_sumcheck_observe_round_poly_async(
        d_c0_out,
        d_c2_out,
        d_sum,
        d_poly,
        d_round_tail,
        &mut gpu_fs.d_challenger_state,
        &gpu_fs.d_p16_rc,
        &gpu_fs.d_p16_mds,
        &gpu_fs.d_p16_sparse,
    );
    if pow_bits > 0 {
        g.graph_pow.grind_from_device_state_device_async(
            &gpu_fs.d_challenger_state,
            8,
            8,
            pow_bits as u32,
            1u64 << 28,
            &mut d_pow_witnesses[round],
            &mut d_pow_flags[round],
        );
        g.graph_sumcheck.challenger_observe_device_scalars_async(
            &mut gpu_fs.d_challenger_state,
            &gpu_fs.d_p16_rc,
            &gpu_fs.d_p16_mds,
            &gpu_fs.d_p16_sparse,
            &d_pow_witnesses[round],
            1,
        );
    }
    g.graph_sumcheck.challenger_sample_exts_device_into_async(
        &mut gpu_fs.d_challenger_state,
        &gpu_fs.d_p16_rc,
        &gpu_fs.d_p16_mds,
        &gpu_fs.d_p16_sparse,
        1,
        d_empty_observe,
        d_round_r,
    );
    g.graph_sumcheck
        .product_sumcheck_update_sum_async(d_poly, d_round_r, d_sum);
    g.graph_sumcheck
        .memcpy_d2d_async(d_round_r, 0, d_round_challenges, round * dim, dim);
}

#[allow(clippy::too_many_arguments)]
fn run_captured_ext_sumcheck_rounds<EF>(
    g: &gpu_backend::GpuBackend,
    gpu_fs: &mut GpuTranscript<'_, EF>,
    d_evals: &mut CudaSlice<u32>,
    d_weights: &mut CudaSlice<u32>,
    d_sum: &mut CudaSlice<u32>,
    n_elements: &mut usize,
    n_rounds: usize,
    pow_bits: usize,
    sumcheck_workspace: Option<GpuWhirCapturedSumcheckWorkspaces>,
) -> Option<CudaSlice<u32>>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: PrimeField64,
{
    let dim = EF::DIMENSION;
    let mut half_lens = Vec::with_capacity(n_rounds);
    let mut current_len = *n_elements;
    for _ in 0..n_rounds {
        let half = current_len / 2;
        half_lens.push(half as u32);
        current_len = half;
    }

    g.graph_stream.join(&g.stream).ok()?;

    let GpuWhirCapturedSumcheckWorkspaces {
        mut eval_states,
        mut weight_states,
        mut d_round_challenges,
        mut d_round_rs,
        mut d_round_tails,
        mut d_pow_witnesses,
        mut d_pow_flags,
        mut d_c0_partials,
        mut d_c2_partials,
        mut d_c0_out,
        mut d_c2_out,
        mut d_poly,
        d_empty_observe,
    } = sumcheck_workspace.unwrap_or_else(|| {
        GpuWhirCapturedSumcheckWorkspaces::allocate(&g.graph_stream, *n_elements, n_rounds, pow_bits, dim)
    });

    if n_rounds == 0 {
        return Some(d_round_challenges);
    }

    let stream = g.graph_stream.clone();
    if let Err(e) = stream.begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED) {
        tracing::warn!("WHIR ext sumcheck capture unavailable; failing GPU proof: {e:?}");
        return None;
    }

    for round in 0..n_rounds {
        let half = half_lens[round];
        if round == 0 {
            let d_eval_next = &mut eval_states[0];
            let d_weight_next = &mut weight_states[0];
            g.graph_sumcheck.product_sumcheck_ext_ext_device_resident_into_async(
                d_evals,
                d_weights,
                half,
                &mut d_c0_partials,
                &mut d_c2_partials,
                &mut d_c0_out,
                &mut d_c2_out,
            );
            capture_sumcheck_round_challenge::<EF>(
                g,
                gpu_fs,
                d_sum,
                round,
                pow_bits,
                &d_c0_out,
                &d_c2_out,
                &mut d_poly,
                &mut d_round_tails[round],
                &mut d_round_rs[round],
                &mut d_round_challenges,
                &mut d_pow_witnesses,
                &mut d_pow_flags,
                &d_empty_observe,
            );
            g.graph_fold
                .fold_ext_device_with_challenge_into_async(d_evals, half, &d_round_rs[round], d_eval_next);
            g.graph_fold
                .fold_ext_device_with_challenge_into_async(d_weights, half, &d_round_rs[round], d_weight_next);
        } else {
            let (eval_done, eval_rest) = eval_states.split_at_mut(round);
            let (weight_done, weight_rest) = weight_states.split_at_mut(round);
            let d_eval_cur = &eval_done[round - 1];
            let d_eval_next = &mut eval_rest[0];
            let d_weight_cur = &weight_done[round - 1];
            let d_weight_next = &mut weight_rest[0];
            g.graph_sumcheck.product_sumcheck_ext_ext_device_resident_into_async(
                d_eval_cur,
                d_weight_cur,
                half,
                &mut d_c0_partials,
                &mut d_c2_partials,
                &mut d_c0_out,
                &mut d_c2_out,
            );
            capture_sumcheck_round_challenge::<EF>(
                g,
                gpu_fs,
                d_sum,
                round,
                pow_bits,
                &d_c0_out,
                &d_c2_out,
                &mut d_poly,
                &mut d_round_tails[round],
                &mut d_round_rs[round],
                &mut d_round_challenges,
                &mut d_pow_witnesses,
                &mut d_pow_flags,
                &d_empty_observe,
            );
            g.graph_fold
                .fold_ext_device_with_challenge_into_async(d_eval_cur, half, &d_round_rs[round], d_eval_next);
            g.graph_fold.fold_ext_device_with_challenge_into_async(
                d_weight_cur,
                half,
                &d_round_rs[round],
                d_weight_next,
            );
        }
    }

    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = match stream.end_capture(graph_flags) {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            tracing::warn!("WHIR ext sumcheck capture produced no graph; failing GPU proof");
            return None;
        }
        Err(e) => {
            tracing::warn!("WHIR ext sumcheck end_capture failed; failing GPU proof: {e:?}");
            return None;
        }
    };
    if let Err(e) = graph.launch() {
        tracing::warn!("WHIR ext sumcheck graph launch failed; failing GPU proof: {e:?}");
        return None;
    }
    if let Err(e) = g.stream.join(&stream) {
        tracing::warn!("WHIR ext sumcheck graph stream join failed; failing GPU proof: {e:?}");
        return None;
    }

    if pow_bits > 0 {
        for (tail, witness) in d_round_tails.into_iter().zip(d_pow_witnesses.into_iter()) {
            gpu_fs.transcript_chunks.push(GpuTranscriptChunk {
                d_words: tail.into(),
                n_words: 10,
            });
            gpu_fs.transcript_chunks.push(GpuTranscriptChunk {
                d_words: witness.into(),
                n_words: 1,
            });
        }
    } else {
        for tail in d_round_tails {
            gpu_fs.transcript_chunks.push(GpuTranscriptChunk {
                d_words: tail.into(),
                n_words: 10,
            });
        }
    }

    *d_evals = eval_states.pop().unwrap();
    *d_weights = weight_states.pop().unwrap();
    *n_elements = current_len;

    Some(d_round_challenges)
}

#[allow(clippy::too_many_arguments)]
fn run_captured_initial_sumcheck_rounds<EF>(
    g: &gpu_backend::GpuBackend,
    gpu_fs: &mut GpuTranscript<'_, EF>,
    d_evals: &mut CudaSlice<u32>,
    d_weights: &mut CudaSlice<u32>,
    d_sum: &mut CudaSlice<u32>,
    n_elements: &mut usize,
    n_rounds: usize,
    pow_bits: usize,
    sumcheck_workspace: Option<GpuWhirCapturedSumcheckWorkspaces>,
) -> Option<CudaSlice<u32>>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: PrimeField64,
{
    let dim = EF::DIMENSION;
    let mut half_lens = Vec::with_capacity(n_rounds);
    let mut current_len = *n_elements;
    for _ in 0..n_rounds {
        let half = current_len / 2;
        half_lens.push(half as u32);
        current_len = half;
    }

    g.graph_stream.join(&g.stream).ok()?;

    let GpuWhirCapturedSumcheckWorkspaces {
        mut eval_states,
        mut weight_states,
        mut d_round_challenges,
        mut d_round_rs,
        mut d_round_tails,
        mut d_pow_witnesses,
        mut d_pow_flags,
        mut d_c0_partials,
        mut d_c2_partials,
        mut d_c0_out,
        mut d_c2_out,
        mut d_poly,
        d_empty_observe,
    } = sumcheck_workspace.unwrap_or_else(|| {
        GpuWhirCapturedSumcheckWorkspaces::allocate(&g.graph_stream, *n_elements, n_rounds, pow_bits, dim)
    });

    if n_rounds == 0 {
        return Some(d_round_challenges);
    }

    let stream = g.graph_stream.clone();
    if let Err(e) = stream.begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED) {
        tracing::warn!("WHIR initial sumcheck capture unavailable; failing GPU proof: {e:?}");
        return None;
    }

    for round in 0..n_rounds {
        let half = half_lens[round];
        if round == 0 {
            let d_eval_next = &mut eval_states[0];
            let d_weight_next = &mut weight_states[0];
            g.graph_sumcheck.product_sumcheck_base_ext_device_resident_into_async(
                d_evals,
                d_weights,
                half,
                &mut d_c0_partials,
                &mut d_c2_partials,
                &mut d_c0_out,
                &mut d_c2_out,
            );
            capture_sumcheck_round_challenge::<EF>(
                g,
                gpu_fs,
                d_sum,
                round,
                pow_bits,
                &d_c0_out,
                &d_c2_out,
                &mut d_poly,
                &mut d_round_tails[round],
                &mut d_round_rs[round],
                &mut d_round_challenges,
                &mut d_pow_witnesses,
                &mut d_pow_flags,
                &d_empty_observe,
            );
            g.graph_fold.fold_base_to_ext_device_with_challenge_into_async(
                d_evals,
                half,
                &d_round_rs[round],
                d_eval_next,
            );
            g.graph_fold
                .fold_ext_device_with_challenge_into_async(d_weights, half, &d_round_rs[round], d_weight_next);
        } else {
            let (eval_done, eval_rest) = eval_states.split_at_mut(round);
            let (weight_done, weight_rest) = weight_states.split_at_mut(round);
            let d_eval_cur = &eval_done[round - 1];
            let d_eval_next = &mut eval_rest[0];
            let d_weight_cur = &weight_done[round - 1];
            let d_weight_next = &mut weight_rest[0];
            g.graph_sumcheck.product_sumcheck_ext_ext_device_resident_into_async(
                d_eval_cur,
                d_weight_cur,
                half,
                &mut d_c0_partials,
                &mut d_c2_partials,
                &mut d_c0_out,
                &mut d_c2_out,
            );
            capture_sumcheck_round_challenge::<EF>(
                g,
                gpu_fs,
                d_sum,
                round,
                pow_bits,
                &d_c0_out,
                &d_c2_out,
                &mut d_poly,
                &mut d_round_tails[round],
                &mut d_round_rs[round],
                &mut d_round_challenges,
                &mut d_pow_witnesses,
                &mut d_pow_flags,
                &d_empty_observe,
            );
            g.graph_fold
                .fold_ext_device_with_challenge_into_async(d_eval_cur, half, &d_round_rs[round], d_eval_next);
            g.graph_fold.fold_ext_device_with_challenge_into_async(
                d_weight_cur,
                half,
                &d_round_rs[round],
                d_weight_next,
            );
        }
    }

    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = match stream.end_capture(graph_flags) {
        Ok(Some(graph)) => graph,
        Ok(None) => {
            tracing::warn!("WHIR initial sumcheck capture produced no graph; failing GPU proof");
            return None;
        }
        Err(e) => {
            tracing::warn!("WHIR initial sumcheck end_capture failed; failing GPU proof: {e:?}");
            return None;
        }
    };
    if let Err(e) = graph.launch() {
        tracing::warn!("WHIR initial sumcheck graph launch failed; failing GPU proof: {e:?}");
        return None;
    }
    if let Err(e) = g.stream.join(&stream) {
        tracing::warn!("WHIR initial sumcheck graph stream join failed; failing GPU proof: {e:?}");
        return None;
    }

    if pow_bits > 0 {
        for (tail, witness) in d_round_tails.into_iter().zip(d_pow_witnesses.into_iter()) {
            gpu_fs.transcript_chunks.push(GpuTranscriptChunk {
                d_words: tail.into(),
                n_words: 10,
            });
            gpu_fs.transcript_chunks.push(GpuTranscriptChunk {
                d_words: witness.into(),
                n_words: 1,
            });
        }
    } else {
        for tail in d_round_tails {
            gpu_fs.transcript_chunks.push(GpuTranscriptChunk {
                d_words: tail.into(),
                n_words: 10,
            });
        }
    }

    *d_evals = eval_states.pop().unwrap();
    *d_weights = weight_states.pop().unwrap();
    *n_elements = current_len;

    Some(d_round_challenges)
}

impl<EF> WhirConfig<EF>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: TwoAdicField + PrimeField64,
{
    /// Complete GPU-accelerated WHIR prove.
    ///
    /// Reimplements `WhirConfig::prove` with GPU-resident `d_evals` and `d_weights`.
    /// Product sumcheck, polynomial folding, DFT, and Merkle tree are all on GPU.
    /// Legacy/public entrypoints may still materialize host transcript/opening
    /// data; the continued-transcript leanVM path keeps Fiat-Shamir sampling on
    /// device and uses combined final WHIR materialization.
    ///
    /// Returns `None` if GPU is unavailable or input is not base-field polynomial.
    #[instrument(name = "GPU WHIR prove", skip_all)]
    pub fn gpu_prove(
        &self,
        prover_state: &mut impl FSProver<EF>,
        statement: Vec<SparseStatement<EF>>,
        witness: Witness<EF>,
        polynomial: &MleRef<'_, EF>,
    ) -> Option<MultilinearPoint<EF>> {
        if std::mem::size_of::<PF<EF>>() != 4 || EF::DIMENSION != 5 {
            return None; // Only KoalaBear+quintic for now
        }
        let g = gpu_backend::gpu()?;

        assert!(self.validate_parameters());
        assert!(self.validate_witness(&witness, polynomial));
        self.validate_statement(&statement);

        // Upload base-field polynomial to GPU.
        let base_evals: &[PF<EF>] = match polynomial {
            MleRef::Base(e) => e,
            _ => return None, // GPU prove requires unpacked base-field input
        };
        let evals_u32: &[u32] = unsafe { std::slice::from_raw_parts(base_evals.as_ptr().cast(), base_evals.len()) };
        let d_evals = g.stream.memcpy_stod(evals_u32).ok()?;
        self.gpu_prove_from_uploaded_base(
            prover_state,
            statement,
            RoundMerkleProverData::Host(witness.prover_data),
            InitialCommitmentOod::Host {
                points: witness.ood_points,
                answers: witness.ood_answers,
            },
            d_evals,
            base_evals.len(),
            None,
            Vec::new(),
            true,
            None,
        )
    }

    /// Complete GPU-accelerated WHIR prove starting from an already-uploaded base polynomial.
    ///
    /// This is the device-native entrypoint used by the end-to-end GPU prover to
    /// avoid rebuilding a host `MleRef` and re-uploading the stacked polynomial.
    #[instrument(name = "GPU WHIR prove (device base polynomial)", skip_all)]
    pub fn gpu_prove_from_device_base(
        &self,
        prover_state: &mut impl FSProver<EF>,
        statement: Vec<SparseStatement<EF>>,
        witness: Witness<EF>,
        d_evals: CudaSlice<u32>,
        n_base_evals: usize,
    ) -> Option<MultilinearPoint<EF>> {
        if std::mem::size_of::<PF<EF>>() != 4 || EF::DIMENSION != 5 {
            return None; // Only KoalaBear+quintic for now
        }
        let _ = gpu_backend::gpu()?;

        assert!(self.validate_parameters());
        assert_eq!(witness.ood_points.len(), witness.ood_answers.len());
        assert_eq!(n_base_evals, 1usize << self.num_variables);
        self.validate_statement(&statement);

        self.gpu_prove_from_uploaded_base(
            prover_state,
            statement,
            RoundMerkleProverData::Host(witness.prover_data),
            InitialCommitmentOod::Host {
                points: witness.ood_points,
                answers: witness.ood_answers,
            },
            d_evals,
            n_base_evals,
            None,
            Vec::new(),
            true,
            None,
        )
    }

    /// Device-native WHIR prove starting from an uploaded base polynomial and a
    /// device-resident initial commitment Merkle tree.
    #[instrument(name = "GPU WHIR prove (device base polynomial + GPU commitment)", skip_all)]
    pub fn gpu_prove_from_device_base_with_gpu_commitment(
        &self,
        prover_state: &mut impl FSProver<EF>,
        statement: Vec<SparseStatement<EF>>,
        initial_prover_data: GpuMerkleProverData,
        initial_ood: GpuInitialOodData,
        d_evals: CudaSlice<u32>,
        n_base_evals: usize,
    ) -> Option<MultilinearPoint<EF>> {
        if std::mem::size_of::<PF<EF>>() != 4 || EF::DIMENSION != 5 {
            return None; // Only KoalaBear+quintic for now
        }
        let _ = gpu_backend::gpu()?;

        assert!(self.validate_parameters());
        assert_eq!(
            initial_ood.d_points.len(),
            initial_ood.n_samples * self.num_variables * EF::DIMENSION
        );
        assert_eq!(initial_ood.d_answers.len(), initial_ood.n_samples * EF::DIMENSION);
        assert_eq!(n_base_evals, 1usize << self.num_variables);
        self.validate_statement(&statement);

        self.gpu_prove_from_uploaded_base(
            prover_state,
            statement,
            RoundMerkleProverData::Device(initial_prover_data),
            InitialCommitmentOod::Device(initial_ood),
            d_evals,
            n_base_evals,
            None,
            Vec::new(),
            true,
            None,
        )
    }

    /// Device-native WHIR prove that continues an existing device-resident
    /// transcript instead of downloading/re-uploading challenger state at the
    /// WHIR boundary.
    #[instrument(name = "GPU WHIR prove (continued device transcript)", skip_all)]
    pub fn gpu_prove_from_device_base_with_gpu_commitment_and_transcript(
        &self,
        prover_state: &mut impl FSProver<EF>,
        initial_prover_data: GpuMerkleProverData,
        initial_ood: GpuInitialOodData,
        d_evals: CudaSlice<u32>,
        n_base_evals: usize,
        transcript_seed: GpuTranscriptSeed,
        device_statement_prefix: Vec<GpuSparseStatement>,
        whir_workspaces: GpuWhirProverWorkspaces,
    ) -> Option<MultilinearPoint<EF>> {
        if std::mem::size_of::<PF<EF>>() != 4 || EF::DIMENSION != 5 {
            return None; // Only KoalaBear+quintic for now
        }
        let _ = gpu_backend::gpu()?;

        assert!(self.validate_parameters());
        assert_eq!(
            initial_ood.d_points.len(),
            initial_ood.n_samples * self.num_variables * EF::DIMENSION
        );
        assert_eq!(initial_ood.d_answers.len(), initial_ood.n_samples * EF::DIMENSION);
        assert_eq!(n_base_evals, 1usize << self.num_variables);

        self.gpu_prove_from_uploaded_base(
            prover_state,
            Vec::new(),
            RoundMerkleProverData::Device(initial_prover_data),
            InitialCommitmentOod::Device(initial_ood),
            d_evals,
            n_base_evals,
            Some(transcript_seed),
            device_statement_prefix,
            false,
            Some(whir_workspaces),
        )
    }

    fn gpu_prove_from_uploaded_base(
        &self,
        prover_state: &mut impl FSProver<EF>,
        statement: Vec<SparseStatement<EF>>,
        initial_prover_data: RoundMerkleProverData<EF>,
        initial_ood: InitialCommitmentOod<EF>,
        mut d_evals: CudaSlice<u32>,
        mut n_elements: usize,
        transcript_seed: Option<GpuTranscriptSeed>,
        device_statement_prefix: Vec<GpuSparseStatement>,
        download_randomness: bool,
        mut whir_workspaces: Option<GpuWhirProverWorkspaces>,
    ) -> Option<MultilinearPoint<EF>> {
        let g = gpu_backend::gpu()?;
        let dim = EF::DIMENSION; // 5
        let require_device_only_path = transcript_seed.is_some() && !download_randomness;
        if require_device_only_path && whir_workspaces.is_none() {
            return None;
        }
        let mut gpu_fs = match transcript_seed {
            Some(seed) => GpuTranscript::from_seed(g, seed),
            None => GpuTranscript::new(g, prover_state),
        };
        let mut pending_merkle_queries = Vec::<PendingMerkleQueries<EF>>::new();
        let mut initial_constraint_guards = Vec::<GpuInitialConstraintGuard>::new();
        let mut device_statement_guards = Vec::<GpuDeviceStatementAccumulationGuard>::new();
        let mut round_constraint_guards = Vec::<GpuRoundConstraintGuard>::new();
        let mut final_query_guards = Vec::<GpuFinalQueryGuards>::new();

        // ═══════════════════════════════════════════════════════════════
        // INITIALIZE: combine statements + initial product sumcheck
        // ═══════════════════════════════════════════════════════════════

        // Build weights on GPU (no upload — built directly on device).
        let (mut d_weights, mut d_sum) = match initial_ood {
            InitialCommitmentOod::Host { points, answers } => {
                if require_device_only_path {
                    return None;
                }
                let gamma = gpu_fs.sample_ext(g);
                let mut all_statements: Vec<SparseStatement<EF>> = points
                    .iter()
                    .zip(&answers)
                    .map(|(&point, &evaluation)| {
                        SparseStatement::dense(
                            MultilinearPoint::expand_from_univariate(point, self.num_variables),
                            evaluation,
                        )
                    })
                    .collect();
                all_statements.extend(statement.iter().cloned());
                let (d_weights, sum) = gpu_combine_statement(&all_statements, gamma, self.num_variables)?;
                let d_sum = g.stream.memcpy_stod(&ef_to_u32::<EF>(&sum)).ok()?;
                (d_weights, d_sum)
            }
            InitialCommitmentOod::Device(initial_ood) => {
                let n_total = 1usize << self.num_variables;
                let mut d_weights = if let Some(workspaces) = whir_workspaces.as_mut() {
                    take_whir_workspace(&mut workspaces.d_weights, "initial weights")
                } else {
                    g.stream.alloc_zeros::<u32>(n_total * dim).ok()?
                };
                let mut d_gamma = if let Some(workspaces) = whir_workspaces.as_mut() {
                    take_whir_workspace(&mut workspaces.d_initial_gamma, "initial gamma")
                } else {
                    g.stream.alloc_zeros::<u32>(dim).ok()?
                };
                let mut d_initial_scalars = if let Some(workspaces) = whir_workspaces.as_mut() {
                    take_whir_workspace(&mut workspaces.d_initial_scalars, "initial gamma powers")
                } else {
                    g.stream.alloc_zeros::<u32>(initial_ood.n_samples * dim).ok()?
                };
                let d_zero_sum = if let Some(workspaces) = whir_workspaces.as_mut() {
                    take_whir_workspace(&mut workspaces.d_zero_sum, "initial zero sum")
                } else {
                    g.stream.alloc_zeros::<u32>(EF::DIMENSION).ok()?
                };
                let mut d_initial_ood_sum = if let Some(workspaces) = whir_workspaces.as_mut() {
                    take_whir_workspace(&mut workspaces.d_initial_ood_sum, "initial OOD sum")
                } else {
                    g.stream.alloc_zeros::<u32>(EF::DIMENSION).ok()?
                };
                let initial_constraint_capture_stream = whir_workspaces.as_ref().map(|_| g.sumcheck.stream().clone());
                if let Some(stream) = &initial_constraint_capture_stream {
                    stream
                        .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
                        .expect("begin WHIR initial constraint graph capture");
                }
                gpu_fs.sample_ext_device_into(g, &mut d_gamma);
                g.sumcheck.extension_powers_device_into_async(
                    &d_gamma,
                    initial_ood.n_samples as u32,
                    &mut d_initial_scalars,
                );
                g.sumcheck.dense_eq_accumulate_from_points_device_async(
                    &mut d_weights,
                    &initial_ood.d_points,
                    &d_initial_scalars,
                    initial_ood.n_samples as u32,
                    self.num_variables as u32,
                    n_total as u32,
                );
                g.sumcheck.ext_dot_accumulate_into_async(
                    &d_zero_sum,
                    &initial_ood.d_answers,
                    &d_initial_scalars,
                    initial_ood.n_samples as u32,
                    &mut d_initial_ood_sum,
                );
                if let Some(stream) = &initial_constraint_capture_stream {
                    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                    let graph = stream
                        .end_capture(graph_flags)
                        .expect("end WHIR initial constraint graph capture")
                        .expect("WHIR initial constraint capture produced no graph");
                    graph.launch().expect("launch WHIR initial constraint graph");
                }
                let mut gamma_start_power = initial_ood.n_samples;
                let d_initial_sum = d_initial_ood_sum;
                if statement.is_empty() {
                    let initial_device_statement_value_count = device_statement_prefix
                        .iter()
                        .map(|statement| statement.values.len())
                        .sum::<usize>();
                    let initial_device_statement_workspaces = whir_workspaces
                        .as_mut()
                        .and_then(|workspaces| workspaces.initial_device_statement_accumulation.take());
                    if require_device_only_path
                        && initial_device_statement_value_count > 0
                        && initial_device_statement_workspaces.is_none()
                    {
                        return None;
                    }
                    let capture_initial_device_statements =
                        initial_device_statement_workspaces.is_some() && initial_device_statement_value_count > 0;
                    let initial_device_statement_capture_stream =
                        capture_initial_device_statements.then(|| g.sumcheck.stream().clone());
                    if let Some(stream) = &initial_device_statement_capture_stream {
                        stream
                            .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
                            .expect("begin WHIR initial device-statement graph capture");
                    }
                    let (d_sum, device_statement_guard) =
                        gpu_accumulate_device_statements_with_device_gamma_into_async::<EF>(
                            &mut d_weights,
                            d_initial_sum,
                            d_gamma,
                            gamma_start_power,
                            &device_statement_prefix,
                            self.num_variables,
                            initial_device_statement_workspaces,
                        )?;
                    if let Some(stream) = &initial_device_statement_capture_stream {
                        let graph_flags =
                            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                        let graph = stream
                            .end_capture(graph_flags)
                            .expect("end WHIR initial device-statement graph capture")
                            .expect("WHIR initial device-statement capture produced no graph");
                        graph.launch().expect("launch WHIR initial device-statement graph");
                    }
                    initial_constraint_guards.push(GpuInitialConstraintGuard {
                        _initial_ood: initial_ood,
                        _d_initial_scalars: d_initial_scalars,
                        _d_zero_sum: d_zero_sum,
                    });
                    device_statement_guards.push(device_statement_guard);
                    (d_weights, d_sum)
                } else {
                    if require_device_only_path {
                        return None;
                    }
                    let d_after_device_sum = gpu_accumulate_device_statements_with_device_gamma::<EF>(
                        &mut d_weights,
                        &d_initial_sum,
                        &d_gamma,
                        gamma_start_power,
                        &device_statement_prefix,
                        self.num_variables,
                    )?;
                    gamma_start_power += device_statement_prefix
                        .iter()
                        .map(|statement| statement.values.len())
                        .sum::<usize>();
                    let d_sum = gpu_accumulate_statement_with_device_gamma(
                        &mut d_weights,
                        &d_after_device_sum,
                        &d_gamma,
                        gamma_start_power,
                        &statement,
                        self.num_variables,
                    )?;
                    (d_weights, d_sum)
                }
            }
        };

        // Run initial product sumcheck rounds on GPU.
        let ff0 = self.folding_factor.at_round(0);
        let pow0 = self.starting_folding_pow_bits;
        let mut d_randomness_words = if let Some(workspaces) = whir_workspaces.as_mut() {
            take_whir_workspace(&mut workspaces.d_randomness_words, "randomness words")
        } else {
            g.stream.alloc_zeros::<u32>(self.num_variables * dim).ok()?
        };
        let mut randomness_count = 0usize;

        tracing::info!(
            "GPU WHIR prove: n_vars={}, ff0={ff0}, n_rounds={}, resident={:.1}MB",
            self.num_variables,
            self.n_rounds(),
            (n_elements * 4) as f64 / 1e6,
        );

        let initial_sumcheck = whir_workspaces
            .as_mut()
            .map(|workspaces| take_whir_workspace(&mut workspaces.initial_sumcheck, "initial sumcheck"));
        let d_initial_challenges = run_captured_initial_sumcheck_rounds(
            g,
            &mut gpu_fs,
            &mut d_evals,
            &mut d_weights,
            &mut d_sum,
            &mut n_elements,
            ff0,
            pow0,
            initial_sumcheck,
        )?;
        if ff0 > 0 {
            g.sumcheck
                .memcpy_d2d_async(&d_initial_challenges, 0, &mut d_randomness_words, 0, ff0 * dim);
            randomness_count += ff0;
        }

        // Round state tracking.
        let mut domain_size = self.starting_domain_size();
        let mut next_domain_gen = PF::<EF>::two_adic_generator(log2_strict_usize(domain_size) - ff0);
        let mut merkle_prover_data = initial_prover_data;

        // ═══════════════════════════════════════════════════════════════
        // WHIR ROUNDS
        // ═══════════════════════════════════════════════════════════════

        for round_index in 0..=self.n_rounds() {
            let num_variables = self.num_variables - self.folding_factor.total_number(round_index);

            // ─── FINAL ROUND ───────────────────────────────────────────
            if round_index == self.n_rounds() {
                let final_sumcheck = if self.final_sumcheck_rounds > 0 {
                    whir_workspaces
                        .as_mut()
                        .map(|workspaces| take_whir_workspace(&mut workspaces.final_sumcheck, "final sumcheck"))
                } else {
                    None
                };
                let final_query_workspaces = whir_workspaces
                    .as_mut()
                    .map(|workspaces| take_whir_workspace(&mut workspaces.final_queries, "final query workspaces"));
                let final_output = self.gpu_final_round(
                    g,
                    &mut gpu_fs,
                    &mut d_evals,
                    &mut d_weights,
                    &mut d_sum,
                    &mut n_elements,
                    &mut d_randomness_words,
                    &mut randomness_count,
                    &merkle_prover_data,
                    domain_size,
                    round_index,
                    final_query_workspaces,
                    final_sumcheck,
                )?;
                pending_merkle_queries.push(PendingMerkleQueries {
                    merkle_data: merkle_prover_data,
                    indices: final_output.indices,
                });
                final_query_guards.push(final_output.guards);
                break;
            }

            // ─── REGULAR ROUND ─────────────────────────────────────────
            let round_params = &self.round_parameters[round_index];
            let ff_next = self.folding_factor.at_round(round_index + 1);
            let mut round_workspaces = whir_workspaces
                .as_mut()
                .map(|workspaces| &mut workspaces.round_workspaces[round_index]);

            // ── DFT: GPU reorder → GPU DFT → GPU Merkle ──
            let domain_reduction = 1 << self.rs_reduction_factor(round_index);
            let new_domain_size = domain_size / domain_reduction;
            let inv_rate = new_domain_size >> num_variables;
            let log_inv_rate = log2_strict_usize(inv_rate);
            let dft_n_cols = 1usize << ff_next;

            let d_dft_output = info_span!("GPU reorder+DFT").in_scope(|| {
                if let Some(workspaces) = round_workspaces.as_mut() {
                    let d_dft_output_workspace =
                        take_whir_workspace(&mut workspaces.d_dft_output, "WHIR round DFT output");
                    let dft_twiddles = workspaces
                        .dft_twiddles
                        .as_ref()
                        .expect("uploaded WHIR round NTT twiddles");
                    g.ntt.reorder_and_dft_ext_device_guarded_with_twiddles_into(
                        &d_evals,
                        n_elements as u32,
                        ff_next,
                        log_inv_rate,
                        dim,
                        dft_twiddles,
                        d_dft_output_workspace,
                    )
                } else {
                    g.ntt
                        .reorder_and_dft_ext_device_guarded(&d_evals, n_elements as u32, ff_next, log_inv_rate, dim)
                }
            });
            let (d_dft, d_dft_guards) = d_dft_output.into_parts();
            let dft_height = (n_elements << log_inv_rate) / dft_n_cols;
            let dft_width = dft_n_cols * dim;

            // GPU Merkle tree on DFT output.
            let (d_root, device_tree) = info_span!("GPU Merkle").in_scope(|| {
                if let Some(workspaces) = round_workspaces.as_mut() {
                    let merkle_layers = take_whir_workspace(&mut workspaces.merkle_layers, "WHIR round Merkle layers");
                    g.merkle.build_tree_from_device_resident_root_device_into(
                        &d_dft,
                        dft_height as u32,
                        dft_width as u32,
                        dft_width as u32,
                        merkle_layers,
                    )
                } else {
                    g.merkle.build_tree_from_device_resident_root_device(
                        &d_dft,
                        dft_height as u32,
                        dft_width as u32,
                        dft_width as u32,
                    )
                }
            });
            let new_merkle_data = RoundMerkleProverData::Device(GpuMerkleProverData::new_extension_with_guards(
                d_dft,
                device_tree,
                dft_height,
                dft_width,
                d_dft_guards,
            ));

            let round_ood_capture_stream = round_workspaces.as_ref().map(|_| g.sumcheck.stream().clone());
            if let Some(stream) = &round_ood_capture_stream {
                stream
                    .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .expect("begin WHIR round OOD graph capture");
            }

            // Send root to Fiat-Shamir.
            gpu_fs.observe_base_scalars_device(g, GpuDeviceSlice::shared(d_root, 0, DIGEST_ELEMS), DIGEST_ELEMS);

            // ── OOD evaluation ──
            let round_ood = if round_params.ood_samples > 0 {
                let mut d_ood_points = if let Some(workspaces) = round_workspaces.as_mut() {
                    take_whir_workspace(&mut workspaces.d_ood_points, "round OOD univariate points")
                } else {
                    g.stream.alloc_zeros::<u32>(round_params.ood_samples * dim).ok()?
                };
                gpu_fs.sample_ext_vec_device_into(g, round_params.ood_samples, &mut d_ood_points);
                let mut d_ood_challenges = if let Some(workspaces) = round_workspaces.as_mut() {
                    take_whir_workspace(&mut workspaces.d_ood_challenges, "round OOD expanded challenges")
                } else {
                    g.stream
                        .alloc_zeros::<u32>(round_params.ood_samples * num_variables * dim)
                        .ok()?
                };
                g.sumcheck.expand_univariate_points_device_into_async(
                    &d_ood_points,
                    round_params.ood_samples as u32,
                    num_variables as u32,
                    &mut d_ood_challenges,
                );
                let d_ood_answers = info_span!("ood evaluation").in_scope(|| {
                    let d_ood_answer_states = round_workspaces.as_mut().map(|workspaces| {
                        workspaces
                            .d_ood_answer_states
                            .iter_mut()
                            .enumerate()
                            .map(|(state_idx, slot)| {
                                take_whir_workspace(slot, &format!("round OOD answer state {state_idx}"))
                            })
                            .collect::<Vec<_>>()
                    });
                    match d_ood_answer_states {
                        Some(states) => gpu_mle_eval_ext_many_device_into::<EF>(
                            g,
                            &d_evals,
                            n_elements,
                            &d_ood_challenges,
                            round_params.ood_samples,
                            num_variables,
                            states,
                        ),
                        None => gpu_mle_eval_ext_many_device::<EF>(
                            g,
                            &d_evals,
                            n_elements,
                            &d_ood_challenges,
                            round_params.ood_samples,
                            num_variables,
                        ),
                    }
                });
                let d_ood_answer_values = Arc::new(d_ood_answers.d_values);
                gpu_fs.observe_extension_scalars_device(
                    g,
                    GpuDeviceSlice::shared(d_ood_answer_values.clone(), 0, round_params.ood_samples * dim),
                    round_params.ood_samples,
                );
                if let Some(stream) = &round_ood_capture_stream {
                    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                    let graph = stream
                        .end_capture(graph_flags)
                        .expect("end WHIR round OOD graph capture")
                        .expect("WHIR round OOD capture produced no graph");
                    graph.launch().expect("launch WHIR round OOD graph");
                }
                Some(GpuRoundOodData {
                    d_challenges: d_ood_challenges,
                    d_answers: GpuDeviceSlice::shared(d_ood_answer_values, 0, round_params.ood_samples * dim),
                    _answer_intermediates: d_ood_answers._intermediates,
                    _d_univariate_points: d_ood_points,
                })
            } else {
                if let Some(stream) = &round_ood_capture_stream {
                    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                    let graph = stream
                        .end_capture(graph_flags)
                        .expect("end WHIR round OOD graph capture")
                        .expect("WHIR round OOD capture produced no graph");
                    graph.launch().expect("launch WHIR round OOD graph");
                }
                None
            };

            // ── STIR queries ──
            let folded_domain_size = domain_size >> self.folding_factor.at_round(round_index);
            let ff_prev = self.folding_factor.at_round(round_index);
            let randomness_start = (randomness_count - ff_prev) * dim;
            let randomness_end = randomness_count * dim;
            let mut d_folding_randomness_guard = None;
            let mut d_query_pow_flag_guard = None;
            let strict_stir_query_workspaces = if matches!(merkle_prover_data, RoundMerkleProverData::Device(_)) {
                round_workspaces.as_mut().map(|workspaces| {
                    (
                        take_whir_workspace(&mut workspaces.d_query_pow_witness, "round query PoW witness"),
                        take_whir_workspace(&mut workspaces.d_query_pow_flag, "round query PoW flag"),
                        take_whir_workspace(&mut workspaces.d_stir_sample_words, "round STIR sample words"),
                        take_whir_workspace(&mut workspaces.d_stir_challenges, "round STIR challenges"),
                        take_whir_workspace(&mut workspaces.d_stir_indices, "round STIR indices"),
                        take_whir_workspace(&mut workspaces.d_stir_evaluations, "round STIR evaluations"),
                    )
                })
            } else {
                None
            };

            let (d_stir_sample_words, d_stir_challenges, d_stir_indices, d_stir_evaluations) = if let Some((
                d_query_pow_witness,
                d_query_pow_flag,
                mut d_stir_sample_words,
                mut d_stir_challenges,
                mut d_stir_indices,
                mut d_stir_evaluations,
            )) =
                strict_stir_query_workspaces
            {
                let RoundMerkleProverData::Device(data) = &merkle_prover_data else {
                    unreachable!("strict STIR query workspaces require device Merkle data");
                };
                let capture_stir_query_work = round_params.query_pow_bits > 0 || round_params.num_queries > 0;
                let stream = g.sumcheck.stream().clone();
                if capture_stir_query_work {
                    stream
                        .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
                        .expect("begin WHIR round STIR query graph capture");
                }
                d_query_pow_flag_guard = gpu_fs.pow_grinding_device_into(
                    g,
                    round_params.query_pow_bits,
                    d_query_pow_witness,
                    d_query_pow_flag,
                );
                gpu_fs.sample_in_range_device_into(g, round_params.num_queries, &mut d_stir_sample_words);
                g.sumcheck.expand_sampled_base_query_points_device_into_async(
                    &d_stir_sample_words,
                    round_params.num_queries as u32,
                    folded_domain_size.ilog2(),
                    kb_u32(next_domain_gen),
                    num_variables as u32,
                    &mut d_stir_challenges,
                    &mut d_stir_indices,
                );
                let d_folding_randomness = d_randomness_words.slice(randomness_start..randomness_end);
                data.eval_rows_at_randomness_device_into(
                    g,
                    &d_stir_indices,
                    round_params.num_queries,
                    &d_folding_randomness,
                    ff_prev,
                    &mut d_stir_evaluations,
                );
                if capture_stir_query_work {
                    let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                    let graph = stream
                        .end_capture(graph_flags)
                        .expect("end WHIR round STIR query graph capture")
                        .expect("WHIR round STIR query capture produced no graph");
                    graph.launch().expect("launch WHIR round STIR query graph");
                }
                (
                    d_stir_sample_words,
                    d_stir_challenges,
                    d_stir_indices,
                    d_stir_evaluations,
                )
            } else {
                gpu_fs.pow_grinding(g, round_params.query_pow_bits);
                let d_stir_sample_words = gpu_fs.sample_in_range_device(g, round_params.num_queries);
                let (d_stir_challenges, d_stir_indices) = g.sumcheck.expand_sampled_base_query_points_device_async(
                    &d_stir_sample_words,
                    round_params.num_queries as u32,
                    folded_domain_size.ilog2(),
                    kb_u32(next_domain_gen),
                    num_variables as u32,
                );

                // Evaluate opened leaves at folding randomness.
                let d_stir_evaluations = match &merkle_prover_data {
                    RoundMerkleProverData::Device(data) => {
                        let d_folding_randomness = d_randomness_words.slice(randomness_start..randomness_end);
                        data.eval_rows_at_randomness_device(
                            g,
                            &d_stir_indices,
                            round_params.num_queries,
                            &d_folding_randomness,
                            ff_prev,
                        )
                    }
                    RoundMerkleProverData::Host(_) => {
                        if require_device_only_path {
                            return None;
                        }
                        let d_folding_randomness = g
                            .stream
                            .clone_dtod(&d_randomness_words.slice(randomness_start..randomness_end))
                            .ok()?;
                        let d_stir_evaluations = merkle_prover_data.eval_rows_at_randomness_device(
                            g,
                            &d_stir_indices,
                            round_params.num_queries,
                            &d_folding_randomness,
                            ff_prev,
                        );
                        d_folding_randomness_guard = Some(d_folding_randomness);
                        d_stir_evaluations
                    }
                };
                (
                    d_stir_sample_words,
                    d_stir_challenges,
                    d_stir_indices,
                    d_stir_evaluations,
                )
            };

            // ── Add new eq constraints to weights ON GPU ──
            let mut d_gamma_comb = if let Some(workspaces) = round_workspaces.as_mut() {
                take_whir_workspace(&mut workspaces.d_gamma_comb, "round gamma combination")
            } else {
                g.stream.alloc_zeros::<u32>(dim).ok()?
            };
            let n_constraint_scalars = round_params.ood_samples + round_params.num_queries;
            let mut d_constraint_scalars = if let Some(workspaces) = round_workspaces.as_mut() {
                take_whir_workspace(&mut workspaces.d_constraint_scalars, "round constraint scalars")
            } else {
                g.stream.alloc_zeros::<u32>(n_constraint_scalars * dim).ok()?
            };
            let mut d_ood_constraint_sum_workspace = if let Some(workspaces) = round_workspaces.as_mut() {
                Some(take_whir_workspace(
                    &mut workspaces.d_ood_constraint_sum,
                    "round OOD constraint sum",
                ))
            } else {
                None
            };
            let mut d_stir_constraint_sum_workspace = if let Some(workspaces) = round_workspaces.as_mut() {
                Some(take_whir_workspace(
                    &mut workspaces.d_stir_constraint_sum,
                    "round STIR constraint sum",
                ))
            } else {
                None
            };
            let round_constraint_capture_stream = round_workspaces.as_ref().map(|_| g.sumcheck.stream().clone());
            if let Some(stream) = &round_constraint_capture_stream {
                stream
                    .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .expect("begin WHIR round constraint graph capture");
            }
            gpu_fs.sample_ext_device_into(g, &mut d_gamma_comb);
            g.sumcheck.extension_powers_device_into_async(
                &d_gamma_comb,
                n_constraint_scalars as u32,
                &mut d_constraint_scalars,
            );
            let mut d_constraint_sum = d_sum;
            let mut dot_guards = Vec::<GpuExtDotAccumulationGuard>::new();
            let mut round_ood_guard = None;

            if let Some(round_ood) = round_ood {
                let d_ood_scalars = d_constraint_scalars.slice(0..round_params.ood_samples * dim);
                g.sumcheck.dense_eq_accumulate_from_points_device_async(
                    &mut d_weights,
                    &round_ood.d_challenges,
                    &d_ood_scalars,
                    round_params.ood_samples as u32,
                    num_variables as u32,
                    n_elements as u32,
                );
                let d_base_sum = d_constraint_sum;
                let d_next_sum = if let Some(mut d_out) = d_ood_constraint_sum_workspace.take() {
                    g.sumcheck.ext_dot_accumulate_into_async(
                        &d_base_sum,
                        &round_ood.d_answers,
                        &d_ood_scalars,
                        round_params.ood_samples as u32,
                        &mut d_out,
                    );
                    d_out
                } else {
                    g.sumcheck.ext_dot_accumulate_device_async(
                        &d_base_sum,
                        &round_ood.d_answers,
                        &d_ood_scalars,
                        round_params.ood_samples as u32,
                    )
                };
                d_constraint_sum = d_next_sum;
                dot_guards.push(GpuExtDotAccumulationGuard {
                    _d_base_sum: d_base_sum,
                    _d_values: None,
                });
                round_ood_guard = Some(round_ood);
            }

            // STIR constraints.
            let mut d_stir_evaluations_guard = None;
            if round_params.num_queries > 0 {
                let stir_scalar_start = round_params.ood_samples * dim;
                let stir_scalar_end = stir_scalar_start + round_params.num_queries * dim;
                let d_stir_scalars = d_constraint_scalars.slice(stir_scalar_start..stir_scalar_end);
                g.sumcheck.dense_eq_accumulate_from_points_device_async(
                    &mut d_weights,
                    &d_stir_challenges,
                    &d_stir_scalars,
                    round_params.num_queries as u32,
                    num_variables as u32,
                    n_elements as u32,
                );
                let d_base_sum = d_constraint_sum;
                let d_next_sum = if let Some(mut d_out) = d_stir_constraint_sum_workspace.take() {
                    g.sumcheck.ext_dot_accumulate_into_async(
                        &d_base_sum,
                        &d_stir_evaluations,
                        &d_stir_scalars,
                        round_params.num_queries as u32,
                        &mut d_out,
                    );
                    d_out
                } else {
                    g.sumcheck.ext_dot_accumulate_device_async(
                        &d_base_sum,
                        &d_stir_evaluations,
                        &d_stir_scalars,
                        round_params.num_queries as u32,
                    )
                };
                d_constraint_sum = d_next_sum;
                dot_guards.push(GpuExtDotAccumulationGuard {
                    _d_base_sum: d_base_sum,
                    _d_values: Some(d_stir_evaluations),
                });
            } else {
                d_stir_evaluations_guard = Some(d_stir_evaluations);
            }
            if let Some(stream) = &round_constraint_capture_stream {
                let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                let graph = stream
                    .end_capture(graph_flags)
                    .expect("end WHIR round constraint graph capture")
                    .expect("WHIR round constraint capture produced no graph");
                graph.launch().expect("launch WHIR round constraint graph");
            }
            round_constraint_guards.push(GpuRoundConstraintGuard {
                _d_gamma_comb: d_gamma_comb,
                _d_constraint_scalars: d_constraint_scalars,
                _round_ood: round_ood_guard,
                _d_stir_challenges: d_stir_challenges,
                _d_query_pow_flag: d_query_pow_flag_guard,
                _d_folding_randomness: d_folding_randomness_guard,
                _d_stir_evaluations: d_stir_evaluations_guard,
                _dot_guards: dot_guards,
            });
            // ── Product sumcheck rounds ON GPU ──
            d_sum = d_constraint_sum;
            let round_sumcheck = round_workspaces
                .as_mut()
                .map(|workspaces| take_whir_workspace(&mut workspaces.sumcheck, "round sumcheck"));
            let d_round_challenges = run_captured_ext_sumcheck_rounds(
                g,
                &mut gpu_fs,
                &mut d_evals,
                &mut d_weights,
                &mut d_sum,
                &mut n_elements,
                ff_next,
                round_params.folding_pow_bits,
                round_sumcheck,
            )?;
            if ff_next > 0 {
                g.sumcheck.memcpy_d2d_async(
                    &d_round_challenges,
                    0,
                    &mut d_randomness_words,
                    randomness_count * dim,
                    ff_next * dim,
                );
                randomness_count += ff_next;
            }

            // Update round state.
            domain_size = new_domain_size;
            next_domain_gen = PF::<EF>::two_adic_generator(log2_strict_usize(new_domain_size) - ff_next);
            let previous_merkle_data = std::mem::replace(&mut merkle_prover_data, new_merkle_data);
            pending_merkle_queries.push(PendingMerkleQueries {
                merkle_data: previous_merkle_data,
                indices: PendingQueryIndices::RawDeviceIndices {
                    d_indices: d_stir_indices,
                    _d_sample_words: Some(d_stir_sample_words),
                },
            });
        }

        let final_materialization_workspace = whir_workspaces.as_mut().map(|workspaces| {
            take_whir_workspace(&mut workspaces.d_final_materialization, "WHIR final materialization")
        });
        let final_opening_workspaces = whir_workspaces
            .as_mut()
            .map(|workspaces| take_whir_workspace(&mut workspaces.final_openings, "WHIR final opening workspaces"));
        finish_whir_materialization(
            g,
            pending_merkle_queries,
            gpu_fs,
            prover_state,
            require_device_only_path,
            final_materialization_workspace,
            final_opening_workspaces,
        )?;
        if !download_randomness {
            return Some(MultilinearPoint(Vec::new()));
        }
        let randomness_words = g
            .stream
            .memcpy_dtov(&d_randomness_words.slice(0..randomness_count * dim))
            .ok()?;
        Some(MultilinearPoint(
            randomness_words
                .chunks_exact(dim)
                .map(|chunk| ef_from_u32::<EF>(chunk.try_into().unwrap()))
                .collect(),
        ))
    }

    /// GPU final round: send coefficients, grind, open final queries, optional sumcheck.
    #[allow(clippy::too_many_arguments)]
    fn gpu_final_round(
        &self,
        g: &gpu_backend::GpuBackend,
        gpu_fs: &mut GpuTranscript<'_, EF>,
        d_evals: &mut CudaSlice<u32>,
        d_weights: &mut CudaSlice<u32>,
        d_sum: &mut CudaSlice<u32>,
        n_elements: &mut usize,
        d_randomness_words: &mut CudaSlice<u32>,
        randomness_count: &mut usize,
        merkle_prover_data: &RoundMerkleProverData<EF>,
        domain_size: usize,
        round_index: usize,
        final_query_workspaces: Option<GpuWhirFinalQueryWorkspaces>,
        final_sumcheck: Option<GpuWhirCapturedSumcheckWorkspaces>,
    ) -> Option<GpuFinalQueryOutput> {
        // Coefficients are transcript/proof data, while final sumcheck still
        // consumes the evaluation buffer below.
        let d_coeffs = g.fold.evals_to_coeffs_ext_device_async(d_evals, *n_elements as u32);
        gpu_fs.observe_base_scalars_device(g, d_coeffs, *n_elements * EF::DIMENSION);

        let final_domain_size = domain_size >> self.folding_factor.at_round(round_index);
        let mut d_final_query_points_guard = None;
        let mut d_final_query_pow_flag_guard = None;
        let (final_index_words, final_indexes) = if let Some(GpuWhirFinalQueryWorkspaces {
            d_query_pow_witness,
            d_query_pow_flag,
            mut d_sample_words,
            mut d_query_points,
            mut d_indices,
        }) = final_query_workspaces
        {
            let capture_final_query_work = self.final_query_pow_bits > 0 || self.final_queries > 0;
            let stream = g.sumcheck.stream().clone();
            if capture_final_query_work {
                stream
                    .begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .expect("begin WHIR final query graph capture");
            }
            d_final_query_pow_flag_guard =
                gpu_fs.pow_grinding_device_into(g, self.final_query_pow_bits, d_query_pow_witness, d_query_pow_flag);
            gpu_fs.sample_in_range_device_into(g, self.final_queries, &mut d_sample_words);
            g.sumcheck.expand_sampled_base_query_points_device_into_async(
                &d_sample_words,
                self.final_queries as u32,
                final_domain_size.ilog2(),
                0,
                0,
                &mut d_query_points,
                &mut d_indices,
            );
            if capture_final_query_work {
                let graph_flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                let graph = stream
                    .end_capture(graph_flags)
                    .expect("end WHIR final query graph capture")
                    .expect("WHIR final query capture produced no graph");
                graph.launch().expect("launch WHIR final query graph");
            }
            d_final_query_points_guard = Some(d_query_points);
            (d_sample_words, d_indices)
        } else {
            gpu_fs.pow_grinding(g, self.final_query_pow_bits);
            let final_index_words = gpu_fs.sample_in_range_device(g, self.final_queries);
            let (_d_unused_points, final_indexes) = g.sumcheck.expand_sampled_base_query_points_device_async(
                &final_index_words,
                self.final_queries as u32,
                final_domain_size.ilog2(),
                0,
                0,
            );
            (final_index_words, final_indexes)
        };
        let _ = merkle_prover_data;

        // Final sumcheck rounds on GPU.
        if self.final_sumcheck_rounds > 0 {
            let dim = EF::DIMENSION;
            let d_round_challenges = run_captured_ext_sumcheck_rounds(
                g,
                gpu_fs,
                d_evals,
                d_weights,
                d_sum,
                n_elements,
                self.final_sumcheck_rounds,
                0,
                final_sumcheck,
            )?;
            g.sumcheck.memcpy_d2d_async(
                &d_round_challenges,
                0,
                d_randomness_words,
                *randomness_count * dim,
                self.final_sumcheck_rounds * dim,
            );
            *randomness_count += self.final_sumcheck_rounds;
        }

        Some(GpuFinalQueryOutput {
            indices: PendingQueryIndices::RawDeviceIndices {
                d_indices: final_indexes,
                _d_sample_words: Some(final_index_words),
            },
            guards: GpuFinalQueryGuards {
                _d_query_points: d_final_query_points_guard,
                _d_query_pow_flag: d_final_query_pow_flag_guard,
            },
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

fn gpu_mle_eval_ext_device<EF: ExtensionField<PF<EF>>>(
    g: &gpu_backend::GpuBackend,
    d_evals: &CudaSlice<u32>,
    n_elements: usize,
    d_point_words: &CudaSlice<u32>,
    n_coords: usize,
) -> CudaSlice<u32> {
    assert_eq!(n_elements, 1usize << n_coords);

    if n_coords == 0 {
        return g
            .stream
            .clone_dtod(&d_evals.slice(0..5))
            .expect("clone constant evaluation");
    }

    let d_first_coord = d_point_words.slice(0..EF::DIMENSION);
    let mut current_len = n_elements;
    let mut current = g
        .fold
        .fold_ext_device_with_challenge(d_evals, (current_len / 2) as u32, &d_first_coord);
    current_len /= 2;

    for coord_idx in 1..n_coords {
        let start = coord_idx * EF::DIMENSION;
        let end = start + EF::DIMENSION;
        let d_coord = d_point_words.slice(start..end);
        current = g
            .fold
            .fold_ext_device_with_challenge(&current, (current_len / 2) as u32, &d_coord);
        current_len /= 2;
    }

    current
}

fn gpu_mle_eval_ext_many_device<EF: ExtensionField<PF<EF>>>(
    g: &gpu_backend::GpuBackend,
    d_evals: &CudaSlice<u32>,
    n_elements: usize,
    d_points_words: &CudaSlice<u32>,
    n_points: usize,
    n_coords: usize,
) -> GpuMleEvalExtManyOutput {
    assert_eq!(n_elements, 1usize << n_coords);

    if n_points == 0 {
        return GpuMleEvalExtManyOutput {
            d_values: g.stream.alloc_zeros::<u32>(0).expect("allocate empty MLE batch"),
            _intermediates: Vec::new(),
        };
    }

    if n_coords == 0 {
        return GpuMleEvalExtManyOutput {
            d_values: g.sumcheck.repeat_ext_value_device_async(d_evals, n_points as u32),
            _intermediates: Vec::new(),
        };
    }

    let point_stride_words = (n_coords * EF::DIMENSION) as u32;
    let mut current_len = n_elements;
    let mut current = g.sumcheck.fold_single_ext_to_many_points_device_async(
        d_evals,
        d_points_words,
        point_stride_words,
        0,
        current_len as u32,
        (current_len / 2) as u32,
        n_points as u32,
    );
    current_len /= 2;

    let mut intermediates = Vec::with_capacity(n_coords.saturating_sub(1));
    for coord_idx in 1..n_coords {
        let previous = current;
        current = g.sumcheck.fold_multi_col_ext_per_point_device_async(
            &previous,
            d_points_words,
            point_stride_words,
            coord_idx as u32,
            current_len as u32,
            (current_len / 2) as u32,
            n_points as u32,
        );
        intermediates.push(previous);
        current_len /= 2;
    }

    GpuMleEvalExtManyOutput {
        d_values: current,
        _intermediates: intermediates,
    }
}

fn gpu_mle_eval_ext_many_device_into<EF: ExtensionField<PF<EF>>>(
    g: &gpu_backend::GpuBackend,
    d_evals: &CudaSlice<u32>,
    n_elements: usize,
    d_points_words: &CudaSlice<u32>,
    n_points: usize,
    n_coords: usize,
    states: Vec<CudaSlice<u32>>,
) -> GpuMleEvalExtManyOutput {
    assert_eq!(n_elements, 1usize << n_coords);
    assert_eq!(
        states.len(),
        n_coords.max(1),
        "WHIR OOD MLE answer state count must match coordinate count"
    );

    let mut states = states.into_iter();
    if n_points == 0 {
        return GpuMleEvalExtManyOutput {
            d_values: states.next().expect("missing empty OOD MLE output state"),
            _intermediates: Vec::new(),
        };
    }

    if n_coords == 0 {
        let mut d_out = states.next().expect("missing constant OOD MLE output state");
        g.sumcheck
            .repeat_ext_value_device_into_async(d_evals, n_points as u32, &mut d_out);
        return GpuMleEvalExtManyOutput {
            d_values: d_out,
            _intermediates: Vec::new(),
        };
    }

    let point_stride_words = (n_coords * EF::DIMENSION) as u32;
    let mut current_len = n_elements;
    let mut current = states.next().expect("missing first OOD MLE fold state");
    g.sumcheck.fold_single_ext_to_many_points_device_into_async(
        d_evals,
        d_points_words,
        point_stride_words,
        0,
        current_len as u32,
        (current_len / 2) as u32,
        n_points as u32,
        &mut current,
    );
    current_len /= 2;

    let mut intermediates = Vec::with_capacity(n_coords.saturating_sub(1));
    for coord_idx in 1..n_coords {
        let previous = current;
        let mut next = states.next().expect("missing subsequent OOD MLE fold state");
        g.sumcheck.fold_multi_col_ext_per_point_device_into_async(
            &previous,
            d_points_words,
            point_stride_words,
            coord_idx as u32,
            current_len as u32,
            (current_len / 2) as u32,
            n_points as u32,
            &mut next,
        );
        intermediates.push(previous);
        current = next;
        current_len /= 2;
    }
    debug_assert!(states.next().is_none());

    GpuMleEvalExtManyOutput {
        d_values: current,
        _intermediates: intermediates,
    }
}

fn gpu_mle_eval_ext<EF: ExtensionField<PF<EF>>>(
    g: &gpu_backend::GpuBackend,
    d_evals: &CudaSlice<u32>,
    n_elements: usize,
    point: &MultilinearPoint<EF>,
) -> EF {
    let point_words: Vec<u32> = point.0.iter().flat_map(|coord| ef_to_u32::<EF>(coord)).collect();
    let d_point_words = g.stream.memcpy_stod(&point_words).expect("upload mle point");
    let current = gpu_mle_eval_ext_device::<EF>(g, d_evals, n_elements, &d_point_words, point.0.len());
    let evals_u32 = g.stream.memcpy_dtov(&current).expect("download mle evaluation");
    ef_from_u32::<EF>(&evals_u32[..5].try_into().unwrap())
}

/// Reconstitute extension field elements from flat u32 array (n*dim u32s → n EF values).
fn reconstitute_ext<EF: ExtensionField<PF<EF>>>(flat: &[u32]) -> Vec<EF> {
    let dim = EF::DIMENSION;
    let n = flat.len() / dim;
    (0..n)
        .map(|i| EF::from_basis_coefficients_fn(|j| unsafe { *(&flat[i * dim + j] as *const u32 as *const PF<EF>) }))
        .collect()
}

/// Convert flat u32 root from GPU to [PF<EF>; DIGEST_ELEMS].
fn u32_slice_to_root<EF: ExtensionField<PF<EF>>>(flat: &[u32]) -> [PF<EF>; DIGEST_ELEMS] {
    let mut root = [PF::<EF>::ZERO; DIGEST_ELEMS];
    for j in 0..DIGEST_ELEMS {
        root[j] = unsafe { std::mem::transmute_copy(&flat[j]) };
    }
    root
}

fn push_device_merkle_openings<EF: ExtensionField<PF<EF>>>(
    merkle_data: &GpuMerkleProverData,
    n_samples: usize,
    log_height: usize,
    row_words: &[u32],
    index_words: &[u32],
    sibling_words: &[u32],
    sibling_words_per_level: usize,
    prover_state: &mut impl FSProver<EF>,
) {
    let mut sibling_by_sample = vec![Vec::<[PF<EF>; DIGEST_ELEMS]>::with_capacity(log_height); n_samples];
    for level in 0..log_height {
        let level_start = level * sibling_words_per_level;
        let level_end = level_start + sibling_words_per_level;
        for (sample_idx, chunk) in sibling_words[level_start..level_end]
            .chunks_exact(DIGEST_ELEMS)
            .enumerate()
        {
            sibling_by_sample[sample_idx].push(u32_slice_to_root::<EF>(chunk));
        }
    }

    match merkle_data.leaf_kind {
        GpuMerkleLeafKind::Base => {
            let mut base_paths = Vec::with_capacity(n_samples);
            for ((leaf_index, row), sibling_hashes) in index_words
                .iter()
                .copied()
                .map(|idx| idx as usize)
                .zip(row_words.chunks_exact(merkle_data.full_leaf_base_width))
                .zip(sibling_by_sample)
            {
                let leaf_data: Vec<PF<EF>> = unsafe { std::mem::transmute::<Vec<u32>, Vec<PF<EF>>>(row.to_vec()) };
                base_paths.push(MerklePath {
                    leaf_data,
                    sibling_hashes,
                    leaf_index,
                });
            }
            if !base_paths.is_empty() {
                prover_state.hint_merkle_paths_base(base_paths);
            }
        }
        GpuMerkleLeafKind::Extension => {
            let mut ext_paths = Vec::with_capacity(n_samples);
            for ((leaf_index, row), sibling_hashes) in index_words
                .iter()
                .copied()
                .map(|idx| idx as usize)
                .zip(row_words.chunks_exact(merkle_data.full_leaf_base_width))
                .zip(sibling_by_sample)
            {
                ext_paths.push(MerklePath {
                    leaf_data: reconstitute_ext::<EF>(row),
                    sibling_hashes,
                    leaf_index,
                });
            }
            if !ext_paths.is_empty() {
                prover_state.hint_merkle_paths_extension(ext_paths);
            }
        }
    }
}

fn all_pending_merkle_queries_are_device<EF: ExtensionField<PF<EF>>>(
    pending_merkle_queries: &[PendingMerkleQueries<EF>],
) -> bool {
    pending_merkle_queries.iter().all(|pending| {
        matches!(
            (&pending.merkle_data, &pending.indices),
            (
                RoundMerkleProverData::Device(_),
                PendingQueryIndices::RawDeviceIndices { .. }
            )
        )
    })
}

fn build_device_merkle_opening_plans<EF: ExtensionField<PF<EF>>>(
    pending_merkle_queries: Vec<PendingMerkleQueries<EF>>,
) -> (Vec<DeviceMerkleOpeningPlan>, usize) {
    let mut plans = Vec::with_capacity(pending_merkle_queries.len());
    let mut total_words = 0usize;
    for PendingMerkleQueries { merkle_data, indices } in pending_merkle_queries {
        let (
            RoundMerkleProverData::Device(merkle_data),
            PendingQueryIndices::RawDeviceIndices {
                d_indices,
                _d_sample_words,
            },
        ) = (merkle_data, indices)
        else {
            unreachable!("all-device merkle opening path was prechecked");
        };
        let n_samples = d_indices.len();
        let row_words_len = n_samples * merkle_data.full_leaf_base_width;
        let sibling_words_per_level = n_samples * DIGEST_ELEMS;
        let log_height = log2_ceil_usize(merkle_data.height);
        let total_plan_words = n_samples + row_words_len + log_height * sibling_words_per_level;
        plans.push(DeviceMerkleOpeningPlan {
            merkle_data,
            d_indices,
            _d_sample_words,
            n_samples,
            offset: total_words,
            total_words: total_plan_words,
            row_words_len,
            sibling_words_per_level,
            log_height,
            _d_row_words: None,
            _d_sibling_layers: Vec::new(),
        });
        total_words += total_plan_words;
    }
    (plans, total_words)
}

fn stage_device_merkle_opening_plans(
    g: &gpu_backend::GpuBackend,
    plans: &mut [DeviceMerkleOpeningPlan],
    d_opening_words: &mut CudaSlice<u32>,
    base_offset: usize,
    mut opening_workspaces: Option<&mut [GpuWhirOpeningWorkspaces]>,
    require_uploaded_workspaces: bool,
) -> Option<()> {
    if require_uploaded_workspaces
        && opening_workspaces.as_ref().map(|workspaces| workspaces.len()) != Some(plans.len())
    {
        return None;
    }
    for (plan_idx, plan) in plans.iter_mut().enumerate() {
        if plan.n_samples == 0 {
            continue;
        }
        let mut opening_workspace = opening_workspaces
            .as_deref_mut()
            .and_then(|workspaces| workspaces.get_mut(plan_idx));
        if require_uploaded_workspaces && opening_workspace.is_none() {
            return None;
        }
        g.sumcheck.memcpy_d2d_async(
            &plan.d_indices,
            0,
            d_opening_words,
            base_offset + plan.offset,
            plan.n_samples,
        );

        let row_offset = base_offset + plan.offset + plan.n_samples;
        let d_row_words = if let Some(workspace) = opening_workspace.as_mut() {
            let mut d_row_words = take_whir_workspace(&mut workspace.d_row_words, "WHIR final opening row words");
            if d_row_words.len() < plan.row_words_len {
                return None;
            }
            g.merkle.gather_rows_device_into_async(
                &plan.merkle_data.d_leaf_matrix,
                &plan.d_indices,
                plan.n_samples as u32,
                plan.merkle_data.full_leaf_base_width as u32,
                &mut d_row_words,
            );
            d_row_words
        } else {
            g.merkle.gather_rows_device_async(
                &plan.merkle_data.d_leaf_matrix,
                &plan.d_indices,
                plan.n_samples as u32,
                plan.merkle_data.full_leaf_base_width as u32,
            )
        };
        g.sumcheck
            .memcpy_d2d_async(&d_row_words, 0, d_opening_words, row_offset, plan.row_words_len);
        plan._d_row_words = Some(d_row_words);

        let sibling_offset = row_offset + plan.row_words_len;
        for level in 0..plan.log_height {
            let d_siblings = if let Some(workspace) = opening_workspace.as_mut() {
                let Some(slot) = workspace.d_sibling_layers.get_mut(level) else {
                    return None;
                };
                let mut d_siblings = take_whir_workspace(slot, "WHIR final opening sibling words");
                if d_siblings.len() < plan.sibling_words_per_level {
                    return None;
                }
                g.merkle.gather_sibling_hashes_device_into_async(
                    &plan.merkle_data.d_digest_layers[level],
                    &plan.d_indices,
                    plan.n_samples as u32,
                    level as u32,
                    &mut d_siblings,
                );
                d_siblings
            } else {
                g.merkle.gather_sibling_hashes_device_async(
                    &plan.merkle_data.d_digest_layers[level],
                    &plan.d_indices,
                    plan.n_samples as u32,
                    level as u32,
                )
            };
            g.sumcheck.memcpy_d2d_async(
                &d_siblings,
                0,
                d_opening_words,
                sibling_offset + level * plan.sibling_words_per_level,
                plan.sibling_words_per_level,
            );
            plan._d_sibling_layers.push(d_siblings);
        }
    }
    Some(())
}

fn push_device_merkle_opening_plans<EF: ExtensionField<PF<EF>>>(
    plans: &[DeviceMerkleOpeningPlan],
    opening_words: &[u32],
    base_offset: usize,
    prover_state: &mut impl FSProver<EF>,
) {
    for plan in plans {
        if plan.n_samples == 0 {
            continue;
        }
        let plan_words = &opening_words[base_offset + plan.offset..base_offset + plan.offset + plan.total_words];
        let (index_words, rest) = plan_words.split_at(plan.n_samples);
        let (row_words, sibling_words) = rest.split_at(plan.row_words_len);
        push_device_merkle_openings::<EF>(
            &plan.merkle_data,
            plan.n_samples,
            plan.log_height,
            row_words,
            index_words,
            sibling_words,
            plan.sibling_words_per_level,
            prover_state,
        );
    }
}

fn inject_gpu_transcript_words<EF: ExtensionField<PF<EF>>>(
    prover_state: &mut impl FSProver<EF>,
    transcript_words: &[u32],
    challenger_state_words: &[u32],
) {
    let transcript_scalars = transcript_words
        .iter()
        .copied()
        .map(kb_from_u32::<PF<EF>>)
        .collect::<Vec<_>>();
    let final_challenger_state = [
        kb_from_u32::<PF<EF>>(challenger_state_words[0]),
        kb_from_u32::<PF<EF>>(challenger_state_words[1]),
        kb_from_u32::<PF<EF>>(challenger_state_words[2]),
        kb_from_u32::<PF<EF>>(challenger_state_words[3]),
        kb_from_u32::<PF<EF>>(challenger_state_words[4]),
        kb_from_u32::<PF<EF>>(challenger_state_words[5]),
        kb_from_u32::<PF<EF>>(challenger_state_words[6]),
        kb_from_u32::<PF<EF>>(challenger_state_words[7]),
    ];
    prover_state.inject_gpu_transcript_state(&transcript_scalars, final_challenger_state);
}

fn finish_whir_materialization<EF>(
    g: &gpu_backend::GpuBackend,
    pending_merkle_queries: Vec<PendingMerkleQueries<EF>>,
    gpu_fs: GpuTranscript<'_, EF>,
    prover_state: &mut impl FSProver<EF>,
    require_device_only_path: bool,
    d_final_materialization: Option<CudaSlice<u32>>,
    final_opening_workspaces: Option<Vec<GpuWhirOpeningWorkspaces>>,
) -> Option<()>
where
    EF: ExtensionField<PF<EF>>,
    PF<EF>: PrimeField64,
{
    if !all_pending_merkle_queries_are_device(&pending_merkle_queries) {
        if require_device_only_path {
            return None;
        }
        hint_merkle_paths_inline_all_device(g, pending_merkle_queries, prover_state);
        gpu_fs.finish(g, prover_state);
        return Some(());
    }

    let (mut plans, merkle_words) = build_device_merkle_opening_plans(pending_merkle_queries);
    let transcript_words: usize = gpu_fs.transcript_chunks.iter().map(|chunk| chunk.n_words).sum();
    let transcript_offset = merkle_words;
    let challenger_offset = transcript_offset + transcript_words;
    let total_words = challenger_offset + 8;
    let mut d_final = if let Some(d_final) = d_final_materialization {
        if d_final.len() < total_words {
            if require_device_only_path {
                return None;
            }
            g.stream
                .alloc_zeros::<u32>(total_words)
                .expect("allocate combined WHIR proof materialization buffer")
        } else {
            d_final
        }
    } else {
        g.stream
            .alloc_zeros::<u32>(total_words)
            .expect("allocate combined WHIR proof materialization buffer")
    };

    let mut final_opening_workspaces = final_opening_workspaces;
    stage_device_merkle_opening_plans(
        g,
        &mut plans,
        &mut d_final,
        0,
        final_opening_workspaces.as_deref_mut(),
        require_device_only_path,
    )?;

    let mut offset = transcript_offset;
    for chunk in &gpu_fs.transcript_chunks {
        g.sumcheck
            .memcpy_d2d_async(&chunk.d_words, 0, &mut d_final, offset, chunk.n_words);
        offset += chunk.n_words;
    }
    g.sumcheck
        .memcpy_d2d_async(&gpu_fs.d_challenger_state, 0, &mut d_final, challenger_offset, 8);

    let final_words = g
        .stream
        .memcpy_dtov(&d_final.slice(0..total_words))
        .expect("download combined WHIR proof materialization");
    push_device_merkle_opening_plans::<EF>(&plans, &final_words, 0, prover_state);
    inject_gpu_transcript_words::<EF>(
        prover_state,
        &final_words[transcript_offset..challenger_offset],
        &final_words[challenger_offset..challenger_offset + 8],
    );
    Some(())
}

/// Materialize all-device Merkle openings with one final packed download.
fn hint_merkle_paths_inline_all_device<EF: ExtensionField<PF<EF>>>(
    g: &gpu_backend::GpuBackend,
    pending_merkle_queries: Vec<PendingMerkleQueries<EF>>,
    prover_state: &mut impl FSProver<EF>,
) {
    if pending_merkle_queries.is_empty() {
        return;
    }

    if !all_pending_merkle_queries_are_device(&pending_merkle_queries) {
        for PendingMerkleQueries { merkle_data, indices } in pending_merkle_queries {
            hint_merkle_paths_inline(g, &merkle_data, prover_state, indices);
        }
        return;
    }

    let (mut plans, total_words) = build_device_merkle_opening_plans(pending_merkle_queries);
    if total_words == 0 {
        return;
    }

    let mut d_opening_words = g
        .stream
        .alloc_zeros::<u32>(total_words)
        .expect("allocate combined merkle opening download buffer");
    stage_device_merkle_opening_plans(g, &mut plans, &mut d_opening_words, 0, None, false)
        .expect("legacy Merkle opening staging should allocate fallback workspaces");

    let opening_words = g
        .stream
        .memcpy_dtov(&d_opening_words)
        .expect("download combined merkle opening data");
    push_device_merkle_opening_plans::<EF>(&plans, &opening_words, 0, prover_state);
}

/// Materialize Merkle proof paths for a batch of previously sampled query indices.
fn hint_merkle_paths_inline<EF: ExtensionField<PF<EF>>>(
    g: &gpu_backend::GpuBackend,
    merkle_data: &RoundMerkleProverData<EF>,
    prover_state: &mut impl FSProver<EF>,
    indices: PendingQueryIndices,
) {
    let mut base_paths = Vec::new();
    let mut ext_paths = Vec::new();
    for (leaf_index, answer, sibling_hashes) in merkle_data.open_batch(g, indices) {
        match answer {
            MleOwned::Base(leaf) => base_paths.push(MerklePath {
                leaf_data: leaf,
                sibling_hashes,
                leaf_index,
            }),
            MleOwned::Extension(leaf) => ext_paths.push(MerklePath {
                leaf_data: leaf,
                sibling_hashes,
                leaf_index,
            }),
            _ => unreachable!(),
        }
    }
    if !base_paths.is_empty() {
        prover_state.hint_merkle_paths_base(base_paths);
    }
    if !ext_paths.is_empty() {
        prover_state.hint_merkle_paths_extension(ext_paths);
    }
}
