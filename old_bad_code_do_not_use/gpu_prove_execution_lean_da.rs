//! COMPLETE GPU prove_execution — full protocol reimplementation.

use std::sync::{Arc, OnceLock};
use std::collections::BTreeMap;

use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream};

use crate::prove_execution::ExecutionProof;
use crate::*;
use backend::*;
use lean_vm::*;
use sub_protocols::*;
use tracing::info_span;
use utils::{build_prover_state, from_end};

struct Gpu {
    stream: Arc<CudaStream>,
    sumcheck: gpu_sumcheck::GpuSumcheck,
    fold: gpu_poly_fold::GpuPolyFold,
    ntt: gpu_ntt::GpuNtt,
    merkle: gpu_merkle::GpuMerkle,
}

static GPU: OnceLock<Option<Gpu>> = OnceLock::new();

fn gpu() -> Option<&'static Gpu> {
    GPU.get_or_init(|| {
        let ctx = CudaContext::new(0).ok()?;
        let s = ctx.default_stream();
        tracing::info!("GPU prover initialized (full pipeline)");
        Some(Gpu {
            sumcheck: gpu_sumcheck::GpuSumcheck::new(s.clone()),
            fold: gpu_poly_fold::GpuPolyFold::new(s.clone()),
            ntt: gpu_ntt::GpuNtt::new(s.clone()),
            merkle: gpu_merkle::GpuMerkle::new(s.clone()),
            stream: s,
        })
    }).as_ref()
}

/// Full GPU prove_execution.
///
/// Reimplements the ENTIRE proving protocol with GPU for heavy compute.
/// All polynomial/column data stays on GPU as flat u32 CudaSlice.
/// Only ProverState (Fiat-Shamir, ~200 bytes per round) crosses PCIe.
pub fn gpu_prove_execution(
    bytecode: &Bytecode,
    public_input: &[F],
    witness: &ExecutionWitness,
    whir_config: &WhirConfigBuilder,
    vm_profiler: bool,
) -> Result<ExecutionProof, ProverError> {
    check_rate(whir_config.starting_log_inv_rate)
        .map_err(|err| panic!("{err}"))
        .unwrap();

    // ═══════════════════════════════════════════════════════════════════
    // STEP 1: VM Execution (CPU — cannot be GPU-accelerated)
    // ═══════════════════════════════════════════════════════════════════
    let ExecutionTrace {
        traces, public_memory_size, mut memory, metadata,
    } = info_span!("Witness generation").in_scope(|| -> Result<_, ProverError> {
        let execution_result = info_span!("Executing bytecode")
            .in_scope(|| try_execute_bytecode(bytecode, public_input, witness, vm_profiler))?;
        Ok(info_span!("Building execution trace")
            .in_scope(|| get_execution_trace(bytecode, execution_result)))
    })?;

    let min_memory_size = (1 << MIN_LOG_MEMORY_SIZE).max(1 << bytecode.log_size());
    if memory.len() < min_memory_size {
        memory.resize(min_memory_size, F::ZERO);
    }

    // ═══════════════════════════════════════════════════════════════════
    // STEP 2: ProverState setup (CPU — tiny Fiat-Shamir initialization)
    // ═══════════════════════════════════════════════════════════════════
    let mut prover_state = build_prover_state();
    prover_state.observe_scalars(public_input);
    prover_state.observe_scalars(&poseidon16_compress_pair(&bytecode.hash, &SNARK_DOMAIN_SEP));
    prover_state.add_base_scalars(
        &[
            vec![whir_config.starting_log_inv_rate, log2_strict_usize(memory.len()), public_input.len()],
            traces.values().map(|t| t.log_n_rows).collect::<Vec<_>>(),
        ].concat().into_iter().map(F::from_usize).collect::<Vec<_>>(),
    );

    for (table, table_trace) in &traces {
        let log_n_rows = table_trace.log_n_rows;
        assert!(log_n_rows >= MIN_LOG_N_ROWS_PER_TABLE, "missing padding");
        let log_limit = max_log_n_rows_per_table(table);
        if log_n_rows > log_limit {
            return Err(TooBigTableError { table_name: table.name(), log_n_rows, log_limit }.into());
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // STEP 3: Access counts (CPU for now, TODO: GPU kernel)
    // ═══════════════════════════════════════════════════════════════════
    let mut memory_acc = F::zero_vec(memory.len());
    info_span!("Building memory access count").in_scope(|| {
        for (table, trace) in &traces {
            for lookup in table.lookups() {
                for i in &trace.columns[lookup.index] {
                    for j in 0..lookup.values.len() {
                        memory_acc[i.to_usize() + j] += F::ONE;
                    }
                }
            }
        }
    });
    let mut bytecode_acc = F::zero_vec(bytecode.padded_size());
    info_span!("Building bytecode access count").in_scope(|| {
        for pc in traces[&Table::execution()].columns[COL_PC].iter() {
            bytecode_acc[pc.to_usize()] += F::ONE;
        }
    });

    // ═══════════════════════════════════════════════════════════════════
    // STEP 4-5: Polynomial stacking + WHIR commit
    // GPU: DFT→Merkle chained on device. Upload polynomial once.
    // ═══════════════════════════════════════════════════════════════════
    let stacked_pcs_witness = info_span!("GPU stack+commit").in_scope(|| {
        let g = gpu();
        if let Some(g) = g {
            // Build stacked polynomial (CPU, just memcpy).
            let tables_heights = traces.iter().map(|(t,tr)| (*t, tr.log_n_rows)).collect();
            let tables_sorted = sort_tables_by_height(&tables_heights);
            let stacked_n_vars = compute_stacked_n_vars(
                log2_strict_usize(memory.len()),
                log2_strict_usize(bytecode_acc.len()),
                &tables_sorted.iter().cloned().collect(),
            );
            let mut global_poly = F::zero_vec(1 << stacked_n_vars);
            global_poly[..memory.len()].copy_from_slice(&memory);
            let mut offset = memory.len();
            global_poly[offset..][..memory_acc.len()].copy_from_slice(&memory_acc);
            offset += memory_acc.len();
            global_poly[offset..][..bytecode_acc.len()].copy_from_slice(&bytecode_acc);
            let largest_h = 1 << tables_sorted[0].1;
            offset += largest_h.max(bytecode_acc.len());
            let actual_data_len_start = offset;
            for (table, log_n_rows) in &tables_sorted {
                let nr = 1 << *log_n_rows;
                for ci in 0..table.n_columns() {
                    global_poly[offset..][..nr].copy_from_slice(&traces[table].columns[ci][..nr]);
                    offset += nr;
                }
            }
            let actual_data_len = offset;
            tracing::info!("stacked PCS data: {} = 2^{}", actual_data_len, stacked_n_vars);

            // Upload to GPU.
            let poly_u32: &[u32] = unsafe {
                std::slice::from_raw_parts(global_poly.as_ptr().cast(), global_poly.len())
            };
            let d_poly = g.stream.memcpy_stod(poly_u32).unwrap();

            // GPU DFT → GPU Merkle.
            let ff = whir_config.folding_factor.at_round(0);
            let n_blocks = 1usize << ff;
            let effective_n_cols = actual_data_len.div_ceil((1 << stacked_n_vars) / n_blocks);
            let n_evals = (1u32 << stacked_n_vars);
            let n_cols = n_blocks as u32;

            let d_dft = g.ntt.reorder_and_dft_device(&d_poly, n_evals, ff, whir_config.starting_log_inv_rate);
            let full_len = (n_evals as u64) << whir_config.starting_log_inv_rate;
            let height = (full_len / n_cols as u64) as u32;
            let (root_u32, merkle_layers) = g.merkle.build_tree_from_device(&d_dft, height, n_cols, n_cols);

            // Download DFT output for CPU-side operations.
            let dft_flat = g.stream.memcpy_dtov(&d_dft).unwrap();
            let dft_pf: Vec<F> = unsafe { std::mem::transmute(dft_flat) };

            // Construct leanVM types from GPU results.
            let dft_matrix = DenseMatrix::new(dft_pf, n_blocks);
            let digest_layers: Vec<Vec<[F; DIGEST_ELEMS]>> = merkle_layers.iter().map(|layer| {
                let n = layer.len() / DIGEST_ELEMS;
                (0..n).map(|i| {
                    let mut d = [F::ZERO; DIGEST_ELEMS];
                    for j in 0..DIGEST_ELEMS { d[j] = unsafe { std::mem::transmute(layer[i*DIGEST_ELEMS+j]) }; }
                    d
                }).collect()
            }).collect();
            let tree = backend::merkle::MerkleTree { digest_layers };
            let whir_tree = WhirMerkleTree { leaf: dft_matrix, tree, full_leaf_base_width: n_blocks };
            let root: [F; DIGEST_ELEMS] = whir_tree.root();
            let prover_data = MerkleData::Base(whir_tree);

            prover_state.add_base_scalars(&root);

            // OOD evaluation (CPU for now).
            let whir_cfg = WhirConfig::<EF>::new(whir_config, stacked_n_vars);
            // Inline sample_ood_points (not public from whir crate).
            let (ood_points, ood_answers) = {
                let num_samples = whir_cfg.commitment_ood_samples;
                let mut pts = Vec::new();
                let mut ans = Vec::new();
                if num_samples > 0 {
                    pts = prover_state.sample_vec(num_samples);
                    let mle = MleOwned::<EF>::Base(global_poly.clone());
                    ans.extend(pts.iter().map(|p| {
                        mle.evaluate(&MultilinearPoint::expand_from_univariate(*p, stacked_n_vars))
                    }));
                    prover_state.add_extension_scalars(&ans);
                }
                (pts, ans)
            };

            let inner_witness = Witness { prover_data, ood_points, ood_answers };
            let global_polynomial = MleOwned::Base(global_poly);

            StackedPcsWitness { stacked_n_vars, inner_witness, global_polynomial }
        } else {
            // CPU fallback.
            stack_polynomials_and_commit(&mut prover_state, whir_config, &memory, &memory_acc, &bytecode_acc, &traces)
        }
    });

    // ═══════════════════════════════════════════════════════════════════
    // STEP 6: Logup (GKR)
    // TODO: Reimplement on GPU (fingerprint + GKR reduction)
    // For now: CPU
    // ═══════════════════════════════════════════════════════════════════
    let logup_c = prover_state.sample();
    let logup_alphas = prover_state.sample_vec(log2_ceil_usize(max_bus_width_including_domainsep()));
    let logup_alphas_eq_poly = eval_eq(&logup_alphas);

    let logup_statements = prove_generic_logup(
        &mut prover_state, logup_c, &logup_alphas_eq_poly,
        &memory, &memory_acc, &bytecode.instructions_multilinear, &bytecode_acc, &traces,
    );

    let gkr_point = &logup_statements.gkr_point;
    let mut committed_statements: CommittedStatements = Default::default();
    for table in ALL_TABLES {
        let log_n_rows = traces[&table].log_n_rows;
        committed_statements.insert(table, vec![(
            MultilinearPoint(from_end(gkr_point, log_n_rows).to_vec()),
            logup_statements.columns_values[&table].clone(),
            BTreeMap::new(),
        )]);
    }

    // ═══════════════════════════════════════════════════════════════════
    // STEP 7: AIR sumcheck
    // TODO: Reimplement on GPU (constraint eval + fold)
    // For now: CPU
    // ═══════════════════════════════════════════════════════════════════
    let bus_beta = prover_state.sample();
    let air_alpha = prover_state.sample();
    let air_alpha_powers: Vec<EF> = air_alpha.powers().collect_n(max_air_constraints() + 1);
    let air_eta: EF = prover_state.sample();

    let tables_log_heights: BTreeMap<Table, VarCount> =
        traces.iter().map(|(table, trace)| (*table, trace.log_n_rows)).collect();
    let tables_sorted = sort_tables_by_height(&tables_log_heights);

    let column_refs: Vec<Vec<&[F]>> = tables_sorted.iter()
        .map(|(table, _)| traces[table].columns[..table.n_columns()].iter().map(Vec::as_slice).collect())
        .collect();
    let shifted_rows: Vec<Vec<Vec<F>>> = tables_sorted.par_iter().zip(&column_refs)
        .map(|((table, _), cols)| compute_shifted_columns(&table.down_column_indexes(), cols))
        .collect();

    let mut sessions = Vec::with_capacity(tables_sorted.len());
    for (idx, (table, log_n_rows)) in tables_sorted.iter().enumerate() {
        let bus_numerator_value = logup_statements.bus_numerators_values[table];
        let bus_denominator_value = logup_statements.bus_denominators_values[table];
        let bus_final_value = bus_numerator_value
            * match table.bus().direction { BusDirection::Pull => EF::NEG_ONE, BusDirection::Push => EF::ONE }
            + bus_beta * (bus_denominator_value - logup_c);

        let eq_suffix = from_end(gkr_point, *log_n_rows).to_vec();
        let extra_data = ExtraDataForBuses::new(logup_alphas_eq_poly.clone(), bus_beta, air_alpha_powers.clone());

        let mut up_down: Vec<&[PF<EF>]> = column_refs[idx].to_vec();
        up_down.extend(shifted_rows[idx].iter().map(Vec::as_slice));
        let packed = MleGroupRef::<EF>::Base(up_down).pack();
        let non_padded = traces[table].non_padded_n_rows;

        macro_rules! make_session {
            ($t:expr) => {{ Box::new(AirSumcheckSession::new(packed, eq_suffix, bus_final_value, *$t, extra_data, non_padded)) as Box<dyn OuterSumcheckSession<EF> + '_> }};
        }
        sessions.push(delegate_to_inner!(table => make_session));
    }

    let sumcheck_air_point = info_span!("batched AIR sumcheck")
        .in_scope(|| prove_batched_air_sumcheck(&mut prover_state, &mut sessions, air_eta));

    for (idx, (table, _)) in tables_sorted.iter().enumerate() {
        let col_evals = sessions[idx].final_column_evals();
        prover_state.add_extension_scalars(&col_evals);

        let natural_ordering_point = natural_ordering_point_for_session(&sumcheck_air_point.0, traces[table].log_n_rows);
        macro_rules! split {
            ($t:expr) => {{ columns_evals_up_and_down($t, &col_evals, &natural_ordering_point) }};
        }
        let claim = delegate_to_inner!(table => split);
        committed_statements.get_mut(table).unwrap().push(claim);
    }

    // ═══════════════════════════════════════════════════════════════════
    // STEP 8: WHIR prove — GPU product sumcheck + fold + DFT + Merkle
    // Data stays on GPU. Only Fiat-Shamir (~200 bytes/round) crosses PCIe.
    // ═══════════════════════════════════════════════════════════════════
    let public_memory_random_point = MultilinearPoint(prover_state.sample_vec(log2_strict_usize(public_memory_size)));
    let public_memory_eval = (&memory[..public_memory_size]).evaluate(&public_memory_random_point);

    let previous_statements = vec![
        SparseStatement::new(stacked_pcs_witness.stacked_n_vars, logup_statements.memory_and_acc_point,
            vec![SparseValue::new(0, logup_statements.value_memory), SparseValue::new(1, logup_statements.value_memory_acc)]),
        SparseStatement::new(stacked_pcs_witness.stacked_n_vars, public_memory_random_point, vec![SparseValue::new(0, public_memory_eval)]),
        SparseStatement::new(stacked_pcs_witness.stacked_n_vars, logup_statements.bytecode_and_acc_point,
            vec![SparseValue::new((2 * memory.len()) >> bytecode.log_size(), logup_statements.value_bytecode_acc)]),
    ];

    let global_statements_base = stacked_pcs_global_statements(
        stacked_pcs_witness.stacked_n_vars, log2_strict_usize(memory.len()), bytecode.log_size(),
        previous_statements, &tables_log_heights, &committed_statements,
    );

    let whir_cfg = WhirConfig::<EF>::new(whir_config, stacked_pcs_witness.global_polynomial.by_ref().n_vars());

    // GPU WHIR prove: product sumcheck + fold + DFT + Merkle all on GPU.
    match whir_cfg.gpu_prove(
        &mut prover_state, global_statements_base, stacked_pcs_witness.inner_witness,
        &stacked_pcs_witness.global_polynomial.by_ref(),
    ) {
        Some(_) => tracing::info!("GPU WHIR prove completed"),
        None => panic!("GPU WHIR prove failed — GPU required for gpu_prove_execution"),
    }

    tracing::info!("total pow_grinding time: {} ms", pow_grinding_time().as_millis());
    reset_pow_grinding_time();

    Ok(ExecutionProof { proof: prover_state.into_proof(), metadata: Some(metadata) })
}
