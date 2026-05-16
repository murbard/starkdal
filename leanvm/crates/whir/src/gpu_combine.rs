//! GPU-accelerated combine_statement.
//!
//! Builds equality polynomial weights directly on GPU.
//! Eliminates the 640MB upload that was bottlenecking the GPU product sumcheck.

use std::sync::Arc;

use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DeviceSlice, SyncOnDrop};
use cudarc::driver::sys;
use field::ExtensionField;
use poly::*;

use crate::gpu_backend;
use crate::{GpuSparseStatement, GpuSparseValue, SparseStatement, SparseValue};
use poly::matrix_next_mle_folded;

enum StatementPoly {
    Owned(CudaSlice<u32>),
    SharedOne(&'static CudaSlice<u32>),
}

impl From<CudaSlice<u32>> for StatementPoly {
    fn from(data: CudaSlice<u32>) -> Self {
        Self::Owned(data)
    }
}

impl DeviceSlice<u32> for StatementPoly {
    fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::SharedOne(data) => data.len(),
        }
    }

    fn stream(&self) -> &Arc<CudaStream> {
        match self {
            Self::Owned(data) => data.stream(),
            Self::SharedOne(data) => data.stream(),
        }
    }
}

impl DevicePtr<u32> for StatementPoly {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        match self {
            Self::Owned(data) => data.device_ptr(stream),
            Self::SharedOne(data) => data.device_ptr(stream),
        }
    }
}

/// Build combined weights on GPU.
///
/// Reimplements combine_statement but keeps the result on GPU as a CudaSlice.
/// For each sparse statement, builds eq(point, x) on GPU and accumulates.
///
/// Returns (d_weights_flat, sum) where d_weights_flat has n_total * 5 u32s
/// (flat extension field, NOT packed).
///
/// Returns None if GPU not available or statements are too complex.
pub(crate) fn gpu_combine_statement<EF>(
    statements: &[SparseStatement<EF>],
    gamma: EF,
    num_variables: usize,
) -> Option<(CudaSlice<u32>, EF)>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION; // 5
    let n_total = 1usize << num_variables;

    // Allocate zero weights on GPU.
    let mut d_weights = g.stream.alloc_zeros::<u32>(n_total * dim).ok()?;
    let mut combined_sum = EF::ZERO;
    let mut gamma_pow = EF::ONE;
    gpu_accumulate_statement_into(
        &mut d_weights,
        &mut combined_sum,
        &mut gamma_pow,
        statements,
        gamma,
        num_variables,
    )?;
    Some((d_weights, combined_sum))
}

pub(crate) fn gpu_accumulate_statement_into<EF>(
    d_weights: &mut CudaSlice<u32>,
    combined_sum: &mut EF,
    gamma_pow: &mut EF,
    statements: &[SparseStatement<EF>],
    gamma: EF,
    num_variables: usize,
) -> Option<()>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION; // 5
    let n_total = 1usize << num_variables;

    for smt in statements {
        let inner_n_vars = smt.inner_num_variables();

        // Build the polynomial on GPU (eq or next_mle depending on is_next).
        let d_poly = if smt.is_next {
            // next_mle: compute on CPU, upload to GPU (small: ~few KB per statement).
            let next_values = matrix_next_mle_folded::<EF>(&smt.point.0);
            let next_u32: &[u32] =
                unsafe { std::slice::from_raw_parts(next_values.as_ptr().cast::<u32>(), next_values.len() * dim) };
            g.stream.memcpy_stod(next_u32).ok()?
        } else {
            // eq polynomial: build on GPU.
            let point_ef_u32: Vec<[u32; 5]> = smt
                .point
                .0
                .iter()
                .map(|p| {
                    let mut out = [0u32; 5];
                    let coeffs = p.as_basis_coefficients_slice();
                    for (j, c) in coeffs.iter().enumerate() {
                        out[j] = unsafe { *(c as *const PF<EF> as *const u32) };
                    }
                    out
                })
                .collect();
            g.sumcheck.eq_polynomial_device(&point_ef_u32)
        };

        for evaluation in &smt.values {
            // Scale by gamma_pow and accumulate into weights at the selector offset.
            let scalar_u32 = {
                let mut out = [0u32; 5];
                let coeffs = gamma_pow.as_basis_coefficients_slice();
                for (j, c) in coeffs.iter().enumerate() {
                    out[j] = unsafe { *(c as *const PF<EF> as *const u32) };
                }
                out
            };

            let selector = evaluation.selector;
            let inner_n = 1usize << inner_n_vars;

            if selector == 0 && inner_n == n_total {
                // Dense case: polynomial covers the full domain. Accumulate directly.
                g.sumcheck
                    .eq_accumulate_device(d_weights, &d_poly, &scalar_u32, n_total as u32);
            } else {
                // Sparse case: eq covers a sub-range at selector offset.
                // Use offset accumulator: weights[selector * inner_n + j] += scalar * eq[j]
                let offset = (selector * inner_n) as u32;
                g.sumcheck
                    .eq_accumulate_offset_device(d_weights, &d_poly, &scalar_u32, offset, inner_n as u32);
            }

            *combined_sum += evaluation.value * *gamma_pow;
            *gamma_pow *= gamma;
        }
    }

    Some(())
}

pub(crate) fn gpu_accumulate_statement_with_device_gamma<EF>(
    d_weights: &mut CudaSlice<u32>,
    d_base_sum: &CudaSlice<u32>,
    d_gamma: &CudaSlice<u32>,
    gamma_start_power: usize,
    statements: &[SparseStatement<EF>],
    num_variables: usize,
) -> Option<CudaSlice<u32>>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION;
    let n_total = 1usize << num_variables;
    let total_values: usize = statements.iter().map(|statement| statement.values.len()).sum();

    if total_values == 0 {
        return g.stream.clone_dtod(d_base_sum).ok();
    }

    let d_gamma_powers = g
        .sumcheck
        .extension_powers_device(d_gamma, (gamma_start_power + total_values) as u32);
    let statement_values = flatten_sparse_values::<EF>(statements);
    let d_statement_values = g.stream.memcpy_stod(&statement_values).ok()?;
    let d_statement_scalars = d_gamma_powers.slice(gamma_start_power * dim..(gamma_start_power + total_values) * dim);

    // Keep asynchronously referenced device buffers alive until the final dot-product
    // synchronization at the end of this helper.
    let mut poly_buffers = Vec::with_capacity(statements.len());
    let mut scalar_idx = 0usize;
    for statement in statements {
        let inner_n_vars = statement.inner_num_variables();
        let d_poly: StatementPoly = build_statement_poly_device(statement)?.into();
        let poly_idx = poly_buffers.len();
        poly_buffers.push(d_poly);

        for SparseValue { selector, .. } in &statement.values {
            let scalar_start = scalar_idx * dim;
            let scalar_end = scalar_start + dim;
            let d_scalar = d_statement_scalars.slice(scalar_start..scalar_end);

            let inner_n = 1usize << inner_n_vars;
            if *selector == 0 && inner_n == n_total {
                g.sumcheck.eq_accumulate_device_scalar_device(
                    d_weights,
                    &poly_buffers[poly_idx],
                    &d_scalar,
                    n_total as u32,
                );
            } else {
                let offset = (*selector * inner_n) as u32;
                g.sumcheck.eq_accumulate_offset_device_scalar_device(
                    d_weights,
                    &poly_buffers[poly_idx],
                    &d_scalar,
                    offset,
                    inner_n as u32,
                );
            }
            scalar_idx += 1;
        }
    }

    Some(g.sumcheck.ext_dot_accumulate_device(
        d_base_sum,
        &d_statement_values,
        &d_statement_scalars,
        total_values as u32,
    ))
}

pub(crate) fn gpu_accumulate_device_statements_with_device_gamma<EF>(
    d_weights: &mut CudaSlice<u32>,
    d_base_sum: &CudaSlice<u32>,
    d_gamma: &CudaSlice<u32>,
    gamma_start_power: usize,
    statements: &[GpuSparseStatement],
    num_variables: usize,
) -> Option<CudaSlice<u32>>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION;
    let n_total = 1usize << num_variables;
    let total_values: usize = statements.iter().map(|statement| statement.values.len()).sum();

    if total_values == 0 {
        return g.stream.clone_dtod(d_base_sum).ok();
    }

    let d_gamma_powers = g
        .sumcheck
        .extension_powers_device(d_gamma, (gamma_start_power + total_values) as u32);
    let d_statement_scalars = d_gamma_powers.slice(gamma_start_power * dim..(gamma_start_power + total_values) * dim);
    let mut d_statement_values = g.stream.alloc_zeros::<u32>(total_values * dim).ok()?;

    let mut poly_buffers = Vec::with_capacity(statements.len());
    let mut scalar_idx = 0usize;
    for statement in statements {
        if statement.total_num_variables != num_variables {
            return None;
        }
        let inner_n_vars = statement.point_len;
        let inner_n = 1usize << inner_n_vars;
        let d_poly = if statement.is_next {
            g.sumcheck
                .next_mle_device_from_point_words(&statement.d_point_words, inner_n_vars)
                .into()
        } else if inner_n_vars == 0 {
            if dim != 5 {
                return None;
            }
            StatementPoly::SharedOne(&g.d_ext_one)
        } else {
            g.sumcheck
                .eq_polynomial_device_from_flat_point_words_async(&statement.d_point_words, inner_n_vars)
                .into()
        };
        let poly_idx = poly_buffers.len();
        poly_buffers.push(d_poly);

        for GpuSparseValue { selector, d_value } in &statement.values {
            if d_value.len() != dim {
                return None;
            }
            g.sumcheck
                .memcpy_d2d_async(d_value, 0, &mut d_statement_values, scalar_idx * dim, dim);

            let scalar_start = scalar_idx * dim;
            let scalar_end = scalar_start + dim;
            let d_scalar = d_statement_scalars.slice(scalar_start..scalar_end);

            if *selector == 0 && inner_n == n_total {
                g.sumcheck.eq_accumulate_device_scalar_device(
                    d_weights,
                    &poly_buffers[poly_idx],
                    &d_scalar,
                    n_total as u32,
                );
            } else {
                let offset = (*selector * inner_n) as u32;
                g.sumcheck.eq_accumulate_offset_device_scalar_device(
                    d_weights,
                    &poly_buffers[poly_idx],
                    &d_scalar,
                    offset,
                    inner_n as u32,
                );
            }
            scalar_idx += 1;
        }
    }

    Some(g.sumcheck.ext_dot_accumulate_device(
        d_base_sum,
        &d_statement_values,
        &d_statement_scalars,
        total_values as u32,
    ))
}

pub(crate) struct GpuDeviceStatementAccumulationGuard {
    _d_base_sum: Option<CudaSlice<u32>>,
    _d_gamma: CudaSlice<u32>,
    _d_gamma_powers: Option<CudaSlice<u32>>,
    _d_statement_values: Option<CudaSlice<u32>>,
    _poly_buffers: Vec<StatementPoly>,
}

pub(crate) struct GpuDeviceStatementAccumulationWorkspaces {
    pub(crate) d_gamma_powers: CudaSlice<u32>,
    pub(crate) d_statement_values: CudaSlice<u32>,
    pub(crate) d_sum: CudaSlice<u32>,
    pub(crate) d_polys: Vec<Option<CudaSlice<u32>>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_accumulate_device_statements_with_device_gamma_into_async<EF>(
    d_weights: &mut CudaSlice<u32>,
    d_base_sum: CudaSlice<u32>,
    d_gamma: CudaSlice<u32>,
    gamma_start_power: usize,
    statements: &[GpuSparseStatement],
    num_variables: usize,
    workspaces: Option<GpuDeviceStatementAccumulationWorkspaces>,
) -> Option<(CudaSlice<u32>, GpuDeviceStatementAccumulationGuard)>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION;
    let n_total = 1usize << num_variables;
    let total_values: usize = statements.iter().map(|statement| statement.values.len()).sum();

    if total_values == 0 {
        return Some((
            d_base_sum,
            GpuDeviceStatementAccumulationGuard {
                _d_base_sum: None,
                _d_gamma: d_gamma,
                _d_gamma_powers: None,
                _d_statement_values: None,
                _poly_buffers: Vec::new(),
            },
        ));
    }

    let has_workspaces = workspaces.is_some();
    let workspace_poly_count = workspaces.as_ref().map(|workspace| workspace.d_polys.len());
    if let Some(poly_count) = workspace_poly_count {
        if poly_count != statements.len() {
            return None;
        }
    }
    let (mut d_gamma_powers, mut d_statement_values, mut d_sum, mut d_polys) = match workspaces {
        Some(workspaces) => {
            if workspaces.d_gamma_powers.len() < (gamma_start_power + total_values) * dim {
                return None;
            }
            if workspaces.d_statement_values.len() < total_values * dim || workspaces.d_sum.len() < dim {
                return None;
            }
            (
                workspaces.d_gamma_powers,
                workspaces.d_statement_values,
                workspaces.d_sum,
                workspaces.d_polys,
            )
        }
        None => {
            let d_gamma_powers = g
                .sumcheck
                .extension_powers_device(&d_gamma, (gamma_start_power + total_values) as u32);
            let d_statement_values = g.stream.alloc_zeros::<u32>(total_values * dim).ok()?;
            let d_sum = g.stream.alloc_zeros::<u32>(dim).ok()?;
            let d_polys = std::iter::repeat_with(|| None).take(statements.len()).collect();
            (d_gamma_powers, d_statement_values, d_sum, d_polys)
        }
    };
    if d_gamma_powers.len() == 0 {
        return None;
    }
    if d_gamma_powers.len() < (gamma_start_power + total_values) * dim {
        return None;
    }
    if d_statement_values.len() < total_values * dim {
        return None;
    }
    if has_workspaces {
        g.sumcheck.extension_powers_device_into_async(
            &d_gamma,
            (gamma_start_power + total_values) as u32,
            &mut d_gamma_powers,
        );
    }
    let d_statement_scalars = d_gamma_powers.slice(gamma_start_power * dim..(gamma_start_power + total_values) * dim);

    let mut poly_buffers = Vec::with_capacity(statements.len());
    let mut scalar_idx = 0usize;
    for (statement_idx, statement) in statements.iter().enumerate() {
        if statement.total_num_variables != num_variables {
            return None;
        }
        let inner_n_vars = statement.point_len;
        let inner_n = 1usize << inner_n_vars;
        let d_poly = if statement.is_next {
            let mut d_poly = d_polys
                .get_mut(statement_idx)
                .and_then(Option::take)
                .unwrap_or_else(|| {
                    g.stream
                        .alloc_zeros::<u32>(inner_n * dim)
                        .expect("alloc device-statement next MLE workspace")
                });
            if d_poly.len() < inner_n * dim {
                return None;
            }
            g.sumcheck
                .next_mle_device_from_point_words_into_async(&statement.d_point_words, inner_n_vars, &mut d_poly);
            d_poly.into()
        } else if inner_n_vars == 0 {
            if dim != 5 {
                return None;
            }
            StatementPoly::SharedOne(&g.d_ext_one)
        } else {
            let mut d_poly = d_polys
                .get_mut(statement_idx)
                .and_then(Option::take)
                .unwrap_or_else(|| {
                    g.stream
                        .alloc_zeros::<u32>(inner_n * dim)
                        .expect("alloc device-statement eq workspace")
                });
            if d_poly.len() < inner_n * dim {
                return None;
            }
            g.sumcheck.eq_polynomial_device_from_flat_point_words_into_async(
                &statement.d_point_words,
                inner_n_vars as u32,
                &mut d_poly,
            );
            d_poly.into()
        };
        let poly_idx = poly_buffers.len();
        poly_buffers.push(d_poly);

        for GpuSparseValue { selector, d_value } in &statement.values {
            if d_value.len() != dim {
                return None;
            }
            g.sumcheck
                .memcpy_d2d_async(d_value, 0, &mut d_statement_values, scalar_idx * dim, dim);

            let scalar_start = scalar_idx * dim;
            let scalar_end = scalar_start + dim;
            let d_scalar = d_statement_scalars.slice(scalar_start..scalar_end);

            if *selector == 0 && inner_n == n_total {
                g.sumcheck.eq_accumulate_device_scalar_device(
                    d_weights,
                    &poly_buffers[poly_idx],
                    &d_scalar,
                    n_total as u32,
                );
            } else {
                let offset = (*selector * inner_n) as u32;
                g.sumcheck.eq_accumulate_offset_device_scalar_device(
                    d_weights,
                    &poly_buffers[poly_idx],
                    &d_scalar,
                    offset,
                    inner_n as u32,
                );
            }
            scalar_idx += 1;
        }
    }

    g.sumcheck.ext_dot_accumulate_into_async(
        &d_base_sum,
        &d_statement_values,
        &d_statement_scalars,
        total_values as u32,
        &mut d_sum,
    );
    Some((
        d_sum,
        GpuDeviceStatementAccumulationGuard {
            _d_base_sum: Some(d_base_sum),
            _d_gamma: d_gamma,
            _d_gamma_powers: Some(d_gamma_powers),
            _d_statement_values: Some(d_statement_values),
            _poly_buffers: poly_buffers,
        },
    ))
}

fn build_statement_poly_device<EF>(statement: &SparseStatement<EF>) -> Option<CudaSlice<u32>>
where
    EF: ExtensionField<PF<EF>>,
{
    let g = gpu_backend::gpu()?;
    let dim = EF::DIMENSION;
    if statement.is_next {
        let next_values = matrix_next_mle_folded::<EF>(&statement.point.0);
        let next_u32: &[u32] =
            unsafe { std::slice::from_raw_parts(next_values.as_ptr().cast::<u32>(), next_values.len() * dim) };
        return g.stream.memcpy_stod(next_u32).ok();
    }

    let point_ef_u32: Vec<[u32; 5]> = statement
        .point
        .0
        .iter()
        .map(|p| {
            let mut out = [0u32; 5];
            let coeffs = p.as_basis_coefficients_slice();
            for (j, c) in coeffs.iter().enumerate() {
                out[j] = unsafe { *(c as *const PF<EF> as *const u32) };
            }
            out
        })
        .collect();
    Some(g.sumcheck.eq_polynomial_device(&point_ef_u32))
}

fn flatten_sparse_values<EF>(statements: &[SparseStatement<EF>]) -> Vec<u32>
where
    EF: ExtensionField<PF<EF>>,
{
    statements
        .iter()
        .flat_map(|statement| {
            statement.values.iter().flat_map(|value| {
                value
                    .value
                    .as_basis_coefficients_slice()
                    .iter()
                    .map(|coeff| unsafe { *(coeff as *const PF<EF> as *const u32) })
            })
        })
        .collect()
}
