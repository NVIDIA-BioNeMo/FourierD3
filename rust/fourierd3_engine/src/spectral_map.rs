// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles a fused FFT / reciprocal-space map / inverse FFT kernel set into
//! serialized execution-plan bytes.

use crate::kernel_compiler::spectral_map::{SpectralMapPipeline, SpectralMapSpec};
use fourierd3_engine::dtype::Dtype;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyAny, PyBytes, PyModule, PyModuleMethods};
use pyo3::{Bound, PyObject, PyResult, Python, pyfunction, wrap_pyfunction};

use crate::buffer::owned_buffers_from_pylist;
use crate::compiler::compile_error;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn spectral_map_plan_bytes(
    fft_lengths: [i64; 3],
    precision: i32,
    sm: i32,
    batch_shape: Vec<i64>,
    grid_in_bufs: &Bound<'_, PyAny>,
    grid_inner_sizes: Vec<u32>,
    input_signs: Vec<i32>,
    n_grid_out: usize,
    output_inner_sizes: Vec<u32>,
    output_signs: Vec<i32>,
    aux_bufs: &Bound<'_, PyAny>,
    aux_dtypes: Vec<i32>,
    aux_inner_shapes: Vec<Vec<u32>>,
    aux_output_dtypes: Vec<i32>,
    aux_output_inner_shapes: Vec<Vec<u32>>,
    device_fn_ir: &str,
    opts: Option<Vec<String>>,
    compile_budget_ms: f64,
) -> PyResult<Vec<u8>> {
    let grid_in_owned = owned_buffers_from_pylist(grid_in_bufs)?;
    let aux_bufs_owned = owned_buffers_from_pylist(aux_bufs)?;
    if grid_in_owned.len() != grid_inner_sizes.len() || grid_in_owned.len() != input_signs.len() {
        return Err(PyRuntimeError::new_err(
            "grid_in_bufs / grid_inner_sizes / input_signs length mismatch",
        ));
    }
    if output_inner_sizes.len() != n_grid_out || output_signs.len() != n_grid_out {
        return Err(PyRuntimeError::new_err(
            "output_inner_sizes / output_signs length must equal n_grid_out",
        ));
    }
    if aux_bufs_owned.len() != aux_dtypes.len() || aux_bufs_owned.len() != aux_inner_shapes.len() {
        return Err(PyRuntimeError::new_err(
            "aux_bufs / aux_dtypes / aux_inner_shapes length mismatch",
        ));
    }
    if aux_output_dtypes.len() != aux_output_inner_shapes.len() {
        return Err(PyRuntimeError::new_err(
            "aux_output_dtypes / aux_output_inner_shapes length mismatch",
        ));
    }

    let precision = Dtype::from_id(precision).map_err(PyRuntimeError::new_err)?;
    if !matches!(precision, Dtype::F32 | Dtype::F64) {
        return Err(PyRuntimeError::new_err(format!(
            "FFT precision must be f32 or f64, got {precision:?}"
        )));
    }
    if sm <= 0 {
        return Err(PyRuntimeError::new_err(format!("non-positive SM {sm}")));
    }
    for (label, length) in ["nx", "ny", "nz"].into_iter().zip(fft_lengths) {
        if length <= 0 {
            return Err(PyRuntimeError::new_err(format!(
                "non-positive {label} = {length}"
            )));
        }
    }
    let problem = SpectralMapSpec {
        fft_lengths: fft_lengths.map(|length| length as u32),
        precision,
        sm: sm as u32,
        n_grid_in: grid_in_owned.len() as u32,
        n_grid_out: n_grid_out as u32,
        n_aux: aux_bufs_owned.len() as u32,
        n_aux_out: aux_output_dtypes.len() as u32,
        batch_shape,
        grid_inner_sizes,
        output_inner_sizes,
        input_signs,
        output_signs,
        grid_in_bufs: grid_in_owned
            .into_iter()
            .map(|buffer| buffer.into_buffer())
            .collect::<Result<_, _>>()
            .map_err(PyRuntimeError::new_err)?,
        aux_bufs: aux_bufs_owned
            .into_iter()
            .map(|buffer| buffer.into_buffer())
            .collect::<Result<_, _>>()
            .map_err(PyRuntimeError::new_err)?,
        aux_inner_shapes,
        aux_dtypes: aux_dtypes
            .into_iter()
            .map(Dtype::from_id)
            .collect::<Result<_, _>>()
            .map_err(PyRuntimeError::new_err)?,
        aux_output_inner_shapes,
        aux_output_dtypes: aux_output_dtypes
            .into_iter()
            .map(Dtype::from_id)
            .collect::<Result<_, _>>()
            .map_err(PyRuntimeError::new_err)?,
    };
    let options = opts.unwrap_or_default();
    let budget = (compile_budget_ms > 0.0).then_some(compile_budget_ms);
    SpectralMapPipeline::emit(
        &problem,
        device_fn_ir,
        &String::from("fft"),
        &options,
        budget,
    )
    .and_then(|pipeline| pipeline.to_plan())
    .and_then(|plan| crate::execution_plan::serialize(&plan).map_err(|e| e.to_string()))
    .map_err(compile_error)
}

#[pyfunction]
#[pyo3(signature = (
    fft_lengths,
    precision,
    sm,
    batch_shape,
    grid_in_bufs,
    grid_inner_sizes,
    input_signs,
    n_grid_out,
    output_inner_sizes,
    output_signs,
    aux_bufs,
    aux_dtypes,
    aux_inner_shapes,
    aux_output_dtypes,
    aux_output_inner_shapes,
    device_fn_ir,
    opts=None,
    compile_budget_ms=0.0,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn compile_spectral_map_to_bytes(
    py: Python<'_>,
    fft_lengths: [i64; 3],
    precision: i32,
    sm: i32,
    batch_shape: Vec<i64>,
    grid_in_bufs: &Bound<'_, PyAny>,
    grid_inner_sizes: Vec<u32>,
    input_signs: Vec<i32>,
    n_grid_out: usize,
    output_inner_sizes: Vec<u32>,
    output_signs: Vec<i32>,
    aux_bufs: &Bound<'_, PyAny>,
    aux_dtypes: Vec<i32>,
    aux_inner_shapes: Vec<Vec<u32>>,
    aux_output_dtypes: Vec<i32>,
    aux_output_inner_shapes: Vec<Vec<u32>>,
    device_fn_ir: &str,
    opts: Option<Vec<String>>,
    compile_budget_ms: f64,
) -> PyResult<PyObject> {
    let bytes = spectral_map_plan_bytes(
        fft_lengths,
        precision,
        sm,
        batch_shape,
        grid_in_bufs,
        grid_inner_sizes,
        input_signs,
        n_grid_out,
        output_inner_sizes,
        output_signs,
        aux_bufs,
        aux_dtypes,
        aux_inner_shapes,
        aux_output_dtypes,
        aux_output_inner_shapes,
        device_fn_ir,
        opts,
        compile_budget_ms,
    )?;
    Ok(PyBytes::new_bound(py, &bytes).into_any().unbind())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_spectral_map_to_bytes, m)?)?;
    Ok(())
}
