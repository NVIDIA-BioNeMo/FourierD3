// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles a periodic-scatter kernel set into serialized execution-plan bytes.

use crate::kernel_compiler::periodic_scatter::{ScatterPlanRequest, compile_scatter_plan};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyAny, PyBytes, PyModule, PyModuleMethods, PyTuple};
use pyo3::{Bound, PyObject, PyResult, Python, pyfunction, wrap_pyfunction};

use crate::buffer::{OwnedBuffer, buffers_from_pylist};

#[allow(clippy::too_many_arguments)]
fn scatter_plan_bytes(
    device_fn_ir: &str,
    support: &[i16],
    cell_idx: &Bound<'_, PyTuple>,
    grid_in: &Bound<'_, PyAny>,
    nongrid_in: &Bound<'_, PyAny>,
    idx_bufs: &Bound<'_, PyAny>,
    grid_out: &Bound<'_, PyAny>,
    nongrid_out: &Bound<'_, PyAny>,
    batch_sizes: &[i64],
    n_backend_arrays: i32,
    cart_order: i32,
    cart_mo: i32,
    grid_shape: [i64; 3],
    cell_grid_shape: Option<[i64; 3]>,
    opts: Option<Vec<String>>,
    compile_budget_ms: f64,
) -> PyResult<Vec<u8>> {
    let _ = compile_budget_ms;
    let request = ScatterPlanRequest {
        device_fn_ir: device_fn_ir.to_string(),
        support: support.to_vec(),
        cell_idx: OwnedBuffer::from_tuple(cell_idx)?
            .into_buffer()
            .map_err(PyRuntimeError::new_err)?,
        grid_in: buffers_from_pylist(grid_in)?,
        nongrid_in: buffers_from_pylist(nongrid_in)?,
        idx_bufs: buffers_from_pylist(idx_bufs)?,
        grid_out: buffers_from_pylist(grid_out)?,
        nongrid_out: buffers_from_pylist(nongrid_out)?,
        batch_sizes: batch_sizes.to_vec(),
        n_backend_arrays: n_backend_arrays.max(0) as usize,
        cartesian: (cart_order > 0).then_some((cart_order, cart_mo)),
        grid_shape,
        cell_grid_shape,
    };
    let options = opts.unwrap_or_default();
    let plan = compile_scatter_plan(request, &options).map_err(PyRuntimeError::new_err)?;
    crate::execution_plan::serialize(&plan)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (
    device_fn_ir,
    support,
    cell_idx,
    grid_in,
    nongrid_in,
    idx_bufs,
    grid_out,
    nongrid_out,
    batch_sizes,
    n_backend_arrays,
    cart_order,
    cart_mo,
    grid_shape,
    cell_grid_shape=None,
    opts=None,
    compile_budget_ms=0.0,
))]
#[allow(clippy::too_many_arguments)]
fn compile_periodic_scatter_to_bytes(
    py: Python<'_>,
    device_fn_ir: &str,
    support: Vec<i16>,
    cell_idx: &Bound<'_, PyTuple>,
    grid_in: &Bound<'_, PyAny>,
    nongrid_in: &Bound<'_, PyAny>,
    idx_bufs: &Bound<'_, PyAny>,
    grid_out: &Bound<'_, PyAny>,
    nongrid_out: &Bound<'_, PyAny>,
    batch_sizes: Vec<i64>,
    n_backend_arrays: i32,
    cart_order: i32,
    cart_mo: i32,
    grid_shape: [i64; 3],
    cell_grid_shape: Option<[i64; 3]>,
    opts: Option<Vec<String>>,
    compile_budget_ms: f64,
) -> PyResult<PyObject> {
    let bytes = scatter_plan_bytes(
        device_fn_ir,
        &support,
        cell_idx,
        grid_in,
        nongrid_in,
        idx_bufs,
        grid_out,
        nongrid_out,
        &batch_sizes,
        n_backend_arrays,
        cart_order,
        cart_mo,
        grid_shape,
        cell_grid_shape,
        opts,
        compile_budget_ms,
    )?;
    Ok(PyBytes::new_bound(py, &bytes).into_any().unbind())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_periodic_scatter_to_bytes, m)?)?;
    Ok(())
}
