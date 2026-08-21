// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA search paths and the error the compile entry points raise when a
//! device cannot run the requested kernel shape at all.

use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyErr, PyResult, pyfunction, wrap_pyfunction};

use std::path::PathBuf;

// `unexpected_cfgs` suppression: pyo3 0.22's `create_exception!`
// expansion includes a `cfg(gil-refs)` gate that the host crate's
// build doesn't declare; the cfg works at runtime but trips the lint
// gate under `-D warnings`.
#[allow(unexpected_cfgs)]
mod kernel_infeasible {
    use super::PyRuntimeError;
    pyo3::create_exception!(
        fourierd3._engine._extension,
        KernelInfeasibleError,
        PyRuntimeError
    );
}
pub(crate) use kernel_infeasible::KernelInfeasibleError;

/// Maps a compiler error onto the Python exception that says whether the
/// device could ever run this kernel shape.
pub(crate) fn compile_error(error: String) -> PyErr {
    if crate::kernel_compiler::infeasibility::is_infeasible(&error) {
        KernelInfeasibleError::new_err(error)
    } else {
        PyRuntimeError::new_err(error)
    }
}

#[pyfunction]
fn add_include_dir(path: &str) -> PyResult<()> {
    crate::kernel_compiler::cuda_toolchain::add_include_dir(PathBuf::from(path))
        .map_err(PyRuntimeError::new_err)
}

#[pyfunction]
fn add_lib_dir(path: &str) -> PyResult<()> {
    crate::kernel_compiler::cuda_toolchain::add_lib_dir(PathBuf::from(path))
        .map_err(PyRuntimeError::new_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_include_dir, m)?)?;
    m.add_function(wrap_pyfunction!(add_lib_dir, m)?)?;
    m.add(
        "KernelInfeasibleError",
        m.py().get_type_bound::<KernelInfeasibleError>(),
    )?;
    Ok(())
}
