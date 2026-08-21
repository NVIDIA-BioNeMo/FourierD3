// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `fourierd3._engine._extension` module.

// pyo3 0.22's `#[pyfunction]` macro expands to code that trips
// `clippy::useless_conversion` on `?`-returning bodies, and (under
// edition 2024) `unsafe_op_in_unsafe_fn` because the macro injects
// unsafe calls into the body of the user-marked function without
// wrapping them. Both lints fire on macro-generated code where
// `#[allow]` on the function doesn't reach, so suppress at crate scope.
#![allow(clippy::useless_conversion, unsafe_op_in_unsafe_fn)]

extern crate self as fourierd3_engine;

use pyo3::types::PyModule;
use pyo3::{Bound, PyResult, pymodule};

mod artifact_cache;
mod buffer;
mod compiler;
mod cuda_compiler;
mod cuda_driver;
mod dtype;
mod dynamic_library;
mod execution_plan;
mod executor;
pub mod ir;
mod kernel_compiler;
mod periodic_scatter;
mod plan_executor;
mod spectral_map;
mod xla_ffi;

#[cfg(test)]
mod tests {
    mod execution_plan_validate;
    mod plan_executor_gpu_smoke;
}

pub use fourierd3_macros::cuda;

#[pymodule]
fn _extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    compiler::register(m)?;
    executor::register(m)?;
    periodic_scatter::register(m)?;
    spectral_map::register(m)?;
    Ok(())
}
