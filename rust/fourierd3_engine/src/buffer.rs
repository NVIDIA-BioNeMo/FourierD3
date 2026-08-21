// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversion of the `(name, dtype, ic, extents, elem_size)` buffer tuples the
//! Python lowering builds into [`crate::kernel_compiler::buffer::Buffer`].

use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyAny, PyAnyMethods, PyList, PyListMethods, PyTuple, PyTupleMethods};
use pyo3::{Bound, PyResult};

use crate::kernel_compiler::buffer::Buffer;

pub(crate) struct OwnedBuffer {
    name: String,
    dtype: i32,
    ic: Vec<i32>,
    extents: Vec<i64>,
    elem_size: i64,
}

impl OwnedBuffer {
    pub(crate) fn from_tuple(t: &Bound<'_, PyTuple>) -> PyResult<Self> {
        if t.len() < 5 {
            return Err(PyRuntimeError::new_err(format!(
                "buffer tuple must have 5 elements, got {}",
                t.len()
            )));
        }
        let name: String = t.get_item(0)?.extract()?;
        let dtype: i32 = t.get_item(1)?.extract()?;
        let ic: Vec<i32> = t.get_item(2)?.extract()?;
        let extents: Vec<i64> = t.get_item(3)?.extract()?;
        let elem_size: i64 = t.get_item(4)?.extract()?;
        if ic.len() != extents.len() {
            return Err(PyRuntimeError::new_err(format!(
                "ic ({}) and extents ({}) length mismatch for buffer {}",
                ic.len(),
                extents.len(),
                name
            )));
        }
        Ok(Self {
            name,
            dtype,
            ic,
            extents,
            elem_size,
        })
    }

    pub(crate) fn into_buffer(self) -> Result<Buffer, String> {
        Ok(Buffer {
            name: self.name,
            dtype: fourierd3_engine::dtype::Dtype::from_id(self.dtype)?,
            ic: self.ic,
            extents: self.extents,
            elem_size: self.elem_size,
        })
    }
}

pub(crate) fn owned_buffers_from_pylist(obj: &Bound<'_, PyAny>) -> PyResult<Vec<OwnedBuffer>> {
    let list: Bound<'_, PyList> = obj.downcast::<PyList>()?.clone();
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let t = item.downcast::<PyTuple>()?;
        out.push(OwnedBuffer::from_tuple(&t.clone())?);
    }
    Ok(out)
}

/// Converts a Python list of buffer tuples straight through to
/// [`Buffer`], reporting the first conversion failure.
pub(crate) fn buffers_from_pylist(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Buffer>> {
    owned_buffers_from_pylist(obj)?
        .into_iter()
        .map(|buffer| buffer.into_buffer().map_err(PyRuntimeError::new_err))
        .collect()
}
