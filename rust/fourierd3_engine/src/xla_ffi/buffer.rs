// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::xla_ffi::dtype::dtype_size;
use crate::xla_ffi::error::{Error, Result};
use crate::xla_ffi::sys::XLA_FFI_Buffer;
#[cfg(test)]
use crate::xla_ffi::sys::{XLA_FFI_Buffer_STRUCT_SIZE, XLA_FFI_DataType_F32};

pub(crate) struct AnyBuffer<'a> {
    raw: &'a XLA_FFI_Buffer,
}

impl<'a> AnyBuffer<'a> {
    pub(crate) fn from_raw(raw: &'a XLA_FFI_Buffer) -> Self {
        Self { raw }
    }

    pub(crate) fn dims(&self) -> &'a [i64] {
        // SAFETY: XLA guarantees `dims` is valid for `rank` entries for
        // the call's lifetime. `rank == 0` would dereference a null
        // pointer, so handle that case explicitly.
        if self.raw.rank == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.raw.dims, self.raw.rank as usize) }
        }
    }

    pub(crate) fn data(&self) -> *mut std::ffi::c_void {
        self.raw.data
    }

    pub(crate) fn num_elements(&self) -> i64 {
        self.dims().iter().product()
    }

    pub(crate) fn size_bytes(&self) -> Result<usize> {
        let esz = dtype_size(self.raw.dtype).ok_or_else(|| {
            Error::invalid_argument(format!("unsupported dtype {}", self.raw.dtype))
        })?;
        if self.dims().iter().any(|&d| d < 0) {
            return Err(Error::invalid_argument(format!(
                "buffer has negative dimension(s) {:?}",
                self.dims()
            )));
        }
        let n = self.num_elements();
        Ok(n as usize * esz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_buffer(dims: &mut [i64]) -> XLA_FFI_Buffer {
        XLA_FFI_Buffer {
            struct_size: XLA_FFI_Buffer_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            dtype: XLA_FFI_DataType_F32,
            data: std::ptr::null_mut(),
            rank: dims.len() as i64,
            dims: if dims.is_empty() {
                std::ptr::null_mut()
            } else {
                dims.as_mut_ptr()
            },
        }
    }

    #[test]
    fn scalar_buffer_has_one_element() {
        let mut dims = [];
        let raw = raw_buffer(&mut dims);
        let b = AnyBuffer::from_raw(&raw);
        assert_eq!(b.num_elements(), 1);
        assert_eq!(b.size_bytes().unwrap(), 4);
    }

    #[test]
    fn zero_extent_buffer_has_zero_bytes() {
        let mut dims = [2, 0, 3];
        let raw = raw_buffer(&mut dims);
        let b = AnyBuffer::from_raw(&raw);
        assert_eq!(b.num_elements(), 0);
        assert_eq!(b.size_bytes().unwrap(), 0);
    }
}
