// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{CStr, c_char};
use std::fmt;

use crate::cuda_driver::{CUDA_SUCCESS, CUresult, CudaDriver};

/// `CUDA_ERROR_INVALID_VALUE` — the code [`Error::other`] reports for failures
/// that originate in this crate rather than the driver (e.g. a NUL in a name).
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;

/// A failed CUDA driver call: the result code, the call site, and the driver's
/// decoded message.
#[derive(Clone, Debug)]
pub(crate) struct Error {
    pub code: CUresult,
    pub context: String,
    pub message: String,
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

impl Error {
    fn from_code(code: CUresult, context: &str) -> Self {
        Self {
            code,
            context: context.to_string(),
            message: decode(code),
        }
    }

    /// An error raised by this crate, not by a driver call.
    pub(crate) fn other(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: CUDA_ERROR_INVALID_VALUE,
            context: context.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "{}: CUDA error {}", self.context, self.code)
        } else {
            write!(
                f,
                "{}: CUDA error {}: {}",
                self.context, self.code, self.message
            )
        }
    }
}

impl std::error::Error for Error {}

/// `Ok(())` on `CUDA_SUCCESS`, otherwise an [`Error`] tagged with `context` and
/// the driver's decoded message. This is the one checkpoint every safe wrapper
/// funnels raw `CUresult`s through.
pub(crate) fn check(code: CUresult, context: &str) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::from_code(code, context))
    }
}

fn decode(code: CUresult) -> String {
    let drv = CudaDriver::get();
    let mut s: *const c_char = std::ptr::null();
    let _ = unsafe { (drv.cuGetErrorString)(code, &mut s) };
    if s.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }
}
