// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CString;

use crate::xla_ffi::sys::{
    XLA_FFI_Api, XLA_FFI_Error, XLA_FFI_Error_Code, XLA_FFI_Error_Code_INTERNAL,
    XLA_FFI_Error_Code_INVALID_ARGUMENT, XLA_FFI_Error_Create_Args,
    XLA_FFI_Error_Create_Args_STRUCT_SIZE,
};

#[derive(Debug, Clone)]
pub(crate) struct Error {
    pub code: XLA_FFI_Error_Code,
    pub message: String,
}

impl Error {
    pub(crate) fn new(code: XLA_FFI_Error_Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn invalid_argument(msg: impl Into<String>) -> Self {
        Self::new(XLA_FFI_Error_Code_INVALID_ARGUMENT, msg)
    }

    pub(crate) fn internal(msg: impl Into<String>) -> Self {
        Self::new(XLA_FFI_Error_Code_INTERNAL, msg)
    }

    pub(crate) unsafe fn into_xla(self, api: *const XLA_FFI_Api) -> *mut XLA_FFI_Error {
        let c = match CString::new(self.message.clone()) {
            Ok(c) => c,
            Err(_) => CString::new("FFI handler error message contained a NUL byte").unwrap(),
        };
        let mut args = XLA_FFI_Error_Create_Args {
            struct_size: XLA_FFI_Error_Create_Args_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            message: c.as_ptr(),
            errc: self.code,
        };
        let create = match unsafe { (*api).XLA_FFI_Error_Create } {
            Some(f) => f,
            None => return std::ptr::null_mut(),
        };
        unsafe { create(&mut args) }
    }
}

pub(crate) type Result<T> = std::result::Result<T, Error>;
