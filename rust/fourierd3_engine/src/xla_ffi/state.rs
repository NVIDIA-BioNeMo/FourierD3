// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use crate::xla_ffi::sys::{
    XLA_FFI_AttrType_SCALAR, XLA_FFI_CallFrame, XLA_FFI_Scalar, XLA_FFI_Stream_Get_Args,
    XLA_FFI_Stream_Get_Args_STRUCT_SIZE,
};

use crate::xla_ffi::dtype::Dtype;
use crate::xla_ffi::error::{Error, Result};

#[derive(Copy, Clone)]
pub(crate) struct Stream(pub *mut c_void);

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

pub(crate) struct DecodeState<'cf> {
    cf: &'cf XLA_FFI_CallFrame,
    arg_cursor: usize,
    ret_cursor: usize,
}

impl<'cf> DecodeState<'cf> {
    pub(crate) fn new(cf: &'cf XLA_FFI_CallFrame) -> Self {
        Self {
            cf,
            arg_cursor: 0,
            ret_cursor: 0,
        }
    }

    pub(crate) fn stream(&self) -> Result<Stream> {
        let mut args = XLA_FFI_Stream_Get_Args {
            struct_size: XLA_FFI_Stream_Get_Args_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            ctx: self.cf.ctx,
            stream: std::ptr::null_mut(),
        };
        let getter = unsafe { (*self.cf.api).XLA_FFI_Stream_Get }
            .ok_or_else(|| Error::internal("XLA_FFI_Stream_Get is null"))?;
        let err = unsafe { getter(&mut args) };
        if err.is_null() {
            Ok(Stream(args.stream))
        } else {
            Err(Error::internal("XLA_FFI_Stream_Get failed"))
        }
    }

    fn n_args(&self) -> usize {
        self.cf.args.size as usize
    }

    fn n_rets(&self) -> usize {
        self.cf.rets.size as usize
    }

    pub(crate) fn remaining_args(&mut self) -> crate::xla_ffi::RemainingArgs<'cf> {
        let start = self.arg_cursor;
        self.arg_cursor = self.n_args();
        crate::xla_ffi::RemainingArgs::new(self.cf, start, self.n_args())
    }

    pub(crate) fn remaining_rets(&mut self) -> crate::xla_ffi::RemainingRets<'cf> {
        let start = self.ret_cursor;
        self.ret_cursor = self.n_rets();
        crate::xla_ffi::RemainingRets::new(self.cf, start, self.n_rets())
    }

    fn find_attr(&self, name: &str) -> Result<usize> {
        let n = self.cf.attrs.size as usize;
        for i in 0..n {
            let span = unsafe { *self.cf.attrs.names.add(i) };
            let s = unsafe { std::slice::from_raw_parts((*span).ptr as *const u8, (*span).len) };
            if s == name.as_bytes() {
                return Ok(i);
            }
        }
        Err(Error::invalid_argument(format!(
            "missing required attr `{name}`"
        )))
    }

    pub(crate) fn attr_scalar<T: Dtype>(&self, name: &str) -> Result<T> {
        let i = self.find_attr(name)?;
        let ty = unsafe { *self.cf.attrs.types.add(i) };
        if ty != XLA_FFI_AttrType_SCALAR {
            return Err(Error::invalid_argument(format!(
                "attr `{name}`: expected SCALAR, got type tag {ty}"
            )));
        }
        let sc = unsafe { &*(*self.cf.attrs.attrs.add(i) as *const XLA_FFI_Scalar) };
        if sc.dtype != T::TAG {
            return Err(Error::invalid_argument(format!(
                "attr `{name}`: expected scalar of {} (tag {}), got tag {}",
                T::NAME,
                T::TAG,
                sc.dtype
            )));
        }
        // SAFETY: XLA owns the scalar payload; we read by value.
        Ok(unsafe { *(sc.value as *const T) })
    }
}
