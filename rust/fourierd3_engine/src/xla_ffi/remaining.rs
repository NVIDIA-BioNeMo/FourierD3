// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::xla_ffi::buffer::AnyBuffer;
use crate::xla_ffi::sys::{XLA_FFI_Buffer, XLA_FFI_CallFrame};

pub(crate) struct RemainingArgs<'cf> {
    cf: &'cf XLA_FFI_CallFrame,
    start: usize,
    end: usize,
}

impl<'cf> RemainingArgs<'cf> {
    pub(crate) fn new(cf: &'cf XLA_FFI_CallFrame, start: usize, end: usize) -> Self {
        Self { cf, start, end }
    }

    pub(crate) fn iter(&self) -> RemainingArgsIter<'cf> {
        RemainingArgsIter {
            cf: self.cf,
            i: self.start,
            end: self.end,
        }
    }
}

pub(crate) struct RemainingArgsIter<'cf> {
    cf: &'cf XLA_FFI_CallFrame,
    i: usize,
    end: usize,
}

impl<'cf> Iterator for RemainingArgsIter<'cf> {
    type Item = AnyBuffer<'cf>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.end {
            return None;
        }
        let raw = unsafe { &*(*self.cf.args.args.add(self.i) as *const XLA_FFI_Buffer) };
        self.i += 1;
        Some(AnyBuffer::from_raw(raw))
    }
}

pub(crate) struct RemainingRets<'cf> {
    cf: &'cf XLA_FFI_CallFrame,
    start: usize,
    end: usize,
}

impl<'cf> RemainingRets<'cf> {
    pub(crate) fn new(cf: &'cf XLA_FFI_CallFrame, start: usize, end: usize) -> Self {
        Self { cf, start, end }
    }

    pub(crate) fn len(&self) -> usize {
        self.end - self.start
    }

    pub(crate) fn iter(&self) -> RemainingRetsIter<'cf> {
        RemainingRetsIter {
            cf: self.cf,
            i: self.start,
            end: self.end,
        }
    }
}

pub(crate) struct RemainingRetsIter<'cf> {
    cf: &'cf XLA_FFI_CallFrame,
    i: usize,
    end: usize,
}

impl<'cf> Iterator for RemainingRetsIter<'cf> {
    type Item = AnyBuffer<'cf>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.end {
            return None;
        }
        let raw = unsafe { &*(*self.cf.rets.rets.add(self.i) as *const XLA_FFI_Buffer) };
        self.i += 1;
        Some(AnyBuffer::from_raw(raw))
    }
}
