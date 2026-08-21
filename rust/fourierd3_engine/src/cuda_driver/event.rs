// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_uint;

use crate::cuda_driver::{CUevent, CudaDriver, Result, StreamRef, check};

const CU_EVENT_DEFAULT: c_uint = 0;
// Barrier-only event, not usable for `elapsed_since` timing.
const CU_EVENT_DISABLE_TIMING: c_uint = 2;

/// A CUDA event, destroyed on drop.
pub(crate) struct Event(CUevent);

unsafe impl Send for Event {}

impl Event {
    pub(crate) fn new() -> Result<Self> {
        Self::with_flags(CU_EVENT_DEFAULT)
    }

    /// An event with timing disabled — cheaper, for ordering/barrier use only.
    pub(crate) fn new_disable_timing() -> Result<Self> {
        Self::with_flags(CU_EVENT_DISABLE_TIMING)
    }

    fn with_flags(flags: c_uint) -> Result<Self> {
        let drv = CudaDriver::get();
        let mut event: CUevent = std::ptr::null_mut();
        check(
            unsafe { (drv.cuEventCreate)(&mut event, flags) },
            "cuEventCreate",
        )?;
        Ok(Event(event))
    }

    pub(crate) fn record(&self, stream: StreamRef) -> Result<()> {
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuEventRecord)(self.0, stream.raw()) },
            "cuEventRecord",
        )
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuEventSynchronize)(self.0) },
            "cuEventSynchronize",
        )
    }

    /// Milliseconds elapsed from `start` to this event (both must be timing-enabled).
    pub(crate) fn elapsed_since(&self, start: &Event) -> Result<f32> {
        let drv = CudaDriver::get();
        let mut ms: f32 = 0.0;
        check(
            unsafe { (drv.cuEventElapsedTime)(&mut ms, start.0, self.0) },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let drv = CudaDriver::get();
        let _ = unsafe { (drv.cuEventDestroy)(self.0) };
    }
}
