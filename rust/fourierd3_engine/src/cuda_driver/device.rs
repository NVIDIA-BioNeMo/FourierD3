// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::os::raw::c_int;

use crate::cuda_driver::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, CUdevice, CudaDriver, Result, check,
};

/// A CUDA device ordinal. Cheap, `Copy`, multi-GPU-ready: every query targets
/// this specific device rather than implicitly the current one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Device(pub(crate) CUdevice);

impl Device {
    /// The device backing the current context.
    pub(crate) fn current() -> Device {
        let drv = CudaDriver::get();
        let mut dev: CUdevice = 0;
        unsafe { (drv.cuCtxGetDevice)(&mut dev) };
        Device(dev)
    }

    pub(crate) fn attribute(&self, attr: c_int) -> Result<i32> {
        let drv = CudaDriver::get();
        let mut v: c_int = 0;
        check(
            unsafe { (drv.cuDeviceGetAttribute)(&mut v, attr, self.0) },
            "cuDeviceGetAttribute",
        )?;
        Ok(v)
    }

    pub(crate) fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok((
            self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        ))
    }

    /// Compute capability as `10 * major + minor` (e.g. 90 for sm_90).
    pub(crate) fn sm_arch(&self) -> Result<i32> {
        let (major, minor) = self.compute_capability()?;
        Ok(10 * major + minor)
    }

    pub(crate) fn max_smem_optin(&self) -> Result<i32> {
        self.attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    }
}
