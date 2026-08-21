// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use crate::cuda_driver::module::Function;
use crate::cuda_driver::{CUdeviceptr, CUstream, Context, CudaDriver, Result, check};

/// Grid/block/shared-memory shape for a kernel launch or graph kernel node.
#[derive(Clone, Copy)]
pub(crate) struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
}

/// A *borrowed* stream — every stream operation lives here. Externally-owned
/// streams (handed in by XLA/PJRT) are used through `StreamRef::from_raw`; only
/// streams this crate created own their handle (see [`Stream`]).
#[derive(Clone, Copy)]
pub(crate) struct StreamRef(pub(crate) CUstream);

unsafe impl Send for StreamRef {}

impl StreamRef {
    pub(crate) fn from_raw(raw: CUstream) -> Self {
        StreamRef(raw)
    }

    pub(crate) fn raw(&self) -> CUstream {
        self.0
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuStreamSynchronize)(self.0) },
            "cuStreamSynchronize",
        )
    }

    /// The context this stream was created in.
    pub(crate) fn context(&self) -> Result<Context> {
        let drv = CudaDriver::get();
        let mut ctx = std::ptr::null_mut();
        check(
            unsafe { (drv.cuStreamGetCtx)(self.0, &mut ctx) },
            "cuStreamGetCtx",
        )?;
        Ok(Context(ctx))
    }

    /// `cuLaunchKernel`. `params` is the array of pointers-to-arguments CUDA
    /// expects; it only needs to outlive the (queued) call.
    pub(crate) fn launch(
        &self,
        func: Function,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let drv = CudaDriver::get();
        let [gx, gy, gz] = cfg.grid;
        let [bx, by, bz] = cfg.block;
        let pp = if params.is_empty() {
            std::ptr::null_mut()
        } else {
            params.as_mut_ptr()
        };
        check(
            unsafe {
                (drv.cuLaunchKernel)(
                    func.raw(),
                    gx,
                    gy,
                    gz,
                    bx,
                    by,
                    bz,
                    cfg.shared_mem,
                    self.0,
                    pp,
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// # Safety
    /// `src` must point to at least `nbytes` host bytes that stay valid until
    /// the copy completes on the stream.
    pub(crate) unsafe fn memcpy_h2d(
        &self,
        dst: CUdeviceptr,
        src: *const c_void,
        nbytes: usize,
    ) -> Result<()> {
        if nbytes == 0 {
            return Ok(());
        }
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuMemcpyHtoDAsync_v2)(dst, src, nbytes, self.0) },
            "cuMemcpyHtoDAsync_v2",
        )
    }
}
