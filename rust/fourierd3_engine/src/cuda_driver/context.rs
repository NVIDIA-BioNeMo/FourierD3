// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::cuda_driver::{CUcontext, CudaDriver, Result, check};

/// A CUDA context handle. Non-owning and `Copy`: the contexts this crate hands
/// out are long-lived (the process-wide default, or a device's primary
/// context), so dropping a `Context` value frees nothing.
#[derive(Clone, Copy)]
pub(crate) struct Context(pub(crate) CUcontext);

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    #[cfg(test)]
    pub(crate) fn current() -> Context {
        let drv = CudaDriver::get();
        let mut ctx = std::ptr::null_mut();
        unsafe { (drv.cuCtxGetCurrent)(&mut ctx) };
        Context(ctx)
    }

    pub(crate) fn set_current(&self) -> Result<()> {
        let drv = CudaDriver::get();
        check(unsafe { (drv.cuCtxSetCurrent)(self.0) }, "cuCtxSetCurrent")
    }

    /// Wrap a raw context handle obtained elsewhere (an FFI boundary, a stream's
    /// owner, …). Lets callers holding a `CUcontext` use the safe API.
    pub(crate) fn from_raw(raw: CUcontext) -> Context {
        Context(raw)
    }

    pub(crate) fn raw(&self) -> CUcontext {
        self.0
    }
}

#[cfg(test)]
pub(crate) fn ensure_context() -> Result<()> {
    use std::sync::OnceLock;

    static CONTEXT: OnceLock<Result<usize>> = OnceLock::new();
    let address = CONTEXT
        .get_or_init(|| {
            let driver = CudaDriver::get();
            check(unsafe { (driver.cuInit)(0) }, "cuInit")?;
            let mut device = 0;
            check(
                unsafe { (driver.cuDeviceGet)(&mut device, 0) },
                "cuDeviceGet",
            )?;
            let mut context = std::ptr::null_mut();
            check(
                unsafe { (driver.cuCtxCreate_v2)(&mut context, 0, device) },
                "cuCtxCreate_v2",
            )?;
            Ok(context as usize)
        })
        .clone()?;
    Context(address as CUcontext).set_current()
}

/// Makes a context current for its lifetime and restores the previously-current
/// one on drop. A no-op (nothing to restore) when the target is already current.
pub(crate) struct ContextGuard {
    prev: CUcontext,
    restore: bool,
}

impl ContextGuard {
    pub(crate) fn enter(ctx: Context) -> Result<Self> {
        let drv = CudaDriver::get();
        let mut prev: CUcontext = std::ptr::null_mut();
        check(
            unsafe { (drv.cuCtxGetCurrent)(&mut prev) },
            "cuCtxGetCurrent",
        )?;
        if prev == ctx.0 {
            return Ok(Self {
                prev,
                restore: false,
            });
        }
        ctx.set_current()?;
        Ok(Self {
            prev,
            restore: true,
        })
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        if self.restore {
            let drv = CudaDriver::get();
            let _ = unsafe { (drv.cuCtxSetCurrent)(self.prev) };
        }
    }
}

/// Run `f` with `ctx` current, restoring the previous context afterward. A null
/// `ctx` runs `f` as-is.
pub(crate) fn with_current_context<R>(ctx: CUcontext, f: impl FnOnce() -> Result<R>) -> Result<R> {
    if ctx.is_null() {
        return f();
    }
    let _guard = ContextGuard::enter(Context(ctx))?;
    f()
}
