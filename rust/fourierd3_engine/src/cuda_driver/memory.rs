// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;

use crate::cuda_driver::context::with_current_context;
use crate::cuda_driver::{CUcontext, CUdeviceptr, CudaDriver, Result, check};

/// An owned device allocation of `len` elements of `T`, freed on drop. Carries
/// the context it was allocated in, so it frees correctly even when dropped on
/// a thread with a different current context (and is multi-GPU-safe).
pub(crate) struct DeviceBuffer<T: Copy> {
    ptr: CUdeviceptr,
    #[cfg(test)]
    len: usize,
    ctx: CUcontext,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy> Send for DeviceBuffer<T> {}
// `&DeviceBuffer` only reads (device→host copies, pointer/len accessors), so
// sharing it across threads is sound.
unsafe impl<T: Copy> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate `len` elements in `ctx`. The caller chooses the context
    /// explicitly (pass `Context::current().raw()` for the current one).
    pub(crate) fn alloc(ctx: CUcontext, len: usize) -> Result<Self> {
        let mut ptr: CUdeviceptr = 0;
        if len != 0 {
            let drv = CudaDriver::get();
            with_current_context(ctx, || {
                check(
                    unsafe { (drv.cuMemAlloc_v2)(&mut ptr, len * std::mem::size_of::<T>()) },
                    "cuMemAlloc_v2",
                )
            })?;
        }
        Ok(Self {
            ptr,
            #[cfg(test)]
            len,
            ctx,
            _marker: PhantomData,
        })
    }

    pub(crate) fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    #[cfg(test)]
    pub(crate) fn to_host(&self) -> Result<Vec<T>> {
        let mut host = Vec::with_capacity(self.len);
        if self.len != 0 {
            let driver = CudaDriver::get();
            with_current_context(self.ctx, || {
                check(
                    unsafe {
                        (driver.cuMemcpyDtoHAsync_v2)(
                            host.as_mut_ptr() as *mut _,
                            self.ptr,
                            self.len * std::mem::size_of::<T>(),
                            std::ptr::null_mut(),
                        )
                    },
                    "cuMemcpyDtoHAsync_v2",
                )?;
                check(
                    unsafe { (driver.cuStreamSynchronize)(std::ptr::null_mut()) },
                    "cuStreamSynchronize",
                )?;
                unsafe { host.set_len(self.len) };
                Ok(())
            })?;
        }
        Ok(host)
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let drv = CudaDriver::get();
            let _ = with_current_context(self.ctx, || {
                check(unsafe { (drv.cuMemFree_v2)(self.ptr) }, "cuMemFree_v2")
            });
        }
    }
}
