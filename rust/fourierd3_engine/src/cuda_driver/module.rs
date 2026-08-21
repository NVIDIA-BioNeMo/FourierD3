// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CString;
use std::os::raw::c_int;
use std::sync::Arc;

use crate::cuda_driver::{
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, CUfunction, CUkernel, CUlibrary, CudaDriver,
    Error, Result, check,
};

struct LibraryInner(CUlibrary);

impl Drop for LibraryInner {
    fn drop(&mut self) {
        let drv = CudaDriver::get();
        let _ = unsafe { (drv.cuLibraryUnload)(self.0) };
    }
}

unsafe impl Send for LibraryInner {}
unsafe impl Sync for LibraryInner {}

/// A loaded CUDA library (PTX/CUBIN image), unloaded when the last clone drops.
/// Shared via `Arc`, so any [`Kernel`] it produced keeps it loaded.
#[derive(Clone)]
pub(crate) struct Module(Arc<LibraryInner>);

impl Module {
    /// Load a library from a CUBIN/PTX image (`cuLibraryLoadData`). A context
    /// must already be current.
    pub(crate) fn load(image: &[u8]) -> Result<Module> {
        let drv = CudaDriver::get();
        let mut lib: CUlibrary = std::ptr::null_mut();
        check(
            unsafe {
                (drv.cuLibraryLoadData)(
                    &mut lib,
                    image.as_ptr() as *const _,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                )
            },
            "cuLibraryLoadData",
        )?;
        Ok(Module(Arc::new(LibraryInner(lib))))
    }

    /// Look up `name` and return a [`Kernel`] that holds this module alive.
    pub(crate) fn kernel(&self, name: &str) -> Result<Kernel> {
        let drv = CudaDriver::get();
        let cname = CString::new(name)
            .map_err(|_| Error::other("kernel name", format!("name {name:?} contains a NUL")))?;
        let mut kernel: CUkernel = std::ptr::null_mut();
        check(
            unsafe { (drv.cuLibraryGetKernel)(&mut kernel, self.raw(), cname.as_ptr()) },
            "cuLibraryGetKernel",
        )?;
        let mut func: CUfunction = std::ptr::null_mut();
        check(
            unsafe { (drv.cuKernelGetFunction)(&mut func, kernel) },
            "cuKernelGetFunction",
        )?;
        Ok(Kernel {
            func,
            _module: self.clone(),
        })
    }

    pub(crate) fn raw(&self) -> CUlibrary {
        self.0.0
    }
}

/// A function looked up from a [`Module`]. Holds a clone of its module, so the
/// library stays loaded as long as the kernel lives.
#[derive(Clone)]
pub(crate) struct Kernel {
    func: CUfunction,
    _module: Module,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub(crate) fn function(&self) -> Function {
        Function(self.func)
    }

    pub(crate) fn set_max_dynamic_shared(&self, bytes: i32) -> Result<()> {
        self.function()
            .set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)
    }
}

/// A borrowed `CUfunction` — the introspection/launch handle. Use it to inspect
/// a function obtained from a graph node, or a [`Kernel`]'s underlying function.
#[derive(Clone, Copy)]
pub(crate) struct Function(pub(crate) CUfunction);

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    pub(crate) fn raw(&self) -> CUfunction {
        self.0
    }

    pub(crate) fn set_attribute(&self, attr: c_int, value: c_int) -> Result<()> {
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuFuncSetAttribute)(self.0, attr, value) },
            "cuFuncSetAttribute",
        )
    }
}
