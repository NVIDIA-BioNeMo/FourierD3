// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw `dlopen` bindings to the libcuda function table — the unsafe substrate
//! under the crate's safe layer. Reach for it (via `crate::cuda_driver::ffi`) only for
//! operations the safe API doesn't cover.

use std::ffi::{c_char, c_void};
use std::os::raw::{c_int, c_uint, c_ulonglong};
#[cfg(test)]
use std::os::raw::{c_uchar, c_ushort};
use std::sync::OnceLock;

use crate::load_sym;
use libloading::Library;

pub(crate) type CUdevice = c_int;
pub(crate) type CUcontext = *mut c_void;
pub(crate) type CUdeviceptr = c_ulonglong;
pub(crate) type CUresult = c_uint;
pub(crate) type CUfunction = *mut c_void;
pub(crate) type CUkernel = *mut c_void;
pub(crate) type CUlibrary = *mut c_void;
pub(crate) type CUstream = *mut c_void;
pub(crate) type CUevent = *mut c_void;
pub(crate) type CUgraph = *mut c_void;
pub(crate) type CUgraphNode = *mut c_void;
pub(crate) type CUgraphExec = *mut c_void;

pub(crate) const CUDA_SUCCESS: CUresult = 0;

pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub(crate) const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;

pub(crate) const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

#[cfg(test)]
pub(crate) type CUmemGenericAllocationHandle = c_ulonglong;
#[cfg(test)]
pub(crate) type CUmemAllocationType = c_uint;
#[cfg(test)]
pub(crate) const CU_MEM_ALLOCATION_TYPE_PINNED: CUmemAllocationType = 0x1;
#[cfg(test)]
pub(crate) type CUmemAllocationHandleType = c_uint;
#[cfg(test)]
pub(crate) const CU_MEM_HANDLE_TYPE_NONE: CUmemAllocationHandleType = 0x0;
#[cfg(test)]
pub(crate) type CUmemLocationType = c_uint;
#[cfg(test)]
pub(crate) const CU_MEM_LOCATION_TYPE_DEVICE: CUmemLocationType = 0x1;
#[cfg(test)]
#[allow(non_camel_case_types)]
pub(crate) type CUmemAccess_flags = c_uint;
#[cfg(test)]
pub(crate) const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: CUmemAccess_flags = 0x3;

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CUmemLocation {
    pub type_: CUmemLocationType,
    pub id: c_int,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)]
pub(crate) struct CUmemAllocationPropAllocFlags {
    pub compressionType: c_uchar,
    pub gpuDirectRDMACapable: c_uchar,
    pub usage: c_ushort,
    pub reserved: [c_uchar; 4],
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)]
pub(crate) struct CUmemAllocationProp {
    pub type_: CUmemAllocationType,
    pub requestedHandleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32HandleMetaData: *mut c_void,
    pub allocFlags: CUmemAllocationPropAllocFlags,
}

#[cfg(test)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: CUmemAccess_flags,
}

#[repr(C)]
#[allow(non_snake_case)]
pub(crate) struct CUDA_KERNEL_NODE_PARAMS {
    pub func: CUfunction,
    pub gridDimX: c_uint,
    pub gridDimY: c_uint,
    pub gridDimZ: c_uint,
    pub blockDimX: c_uint,
    pub blockDimY: c_uint,
    pub blockDimZ: c_uint,
    pub sharedMemBytes: c_uint,
    pub kernelParams: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}

#[repr(C)]
#[allow(non_snake_case)]
pub(crate) struct CUDA_MEMSET_NODE_PARAMS {
    pub dst: CUdeviceptr,
    pub pitch: usize,
    pub value: c_uint,
    pub elementSize: c_uint,
    pub width: usize,
    pub height: usize,
}

#[allow(non_snake_case)]
pub(crate) struct CudaDriver {
    _lib: Library,

    #[cfg(test)]
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    #[cfg(test)]
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    #[cfg(test)]
    pub cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxGetCurrent: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuCtxGetDevice: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuStreamGetCtx: unsafe extern "C" fn(CUstream, *mut CUcontext) -> CUresult,

    pub cuLibraryLoadData: unsafe extern "C" fn(
        *mut CUlibrary,
        *const c_void,
        *const c_void,
        *const c_void,
        c_uint,
        *const c_void,
        *const c_void,
        c_uint,
    ) -> CUresult,
    pub cuLibraryUnload: unsafe extern "C" fn(CUlibrary) -> CUresult,
    pub cuLibraryGetKernel:
        unsafe extern "C" fn(*mut CUkernel, CUlibrary, *const c_char) -> CUresult,
    pub cuKernelGetFunction: unsafe extern "C" fn(*mut CUfunction, CUkernel) -> CUresult,

    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,

    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,
    pub cuEventDestroy: unsafe extern "C" fn(CUevent) -> CUresult,

    pub cuGraphCreate: unsafe extern "C" fn(*mut CUgraph, c_uint) -> CUresult,
    pub cuGraphDestroy: unsafe extern "C" fn(CUgraph) -> CUresult,
    pub cuGraphAddKernelNode_v2: unsafe extern "C" fn(
        *mut CUgraphNode,
        CUgraph,
        *const CUgraphNode,
        usize,
        *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult,
    pub cuGraphAddMemsetNode: unsafe extern "C" fn(
        *mut CUgraphNode,
        CUgraph,
        *const CUgraphNode,
        usize,
        *const CUDA_MEMSET_NODE_PARAMS,
        CUcontext,
    ) -> CUresult,
    pub cuGraphInstantiateWithFlags:
        unsafe extern "C" fn(*mut CUgraphExec, CUgraph, c_ulonglong) -> CUresult,
    pub cuGraphLaunch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,
    pub cuGraphExecDestroy: unsafe extern "C" fn(CUgraphExec) -> CUresult,
    pub cuGraphExecKernelNodeSetParams_v2:
        unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_KERNEL_NODE_PARAMS) -> CUresult,
    pub cuGraphExecMemsetNodeSetParams: unsafe extern "C" fn(
        CUgraphExec,
        CUgraphNode,
        *const CUDA_MEMSET_NODE_PARAMS,
        CUcontext,
    ) -> CUresult,
    pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,

    #[cfg(test)]
    pub cuMemGetAllocationGranularity:
        unsafe extern "C" fn(*mut usize, *const CUmemAllocationProp, c_uint) -> CUresult,
    #[cfg(test)]
    pub cuMemAddressReserve:
        unsafe extern "C" fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, c_ulonglong) -> CUresult,
    #[cfg(test)]
    pub cuMemCreate: unsafe extern "C" fn(
        *mut CUmemGenericAllocationHandle,
        usize,
        *const CUmemAllocationProp,
        c_ulonglong,
    ) -> CUresult,
    #[cfg(test)]
    pub cuMemMap: unsafe extern "C" fn(
        CUdeviceptr,
        usize,
        usize,
        CUmemGenericAllocationHandle,
        c_ulonglong,
    ) -> CUresult,
    #[cfg(test)]
    pub cuMemSetAccess:
        unsafe extern "C" fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult,

    pub cuFuncSetAttribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
    pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CUresult,
    #[cfg(test)]
    pub cuMemsetD8Async: unsafe extern "C" fn(CUdeviceptr, c_uchar, usize, CUstream) -> CUresult,
    pub cuMemcpyHtoDAsync_v2:
        unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    #[cfg(test)]
    pub cuMemcpyDtoHAsync_v2:
        unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,

    pub cuGetErrorString: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    #[allow(non_snake_case)]
    fn load() -> Option<Self> {
        let lib = crate::dynamic_library::open_named("libcuda", &["1"])?;

        #[cfg(test)]
        let cuInit = load_sym!(&lib, "cuInit", unsafe extern "C" fn(c_uint) -> CUresult);
        #[cfg(test)]
        let cuDeviceGet = load_sym!(
            &lib,
            "cuDeviceGet",
            unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult
        );
        #[cfg(test)]
        let cuCtxCreate_v2 = load_sym!(
            &lib,
            "cuCtxCreate_v2",
            unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult
        );
        let cuCtxSetCurrent = load_sym!(
            &lib,
            "cuCtxSetCurrent",
            unsafe extern "C" fn(CUcontext) -> CUresult
        );
        let cuCtxGetCurrent = load_sym!(
            &lib,
            "cuCtxGetCurrent",
            unsafe extern "C" fn(*mut CUcontext) -> CUresult
        );
        let cuCtxGetDevice = load_sym!(
            &lib,
            "cuCtxGetDevice",
            unsafe extern "C" fn(*mut c_int) -> CUresult
        );
        let cuStreamGetCtx = load_sym!(
            &lib,
            "cuStreamGetCtx",
            unsafe extern "C" fn(CUstream, *mut CUcontext) -> CUresult
        );
        let cuLibraryLoadData = load_sym!(
            &lib,
            "cuLibraryLoadData",
            unsafe extern "C" fn(
                *mut CUlibrary,
                *const c_void,
                *const c_void,
                *const c_void,
                c_uint,
                *const c_void,
                *const c_void,
                c_uint,
            ) -> CUresult
        );
        let cuLibraryUnload = load_sym!(
            &lib,
            "cuLibraryUnload",
            unsafe extern "C" fn(CUlibrary) -> CUresult
        );
        let cuLibraryGetKernel = load_sym!(
            &lib,
            "cuLibraryGetKernel",
            unsafe extern "C" fn(*mut CUkernel, CUlibrary, *const c_char) -> CUresult
        );
        let cuKernelGetFunction = load_sym!(
            &lib,
            "cuKernelGetFunction",
            unsafe extern "C" fn(*mut CUfunction, CUkernel) -> CUresult
        );
        let cuLaunchKernel = load_sym!(
            &lib,
            "cuLaunchKernel",
            unsafe extern "C" fn(
                CUfunction,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                CUstream,
                *mut *mut c_void,
                *mut *mut c_void,
            ) -> CUresult
        );

        let cuEventCreate = load_sym!(
            &lib,
            "cuEventCreate",
            unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult
        );
        let cuEventRecord = load_sym!(
            &lib,
            "cuEventRecord",
            unsafe extern "C" fn(CUevent, CUstream) -> CUresult
        );
        let cuEventSynchronize = load_sym!(
            &lib,
            "cuEventSynchronize",
            unsafe extern "C" fn(CUevent) -> CUresult
        );
        let cuEventElapsedTime = load_sym!(
            &lib,
            "cuEventElapsedTime",
            unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult
        );
        let cuEventDestroy = load_sym!(
            &lib,
            "cuEventDestroy",
            unsafe extern "C" fn(CUevent) -> CUresult
        );

        let cuGraphCreate = load_sym!(
            &lib,
            "cuGraphCreate",
            unsafe extern "C" fn(*mut CUgraph, c_uint) -> CUresult
        );
        let cuGraphDestroy = load_sym!(
            &lib,
            "cuGraphDestroy",
            unsafe extern "C" fn(CUgraph) -> CUresult
        );
        let cuGraphAddKernelNode_v2 = load_sym!(
            &lib,
            "cuGraphAddKernelNode_v2",
            unsafe extern "C" fn(
                *mut CUgraphNode,
                CUgraph,
                *const CUgraphNode,
                usize,
                *const CUDA_KERNEL_NODE_PARAMS,
            ) -> CUresult
        );
        let cuGraphAddMemsetNode = load_sym!(
            &lib,
            "cuGraphAddMemsetNode",
            unsafe extern "C" fn(
                *mut CUgraphNode,
                CUgraph,
                *const CUgraphNode,
                usize,
                *const CUDA_MEMSET_NODE_PARAMS,
                CUcontext,
            ) -> CUresult
        );
        // `cuGraphInstantiate` in the ABI is the legacy 5-arg form; cuda.h
        // redirects the name to `cuGraphInstantiateWithFlags` via a macro that
        // dlsym bypasses. Binding "cuGraphInstantiate" as 3-arg calls the 5-arg
        // function with two garbage trailing args, causing driver writes to a
        // garbage log-buffer pointer and segfaults on some GPUs.
        let cuGraphInstantiateWithFlags = load_sym!(
            &lib,
            "cuGraphInstantiateWithFlags",
            unsafe extern "C" fn(*mut CUgraphExec, CUgraph, c_ulonglong) -> CUresult
        );
        let cuGraphLaunch = load_sym!(
            &lib,
            "cuGraphLaunch",
            unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult
        );
        let cuGraphExecDestroy = load_sym!(
            &lib,
            "cuGraphExecDestroy",
            unsafe extern "C" fn(CUgraphExec) -> CUresult
        );
        let cuGraphExecKernelNodeSetParams_v2 = load_sym!(
            &lib,
            "cuGraphExecKernelNodeSetParams_v2",
            unsafe extern "C" fn(
                CUgraphExec,
                CUgraphNode,
                *const CUDA_KERNEL_NODE_PARAMS,
            ) -> CUresult
        );
        let cuGraphExecMemsetNodeSetParams = load_sym!(
            &lib,
            "cuGraphExecMemsetNodeSetParams",
            unsafe extern "C" fn(
                CUgraphExec,
                CUgraphNode,
                *const CUDA_MEMSET_NODE_PARAMS,
                CUcontext,
            ) -> CUresult
        );
        let cuMemAlloc_v2 = load_sym!(
            &lib,
            "cuMemAlloc_v2",
            unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult
        );
        let cuMemFree_v2 = load_sym!(
            &lib,
            "cuMemFree_v2",
            unsafe extern "C" fn(CUdeviceptr) -> CUresult
        );
        #[cfg(test)]
        let cuMemGetAllocationGranularity = load_sym!(
            &lib,
            "cuMemGetAllocationGranularity",
            unsafe extern "C" fn(*mut usize, *const CUmemAllocationProp, c_uint) -> CUresult
        );
        #[cfg(test)]
        let cuMemAddressReserve = load_sym!(
            &lib,
            "cuMemAddressReserve",
            unsafe extern "C" fn(
                *mut CUdeviceptr,
                usize,
                usize,
                CUdeviceptr,
                c_ulonglong,
            ) -> CUresult
        );
        #[cfg(test)]
        let cuMemCreate = load_sym!(
            &lib,
            "cuMemCreate",
            unsafe extern "C" fn(
                *mut CUmemGenericAllocationHandle,
                usize,
                *const CUmemAllocationProp,
                c_ulonglong,
            ) -> CUresult
        );
        #[cfg(test)]
        let cuMemMap = load_sym!(
            &lib,
            "cuMemMap",
            unsafe extern "C" fn(
                CUdeviceptr,
                usize,
                usize,
                CUmemGenericAllocationHandle,
                c_ulonglong,
            ) -> CUresult
        );
        #[cfg(test)]
        let cuMemSetAccess = load_sym!(
            &lib,
            "cuMemSetAccess",
            unsafe extern "C" fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult
        );
        let cuFuncSetAttribute = load_sym!(
            &lib,
            "cuFuncSetAttribute",
            unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult
        );
        let cuDeviceGetAttribute = load_sym!(
            &lib,
            "cuDeviceGetAttribute",
            unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CUresult
        );
        #[cfg(test)]
        let cuMemsetD8Async = load_sym!(
            &lib,
            "cuMemsetD8Async",
            unsafe extern "C" fn(CUdeviceptr, c_uchar, usize, CUstream) -> CUresult
        );
        let cuMemcpyHtoDAsync_v2 = load_sym!(
            &lib,
            "cuMemcpyHtoDAsync_v2",
            unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult
        );
        #[cfg(test)]
        let cuMemcpyDtoHAsync_v2 = load_sym!(
            &lib,
            "cuMemcpyDtoHAsync_v2",
            unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult
        );
        let cuStreamSynchronize = load_sym!(
            &lib,
            "cuStreamSynchronize",
            unsafe extern "C" fn(CUstream) -> CUresult
        );

        let cuGetErrorString = load_sym!(
            &lib,
            "cuGetErrorString",
            unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult
        );

        Some(Self {
            _lib: lib,
            #[cfg(test)]
            cuInit,
            #[cfg(test)]
            cuDeviceGet,
            #[cfg(test)]
            cuCtxCreate_v2,
            cuCtxSetCurrent,
            cuCtxGetCurrent,
            cuCtxGetDevice,
            cuStreamGetCtx,
            cuLibraryLoadData,
            cuLibraryUnload,
            cuLibraryGetKernel,
            cuKernelGetFunction,
            cuLaunchKernel,
            cuEventCreate,
            cuEventRecord,
            cuEventSynchronize,
            cuEventElapsedTime,
            cuEventDestroy,
            cuGraphCreate,
            cuGraphDestroy,
            cuGraphAddKernelNode_v2,
            cuGraphAddMemsetNode,
            cuGraphInstantiateWithFlags,
            cuGraphLaunch,
            cuGraphExecDestroy,
            cuGraphExecKernelNodeSetParams_v2,
            cuGraphExecMemsetNodeSetParams,
            cuMemAlloc_v2,
            cuMemFree_v2,
            #[cfg(test)]
            cuMemGetAllocationGranularity,
            #[cfg(test)]
            cuMemAddressReserve,
            #[cfg(test)]
            cuMemCreate,
            #[cfg(test)]
            cuMemMap,
            #[cfg(test)]
            cuMemSetAccess,
            cuFuncSetAttribute,
            cuDeviceGetAttribute,
            #[cfg(test)]
            cuMemsetD8Async,
            cuMemcpyHtoDAsync_v2,
            #[cfg(test)]
            cuMemcpyDtoHAsync_v2,
            cuStreamSynchronize,
            cuGetErrorString,
        })
    }

    pub(crate) fn get() -> &'static Self {
        static INST: OnceLock<CudaDriver> = OnceLock::new();
        INST.get_or_init(|| {
            CudaDriver::load().expect("libcuda not found. Tried libcuda.so, libcuda.so.1.")
        })
    }
}
