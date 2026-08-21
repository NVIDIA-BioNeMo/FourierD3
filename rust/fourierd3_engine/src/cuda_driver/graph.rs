// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{c_uint, c_void};

use crate::cuda_driver::module::Function;
use crate::cuda_driver::{
    CUDA_KERNEL_NODE_PARAMS, CUDA_MEMSET_NODE_PARAMS, CUgraph, CUgraphExec, CUgraphNode, Context,
    CudaDriver, Result, StreamRef, check,
};

/// A node handle inside a [`Graph`]. Owned by its graph (freed with it), so this
/// is a cheap `Copy` reference.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphNode(pub(crate) CUgraphNode);

/// A kernel node to add or re-point. `params` is the array of
/// pointers-to-arguments CUDA expects; it only needs to outlive the call.
pub(crate) struct KernelNode<'a> {
    pub func: Function,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
    pub params: &'a [*mut c_void],
}

impl KernelNode<'_> {
    fn to_raw(&self) -> CUDA_KERNEL_NODE_PARAMS {
        CUDA_KERNEL_NODE_PARAMS {
            func: self.func.raw(),
            gridDimX: self.grid[0],
            gridDimY: self.grid[1],
            gridDimZ: self.grid[2],
            blockDimX: self.block[0],
            blockDimY: self.block[1],
            blockDimZ: self.block[2],
            sharedMemBytes: self.shared_mem,
            kernelParams: if self.params.is_empty() {
                std::ptr::null_mut()
            } else {
                self.params.as_ptr() as *mut *mut c_void
            },
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        }
    }
}

/// A linear device memset node.
#[derive(Clone, Copy)]
pub(crate) struct MemsetNode {
    pub dst: crate::cuda_driver::CUdeviceptr,
    pub value: u8,
    pub nbytes: usize,
}

impl MemsetNode {
    fn to_raw(self) -> CUDA_MEMSET_NODE_PARAMS {
        CUDA_MEMSET_NODE_PARAMS {
            dst: self.dst,
            pitch: 0,
            value: self.value as c_uint,
            elementSize: 1,
            width: self.nbytes,
            height: 1,
        }
    }
}

/// A CUDA graph, destroyed on drop. Carries the context its memset/memcpy nodes
/// run in.
pub(crate) struct Graph {
    raw: CUgraph,
    ctx: Context,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Graph {
    pub(crate) fn new(ctx: Context) -> Result<Graph> {
        let drv = CudaDriver::get();
        let mut raw: CUgraph = std::ptr::null_mut();
        check(unsafe { (drv.cuGraphCreate)(&mut raw, 0) }, "cuGraphCreate")?;
        Ok(Graph { raw, ctx })
    }

    pub(crate) fn add_kernel_node(
        &mut self,
        deps: &[GraphNode],
        node: &KernelNode,
    ) -> Result<GraphNode> {
        let drv = CudaDriver::get();
        let deps = raw_nodes(deps);
        let params = node.to_raw();
        let mut handle: CUgraphNode = std::ptr::null_mut();
        check(
            unsafe {
                (drv.cuGraphAddKernelNode_v2)(
                    &mut handle,
                    self.raw,
                    deps.as_ptr(),
                    deps.len(),
                    &params,
                )
            },
            "cuGraphAddKernelNode_v2",
        )?;
        Ok(GraphNode(handle))
    }

    pub(crate) fn add_memset_node(
        &mut self,
        deps: &[GraphNode],
        node: &MemsetNode,
    ) -> Result<GraphNode> {
        let drv = CudaDriver::get();
        let deps = raw_nodes(deps);
        let params = node.to_raw();
        let mut handle: CUgraphNode = std::ptr::null_mut();
        check(
            unsafe {
                (drv.cuGraphAddMemsetNode)(
                    &mut handle,
                    self.raw,
                    deps.as_ptr(),
                    deps.len(),
                    &params,
                    self.ctx.raw(),
                )
            },
            "cuGraphAddMemsetNode",
        )?;
        Ok(GraphNode(handle))
    }

    pub(crate) fn instantiate(&self) -> Result<GraphExec> {
        let drv = CudaDriver::get();
        let mut exec: CUgraphExec = std::ptr::null_mut();
        check(
            unsafe { (drv.cuGraphInstantiateWithFlags)(&mut exec, self.raw, 0) },
            "cuGraphInstantiateWithFlags",
        )?;
        Ok(GraphExec {
            raw: exec,
            ctx: self.ctx,
        })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let drv = CudaDriver::get();
        let _ = unsafe { (drv.cuGraphDestroy)(self.raw) };
    }
}

/// An instantiated, launchable graph. Destroyed on drop. Node parameters can be
/// re-pointed in place between launches.
pub(crate) struct GraphExec {
    raw: CUgraphExec,
    ctx: Context,
}

unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

impl GraphExec {
    pub(crate) fn launch(&self, stream: StreamRef) -> Result<()> {
        let drv = CudaDriver::get();
        check(
            unsafe { (drv.cuGraphLaunch)(self.raw, stream.raw()) },
            "cuGraphLaunch",
        )
    }

    pub(crate) fn set_kernel_node_params(&self, node: GraphNode, desc: &KernelNode) -> Result<()> {
        let drv = CudaDriver::get();
        let params = desc.to_raw();
        check(
            unsafe { (drv.cuGraphExecKernelNodeSetParams_v2)(self.raw, node.0, &params) },
            "cuGraphExecKernelNodeSetParams_v2",
        )
    }

    pub(crate) fn set_memset_node_params(&self, node: GraphNode, desc: &MemsetNode) -> Result<()> {
        let drv = CudaDriver::get();
        let params = desc.to_raw();
        check(
            unsafe {
                (drv.cuGraphExecMemsetNodeSetParams)(self.raw, node.0, &params, self.ctx.raw())
            },
            "cuGraphExecMemsetNodeSetParams",
        )
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        let drv = CudaDriver::get();
        let _ = unsafe { (drv.cuGraphExecDestroy)(self.raw) };
    }
}

fn raw_nodes(nodes: &[GraphNode]) -> Vec<CUgraphNode> {
    nodes.iter().map(|n| n.0).collect()
}
