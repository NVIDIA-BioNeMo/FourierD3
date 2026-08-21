// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `cuda_driver` — a safe, idiomatic Rust layer over libcuda.
//!
//! Implement nothing: just use the safe types. [`Error`]/[`Result`] wrap every
//! `CUresult`; [`Device`]/[`Context`] name what you run on; and the RAII handles
//! ([`Stream`]/[`StreamRef`], [`Event`], [`Module`]/[`Kernel`], [`Graph`]/
//! [`GraphExec`], [`DeviceBuffer`]) free on drop and carry their owning context,
//! so the API is multi-GPU-ready. Contexts/devices are always passed explicitly
//! — nothing reads or mutates a hidden "current" global behind your back.
//!
//! The raw `dlopen` function table lives in [`ffi`]; reach for it only for
//! operations this layer doesn't cover.

pub(crate) mod ffi;

mod context;
mod device;
mod error;
mod event;
mod graph;
mod memory;
mod module;
mod stream;

// Bring the raw substrate into crate scope (crate-internal) so the safe modules
// can name `crate::cuda_driver::CudaDriver`, `crate::cuda_driver::CUstream`, the constants, etc.
pub(crate) use ffi::*;

// The only raw items in the *public* root surface: the handle/result aliases the
// safe signatures expose, plus the `CudaDriver` table as an escape hatch.
// Everything else raw (constants, `CUDA_*_NODE_PARAMS`, …) is reached via `ffi`.
pub(crate) use ffi::{
    CUcontext, CUdevice, CUdeviceptr, CUevent, CUfunction, CUgraph, CUgraphExec, CUgraphNode,
    CUlibrary, CUresult, CUstream, CudaDriver,
};

pub(crate) use context::Context;
#[cfg(test)]
pub(crate) use context::ensure_context;
pub(crate) use device::Device;
pub(crate) use error::{Error, Result, check};
pub(crate) use event::Event;
pub(crate) use graph::{Graph, GraphExec, GraphNode, KernelNode, MemsetNode};
pub(crate) use memory::DeviceBuffer;
pub(crate) use module::{Function, Kernel, Module};
pub(crate) use stream::{LaunchConfig, StreamRef};
