// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles JAX-lowered LLVM IR into serialized [`execution_plan`] values:
//! CUDA source emit, NVRTC compilation, cubin caching, and enumeration of the
//! autotune candidates the executor later measures. Stateless — it compiles
//! and does not launch.

extern crate self as kernel_compiler;

pub(crate) mod batch_indexing;
pub(crate) mod buffer;
pub(crate) mod cuda_toolchain;
pub(crate) mod infeasibility;
pub(crate) mod llvm;
pub(crate) mod periodic_scatter;
pub(crate) mod spectral_map;

pub(crate) mod libmathdx;
