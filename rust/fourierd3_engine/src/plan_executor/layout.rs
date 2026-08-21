// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::cuda_driver::CUdeviceptr;

use crate::execution_plan::layout::workspace_offsets;

pub(crate) fn carve(base: CUdeviceptr, sizes: impl Iterator<Item = usize>) -> Vec<CUdeviceptr> {
    workspace_offsets(sizes)
        .0
        .iter()
        .map(|&off| base + off as CUdeviceptr)
        .collect()
}
