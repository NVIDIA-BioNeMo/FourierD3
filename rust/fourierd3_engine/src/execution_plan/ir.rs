// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::execution_plan::blob::Blob;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExecutionPlan {
    pub modules: Vec<KernelModule>,
    pub workspace: Vec<WorkspaceBuf>,
    // A node's `deps` reference strictly-earlier node indices,
    // so the natural index order is a valid topological order.
    pub nodes: Vec<Node>,
}

impl ExecutionPlan {
    /// Re-owns every payload blob so the plan stops referencing (and
    /// pinning) the shared buffer it was decoded from.
    pub(crate) fn detach_blobs(&mut self) {
        for m in &mut self.modules {
            m.cubin = m.cubin.detached();
        }
        for w in &mut self.workspace {
            if let Some(init) = &mut w.init {
                *init = init.detached();
            }
        }
        for n in &mut self.nodes {
            if let Op::Choice { candidates, .. } = &mut n.op {
                for c in candidates {
                    c.detach_blobs();
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct KernelModule {
    pub cubin: Blob,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceBuf {
    pub nbytes: usize,
    pub init: Option<Blob>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Node {
    pub op: Op,
    pub deps: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    KernelLaunch {
        module: usize,
        entry: String,
        grid: [u32; 3],
        block: [u32; 3],
        shmem: u32,
        args: Vec<Arg>,
    },
    Memset {
        target: WritableBuf,
        value: u8,
        nbytes: usize,
    },
    Choice {
        candidates: Vec<ExecutionPlan>,
        input_binding: Vec<BufRef>,
        output_binding: Vec<BufRef>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Arg {
    pub buf: BufRef,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BufRef {
    Input(usize),
    Output(usize),
    Workspace(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WritableBuf {
    Output(usize),
    Workspace(usize),
}
