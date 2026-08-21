// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-indexing a sub-plan into a host plan's index space.
//!
//! A sub-plan names buffers in its own formal slot space: `Input(i)`,
//! `Output(j)`, `Workspace(k)`. Splicing rewrites those formal slots through
//! the host's `input_binding`/`output_binding` and shifts workspace and module
//! indices by the host's bases.

use crate::execution_plan::ir::{Arg, BufRef, Node, Op, WritableBuf};

/// `is_sink[n]` iff no other node in `nodes` depends on node `n`.
pub(crate) fn sink_mask(nodes: &[Node]) -> Vec<bool> {
    let mut is_sink = vec![true; nodes.len()];
    for node in nodes {
        for &dep in &node.deps {
            is_sink[dep] = false;
        }
    }
    is_sink
}

fn rewrite_bufref(
    buf: &BufRef,
    input_binding: &[BufRef],
    output_binding: &[BufRef],
    ws_base: usize,
) -> BufRef {
    match buf {
        BufRef::Input(i) => input_binding[*i],
        BufRef::Output(j) => output_binding[*j],
        BufRef::Workspace(k) => BufRef::Workspace(k + ws_base),
    }
}

fn rewrite_arg(
    arg: &Arg,
    input_binding: &[BufRef],
    output_binding: &[BufRef],
    ws_base: usize,
) -> Arg {
    Arg {
        buf: rewrite_bufref(&arg.buf, input_binding, output_binding, ws_base),
        offset: arg.offset,
    }
}

fn rewrite_writablebuf(
    buf: &WritableBuf,
    output_binding: &[BufRef],
    ws_base: usize,
) -> WritableBuf {
    match buf {
        WritableBuf::Workspace(k) => WritableBuf::Workspace(k + ws_base),
        WritableBuf::Output(j) => match output_binding[*j] {
            BufRef::Output(p) => WritableBuf::Output(p),
            BufRef::Workspace(p) => WritableBuf::Workspace(p),
            BufRef::Input(p) => {
                panic!(
                    "candidate memset buffer (formal output {j}) binds to parent input {p}, which is read-only"
                )
            }
        },
    }
}

/// Rewrite one sub-plan node into the host index space.
///
/// `external` are the host node indices the spliced region depends on; a
/// dep-less source node inherits them. A source node with deps has its deps
/// shifted by `node_base`.
///
/// A `Choice` node survives splicing: only its own `input_binding`/
/// `output_binding` are rewritten through the host binding; its candidates'
/// internal nodes (and the candidates' own modules) stay in the candidate's
/// private formal space, to be resolved later when the Choice is inlined/tuned.
pub(crate) fn rewrite_node(
    node: &Node,
    input_binding: &[BufRef],
    output_binding: &[BufRef],
    module_base: usize,
    ws_base: usize,
    node_base: usize,
    external: &[usize],
) -> Node {
    let deps = if node.deps.is_empty() {
        external.to_vec()
    } else {
        node.deps.iter().map(|ld| ld + node_base).collect()
    };
    let op = match &node.op {
        Op::KernelLaunch {
            module,
            entry,
            grid,
            block,
            shmem,
            args,
        } => Op::KernelLaunch {
            module: module + module_base,
            entry: entry.clone(),
            grid: *grid,
            block: *block,
            shmem: *shmem,
            args: args
                .iter()
                .map(|a| rewrite_arg(a, input_binding, output_binding, ws_base))
                .collect(),
        },
        Op::Memset {
            target,
            value,
            nbytes,
        } => Op::Memset {
            target: rewrite_writablebuf(target, output_binding, ws_base),
            value: *value,
            nbytes: *nbytes,
        },
        Op::Choice {
            candidates,
            input_binding: cand_in,
            output_binding: cand_out,
        } => Op::Choice {
            candidates: candidates.clone(),
            input_binding: cand_in
                .iter()
                .map(|b| rewrite_bufref(b, input_binding, output_binding, ws_base))
                .collect(),
            output_binding: cand_out
                .iter()
                .map(|b| rewrite_bufref(b, input_binding, output_binding, ws_base))
                .collect(),
        },
    };
    Node { op, deps }
}
