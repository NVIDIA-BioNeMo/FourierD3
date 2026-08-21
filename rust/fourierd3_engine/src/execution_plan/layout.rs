// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::execution_plan::{ExecutionPlan, Op};

const WORKSPACE_ALIGN: usize = 256;

pub(crate) fn workspace_offsets(sizes: impl Iterator<Item = usize>) -> (Vec<usize>, usize) {
    let mut offsets = Vec::new();
    let mut cursor = 0usize;
    for s in sizes {
        cursor = cursor.next_multiple_of(WORKSPACE_ALIGN);
        offsets.push(cursor);
        cursor += s;
    }
    (offsets, cursor.next_multiple_of(WORKSPACE_ALIGN))
}

fn aligned_total(sizes: impl Iterator<Item = usize>) -> usize {
    workspace_offsets(sizes).1
}

pub(crate) fn workspace_upper_bound(plan: &ExecutionPlan) -> usize {
    let parent = aligned_total(plan.workspace.iter().map(|w| w.nbytes));
    let choices: usize = plan
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Op::Choice { candidates, .. } => Some(
                candidates
                    .iter()
                    .map(workspace_upper_bound)
                    .max()
                    .unwrap_or(0),
            ),
            _ => None,
        })
        .sum();
    parent + choices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::{Arg, BufRef, KernelModule, Node, WorkspaceBuf};

    fn module(byte: u8) -> KernelModule {
        KernelModule {
            cubin: vec![byte].into(),
        }
    }

    fn ws(nbytes: usize) -> WorkspaceBuf {
        WorkspaceBuf { nbytes, init: None }
    }

    fn launch(module: usize, entry: &str, args: Vec<Arg>, deps: Vec<usize>) -> Node {
        Node {
            op: Op::KernelLaunch {
                module,
                entry: entry.into(),
                grid: [1, 1, 1],
                block: [1, 1, 1],
                shmem: 0,
                args,
            },
            deps,
        }
    }

    #[test]
    fn workspace_upper_bound_dominates_every_choice() {
        let cand = |entry: &str, ws_size: usize| ExecutionPlan {
            modules: vec![module(0)],
            workspace: vec![ws(ws_size)],
            nodes: vec![launch(
                0,
                entry,
                vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
                vec![],
            )],
        };

        let plan = ExecutionPlan {
            modules: vec![module(0)],
            workspace: vec![ws(100), ws(40)],
            nodes: vec![
                Node {
                    op: Op::Choice {
                        candidates: vec![cand("a0", 10), cand("a1", 300)],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    deps: vec![],
                },
                Node {
                    op: Op::Choice {
                        candidates: vec![cand("b0", 500), cand("b1", 20)],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    deps: vec![0],
                },
            ],
        };

        // parent (100 -> 256, then 40 -> +256 = 512) + worst of {10,300}=300->512
        // + worst of {500,20}=500->512.
        assert_eq!(workspace_upper_bound(&plan), 512 + 512 + 512);
    }

    #[test]
    fn aligned_total_packs_to_256() {
        assert_eq!(aligned_total(std::iter::empty()), 0);
        assert_eq!(aligned_total([1usize].into_iter()), 256);
        // 300 then 1: 300 starts at 0, total 301 -> 512; next buffer would
        // start at 512.
        assert_eq!(aligned_total([300usize, 1].into_iter()), 512 + 256);
    }
}
