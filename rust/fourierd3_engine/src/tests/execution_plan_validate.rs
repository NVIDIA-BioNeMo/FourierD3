// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::execution_plan::{
    Arg, Buf, BufRef, KernelModule, Node, Op, PlanError, WorkspaceBuf, WritableBuf,
};
use crate::execution_plan::{ExecutionPlan, PlanBuilder};

#[test]
fn memset_only_recipe_validates() {
    let plan = ExecutionPlan {
        modules: vec![],
        workspace: vec![WorkspaceBuf {
            nbytes: 256,
            init: None,
        }],
        nodes: vec![Node {
            op: Op::Memset {
                target: WritableBuf::Workspace(0),
                value: 0,
                nbytes: 256,
            },
            deps: vec![],
        }],
    };
    assert_eq!(plan.validate(0, 0), Ok(()));
}

#[test]
fn dep_out_of_range_is_rejected() {
    let plan = ExecutionPlan {
        modules: vec![],
        workspace: vec![WorkspaceBuf {
            nbytes: 16,
            init: None,
        }],
        nodes: vec![Node {
            op: Op::Memset {
                target: WritableBuf::Workspace(0),
                value: 0,
                nbytes: 16,
            },
            deps: vec![1],
        }],
    };
    assert_eq!(
        plan.validate(0, 0),
        Err(PlanError::DepNotEarlier { node: 0, dep: 1 })
    );
}

#[test]
fn workspace_index_out_of_range_is_rejected() {
    let plan = ExecutionPlan {
        modules: vec![],
        workspace: vec![],
        nodes: vec![Node {
            op: Op::Memset {
                target: WritableBuf::Workspace(0),
                value: 0,
                nbytes: 16,
            },
            deps: vec![],
        }],
    };
    assert_eq!(
        plan.validate(0, 0),
        Err(PlanError::WorkspaceOutOfRange { node: 0, index: 0 })
    );
}

#[test]
fn whole_recipe_choice_wraps_whole_recipe_candidates() {
    let candidate = |entry: &str| ExecutionPlan {
        modules: vec![KernelModule {
            cubin: vec![0xAA].into(),
        }],
        workspace: vec![WorkspaceBuf {
            nbytes: 128,
            init: None,
        }],
        nodes: vec![Node {
            op: Op::KernelLaunch {
                module: 0,
                entry: entry.into(),
                grid: [1, 1, 1],
                block: [64, 1, 1],
                shmem: 0,
                args: vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
            },
            deps: vec![],
        }],
    };

    let mut b = PlanBuilder::new();
    b.whole_plan_choice(vec![candidate("a"), candidate("b")], 1, 1);
    let plan = b.finish().unwrap();

    assert!(plan.modules.is_empty());
    assert!(plan.workspace.is_empty());
    let [
        Node {
            op:
                Op::Choice {
                    candidates,
                    input_binding,
                    output_binding,
                },
            deps,
        },
    ] = &plan.nodes[..]
    else {
        panic!("expected a single Choice node, got {:?}", plan.nodes);
    };
    assert!(deps.is_empty());
    assert_eq!(candidates.len(), 2);
    assert_eq!(input_binding, &vec![BufRef::Input(0)]);
    assert_eq!(output_binding, &vec![BufRef::Output(0)]);
    plan.validate(1, 1).unwrap();
}

#[test]
fn plan_builder_assembles_and_indexes() {
    let mut b = PlanBuilder::new();
    let m = b.module(vec![0xDE, 0xAD]);
    let w = b.scratch(64);
    let init = b.memset(Buf::Workspace(w), 0, 64);
    let launch = b
        .kernel(m, "k")
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .read(Buf::Input(0))
        .read(w)
        .args([Arg::input(0)])
        .write(Buf::Output(0))
        .add();
    assert_eq!(
        (m.index(), w.index(), init.index(), launch.index()),
        (0, 0, 0, 1)
    );

    let plan = b.finish().unwrap();
    assert_eq!(plan.modules.len(), 1);
    assert_eq!(plan.workspace.len(), 1);
    assert_eq!(plan.nodes.len(), 2);
    assert_eq!(plan.nodes[1].deps, vec![init.index()]);
    let Op::KernelLaunch { args, .. } = &plan.nodes[1].op else {
        unreachable!()
    };
    assert_eq!(args[2], Arg::input(0));
    plan.validate(1, 1).unwrap();
}
