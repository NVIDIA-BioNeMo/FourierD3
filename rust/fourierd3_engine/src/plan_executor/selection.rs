// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::execution_plan::{ExecutionPlan, NodeId, Op, PlanBuilder};

pub(super) fn choice_indices(plan: &ExecutionPlan) -> Vec<usize> {
    plan.nodes
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| matches!(node.op, Op::Choice { .. }).then_some(idx))
        .collect()
}

fn remap_deps(deps: &[usize], sinks: &[Vec<usize>]) -> Vec<usize> {
    let mut out = Vec::new();
    for &dep in deps {
        for &sink in &sinks[dep] {
            if !out.contains(&sink) {
                out.push(sink);
            }
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Selection {
    pub candidate: usize,
    pub nested: Vec<Selection>,
}

impl Selection {
    #[cfg(test)]
    pub(super) fn leaf(candidate: usize) -> Self {
        Selection {
            candidate,
            nested: Vec::new(),
        }
    }
}

pub(super) fn default_selection(plan: &ExecutionPlan) -> Vec<Selection> {
    choice_indices(plan)
        .iter()
        .map(|&ci| match &plan.nodes[ci].op {
            Op::Choice { candidates, .. } => Selection {
                candidate: 0,
                nested: default_selection(&candidates[0]),
            },
            _ => unreachable!("choice_indices points only at Choice nodes"),
        })
        .collect()
}

pub(super) fn inline(plan: &ExecutionPlan, selection: &[Selection]) -> ExecutionPlan {
    let choices = choice_indices(plan);
    assert_eq!(
        selection.len(),
        choices.len(),
        "selection length {} does not match the {} top-level choices",
        selection.len(),
        choices.len()
    );

    let mut out = PlanBuilder::new();
    for module in &plan.modules {
        out.module(module.cubin.clone());
    }
    for ws in &plan.workspace {
        match &ws.init {
            Some(bytes) => out.scratch_init(bytes.clone()),
            None => out.scratch(ws.nbytes),
        };
    }

    let mut sinks: Vec<Vec<usize>> = vec![Vec::new(); plan.nodes.len()];
    let mut choice_of = vec![usize::MAX; plan.nodes.len()];
    for (k, &ci) in choices.iter().enumerate() {
        choice_of[ci] = k;
    }

    for (orig_idx, node) in plan.nodes.iter().enumerate() {
        let deps = remap_deps(&node.deps, &sinks);
        if let Op::Choice {
            candidates,
            input_binding,
            output_binding,
        } = &node.op
        {
            let sel = &selection[choice_of[orig_idx]];
            // Resolve the chosen candidate's own choices first, so what we
            // splice is a flat sub-plan in the candidate's formal index space.
            let resolved = inline(&candidates[sel.candidate], &sel.nested);
            let external: Vec<NodeId> = deps.iter().map(|&d| NodeId::from_index(d)).collect();
            let cand_sinks = out.splice(&resolved, input_binding, output_binding, &external);
            sinks[orig_idx] = cand_sinks.iter().map(|n| n.index()).collect();
        } else {
            let new = out.push_node(node.op.clone(), deps);
            sinks[orig_idx] = vec![new.index()];
        }
    }

    out.finish().expect("inline builds a well-formed plan")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::{Arg, BufRef, KernelModule, Node, WorkspaceBuf, WritableBuf};

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

    fn single_node_plan(entry: &str) -> ExecutionPlan {
        ExecutionPlan {
            modules: vec![module(0xAA)],
            workspace: vec![ws(64)],
            nodes: vec![launch(
                0,
                entry,
                vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
                vec![],
            )],
        }
    }

    fn io01() -> (Vec<BufRef>, Vec<BufRef>) {
        (vec![BufRef::Input(0)], vec![BufRef::Output(0)])
    }

    #[test]
    fn choice_indices_finds_choice_nodes() {
        let plan = ExecutionPlan {
            modules: vec![module(1)],
            workspace: vec![],
            nodes: vec![
                launch(0, "a", vec![], vec![]),
                Node {
                    op: Op::Choice {
                        candidates: vec![single_node_plan("c")],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    deps: vec![0],
                },
                launch(0, "b", vec![], vec![0]),
                Node {
                    op: Op::Choice {
                        candidates: vec![single_node_plan("d")],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    deps: vec![],
                },
            ],
        };
        assert_eq!(choice_indices(&plan), vec![1, 3]);
    }

    #[test]
    fn inline_single_node_candidate() {
        let plan = ExecutionPlan {
            modules: vec![module(0x11)],
            workspace: vec![ws(32)],
            nodes: vec![
                launch(0, "producer", vec![Arg::output(0)], vec![]),
                Node {
                    op: Op::Choice {
                        candidates: vec![single_node_plan("cand")],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    deps: vec![0],
                },
                launch(0, "consumer", vec![Arg::output(0)], vec![1]),
            ],
        };

        let out = inline(&plan, &[Selection::leaf(0)]);

        assert_eq!(out.nodes.len(), 3);
        assert_eq!(out.modules.len(), 2);
        assert_eq!(out.workspace.len(), 2);

        match &out.nodes[1].op {
            Op::KernelLaunch { module, args, .. } => {
                let deps = &out.nodes[1].deps;
                assert_eq!(*module, 1);
                assert_eq!(
                    args,
                    &vec![Arg::input(0), Arg::workspace(1), Arg::output(0)]
                );
                assert_eq!(deps, &vec![0]);
            }
            other => panic!("expected KernelLaunch, got {other:?}"),
        }

        assert_eq!(out.nodes[2].deps, [1]);

        assert!(choice_indices(&out).is_empty());
        out.validate(1, 1).unwrap();
    }

    #[test]
    fn inline_multi_node_candidate() {
        let cand = ExecutionPlan {
            modules: vec![module(0xBB)],
            workspace: vec![ws(16)],
            nodes: vec![
                Node {
                    op: Op::Memset {
                        target: WritableBuf::Output(0),
                        value: 0,
                        nbytes: 16,
                    },
                    deps: vec![],
                },
                launch(0, "work", vec![Arg::output(0), Arg::workspace(0)], vec![0]),
            ],
        };

        let plan = ExecutionPlan {
            modules: vec![module(0x22)],
            workspace: vec![ws(8)],
            nodes: vec![
                launch(0, "producer", vec![Arg::output(0)], vec![]),
                Node {
                    op: Op::Choice {
                        candidates: vec![cand],
                        input_binding: vec![BufRef::Input(0)],
                        output_binding: vec![BufRef::Output(0)],
                    },
                    deps: vec![0],
                },
                launch(0, "consumer", vec![Arg::output(0)], vec![1]),
            ],
        };

        let out = inline(&plan, &[Selection::leaf(0)]);

        assert_eq!(out.nodes.len(), 4);

        match &out.nodes[1].op {
            Op::Memset { target, .. } => {
                assert_eq!(out.nodes[1].deps, [0]);
                assert_eq!(target, &WritableBuf::Output(0));
            }
            other => panic!("expected Memset, got {other:?}"),
        }

        match &out.nodes[2].op {
            Op::KernelLaunch { args, .. } => {
                assert_eq!(out.nodes[2].deps, [1]);
                assert_eq!(args, &vec![Arg::output(0), Arg::workspace(1)]);
            }
            other => panic!("expected KernelLaunch, got {other:?}"),
        }

        assert_eq!(out.nodes[3].deps, [2]);

        assert!(choice_indices(&out).is_empty());
        out.validate(1, 1).unwrap();
    }

    #[test]
    fn inline_two_choices() {
        let choice = |entry_a: &str, entry_b: &str, deps: Vec<usize>| {
            let (input_binding, output_binding) = io01();
            Node {
                op: Op::Choice {
                    candidates: vec![single_node_plan(entry_a), single_node_plan(entry_b)],
                    input_binding,
                    output_binding,
                },
                deps,
            }
        };

        let plan = ExecutionPlan {
            modules: vec![module(0x33)],
            workspace: vec![ws(8)],
            nodes: vec![
                launch(0, "producer", vec![Arg::output(0)], vec![]),
                choice("a0", "a1", vec![0]),
                choice("b0", "b1", vec![1]),
            ],
        };

        let out = inline(&plan, &[Selection::leaf(1), Selection::leaf(0)]);
        assert_eq!(out.nodes.len(), 3);
        assert_eq!(out.modules.len(), 3);
        assert_eq!(out.workspace.len(), 3);

        match &out.nodes[1].op {
            Op::KernelLaunch {
                entry,
                module,
                args,
                ..
            } => {
                assert_eq!(entry, "a1");
                assert_eq!(*module, 1);
                assert_eq!(
                    args,
                    &vec![Arg::input(0), Arg::workspace(1), Arg::output(0)]
                );
                assert_eq!(out.nodes[1].deps, [0]);
            }
            other => panic!("expected KernelLaunch, got {other:?}"),
        }
        match &out.nodes[2].op {
            Op::KernelLaunch {
                entry,
                module,
                args,
                ..
            } => {
                assert_eq!(entry, "b0");
                assert_eq!(*module, 2);
                assert_eq!(
                    args,
                    &vec![Arg::input(0), Arg::workspace(2), Arg::output(0)]
                );
                assert_eq!(out.nodes[2].deps, [1]);
            }
            other => panic!("expected KernelLaunch, got {other:?}"),
        }

        assert!(choice_indices(&out).is_empty());
        out.validate(1, 1).unwrap();
    }

    #[test]
    fn inline_nested_choice() {
        let inner = ExecutionPlan {
            modules: vec![],
            workspace: vec![],
            nodes: vec![Node {
                op: Op::Choice {
                    candidates: vec![single_node_plan("inner_a"), single_node_plan("inner_b")],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Output(0)],
                },
                deps: vec![],
            }],
        };
        let plan = ExecutionPlan {
            modules: vec![],
            workspace: vec![],
            nodes: vec![Node {
                op: Op::Choice {
                    candidates: vec![inner],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Output(0)],
                },
                deps: vec![],
            }],
        };

        let out = inline(
            &plan,
            &[Selection {
                candidate: 0,
                nested: vec![Selection::leaf(1)],
            }],
        );

        assert!(
            choice_indices(&out).is_empty(),
            "nested choice must be fully resolved"
        );
        assert_eq!(out.nodes.len(), 1);
        assert_eq!(out.modules.len(), 1);
        assert_eq!(out.workspace.len(), 1);
        match &out.nodes[0].op {
            Op::KernelLaunch { entry, args, .. } => {
                assert_eq!(entry, "inner_b"); // nested selection candidate 1
                assert_eq!(
                    args,
                    &vec![Arg::input(0), Arg::workspace(0), Arg::output(0)]
                );
            }
            other => panic!("expected KernelLaunch, got {other:?}"),
        }
        out.validate(1, 1).unwrap();
    }
}
