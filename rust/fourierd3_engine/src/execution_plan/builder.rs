// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::execution_plan::PlanError;
use crate::execution_plan::ir::{
    Arg, BufRef, ExecutionPlan, KernelModule, Node, Op, WorkspaceBuf, WritableBuf,
};
use crate::execution_plan::splice::{rewrite_node, sink_mask};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModuleId(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceId(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct NodeId(usize);

impl ModuleId {
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

impl WorkspaceId {
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

impl NodeId {
    pub(crate) fn from_index(index: usize) -> NodeId {
        NodeId(index)
    }

    pub(crate) fn index(self) -> usize {
        self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Buf {
    Input(usize),
    Output(usize),
    Workspace(WorkspaceId),
}

impl From<WorkspaceId> for Buf {
    fn from(w: WorkspaceId) -> Buf {
        Buf::Workspace(w)
    }
}

impl Buf {
    fn bufref(self) -> BufRef {
        match self {
            Buf::Input(i) => BufRef::Input(i),
            Buf::Output(i) => BufRef::Output(i),
            Buf::Workspace(w) => BufRef::Workspace(w.index()),
        }
    }

    fn track_key(self) -> Option<Tracked> {
        match self {
            Buf::Input(_) => None,
            Buf::Output(i) => Some(Tracked::Output(i)),
            Buf::Workspace(w) => Some(Tracked::Workspace(w.index())),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Tracked {
    Output(usize),
    Workspace(usize),
}

#[derive(Default)]
struct History {
    last_writer: Option<NodeId>,
    readers_since_write: Vec<NodeId>,
}

#[derive(Default)]
pub(crate) struct PlanBuilder {
    modules: Vec<KernelModule>,
    workspace: Vec<WorkspaceBuf>,
    nodes: Vec<Node>,
    history: std::collections::HashMap<Tracked, History>,
}

impl PlanBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn module(&mut self, cubin: impl Into<crate::execution_plan::Blob>) -> ModuleId {
        self.modules.push(KernelModule {
            cubin: cubin.into(),
        });
        ModuleId(self.modules.len() - 1)
    }

    pub(crate) fn scratch(&mut self, nbytes: usize) -> WorkspaceId {
        self.workspace.push(WorkspaceBuf { nbytes, init: None });
        WorkspaceId(self.workspace.len() - 1)
    }

    pub(crate) fn scratch_init(
        &mut self,
        bytes: impl Into<crate::execution_plan::Blob>,
    ) -> WorkspaceId {
        let bytes = bytes.into();
        self.workspace.push(WorkspaceBuf {
            nbytes: bytes.len(),
            init: Some(bytes),
        });
        WorkspaceId(self.workspace.len() - 1)
    }

    pub(crate) fn kernel(&mut self, module: ModuleId, entry: &str) -> Kernel<'_> {
        Kernel {
            builder: self,
            module,
            entry: entry.to_string(),
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shmem: 0,
            args: Vec::new(),
            touches: Touches::default(),
            extra_deps: Vec::new(),
        }
    }

    pub(crate) fn memset(&mut self, target: Buf, value: u8, nbytes: usize) -> NodeId {
        let mut touches = Touches::default();
        touches.write(target);
        let deps = self.infer_deps(&touches, &[]);
        self.commit(&touches, &deps);
        self.push(Node {
            op: Op::Memset {
                target: writable(target),
                value,
                nbytes,
            },
            deps,
        })
    }

    pub(crate) fn choice(&mut self, candidates: Vec<ExecutionPlan>) -> Choice<'_> {
        Choice {
            builder: self,
            candidates,
            reads: Vec::new(),
            writes: Vec::new(),
            extra_deps: Vec::new(),
        }
    }

    pub(crate) fn whole_plan_choice(
        &mut self,
        candidates: Vec<ExecutionPlan>,
        n_inputs: usize,
        n_outputs: usize,
    ) -> NodeId {
        let mut c = self.choice(candidates);
        for i in 0..n_inputs {
            c = c.reads([Buf::Input(i)]);
        }
        for j in 0..n_outputs {
            c = c.writes([Buf::Output(j)]);
        }
        c.add()
    }

    /// Append `sub` into this builder, re-indexing it into the builder's slot
    /// space: modules shift by `module_base`, workspace by `ws_base`; `sub`'s
    /// formal `BufRef`s map `Input(i)->input_binding[i]`,
    /// `Output(j)->output_binding[j]`, `Workspace(k)->Workspace(k+ws_base)`;
    /// internal node deps shift by the current node base, and dep-less nodes
    /// inherit `external`. Returns the `NodeId`s of `sub`'s sink nodes.
    ///
    /// Spliced nodes carry explicit deps and bypass the `history` auto-dep map:
    /// the caller serializes regions through `external`.
    pub(crate) fn splice(
        &mut self,
        sub: &ExecutionPlan,
        input_binding: &[BufRef],
        output_binding: &[BufRef],
        external: &[NodeId],
    ) -> Vec<NodeId> {
        let module_base = self.modules.len();
        let ws_base = self.workspace.len();
        let node_base = self.nodes.len();
        self.modules.extend(sub.modules.iter().cloned());
        self.workspace.extend(sub.workspace.iter().cloned());

        let external: Vec<usize> = external.iter().map(|n| n.index()).collect();
        for node in &sub.nodes {
            let rewritten = rewrite_node(
                node,
                input_binding,
                output_binding,
                module_base,
                ws_base,
                node_base,
                &external,
            );
            self.nodes.push(rewritten);
        }

        sink_mask(&sub.nodes)
            .iter()
            .enumerate()
            .filter(|&(_, &sink)| sink)
            .map(|(l, _)| NodeId(node_base + l))
            .collect()
    }

    /// Append a pre-built op with explicit deps, bypassing the `history`
    /// auto-dep map. Modules and workspace referenced by `op` must already be
    /// present (`module`/`scratch`/`scratch_init` or a prior `splice`).
    pub(crate) fn push_node(&mut self, op: Op, deps: Vec<usize>) -> NodeId {
        self.push(Node { op, deps })
    }

    pub(crate) fn finish(self) -> Result<ExecutionPlan, PlanError> {
        Ok(ExecutionPlan {
            modules: self.modules,
            workspace: self.workspace,
            nodes: self.nodes,
        })
    }

    fn push(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        NodeId(self.nodes.len() - 1)
    }

    fn infer_deps(&self, touches: &Touches, extra: &[NodeId]) -> Vec<usize> {
        let mut deps: Vec<usize> = Vec::new();
        let mut add = |n: NodeId| deps.push(n.index());

        for &buf in &touches.reads {
            if let Some(h) = self.history.get(&buf)
                && let Some(w) = h.last_writer
            {
                add(w);
            }
        }
        for &buf in &touches.writes {
            if let Some(h) = self.history.get(&buf) {
                if let Some(w) = h.last_writer {
                    add(w);
                }
                for &r in &h.readers_since_write {
                    add(r);
                }
            }
        }
        for &n in extra {
            add(n);
        }
        deps.sort_unstable();
        deps.dedup();
        deps
    }

    fn commit(&mut self, touches: &Touches, deps: &[usize]) {
        let this = NodeId(self.nodes.len());
        let _ = deps;
        for &buf in &touches.reads {
            self.history
                .entry(buf)
                .or_default()
                .readers_since_write
                .push(this);
        }
        for &buf in &touches.writes {
            let h = self.history.entry(buf).or_default();
            h.last_writer = Some(this);
            h.readers_since_write.clear();
        }
    }
}

#[derive(Default)]
struct Touches {
    reads: Vec<Tracked>,
    writes: Vec<Tracked>,
}

impl Touches {
    fn read(&mut self, buf: Buf) {
        if let Some(k) = buf.track_key() {
            self.reads.push(k);
        }
    }
    fn write(&mut self, buf: Buf) {
        if let Some(k) = buf.track_key() {
            self.writes.push(k);
        }
    }
}

fn writable(buf: Buf) -> WritableBuf {
    match buf {
        Buf::Output(i) => WritableBuf::Output(i),
        Buf::Workspace(w) => WritableBuf::Workspace(w.index()),
        Buf::Input(_) => panic!("memset cannot target a read-only input buffer"),
    }
}

#[must_use]
pub(crate) struct Kernel<'a> {
    builder: &'a mut PlanBuilder,
    module: ModuleId,
    entry: String,
    grid: [u32; 3],
    block: [u32; 3],
    shmem: u32,
    args: Vec<Arg>,
    touches: Touches,
    extra_deps: Vec<NodeId>,
}

impl Kernel<'_> {
    pub(crate) fn grid(mut self, grid: [u32; 3]) -> Self {
        self.grid = grid;
        self
    }
    pub(crate) fn block(mut self, block: [u32; 3]) -> Self {
        self.block = block;
        self
    }
    pub(crate) fn shmem(mut self, shmem: u32) -> Self {
        self.shmem = shmem;
        self
    }

    pub(crate) fn read(self, buf: impl Into<Buf>) -> Self {
        self.read_at(buf, 0)
    }
    pub(crate) fn write(self, buf: impl Into<Buf>) -> Self {
        self.write_at(buf, 0)
    }
    pub(crate) fn read_write(self, buf: impl Into<Buf>) -> Self {
        self.read_write_at(buf, 0)
    }

    pub(crate) fn read_at(mut self, buf: impl Into<Buf>, offset: usize) -> Self {
        let buf = buf.into();
        self.touches.read(buf);
        self.args.push(Arg::pointer(buf.bufref(), offset));
        self
    }
    pub(crate) fn write_at(mut self, buf: impl Into<Buf>, offset: usize) -> Self {
        let buf = buf.into();
        self.touches.write(buf);
        self.args.push(Arg::pointer(buf.bufref(), offset));
        self
    }
    pub(crate) fn read_write_at(mut self, buf: impl Into<Buf>, offset: usize) -> Self {
        let buf = buf.into();
        self.touches.read(buf);
        self.touches.write(buf);
        self.args.push(Arg::pointer(buf.bufref(), offset));
        self
    }

    pub(crate) fn args(mut self, args: impl IntoIterator<Item = Arg>) -> Self {
        for arg in args {
            let b = match arg.buf {
                BufRef::Input(i) => Buf::Input(i),
                BufRef::Output(i) => Buf::Output(i),
                BufRef::Workspace(i) => Buf::Workspace(WorkspaceId(i)),
            };
            self.touches.read(b);
            self.touches.write(b);
            self.args.push(arg);
        }
        self
    }

    pub(crate) fn add(self) -> NodeId {
        let Kernel {
            builder,
            module,
            entry,
            grid,
            block,
            shmem,
            args,
            touches,
            extra_deps,
        } = self;
        let deps = builder.infer_deps(&touches, &extra_deps);
        builder.commit(&touches, &deps);
        builder.push(Node {
            op: Op::KernelLaunch {
                module: module.index(),
                entry,
                grid,
                block,
                shmem,
                args,
            },
            deps,
        })
    }
}

#[must_use]
pub(crate) struct Choice<'a> {
    builder: &'a mut PlanBuilder,
    candidates: Vec<ExecutionPlan>,
    reads: Vec<Buf>,
    writes: Vec<(Buf, bool)>,
    extra_deps: Vec<NodeId>,
}

impl Choice<'_> {
    pub(crate) fn reads(mut self, bufs: impl IntoIterator<Item = Buf>) -> Self {
        self.reads.extend(bufs);
        self
    }
    pub(crate) fn writes(mut self, bufs: impl IntoIterator<Item = Buf>) -> Self {
        self.writes.extend(bufs.into_iter().map(|buf| (buf, true)));
        self
    }
    pub(crate) fn add(self) -> NodeId {
        let Choice {
            builder,
            candidates,
            reads,
            writes,
            extra_deps,
        } = self;
        let mut touches = Touches::default();
        for &b in &reads {
            touches.read(b);
        }
        for &(b, written) in &writes {
            if written {
                touches.write(b);
            }
        }
        let deps = builder.infer_deps(&touches, &extra_deps);
        builder.commit(&touches, &deps);
        builder.push(Node {
            op: Op::Choice {
                candidates,
                input_binding: reads.iter().map(|b| b.bufref()).collect(),
                output_binding: writes.iter().map(|(b, _)| b.bufref()).collect(),
            },
            deps,
        })
    }
}

impl Arg {
    pub(crate) fn pointer(buf: BufRef, offset: usize) -> Arg {
        Arg { buf, offset }
    }

    pub(crate) fn input(i: usize) -> Arg {
        Arg::pointer(BufRef::Input(i), 0)
    }

    pub(crate) fn output(i: usize) -> Arg {
        Arg::pointer(BufRef::Output(i), 0)
    }

    pub(crate) fn workspace(i: usize) -> Arg {
        Arg::pointer(BufRef::Workspace(i), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(b: &mut PlanBuilder) -> ModuleId {
        b.module(vec![0xAA])
    }

    #[test]
    fn split_k_chain_infers_raw() {
        let mut b = PlanBuilder::new();
        let m = module(&mut b);
        let ws = b.scratch(256);
        let gemm = b
            .kernel(m, "gemm")
            .read(Buf::Input(0))
            .read(Buf::Input(1))
            .write(ws)
            .add();
        let reduce = b.kernel(m, "reduce").read(ws).write(Buf::Output(0)).add();
        let plan = b.finish().unwrap();
        assert_eq!(plan.nodes[reduce.index()].deps, vec![gemm.index()]);
    }

    #[test]
    fn independent_outputs_share_inputs_without_a_dep() {
        let mut b = PlanBuilder::new();
        let m = module(&mut b);
        let a = b
            .kernel(m, "a")
            .read(Buf::Input(0))
            .read(Buf::Input(1))
            .write(Buf::Output(0))
            .add();
        let c = b
            .kernel(m, "c")
            .read(Buf::Input(0))
            .read(Buf::Input(1))
            .write(Buf::Output(1))
            .add();
        let plan = b.finish().unwrap();
        assert!(plan.nodes[a.index()].deps.is_empty());
        assert!(plan.nodes[c.index()].deps.is_empty());
    }

    #[test]
    fn write_after_read_infers_war() {
        let mut b = PlanBuilder::new();
        let m = module(&mut b);
        let ws = b.scratch(64);
        let seed = b.kernel(m, "seed").write(ws).add();
        let a = b.kernel(m, "a").read(ws).write(Buf::Output(0)).add();
        let bw = b.kernel(m, "b").write(ws).add();
        let plan = b.finish().unwrap();
        assert_eq!(plan.nodes[a.index()].deps, vec![seed.index()]);
        assert!(plan.nodes[bw.index()].deps.contains(&a.index()));
    }

    #[test]
    fn reader_reader_on_workspace_has_no_edge() {
        let mut b = PlanBuilder::new();
        let m = module(&mut b);
        let ws = b.scratch(64);
        let seed = b.kernel(m, "seed").write(ws).add();
        let r0 = b.kernel(m, "r0").read(ws).write(Buf::Output(0)).add();
        let r1 = b.kernel(m, "r1").read(ws).write(Buf::Output(1)).add();
        let plan = b.finish().unwrap();
        assert_eq!(plan.nodes[r0.index()].deps, vec![seed.index()]);
        assert_eq!(plan.nodes[r1.index()].deps, vec![seed.index()]);
    }

    #[test]
    fn memset_then_kernel_chains_on_the_zero_fill() {
        let mut b = PlanBuilder::new();
        let m = module(&mut b);
        let zero = b.memset(Buf::Output(0), 0, 16);
        let k = b
            .kernel(m, "scatter")
            .read(Buf::Input(0))
            .read_write(Buf::Output(0))
            .add();
        let plan = b.finish().unwrap();
        assert_eq!(plan.nodes[k.index()].deps, vec![zero.index()]);
    }

    #[test]
    fn whole_plan_choice_binds_identity() {
        let candidate = |entry: &str| ExecutionPlan {
            modules: vec![KernelModule {
                cubin: vec![0xAA].into(),
            }],
            workspace: vec![],
            nodes: vec![Node {
                op: Op::KernelLaunch {
                    module: 0,
                    entry: entry.into(),
                    grid: [1, 1, 1],
                    block: [64, 1, 1],
                    shmem: 0,
                    args: vec![Arg::input(0), Arg::output(0)],
                },
                deps: vec![],
            }],
        };
        let mut b = PlanBuilder::new();
        b.whole_plan_choice(vec![candidate("a"), candidate("b")], 1, 1);
        let plan = b.finish().unwrap();
        let Op::Choice {
            input_binding,
            output_binding,
            ..
        } = &plan.nodes[0].op
        else {
            panic!("expected a Choice");
        };
        assert_eq!(input_binding, &vec![BufRef::Input(0)]);
        assert_eq!(output_binding, &vec![BufRef::Output(0)]);
    }

    #[test]
    fn args_bulk_path_treats_pointers_as_read_write() {
        let mut b = PlanBuilder::new();
        let m = module(&mut b);
        let ws = b.scratch(32);
        let seed = b.kernel(m, "seed").write(ws).add();
        let k = b
            .kernel(m, "k")
            .args([Arg::input(0), Arg::workspace(ws.index())])
            .add();
        let plan = b.finish().unwrap();
        assert_eq!(plan.nodes[k.index()].deps, vec![seed.index()]);
    }

    #[test]
    fn splice_keeps_choice_and_remaps_its_bindings() {
        // A candidate carries its own module; the parent splice must not
        // re-index it (only `sub.modules` merge with +module_base).
        let candidate = |entry: &str| ExecutionPlan {
            modules: vec![KernelModule {
                cubin: vec![0xCC].into(),
            }],
            workspace: vec![],
            nodes: vec![Node {
                op: Op::KernelLaunch {
                    module: 0,
                    entry: entry.into(),
                    grid: [1, 1, 1],
                    block: [1, 1, 1],
                    shmem: 0,
                    args: vec![Arg::input(0), Arg::output(0), Arg::output(1)],
                },
                deps: vec![],
            }],
        };
        // `sub`'s single top-level node is a Choice naming `sub`'s formal slots:
        // input 0, output 0, and workspace 0 (its output binding slot 1).
        let sub = ExecutionPlan {
            modules: vec![KernelModule {
                cubin: vec![0xDD].into(),
            }],
            workspace: vec![WorkspaceBuf {
                nbytes: 128,
                init: None,
            }],
            nodes: vec![Node {
                op: Op::Choice {
                    candidates: vec![candidate("a"), candidate("b")],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Output(0), BufRef::Workspace(0)],
                },
                deps: vec![],
            }],
        };

        let mut b = PlanBuilder::new();
        let m = b.module(vec![0x11]);
        let parent_ws = b.scratch(64);
        let producer = b.kernel(m, "producer").write(Buf::Output(0)).add();

        let sinks = b.splice(
            &sub,
            &[BufRef::Input(0)],
            &[BufRef::Output(0), BufRef::Workspace(parent_ws.index())],
            &[producer],
        );
        let plan = b.finish().unwrap();

        // Top-level module merged (+module_base); candidate modules untouched.
        assert_eq!(plan.modules.len(), 2);
        // Parent workspace + sub workspace.
        assert_eq!(plan.workspace.len(), 2);
        assert_eq!(sinks, vec![NodeId(1)]);

        let spliced = &plan.nodes[1];
        assert_eq!(spliced.deps, vec![producer.index()]);
        let Op::Choice {
            candidates,
            input_binding,
            output_binding,
        } = &spliced.op
        else {
            panic!("the Choice must survive splicing");
        };
        // Bindings remapped through the parent: Input/Output pass through,
        // Workspace shifts by ws_base (parent had 1 workspace).
        assert_eq!(input_binding, &vec![BufRef::Input(0)]);
        assert_eq!(
            output_binding,
            &vec![BufRef::Output(0), BufRef::Workspace(1)]
        );
        // Candidates' internal nodes stay in the candidate's private slot space.
        assert_eq!(candidates.len(), 2);
        let Op::KernelLaunch { module, args, .. } = &candidates[0].nodes[0].op else {
            panic!("expected KernelLaunch in candidate");
        };
        assert_eq!(*module, 0);
        assert_eq!(args, &vec![Arg::input(0), Arg::output(0), Arg::output(1)]);

        plan.validate(1, 1).unwrap();
    }
}
