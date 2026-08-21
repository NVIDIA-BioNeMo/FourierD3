// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::cuda_driver::{CUcontext, CUdeviceptr, Context, DeviceBuffer, Kernel, Module};

use crate::execution_plan::{BufRef, ExecutionPlan, FlatPlan, Op, WritableBuf};

use crate::plan_executor::Error;
use crate::plan_executor::bind::Ports;
use crate::plan_executor::exec::{ExecSlot, GRAPH_RING_DEPTH, GraphRing, build_graph};

const STATIC_SHMEM_CAP: u32 = 49152;

fn apply_dynamic_shmem(kernel: &Kernel, shmem: u32) -> Result<(), Error> {
    if shmem <= STATIC_SHMEM_CAP {
        return Ok(());
    }
    kernel
        .set_max_dynamic_shared(shmem as i32)
        .map_err(Error::Driver)
}

pub(crate) struct ResidentPlan {
    ctx: CUcontext,
    pub(crate) plan: ExecutionPlan,
    // Field order is the drop order: the graph (and its execs/events) must be
    // freed before the kernels and modules whose code its nodes reference.
    pub(crate) graph: GraphRing,
    pub(crate) functions: HashMap<(usize, String), Kernel>,
    // Kept alive so the kernels (and their graph nodes) stay loaded.
    _modules: Vec<Module>,
}

impl ResidentPlan {
    pub(crate) unsafe fn new(ctx: CUcontext, flat: FlatPlan) -> Result<Self, Error> {
        let plan = flat.into_plan();
        let context = Context::from_raw(ctx);
        context.set_current().map_err(Error::Driver)?;

        let mut modules: Vec<Module> = Vec::with_capacity(plan.modules.len());
        for m in &plan.modules {
            modules.push(Module::load(&m.cubin).map_err(Error::Driver)?);
        }

        let mut functions: HashMap<(usize, String), Kernel> = HashMap::new();
        for node in &plan.nodes {
            if let Op::KernelLaunch {
                module,
                entry,
                shmem,
                ..
            } = &node.op
            {
                let key = (*module, entry.clone());
                if functions.contains_key(&key) {
                    continue;
                }
                let kernel = modules[*module]
                    .kernel(entry)
                    .map_err(|_| Error::MissingSymbol(entry.clone()))?;
                apply_dynamic_shmem(&kernel, *shmem)?;
                functions.insert(key, kernel);
            }
        }

        let placeholder = PlaceholderPorts::for_plan(ctx, &plan)?;
        let ports = placeholder.ports();
        let (graph, nodes) = build_graph(&plan, &functions, &ports, context)?;
        let mut execs: Vec<ExecSlot> = Vec::with_capacity(GRAPH_RING_DEPTH);
        for _ in 0..GRAPH_RING_DEPTH {
            let exec = graph.instantiate().map_err(Error::Driver)?;
            execs.push(ExecSlot { exec, done: None });
        }
        let graph = GraphRing {
            _graph: graph,
            nodes,
            execs,
            next: 0,
        };

        Ok(Self {
            ctx,
            plan,
            graph,
            functions,
            _modules: modules,
        })
    }

    pub(crate) fn ctx(&self) -> CUcontext {
        self.ctx
    }

    pub(crate) fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }
}

impl Drop for ResidentPlan {
    fn drop(&mut self) {
        // Wait out any in-flight launch so the events and graph execs are not
        // freed while the device still references them. The owned Event,
        // GraphExec, Graph and Module values free themselves afterward.
        for slot in &self.graph.execs {
            if let Some(done) = &slot.done {
                let _ = done.synchronize();
            }
        }
    }
}

struct PlaceholderPorts {
    inputs: Vec<CUdeviceptr>,
    outputs: Vec<CUdeviceptr>,
    workspace: Vec<CUdeviceptr>,
    _dummy: DeviceBuffer<u8>,
}

impl PlaceholderPorts {
    fn for_plan(ctx: CUcontext, plan: &ExecutionPlan) -> Result<Self, Error> {
        let mut n_inputs = 0usize;
        let mut n_outputs = 0usize;
        let mut dummy_bytes = 256usize;
        for node in &plan.nodes {
            match &node.op {
                Op::KernelLaunch { args, .. } => {
                    for arg in args {
                        match &arg.buf {
                            BufRef::Input(i) => n_inputs = n_inputs.max(i + 1),
                            BufRef::Output(i) => n_outputs = n_outputs.max(i + 1),
                            BufRef::Workspace(_) => {}
                        }
                    }
                }
                Op::Memset { target, nbytes, .. } => {
                    if let WritableBuf::Output(i) = target {
                        n_outputs = n_outputs.max(i + 1);
                    }
                    dummy_bytes = dummy_bytes.max(*nbytes);
                }
                Op::Choice { .. } => {}
            }
        }
        let dummy = DeviceBuffer::<u8>::alloc(ctx, dummy_bytes).map_err(Error::Driver)?;
        let p = dummy.ptr();
        Ok(Self {
            inputs: vec![p; n_inputs],
            outputs: vec![p; n_outputs],
            workspace: vec![p; plan.workspace.len()],
            _dummy: dummy,
        })
    }

    fn ports(&self) -> Ports<'_> {
        Ports {
            inputs: &self.inputs,
            outputs: &self.outputs,
            workspace: &self.workspace,
        }
    }
}
