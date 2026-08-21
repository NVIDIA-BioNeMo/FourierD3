// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::cuda_driver::{
    CUstream, Context, Event, Function, Graph, GraphExec, GraphNode, Kernel, KernelNode,
    LaunchConfig, MemsetNode, StreamRef,
};

use crate::execution_plan::{ExecutionPlan, Op};

use crate::plan_executor::bind::{Ports, buf_ptr, kernel_params, seed_workspace};
use crate::plan_executor::device::context_of_stream;
use crate::plan_executor::layout::carve;
use crate::plan_executor::{Bindings, Error, ResidentPlan};

fn function(
    functions: &HashMap<(usize, String), Kernel>,
    module: usize,
    entry: &str,
) -> Result<Function, Error> {
    functions
        .get(&(module, entry.to_string()))
        .map(|k| k.function())
        .ok_or_else(|| Error::MissingSymbol(entry.to_string()))
}

// 2 is enough to overlap one re-point with one in-flight launch.
pub(crate) const GRAPH_RING_DEPTH: usize = 2;

pub(crate) struct ExecSlot {
    pub(crate) exec: GraphExec,
    pub(crate) done: Option<Event>,
}

pub(crate) struct GraphRing {
    // Field order is the drop order: tear the execs (and their events) down
    // before the graph they were instantiated from.
    pub(crate) execs: Vec<ExecSlot>,
    // The graph is kept alive so its node handles stay valid for re-pointing.
    pub(crate) _graph: Graph,
    pub(crate) nodes: Vec<GraphNode>,
    pub(crate) next: usize,
}

pub(crate) unsafe fn execute(
    loaded: &mut ResidentPlan,
    bindings: &Bindings,
    stream: CUstream,
) -> Result<(), Error> {
    let workspace = carve(
        bindings.workspace,
        loaded.plan().workspace.iter().map(|w| w.nbytes),
    );
    let ports = Ports {
        inputs: bindings.inputs,
        outputs: bindings.outputs,
        workspace: &workspace,
    };
    unsafe { run(loaded, &ports, stream) }
}

pub(crate) unsafe fn run(
    loaded: &mut ResidentPlan,
    ports: &Ports,
    stream: CUstream,
) -> Result<(), Error> {
    // The stream is owned by the caller (XLA), so only borrow it.
    let stream = StreamRef::from_raw(stream);

    let ctx = loaded.ctx();
    if unsafe { context_of_stream(stream.raw())? } != ctx {
        return Err(Error::ContextMismatch);
    }
    Context::from_raw(ctx)
        .set_current()
        .map_err(Error::Driver)?;

    unsafe { seed_workspace(loaded.plan(), ports, stream.raw())? };

    if let [node] = loaded.plan.nodes.as_slice()
        && let Op::KernelLaunch {
            module,
            entry,
            grid,
            block,
            shmem,
            args,
        } = &node.op
    {
        let func = function(&loaded.functions, *module, entry)?;
        let mut params = kernel_params(args, ports)?;
        stream
            .launch(
                func,
                LaunchConfig {
                    grid: *grid,
                    block: *block,
                    shared_mem: *shmem,
                },
                params.as_mut_slice(),
            )
            .map_err(Error::Driver)?;
        return Ok(());
    }

    // Split the borrows: `repoint` reads the plan and function map while
    // mutating the ring, so hand it the pieces rather than `&mut loaded`.
    let plan = &loaded.plan;
    let functions = &loaded.functions;
    let ring = &mut loaded.graph;

    let i = ring.next;
    ring.next = (ring.next + 1) % ring.execs.len();

    if let Some(prev) = ring.execs[i].done.take() {
        prev.synchronize().map_err(Error::Driver)?;
    }

    repoint(&ring.execs[i].exec, &ring.nodes, plan, functions, ports)?;

    ring.execs[i].exec.launch(stream).map_err(Error::Driver)?;

    let done = Event::new_disable_timing().map_err(Error::Driver)?;
    done.record(stream).map_err(Error::Driver)?;
    ring.execs[i].done = Some(done);

    Ok(())
}

fn repoint(
    exec: &GraphExec,
    nodes: &[GraphNode],
    plan: &ExecutionPlan,
    functions: &HashMap<(usize, String), Kernel>,
    ports: &Ports,
) -> Result<(), Error> {
    for (idx, (node, &handle)) in plan.nodes.iter().zip(nodes).enumerate() {
        match &node.op {
            Op::KernelLaunch {
                module,
                entry,
                grid,
                block,
                shmem,
                args,
            } => {
                let func = function(functions, *module, entry)?;
                // The exec copies the params at the call, so the backing
                // store in `params` only needs to outlive this call.
                let params = kernel_params(args, ports)?;
                exec.set_kernel_node_params(
                    handle,
                    &KernelNode {
                        func,
                        grid: *grid,
                        block: *block,
                        shared_mem: *shmem,
                        params: params.as_slice(),
                    },
                )
                .map_err(Error::Driver)?;
            }
            Op::Memset {
                target,
                value,
                nbytes,
            } => {
                let dst = buf_ptr(ports, target)?;
                exec.set_memset_node_params(
                    handle,
                    &MemsetNode {
                        dst,
                        value: *value,
                        nbytes: *nbytes,
                    },
                )
                .map_err(Error::Driver)?;
            }
            Op::Choice { .. } => return Err(Error::UnresolvedChoice { node: idx }),
        }
    }
    Ok(())
}

pub(crate) fn build_graph(
    plan: &ExecutionPlan,
    functions: &HashMap<(usize, String), Kernel>,
    ports: &Ports,
    ctx: Context,
) -> Result<(Graph, Vec<GraphNode>), Error> {
    let mut graph = Graph::new(ctx).map_err(Error::Driver)?;
    let mut handles: Vec<GraphNode> = Vec::with_capacity(plan.nodes.len());

    for (idx, node) in plan.nodes.iter().enumerate() {
        let deps: Vec<GraphNode> = node.deps.iter().map(|&d| handles[d]).collect();

        let handle = match &node.op {
            Op::KernelLaunch {
                module,
                entry,
                grid,
                block,
                shmem,
                args,
            } => {
                let func = function(functions, *module, entry)?;
                // The graph copies the params at add time, so the backing
                // store in `params` only needs to outlive the add call.
                let params = kernel_params(args, ports)?;
                graph
                    .add_kernel_node(
                        &deps,
                        &KernelNode {
                            func,
                            grid: *grid,
                            block: *block,
                            shared_mem: *shmem,
                            params: params.as_slice(),
                        },
                    )
                    .map_err(Error::Driver)?
            }
            Op::Memset {
                target,
                value,
                nbytes,
            } => {
                let dst = buf_ptr(ports, target)?;
                graph
                    .add_memset_node(
                        &deps,
                        &MemsetNode {
                            dst,
                            value: *value,
                            nbytes: *nbytes,
                        },
                    )
                    .map_err(Error::Driver)?
            }
            Op::Choice { .. } => return Err(Error::UnresolvedChoice { node: idx }),
        };

        handles.push(handle);
    }

    Ok((graph, handles))
}
