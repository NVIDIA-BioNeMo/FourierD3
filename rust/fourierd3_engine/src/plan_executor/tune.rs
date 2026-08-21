// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::cuda_driver::{CUcontext, CUdeviceptr, CUstream, Context, Event, StreamRef};

use crate::execution_plan::{BufRef, ExecutionPlan, Op};

use crate::execution_plan::FlatPlan;
use crate::execution_plan::layout::workspace_offsets;
use crate::plan_executor::bind::{Ports, buf_ptr, bufref_ptr};
use crate::plan_executor::device::context_of_stream;
use crate::plan_executor::exec::run;
use crate::plan_executor::layout::carve;
use crate::plan_executor::selection::{Selection, choice_indices, default_selection, inline};
use crate::plan_executor::{Bindings, Error, ResidentPlan};

const TARGET_BATCH_NS: u64 = 250_000;

const MAX_PACK: usize = 64;

const WARMUP: usize = 3;

const BATCHES: usize = 5;

const PROBE_ABORT_FACTOR: u64 = 10;
const BATCH_ABORT_FACTOR: u64 = 2;

/// Per-candidate autotune timing, off by default (`tune`); opt in via
/// `tune_reported`.
#[derive(Default, Debug)]
pub(crate) struct TuneReport {
    pub choices: Vec<ChoiceReport>,
}

#[derive(Debug)]
pub(crate) struct ChoiceReport {
    pub depth: usize,
    pub winner: Option<usize>,
    pub candidates: Vec<CandidateReport>,
}

#[derive(Debug)]
pub(crate) struct CandidateReport {
    pub ns: Option<u64>,
    /// The measurement was reused from an identical earlier candidate.
    pub cached: bool,
    pub entries: Vec<String>,
}

/// Every `Op::KernelLaunch` entry point in a flat plan, in node order.
pub(crate) fn kernel_entries(plan: &ExecutionPlan) -> Vec<String> {
    plan.nodes
        .iter()
        .filter_map(|node| match &node.op {
            Op::KernelLaunch { entry, .. } => Some(entry.clone()),
            _ => None,
        })
        .collect()
}

fn parent_ptr(
    buf: &BufRef,
    inputs: &[CUdeviceptr],
    outputs: &[CUdeviceptr],
    parent_workspace: &[CUdeviceptr],
) -> std::result::Result<CUdeviceptr, Error> {
    match buf {
        BufRef::Input(i) => inputs.get(*i).copied().ok_or(Error::InputUnbound(*i)),
        BufRef::Output(i) => outputs.get(*i).copied().ok_or(Error::OutputUnbound(*i)),
        BufRef::Workspace(i) => parent_workspace
            .get(*i)
            .copied()
            .ok_or(Error::WorkspaceUnbound(*i)),
    }
}

/// A completed measurement went through every batch; an aborted one carries a
/// ceiling-truncated value that must not be reused for another choice.
unsafe fn time_plan(
    loaded: &mut ResidentPlan,
    ports: &Ports,
    stream: CUstream,
    probe_ceil: Option<u64>,
    batch_ceil: Option<u64>,
) -> std::result::Result<(u64, bool), Error> {
    let stream_ref = StreamRef::from_raw(stream);
    let start = Event::new().map_err(Error::Driver)?;
    let stop = Event::new().map_err(Error::Driver)?;

    let bracket = |loaded: &mut ResidentPlan, n: usize| -> std::result::Result<u64, Error> {
        start.record(stream_ref).map_err(Error::Driver)?;
        for _ in 0..n {
            unsafe { run(loaded, ports, stream) }?;
        }
        stop.record(stream_ref).map_err(Error::Driver)?;
        stop.synchronize().map_err(Error::Driver)?;
        let ms = stop.elapsed_since(&start).map_err(Error::Driver)?;
        Ok((ms as f64 * 1.0e6 / n as f64) as u64)
    };

    let probe = bracket(loaded, 1)?;
    if probe_ceil.is_some_and(|c| probe > c) {
        return Ok((probe, false));
    }
    let pack = (TARGET_BATCH_NS / probe.max(1)).clamp(1, MAX_PACK as u64) as usize;

    for _ in 0..WARMUP {
        unsafe { run(loaded, ports, stream) }?;
    }
    let mut samples = [0; BATCHES];
    let mut min_ns = u64::MAX;
    for sample in &mut samples {
        *sample = bracket(loaded, pack)?;
        min_ns = min_ns.min(*sample);
        if batch_ceil.is_some_and(|ceiling| min_ns > ceiling) {
            return Ok((min_ns, false));
        }
    }
    samples.sort_unstable();
    Ok((samples[BATCHES / 2], true))
}

/// Two candidates with identical execution content — modules, workspace,
/// nodes — bound to identical resolved addresses are the same program, so the
/// first complete measurement serves every recurrence. `None` when a binding
/// does not resolve (the run would fail the same way).
fn candidate_key(flat: &ExecutionPlan, ports: &Ports) -> Option<u64> {
    let mut hasher = DefaultHasher::new();
    for module in &flat.modules {
        module.cubin.as_ref().hash(&mut hasher);
    }
    for buf in &flat.workspace {
        buf.nbytes.hash(&mut hasher);
        buf.init
            .as_ref()
            .map(|init| init.as_ref())
            .hash(&mut hasher);
    }
    for node in &flat.nodes {
        node.deps.hash(&mut hasher);
        match &node.op {
            Op::KernelLaunch {
                module,
                entry,
                grid,
                block,
                shmem,
                args,
            } => {
                (0u8, module, entry, grid, block, shmem).hash(&mut hasher);
                for arg in args {
                    let ptr = bufref_ptr(ports, &arg.buf).ok()?;
                    (ptr + arg.offset as CUdeviceptr).hash(&mut hasher);
                }
            }
            Op::Memset {
                target,
                value,
                nbytes,
            } => (1u8, buf_ptr(ports, target).ok()?, value, nbytes).hash(&mut hasher),
            Op::Choice { .. } => return None,
        }
    }
    Some(hasher.finish())
}

struct CandidateRun {
    nested: Vec<Selection>,
    ns: u64,
}

struct CandidateMeasurement {
    run: CandidateRun,
    entries: Vec<String>,
    cached: bool,
}

struct CandidateLimits {
    bound: Option<u64>,
    probe: Option<u64>,
    batch: Option<u64>,
}

struct ChoiceState {
    best: Option<(usize, CandidateRun)>,
    last_error: Option<Error>,
    pruned: bool,
    report: ChoiceReport,
}

impl ChoiceState {
    fn new(depth: usize, candidates: usize) -> Self {
        Self {
            best: None,
            last_error: None,
            pruned: false,
            report: ChoiceReport {
                depth,
                winner: None,
                candidates: Vec::with_capacity(candidates),
            },
        }
    }

    fn best_ns(&self) -> Option<u64> {
        self.best.as_ref().map(|(_, run)| run.ns)
    }

    fn record(
        &mut self,
        index: usize,
        measured: std::result::Result<Option<CandidateMeasurement>, Error>,
        include_time: bool,
    ) {
        match measured {
            Ok(Some(measurement)) => {
                self.report.candidates.push(CandidateReport {
                    ns: include_time.then_some(measurement.run.ns),
                    cached: measurement.cached,
                    entries: measurement.entries,
                });
                if self.best_ns().is_none_or(|ns| measurement.run.ns < ns) {
                    self.best = Some((index, measurement.run));
                }
            }
            Ok(None) => {
                self.pruned = true;
                self.report.candidates.push(failed_candidate_report());
            }
            Err(error) => {
                self.last_error = Some(error);
                self.report.candidates.push(failed_candidate_report());
            }
        }
    }

    fn finish(mut self) -> std::result::Result<Option<(usize, CandidateRun, ChoiceReport)>, Error> {
        let Some((index, run)) = self.best else {
            if self.pruned {
                return Ok(None);
            }
            return Err(self
                .last_error
                .expect("a Choice always has at least one candidate"));
        };
        self.report.winner = Some(index);
        Ok(Some((index, run, self.report)))
    }
}

fn failed_candidate_report() -> CandidateReport {
    CandidateReport {
        ns: None,
        cached: false,
        entries: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn pick_winners(
    ctx: CUcontext,
    plan: &ExecutionPlan,
    in_ptrs: &[CUdeviceptr],
    out_ptrs: &[CUdeviceptr],
    ws_base: CUdeviceptr,
    stream: CUstream,
    bound: Option<u64>,
    depth: usize,
    memo: &mut HashMap<u64, u64>,
    mut report: Option<&mut TuneReport>,
) -> std::result::Result<Option<Vec<Selection>>, Error> {
    let choices = choice_indices(plan);
    if choices.is_empty() {
        return Ok(Some(Vec::new()));
    }

    // Candidates are timed against the caller's real buffers: the plan's own
    // workspace at the base of the bound region, each candidate's private
    // workspace carved right after it — the same arena, and for the winner the
    // same addresses, that execution will use. The region covers every
    // selection by `workspace_upper_bound`, which sizes the binding.
    let own_sizes: Vec<usize> = plan.workspace.iter().map(|w| w.nbytes).collect();
    let (_, own_total) = workspace_offsets(own_sizes.iter().copied());
    let own_ws = carve(ws_base, own_sizes.iter().copied());
    let child_base = ws_base + own_total as CUdeviceptr;

    unsafe { warm_up_default(ctx, plan, in_ptrs, out_ptrs, &own_ws, child_base, stream)? };

    let mut winners = Vec::with_capacity(choices.len());
    let mut committed = 0u64;
    for &ci in &choices {
        let Op::Choice {
            candidates,
            input_binding,
            output_binding,
        } = &plan.nodes[ci].op
        else {
            unreachable!("choice_indices points only at Choice nodes")
        };
        let c_in = resolve_bindings(input_binding, in_ptrs, out_ptrs, &own_ws)?;
        let c_out = resolve_bindings(output_binding, in_ptrs, out_ptrs, &own_ws)?;
        let mut state = ChoiceState::new(depth, candidates.len());
        // A lone candidate needs no measurement to win; it still recurses for
        // its nested selections. Under a bound its time still feeds
        // `committed` — that is what strangles doomed sibling plans early —
        // so one probe run (a zero probe ceiling stops after it) estimates it;
        // without a bound the estimate would inform nothing and no run
        // happens.
        let sole = candidates.len() == 1;
        let measure = !sole || bound.is_some();
        let residual = bound.map(|b| b.saturating_sub(committed));
        for (c, cand) in candidates.iter().enumerate() {
            let limits = candidate_limits(sole, residual, state.best_ns());
            let timed = unsafe {
                measure_candidate(
                    ctx,
                    cand,
                    &c_in,
                    &c_out,
                    child_base,
                    stream,
                    limits,
                    depth,
                    memo,
                    report.as_deref_mut(),
                    measure,
                )
            };
            state.record(c, timed, measure);
        }
        let Some((bc, run, choice_report)) = state.finish()? else {
            return Ok(None);
        };
        if let Some(r) = report.as_deref_mut() {
            r.choices.push(choice_report);
        }
        winners.push(Selection {
            candidate: bc,
            nested: run.nested,
        });
        committed = committed.saturating_add(run.ns);
        if bound.is_some_and(|b| committed >= b) {
            return Ok(None);
        }
    }
    Ok(Some(winners))
}

#[allow(clippy::too_many_arguments)]
unsafe fn warm_up_default(
    ctx: CUcontext,
    plan: &ExecutionPlan,
    inputs: &[CUdeviceptr],
    outputs: &[CUdeviceptr],
    own_workspace: &[CUdeviceptr],
    child_base: CUdeviceptr,
    stream: CUstream,
) -> std::result::Result<(), Error> {
    let flat = inline(plan, &default_selection(plan));
    let private_sizes = flat.workspace[own_workspace.len()..]
        .iter()
        .map(|workspace| workspace.nbytes);
    let mut workspace = own_workspace.to_vec();
    workspace.extend(carve(child_base, private_sizes));
    let ports = Ports {
        inputs,
        outputs,
        workspace: &workspace,
    };
    let mut loaded = unsafe { ResidentPlan::new(ctx, FlatPlan::assume_flat(flat))? };
    unsafe { run(&mut loaded, &ports, stream)? };
    StreamRef::from_raw(stream)
        .synchronize()
        .map_err(Error::Driver)
}

fn resolve_bindings(
    bindings: &[BufRef],
    inputs: &[CUdeviceptr],
    outputs: &[CUdeviceptr],
    workspace: &[CUdeviceptr],
) -> std::result::Result<Vec<CUdeviceptr>, Error> {
    bindings
        .iter()
        .map(|binding| parent_ptr(binding, inputs, outputs, workspace))
        .collect()
}

fn candidate_limits(sole: bool, residual: Option<u64>, best: Option<u64>) -> CandidateLimits {
    CandidateLimits {
        bound: tighter(residual, best),
        probe: (!sole)
            .then(|| best.map(|ns| ns.saturating_mul(PROBE_ABORT_FACTOR)))
            .flatten()
            .or(sole.then_some(0)),
        batch: (!sole)
            .then(|| {
                tighter(
                    best.map(|ns| ns.saturating_mul(BATCH_ABORT_FACTOR)),
                    residual,
                )
            })
            .flatten(),
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn measure_candidate(
    ctx: CUcontext,
    candidate: &ExecutionPlan,
    inputs: &[CUdeviceptr],
    outputs: &[CUdeviceptr],
    workspace_base: CUdeviceptr,
    stream: CUstream,
    limits: CandidateLimits,
    depth: usize,
    memo: &mut HashMap<u64, u64>,
    report: Option<&mut TuneReport>,
    measure: bool,
) -> std::result::Result<Option<CandidateMeasurement>, Error> {
    let Some(nested) = (unsafe {
        pick_winners(
            ctx,
            candidate,
            inputs,
            outputs,
            workspace_base,
            stream,
            limits.bound,
            depth + 1,
            memo,
            report,
        )?
    }) else {
        return Ok(None);
    };
    let flat = inline(candidate, &nested);
    let entries = kernel_entries(&flat);
    if !measure {
        return Ok(Some(CandidateMeasurement {
            run: CandidateRun { nested, ns: 0 },
            entries,
            cached: false,
        }));
    }
    let workspace_sizes = flat.workspace.iter().map(|workspace| workspace.nbytes);
    let workspace = carve(workspace_base, workspace_sizes);
    let ports = Ports {
        inputs,
        outputs,
        workspace: &workspace,
    };
    let key = candidate_key(&flat, &ports);
    if let Some(ns) = key.and_then(|key| memo.get(&key)).copied() {
        return Ok(Some(CandidateMeasurement {
            run: CandidateRun { nested, ns },
            entries,
            cached: true,
        }));
    }
    let mut loaded = unsafe { ResidentPlan::new(ctx, FlatPlan::assume_flat(flat))? };
    let (ns, complete) =
        unsafe { time_plan(&mut loaded, &ports, stream, limits.probe, limits.batch)? };
    if complete && let Some(key) = key {
        memo.insert(key, ns);
    }
    Ok(Some(CandidateMeasurement {
        run: CandidateRun { nested, ns },
        entries,
        cached: false,
    }))
}

fn tighter(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

unsafe fn tune_impl(
    plan: &ExecutionPlan,
    bindings: &Bindings,
    stream: CUstream,
    mut report: Option<TuneReport>,
) -> std::result::Result<(FlatPlan, Option<TuneReport>), Error> {
    if choice_indices(plan).is_empty() {
        return Ok((FlatPlan::assume_flat(plan.clone()), report));
    }

    let ctx = unsafe { context_of_stream(stream)? };
    Context::from_raw(ctx)
        .set_current()
        .map_err(Error::Driver)?;

    let mut memo = HashMap::new();
    let winners = unsafe {
        pick_winners(
            ctx,
            plan,
            bindings.inputs,
            bindings.outputs,
            bindings.workspace,
            stream,
            None,
            0,
            &mut memo,
            report.as_mut(),
        )?
    }
    .expect("the top-level plan has no bound, so it is never pruned");
    let mut flat = inline(plan, &winners);
    // The winner re-owns its payload bytes so the pending plan's backing
    // buffer (every loser candidate) can be released.
    flat.detach_blobs();
    Ok((FlatPlan::assume_flat(flat), report))
}

/// Tune every `Choice` against the caller's real bindings — the buffers the
/// tuned plan will execute with. Candidate runs write `bindings.outputs`
/// and the workspace; the caller executes the winner afterwards, so both are
/// scratch until then. The workspace must cover the plan's
/// `workspace_upper_bound`.
pub(crate) unsafe fn tune(
    plan: &ExecutionPlan,
    bindings: &Bindings,
    stream: CUstream,
) -> std::result::Result<FlatPlan, Error> {
    unsafe { tune_impl(plan, bindings, stream, None) }.map(|(flat, _)| flat)
}

/// Same as `tune`, but also returns per-candidate timing telemetry: an empty
/// report when `plan` has no `Choice` nodes.
pub(crate) unsafe fn tune_reported(
    plan: &ExecutionPlan,
    bindings: &Bindings,
    stream: CUstream,
) -> std::result::Result<(FlatPlan, TuneReport), Error> {
    let (flat, report) = unsafe { tune_impl(plan, bindings, stream, Some(TuneReport::default())) }?;
    Ok((flat, report.unwrap_or_default()))
}
