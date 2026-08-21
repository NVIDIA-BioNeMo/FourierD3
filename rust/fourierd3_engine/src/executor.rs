// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loads serialized execution plans onto the device and runs them behind the
//! XLA FFI, autotuning the first time a plan is bound to real buffers.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyBytes, PyModule, PyModuleMethods};
use pyo3::{Bound, Py, PyErr, PyObject, PyResult, Python, pyfunction, wrap_pyfunction};

use crate::cuda_driver::{CUcontext, CUdeviceptr, CUstream, StreamRef};
use crate::execution_plan::{ExecutionPlan, serialize};
use crate::plan_executor::{
    Bindings, ChoiceReport, ResidentPlan, TuneReport, execute, kernel_entries, tune, tune_reported,
};
use crate::xla_ffi::{Error, RemainingArgs, RemainingRets, Result, Stream, handler};

pub(crate) enum RunnablePlan {
    Pending {
        source: ExecutionPlan,
        report_to: Option<PathBuf>,
    },
    Resident(ResidentPlan),
}

fn json_escape(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
}

fn json_string_array(out: &mut String, items: &[String]) {
    out.push('[');
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        json_escape(out, item);
    }
    out.push(']');
}

fn write_tune_report(path: &Path, report: &TuneReport, flat: &ExecutionPlan) {
    // Fresh measurements only: memo-cached candidates cost no GPU time.
    let num_timed: usize = report
        .choices
        .iter()
        .flat_map(|c| &c.candidates)
        .filter(|c| c.ns.is_some() && !c.cached)
        .count();

    let mut out = String::new();
    out.push_str(&format!("{{\"num_timed\":{num_timed},\"choices\":["));
    for (i, choice) in report.choices.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let ChoiceReport {
            depth,
            winner,
            candidates,
        } = choice;
        out.push_str(&format!("{{\"depth\":{depth},\"winner\":"));
        match winner {
            Some(w) => out.push_str(&w.to_string()),
            None => out.push_str("null"),
        }
        out.push_str(",\"candidates\":[");
        for (j, cand) in candidates.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str("{\"ns\":");
            match cand.ns {
                Some(ns) => out.push_str(&ns.to_string()),
                None => out.push_str("null"),
            }
            out.push_str(&format!(",\"cached\":{}", cand.cached));
            out.push_str(",\"entries\":");
            json_string_array(&mut out, &cand.entries);
            out.push('}');
        }
        out.push_str("]}");
    }
    out.push_str("],\"final_entries\":");
    json_string_array(&mut out, &kernel_entries(flat));
    out.push('}');

    if let Err(e) = std::fs::write(path, out) {
        eprintln!(
            "[{}] failed to write tune report to {}: {e}",
            env!("CARGO_PKG_NAME"),
            path.display()
        );
    }
}

unsafe fn set_stream_context(stream: CUstream) -> Result<CUcontext> {
    let ctx = StreamRef::from_raw(stream)
        .context()
        .map_err(|e| Error::internal(format!("cuStreamGetCtx failed: {e}")))?;
    ctx.set_current()
        .map_err(|e| Error::internal(format!("cuCtxSetCurrent failed: {e}")))?;
    Ok(ctx.raw())
}

unsafe fn plan_handle_from_u64(handle: u64) -> &'static Mutex<RunnablePlan> {
    unsafe { &*(handle as usize as *const Mutex<RunnablePlan>) }
}

#[handler]
pub fn run_plan(
    stream: Stream,
    #[attr("plan")] plan: u64,
    inputs: RemainingArgs<'_>,
    outputs: RemainingRets<'_>,
) -> Result<()> {
    let handle = unsafe { plan_handle_from_u64(plan) };

    let in_ptrs: Vec<CUdeviceptr> = inputs.iter().map(|b| b.data() as CUdeviceptr).collect();
    let mut raw_out_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(outputs.len());
    let mut workspace_size = None;
    for b in outputs.iter() {
        if workspace_size.is_none() {
            workspace_size = Some(b.size_bytes()?);
        }
        raw_out_ptrs.push(b.data() as CUdeviceptr);
    }

    let Some((&workspace_base, real_out_ptrs)) = raw_out_ptrs.split_first() else {
        return Err(Error::invalid_argument("missing workspace output"));
    };
    let workspace_size = workspace_size.expect("outputs are non-empty past the split");

    let stream = stream.0 as CUstream;

    let mut state = handle
        .lock()
        .map_err(|_| Error::internal("plan handle lock poisoned"))?;

    if let RunnablePlan::Pending { source, report_to } = &*state {
        // Tuning times candidates against the real bindings, so the buffer
        // must already cover the pending plan's worst selection.
        let required = crate::execution_plan::layout::workspace_upper_bound(source);
        if workspace_size < required {
            return Err(Error::invalid_argument(format!(
                "workspace output is {workspace_size} bytes, plan needs {required}"
            )));
        }
        let ctx = unsafe { set_stream_context(stream)? };
        let bindings = Bindings {
            inputs: &in_ptrs,
            outputs: real_out_ptrs,
            workspace: workspace_base,
        };
        let flat = if let Some(report_path) = report_to {
            let (flat, report) = unsafe { tune_reported(source, &bindings, stream) }
                .map_err(|e| Error::internal(format!("tune failed: {e:?}")))?;
            write_tune_report(report_path, &report, &flat);
            flat
        } else {
            unsafe { tune(source, &bindings, stream) }
                .map_err(|e| Error::internal(format!("tune failed: {e:?}")))?
        };
        let resident = unsafe { ResidentPlan::new(ctx, flat) }
            .map_err(|e| Error::internal(format!("load failed: {e:?}")))?;
        *state = RunnablePlan::Resident(resident);
    }
    let RunnablePlan::Resident(loaded) = &mut *state else {
        unreachable!("just transitioned to Resident");
    };

    // The resident plan declares its full workspace (including the winning
    // candidates' formerly-private scratch); validate the XLA buffer covers it.
    let required = crate::execution_plan::layout::workspace_upper_bound(loaded.plan());
    if workspace_size < required {
        return Err(Error::invalid_argument(format!(
            "workspace output is {workspace_size} bytes, plan needs {required}"
        )));
    }

    let bindings = Bindings {
        inputs: &in_ptrs,
        outputs: real_out_ptrs,
        workspace: workspace_base,
    };
    unsafe { execute(loaded, &bindings, stream) }
        .map_err(|e| Error::internal(format!("execute failed: {e:?}")))?;
    Ok(())
}

fn serialized_tuned_plan(
    handle: &Mutex<RunnablePlan>,
) -> std::result::Result<Option<Vec<u8>>, crate::execution_plan::WireError> {
    match &*handle.lock().expect("plan handle lock poisoned") {
        RunnablePlan::Pending { .. } => Ok(None),
        RunnablePlan::Resident(resident) => serialize(resident.plan()).map(Some),
    }
}

unsafe fn capsule_for_fn(py: Python<'_>, fn_ptr: *mut c_void) -> PyResult<PyObject> {
    let cap = unsafe { pyo3::ffi::PyCapsule_New(fn_ptr, std::ptr::null(), None) };
    if cap.is_null() {
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { PyObject::from_owned_ptr(py, cap) })
}

#[pyfunction]
fn run_plan_capsule(py: Python<'_>) -> PyResult<PyObject> {
    unsafe { capsule_for_fn(py, run_plan as *mut c_void) }
}

/// Returns `(handle, workspace_nbytes)`.
#[pyfunction]
#[pyo3(signature = (data, tune_report_to=None))]
fn load_plan(data: &[u8], tune_report_to: Option<PathBuf>) -> PyResult<(u64, usize)> {
    let source = crate::execution_plan::Blob::from_vec(data.to_vec());
    let plan = crate::execution_plan::deserialize_shared(source)
        .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
    let workspace_nbytes = crate::execution_plan::layout::workspace_upper_bound(&plan);
    let handle = Box::new(Mutex::new(RunnablePlan::Pending {
        source: plan,
        report_to: tune_report_to,
    }));
    Ok((Box::into_raw(handle) as usize as u64, workspace_nbytes))
}

#[pyfunction]
fn tuned_plan_bytes(py: Python<'_>, handle: u64) -> PyResult<Option<Py<PyBytes>>> {
    let handle = unsafe { plan_handle_from_u64(handle) };
    let bytes =
        serialized_tuned_plan(handle).map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
    Ok(bytes.map(|bytes| PyBytes::new_bound(py, &bytes).unbind()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_plan_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(load_plan, m)?)?;
    m.add_function(wrap_pyfunction!(tuned_plan_bytes, m)?)?;
    Ok(())
}
