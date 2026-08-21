// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::dtype::Dtype;

pub(crate) mod emit_cuda;
pub(crate) mod hoist_loop_invariants;
pub(crate) mod parse;

pub(crate) use emit_cuda::emit_cuda;
pub(crate) use hoist_loop_invariants::split_loop_invariant;
pub(crate) use parse::{LlvmModule, parse_module};

#[derive(Debug, Clone)]
pub(crate) struct LoweredLicmSplit {
    pub pre_cuda: String,
    pub step_cuda: String,
    pub n_state: usize,
    pub state_dtypes: Vec<Dtype>,
    pub pre_indices: Vec<usize>,
    pub direct_indices: Vec<usize>,
}

pub(crate) fn split_and_lower_llvm_ir(
    ir_text: &str,
    loop_varying_param_indices: &[usize],
    n_out: usize,
) -> Result<LoweredLicmSplit, String> {
    let module = parse_module(ir_text)?;
    let split = split_loop_invariant(&module.function, loop_varying_param_indices, n_out)?;
    let n_pre_in = split.pre_indices.len();

    // Partition globals: each `__constant__` table lives in the
    // emitted CUDA of whichever half (pre / step) actually accesses
    // it. If both halves access the same table, attach to pre and
    // also leave a `extern __constant__` declaration in step so its
    // body can read it.
    let pre_globals = globals_referenced(&module.globals, &split.pre.instrs);
    let step_globals = globals_referenced(&module.globals, &split.step.instrs);
    let pre_names: std::collections::HashSet<&str> =
        pre_globals.iter().map(|g| g.name.as_str()).collect();
    let step_only: Vec<parse::LlvmGlobal> = step_globals
        .into_iter()
        .filter(|g| !pre_names.contains(g.name.as_str()))
        .collect();

    let pre_module = LlvmModule {
        globals: pre_globals,
        function: split.pre.clone(),
    };
    let step_module = LlvmModule {
        globals: step_only,
        function: split.step.clone(),
    };
    let pre_cuda = emit_cuda(&pre_module, n_pre_in);
    let n_step_inputs = split.step.params.len() - n_out;
    let step_cuda = emit_cuda(&step_module, n_step_inputs);
    let state_dtypes: Vec<Dtype> = split
        .pre
        .params
        .iter()
        .skip(n_pre_in)
        .map(|p| p.elem_ty.dtype())
        .collect();
    Ok(LoweredLicmSplit {
        pre_cuda,
        step_cuda,
        n_state: split.n_state,
        state_dtypes,
        pre_indices: split.pre_indices,
        direct_indices: split.direct_indices,
    })
}

pub(crate) fn globals_referenced(
    globals: &[parse::LlvmGlobal],
    instrs: &[parse::Instr],
) -> Vec<parse::LlvmGlobal> {
    let mut used: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for i in instrs {
        if let parse::Instr::Gep {
            base,
            base_is_global,
            ..
        } = i
            && *base_is_global
        {
            used.insert(base.as_str());
        }
    }
    globals
        .iter()
        .filter(|g| used.contains(g.name.as_str()))
        .cloned()
        .collect()
}
