// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kernel_compiler::buffer::Buffer;

use super::{CompileRequest, IndexLayout, ScatterSpec, compile_candidates};

pub(crate) struct ScatterPlanRequest {
    pub device_fn_ir: String,
    pub support: Vec<i16>,
    pub cell_idx: Buffer,
    pub grid_in: Vec<Buffer>,
    pub nongrid_in: Vec<Buffer>,
    pub idx_bufs: Vec<Buffer>,
    pub grid_out: Vec<Buffer>,
    pub nongrid_out: Vec<Buffer>,
    pub batch_sizes: Vec<i64>,
    pub n_backend_arrays: usize,
    pub cartesian: Option<(i32, i32)>,
    pub grid_shape: [i64; 3],
    pub cell_grid_shape: Option<[i64; 3]>,
}

pub(crate) fn compile_scatter_plan(
    request: ScatterPlanRequest,
    opts: &[String],
) -> Result<crate::execution_plan::ExecutionPlan, String> {
    if !request.support.len().is_multiple_of(3) {
        return Err(format!(
            "support length {} must be a multiple of 3",
            request.support.len()
        ));
    }
    let s_support = i32::try_from(request.support.len() / 3)
        .map_err(|_| "support length does not fit in i32".to_string())?;
    let mut problem = problem_from_request(&request, s_support);
    let device_fn_src = consume_device_fn_ir(&mut problem, &request.device_fn_ir)?;
    let candidates = compile_candidates(
        CompileRequest {
            device_fn_src: &device_fn_src,
            problem: &problem,
            support: &request.support,
            grid_shape: request.grid_shape,
            s_support,
            opts,
        },
        &request.grid_out,
        &request.nongrid_out,
    )?;
    let first = candidates
        .first()
        .ok_or_else(|| "no grid candidate produced".to_string())?;
    let mut builder = crate::execution_plan::PlanBuilder::new();
    builder.whole_plan_choice(
        candidates
            .iter()
            .map(|candidate| candidate.to_plan())
            .collect::<Result<Vec<_>, _>>()?,
        first.n_inputs(),
        first.n_outputs(),
    );
    builder.finish().map_err(|error| format!("{error:?}"))
}

fn problem_from_request(request: &ScatterPlanRequest, s_support: i32) -> ScatterSpec {
    let groups: [&[Buffer]; 6] = [
        std::slice::from_ref(&request.cell_idx),
        &request.grid_in,
        &request.nongrid_in,
        &request.grid_out,
        &request.nongrid_out,
        &request.idx_bufs,
    ];
    let ic = groups
        .iter()
        .flat_map(|group| group.iter().map(|buffer| buffer.ic.clone()))
        .collect();
    let buf_batch_extents = groups
        .iter()
        .flat_map(|group| group.iter().map(|buffer| buffer.extents.clone()))
        .collect();
    let layout = IndexLayout {
        ic,
        n_grid_in: request.grid_in.len(),
        n_nongrid_in: request.nongrid_in.len(),
        n_grid_out: request.grid_out.len(),
        n_nongrid_out: request.nongrid_out.len(),
        n_index: request.idx_bufs.len(),
    };
    ScatterSpec {
        batch_sizes: request.batch_sizes.clone(),
        layout,
        buf_batch_extents,
        n_backend_arrays: request.n_backend_arrays,
        nongrid_in_sizes: request
            .nongrid_in
            .iter()
            .map(|buffer| buffer.elem_size)
            .collect(),
        nongrid_in_dtypes: request
            .nongrid_in
            .iter()
            .map(|buffer| buffer.dtype)
            .collect(),
        grid_in_inner_sizes: request
            .grid_in
            .iter()
            .map(|buffer| buffer.elem_size)
            .collect(),
        grid_in_dtypes: request.grid_in.iter().map(|buffer| buffer.dtype).collect(),
        grid_out_inner_sizes: request
            .grid_out
            .iter()
            .map(|buffer| buffer.elem_size)
            .collect(),
        grid_out_dtypes: request.grid_out.iter().map(|buffer| buffer.dtype).collect(),
        nongrid_out_sizes: request
            .nongrid_out
            .iter()
            .map(|buffer| buffer.elem_size)
            .collect(),
        nongrid_out_dtypes: request
            .nongrid_out
            .iter()
            .map(|buffer| buffer.dtype)
            .collect(),
        nongrid_scatter_flags: request
            .nongrid_out
            .iter()
            .map(|buffer| buffer.needs_scatter(&request.batch_sizes))
            .collect(),
        n_state: 0,
        state_sizes: Vec::new(),
        state_dtypes: Vec::new(),
        pre_ngin_indices: Vec::new(),
        direct_ngin_indices: Vec::new(),
        k: 1,
        s_support,
        cartesian: request.cartesian,
        grid_shape: request.grid_shape,
        cell_grid_shape: request
            .cell_grid_shape
            .filter(|shape| shape.iter().all(|&extent| extent > 0)),
    }
}

fn consume_device_fn_ir(problem: &mut ScatterSpec, ir_text: &str) -> Result<String, String> {
    let n_grid_in = problem.layout.n_grid_in;
    let n_nongrid_in = problem.layout.n_nongrid_in;
    let n_out = problem.layout.n_grid_out + problem.layout.n_nongrid_out;
    let nongrid_input_start = 2 + n_grid_in;
    let loop_varying: Vec<usize> = (0..nongrid_input_start).collect();
    let split =
        crate::kernel_compiler::llvm::split_and_lower_llvm_ir(ir_text, &loop_varying, n_out)?;

    problem.n_state = split.n_state;
    problem.state_sizes = vec![1; split.n_state];
    problem.state_dtypes = split.state_dtypes;
    problem.pre_ngin_indices = split
        .pre_indices
        .iter()
        .filter_map(|&parameter| {
            (nongrid_input_start..nongrid_input_start + n_nongrid_in)
                .contains(&parameter)
                .then_some(parameter - nongrid_input_start)
        })
        .collect();
    problem.direct_ngin_indices = split
        .direct_indices
        .iter()
        .filter_map(|&parameter| {
            (nongrid_input_start..nongrid_input_start + n_nongrid_in)
                .contains(&parameter)
                .then_some(parameter - nongrid_input_start)
        })
        .collect();
    Ok(format!("{}\n{}", split.pre_cuda, split.step_cuda))
}
