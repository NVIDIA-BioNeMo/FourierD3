// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kernel_compiler::buffer::{Buffer, buffer_nbytes};
use fourierd3_engine::ir::stmt::render_module;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::accumulator::{Accumulator, all_accumulators, atoms_per_block};
use super::{ScatterSpec, build_param_list, build_preamble, emit_full_grid_kernel};

pub(crate) struct CompileRequest<'a> {
    pub device_fn_src: &'a str,
    pub problem: &'a ScatterSpec,
    pub support: &'a [i16],
    pub grid_shape: [i64; 3],
    pub s_support: i32,
    pub opts: &'a [String],
}

fn output_nbytes(
    grid_out: &[Buffer],
    nongrid_out: &[Buffer],
    grid_shape: [i64; 3],
) -> Result<Vec<usize>, String> {
    let grid_volume: i64 = grid_shape.iter().product();
    let mut sizes = Vec::with_capacity(grid_out.len() + nongrid_out.len());
    for buf in grid_out {
        sizes.push(buffer_nbytes(buf, grid_volume)?);
    }
    for buf in nongrid_out {
        sizes.push(buffer_nbytes(buf, 1)?);
    }
    Ok(sizes)
}

pub(crate) fn compile_candidates(
    req: CompileRequest<'_>,
    grid_out: &[Buffer],
    nongrid_out: &[Buffer],
) -> Result<Vec<Candidate>, String> {
    let preamble = build_preamble(req.grid_shape, req.s_support, req.support);
    let output_nbytes = output_nbytes(grid_out, nongrid_out, req.grid_shape)?;

    let accumulators: Vec<Box<dyn Accumulator>> = all_accumulators()
        .into_iter()
        .filter(|accumulator| accumulator.is_applicable(req.problem))
        .collect();
    if accumulators.is_empty() {
        return Err("no applicable grid accumulator for the described problem".into());
    }

    let kernel_name = "grid";

    let mut pending: Vec<Pending> = Vec::new();
    for accumulator in &accumulators {
        let block_size = accumulator.block_size(req.problem);
        for k in accumulator.k_candidates(req.problem) {
            let mut problem_k = req.problem.clone();
            problem_k.k = k;

            let body = accumulator.emit_body(&problem_k);
            let params = build_param_list(&problem_k);
            let launch_bounds = accumulator.launch_bounds(block_size, k);
            let pre_kernel_src = accumulator.pre_kernel_source(&problem_k);
            let entry = format!("{kernel_name}_{}_k{}", accumulator.name(), k);
            let module = emit_full_grid_kernel(
                &preamble,
                req.device_fn_src,
                &pre_kernel_src,
                &launch_bounds,
                &params,
                &body,
                &entry,
            );
            pending.push(Pending {
                entry,
                source: render_module(&module),
                block_size,
                atoms_per_block: atoms_per_block(block_size, k),
                shmem: 0,
                shape: CandidateShape {
                    total: problem_k.batch_sizes.iter().product(),
                    n_grid_in: problem_k.layout.n_grid_in,
                    n_nongrid_in: problem_k.layout.n_nongrid_in,
                    n_index: problem_k.layout.n_index,
                    n_backend_arrays: problem_k.n_backend_arrays,
                    n_grid_out: problem_k.layout.n_grid_out,
                    nongrid_scatter_flags: problem_k.nongrid_scatter_flags.clone(),
                    output_nbytes: output_nbytes.clone(),
                },
            });
        }
    }

    let candidates: Vec<Candidate> = pending
        .into_par_iter()
        .map(|p| {
            let cubin = crate::kernel_compiler::cuda_toolchain::compile_cubin(
                &p.source,
                Some(p.entry.as_str()),
                req.opts,
                &[],
            )?;
            Ok(Candidate {
                kernel: CompiledKernel {
                    cubin,
                    entry: p.entry,
                },
                block_size: p.block_size,
                atoms_per_block: p.atoms_per_block,
                shmem: p.shmem,
                shape: p.shape,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(candidates)
}

struct CompiledKernel {
    cubin: Vec<u8>,
    entry: String,
}

struct CandidateShape {
    total: i64,
    n_grid_in: usize,
    n_nongrid_in: usize,
    n_index: usize,
    n_backend_arrays: usize,
    n_grid_out: usize,
    nongrid_scatter_flags: Vec<bool>,
    output_nbytes: Vec<usize>,
}

struct Pending {
    entry: String,
    source: Vec<u8>,
    block_size: i32,
    atoms_per_block: i32,
    shmem: u32,
    shape: CandidateShape,
}

pub(crate) struct Candidate {
    kernel: CompiledKernel,
    block_size: i32,
    atoms_per_block: i32,
    shmem: u32,
    shape: CandidateShape,
}

impl Candidate {
    pub(crate) fn n_inputs(&self) -> usize {
        grid_input_count(&self.shape)
    }

    pub(crate) fn n_outputs(&self) -> usize {
        self.shape.output_nbytes.len()
    }

    pub(crate) fn to_plan(&self) -> Result<crate::execution_plan::ExecutionPlan, String> {
        let s = &self.shape;
        let n_inputs = grid_input_count(s);
        let n_outputs = s.output_nbytes.len();

        let mut b = crate::execution_plan::PlanBuilder::new();
        let module = b.module(self.kernel.cubin.clone());

        let total = i32::try_from(s.total).map_err(|e| e.to_string())?;
        let n_params_ws = b.scratch_init(total.to_le_bytes().to_vec());

        for &j in &self.zero_outputs() {
            b.memset(crate::execution_plan::Buf::Output(j), 0, s.output_nbytes[j]);
        }

        let args = self.kernel_args(n_params_ws);
        b.kernel(module, &self.kernel.entry)
            .grid([self.grid_x()?, 1, 1])
            .block([self.block_size as u32, 1, 1])
            .shmem(self.shmem)
            .args(args)
            .add();

        let plan = b.finish().map_err(|e| format!("{e:?}"))?;
        plan.validate(n_inputs, n_outputs)
            .map_err(|e| format!("{e:?}"))?;
        Ok(plan)
    }

    fn zero_outputs(&self) -> Vec<usize> {
        let s = &self.shape;
        let mut zero = Vec::new();
        zero.extend(0..s.n_grid_out);
        for (k, &is_scatter) in s.nongrid_scatter_flags.iter().enumerate() {
            if is_scatter {
                zero.push(s.n_grid_out + k);
            }
        }
        zero
    }

    fn kernel_args(
        &self,
        n_params_ws: crate::execution_plan::WorkspaceId,
    ) -> Vec<crate::execution_plan::Arg> {
        let s = &self.shape;
        let mut args: Vec<crate::execution_plan::Arg> = Vec::new();
        let n_input_bufs = 1 + s.n_grid_in + s.n_nongrid_in + s.n_index;
        for i in 0..n_input_bufs {
            args.push(crate::execution_plan::Arg::input(i));
        }
        args.push(crate::execution_plan::Arg::workspace(n_params_ws.index()));
        for i in 0..s.n_backend_arrays {
            args.push(crate::execution_plan::Arg::input(n_input_bufs + i));
        }
        for j in 0..s.output_nbytes.len() {
            args.push(crate::execution_plan::Arg::output(j));
        }
        args
    }

    fn grid_x(&self) -> Result<u32, String> {
        if self.atoms_per_block <= 0 {
            return Err(format!(
                "atoms_per_block must be positive, got {}",
                self.atoms_per_block
            ));
        }
        let apb = self.atoms_per_block as i64;
        let g = (self.shape.total.max(0) + apb - 1) / apb;
        u32::try_from(g.max(1)).map_err(|_| format!("grid dimension {g} does not fit in u32"))
    }
}

fn grid_input_count(s: &CandidateShape) -> usize {
    1 + s.n_grid_in + s.n_nongrid_in + s.n_index + s.n_backend_arrays
}
