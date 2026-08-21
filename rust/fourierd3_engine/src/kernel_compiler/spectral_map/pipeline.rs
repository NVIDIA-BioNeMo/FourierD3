// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::kernel_compiler::spectral_map::forward_zy_slabs::{self, ZyFwdSlabSpec};
use crate::kernel_compiler::spectral_map::fused_x_transform::{self, XFusedSpec};
use crate::kernel_compiler::spectral_map::inverse_yz_slabs::{self, YzInvSlabSpec};
use crate::kernel_compiler::spectral_map::separate_x_transform;
use crate::kernel_compiler::spectral_map::specification::{self, SpectralMapSpec, complex_bytes};
use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::stmt::{Stmt, render_module};

#[derive(Clone, Debug)]
struct CompiledKernel {
    cubin: Vec<u8>,
    entry: String,
}

impl CompiledKernel {
    fn compile(
        module: &[Stmt],
        entry: &str,
        ltoir_blobs: &[&[u8]],
        opts: &[String],
    ) -> Result<Self, String> {
        let source = render_module(module);
        let cubin = crate::kernel_compiler::cuda_toolchain::compile_cubin(
            &source,
            Some(entry),
            opts,
            ltoir_blobs,
        )?;
        Ok(Self {
            cubin,
            entry: entry.to_string(),
        })
    }
}

#[derive(Clone, Debug)]
struct Launch {
    grid: [u32; 3],
    block: [u32; 3],
    shmem: u32,
}

#[derive(Clone, Debug)]
struct FwdGroup {
    grid_indices: Vec<u32>,
    variants: Vec<(CompiledKernel, Launch)>,
}

#[derive(Clone, Debug)]
struct InvGroup {
    grid_output_index: u32,
    variants: Vec<(CompiledKernel, Launch)>,
}

#[derive(Clone, Debug)]
struct DefusedX {
    fwd: (CompiledKernel, Launch),
    contract: (CompiledKernel, Launch),
    inv: Option<(CompiledKernel, Launch)>,
}

fn freq_elems(problem: &SpectralMapSpec, inner_size: u32) -> i64 {
    problem.total_batch()
        * problem.nx() as i64
        * problem.nz_half() as i64
        * problem.ny() as i64
        * inner_size as i64
}

#[derive(Clone, Debug)]
pub(crate) struct SpectralMapPipeline {
    problem: SpectralMapSpec,
    fwd_zy: Vec<FwdGroup>,
    fused_x: Vec<(CompiledKernel, Launch)>,
    defused_x: DefusedX,
    inv_yz: Vec<InvGroup>,
}

impl SpectralMapPipeline {
    pub(crate) fn emit(
        problem: &SpectralMapSpec,
        device_fn_ir: &str,
        kernel_name_prefix: &str,
        opts: &[String],
        compile_budget_ms: Option<f64>,
    ) -> Result<Self, String> {
        let max_candidates =
            crate::kernel_compiler::spectral_map::candidate_budget::candidates_within_budget(
                compile_budget_ms,
            );
        let max_smem_bytes = crate::cuda_driver::Device::current()
            .max_smem_optin()
            .unwrap_or(49152) as u32;

        let mut inner_groups: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        for (j, &isz) in problem.grid_inner_sizes.iter().enumerate() {
            inner_groups.entry(isz).or_default().push(j as u32);
        }
        let max_grids_per_launch = {
            let ny = problem.ny();
            let nz_half = problem.nz_half();
            let per_grid_min_bytes = ny * nz_half * complex_bytes(problem.precision);
            (max_smem_bytes / per_grid_min_bytes.max(1)).max(1) as usize
        };
        let mut fwd_zy = Vec::new();
        for (inner_size, idxs) in inner_groups {
            for (chunk_i, chunk_idxs) in idxs.chunks(max_grids_per_launch).enumerate() {
                let bufs: Vec<&_> = chunk_idxs
                    .iter()
                    .map(|&j| &problem.grid_in_bufs[j as usize])
                    .collect();
                let kernel_name = format!("{kernel_name_prefix}_fwd_zy_i{inner_size}_g{chunk_i}");
                let emits = forward_zy_slabs::emit_variants(&ZyFwdSlabSpec {
                    batch_shape: &problem.batch_shape,
                    grid_in_bufs: &bufs,
                    inner_size,
                    fft_lengths: problem.fft_lengths,
                    precision: problem.precision,
                    sm: problem.sm,
                    max_smem_bytes,
                    kernel_name: kernel_name.clone(),
                    max_candidates,
                })?;
                let variants = emits
                    .into_par_iter()
                    .map(|emit| {
                        let ltoir: Vec<&[u8]> = emit.ltoir.iter().map(|b| b.as_slice()).collect();
                        let kernel =
                            CompiledKernel::compile(&emit.source, &emit.kernel_name, &ltoir, opts)?;
                        Ok((
                            kernel,
                            Launch {
                                grid: [emit.grid_size, 1, 1],
                                block: emit.block_dim,
                                shmem: emit.shared_bytes,
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                fwd_zy.push(FwdGroup {
                    grid_indices: chunk_idxs.to_vec(),
                    variants,
                });
            }
        }

        let aux_dtypes: Vec<Dtype> = problem.aux_dtypes.clone();
        let aux_output_dtypes: Vec<Dtype> = problem.aux_output_dtypes.clone();
        let aux_sizes: Vec<u32> = problem
            .aux_inner_shapes
            .iter()
            .map(|s| s.iter().product::<u32>().max(1))
            .collect();
        let aux_output_sizes: Vec<u32> = problem
            .aux_output_inner_shapes
            .iter()
            .map(|s| s.iter().product::<u32>().max(1))
            .collect();
        let aux_bufs: Vec<&_> = problem.aux_bufs.iter().collect();

        let fused_kernel_name = format!("{kernel_name_prefix}_x_fused");
        let xs = fused_x_transform::emit_variants(&XFusedSpec {
            fft_lengths: problem.fft_lengths,
            precision: problem.precision,
            sm: problem.sm,
            max_smem_bytes,
            batch_shape: &problem.batch_shape,
            grid_inner_sizes: &problem.grid_inner_sizes,
            output_inner_sizes: &problem.output_inner_sizes,
            input_signs: &problem.input_signs,
            output_signs: &problem.output_signs,
            aux_sizes: &aux_sizes,
            aux_dtypes: &aux_dtypes,
            aux_bufs: &aux_bufs,
            aux_output_sizes: &aux_output_sizes,
            aux_output_dtypes: &aux_output_dtypes,
            device_fn_ir,
            kernel_name: fused_kernel_name.clone(),
            max_candidates,
        })?;
        let fused_x = xs
            .into_par_iter()
            .map(|x| {
                let kernel = CompiledKernel::compile(
                    &x.source,
                    &x.kernel_name,
                    &[&x.ltoir_fwd, &x.ltoir_inv],
                    opts,
                )?;
                Ok((
                    kernel,
                    Launch {
                        grid: [x.grid_size, 1, 1],
                        block: x.block_dim,
                        shmem: x.shared_bytes,
                    },
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let defused_spec = |name: String| XFusedSpec {
            fft_lengths: problem.fft_lengths,
            precision: problem.precision,
            sm: problem.sm,
            max_smem_bytes,
            batch_shape: &problem.batch_shape,
            grid_inner_sizes: &problem.grid_inner_sizes,
            output_inner_sizes: &problem.output_inner_sizes,
            input_signs: &problem.input_signs,
            output_signs: &problem.output_signs,
            aux_sizes: &aux_sizes,
            aux_dtypes: &aux_dtypes,
            aux_bufs: &aux_bufs,
            aux_output_sizes: &aux_output_sizes,
            aux_output_dtypes: &aux_output_dtypes,
            device_fn_ir,
            kernel_name: name,
            max_candidates,
        };
        let compile_defused =
            |emit: separate_x_transform::XDefusedEmit| -> Result<(CompiledKernel, Launch), String> {
                let ltoir: Vec<&[u8]> = emit.ltoir.iter().map(|b| b.as_slice()).collect();
                let kernel =
                    CompiledKernel::compile(&emit.source, &emit.kernel_name, &ltoir, opts)?;
                Ok((
                    kernel,
                    Launch {
                        grid: [emit.grid_size, 1, 1],
                        block: emit.block_dim,
                        shmem: emit.shared_bytes,
                    },
                ))
            };
        let defused_x = DefusedX {
            fwd: compile_defused(separate_x_transform::emit_x_fwd(&defused_spec(format!(
                "{kernel_name_prefix}_x_dfwd"
            )))?)?,
            contract: compile_defused(separate_x_transform::emit_x_contract(&defused_spec(
                format!("{kernel_name_prefix}_x_dcontract"),
            ))?)?,
            inv: if problem.n_grid_out > 0 {
                Some(compile_defused(separate_x_transform::emit_x_inv(
                    &defused_spec(format!("{kernel_name_prefix}_x_dinv")),
                )?)?)
            } else {
                None
            },
        };

        let total_slabs = problem.total_slabs() as u32;
        let mut inv_yz = Vec::with_capacity(problem.output_inner_sizes.len());
        for (j, &inner_size) in problem.output_inner_sizes.iter().enumerate() {
            let kernel_name = format!("{kernel_name_prefix}_inv_yz_o{j}");
            let emits = inverse_yz_slabs::emit_variants(&YzInvSlabSpec {
                total_slabs,
                inner_size,
                fft_lengths: problem.fft_lengths,
                precision: problem.precision,
                sm: problem.sm,
                max_smem_bytes,
                kernel_name: kernel_name.clone(),
                max_candidates,
            })?;
            let variants = emits
                .into_par_iter()
                .map(|emit| {
                    let ltoir: Vec<&[u8]> = emit.ltoir.iter().map(|b| b.as_slice()).collect();
                    let kernel =
                        CompiledKernel::compile(&emit.source, &emit.kernel_name, &ltoir, opts)?;
                    Ok((
                        kernel,
                        Launch {
                            grid: [emit.grid_size, 1, 1],
                            block: emit.block_dim,
                            shmem: emit.shared_bytes,
                        },
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            inv_yz.push(InvGroup {
                grid_output_index: j as u32,
                variants,
            });
        }

        Ok(SpectralMapPipeline {
            problem: problem.clone(),
            fwd_zy,
            fused_x,
            defused_x,
            inv_yz,
        })
    }

    pub(crate) fn to_plan(&self) -> Result<crate::execution_plan::ExecutionPlan, String> {
        let p = &self.problem;
        let n_grid_in = p.n_grid_in as usize;
        let n_aux = p.n_aux as usize;
        let n_grid_out = p.n_grid_out as usize;
        let n_aux_out = p.n_aux_out as usize;

        let n_inputs = n_grid_in + n_aux;
        let n_outputs = n_grid_out + n_aux_out;
        let complex_bytes = specification::complex_bytes(p.precision) as i64;

        let mut b = crate::execution_plan::PlanBuilder::new();

        let freq_ws: Vec<crate::execution_plan::WorkspaceId> = p
            .grid_inner_sizes
            .iter()
            .map(|&isz| {
                Ok(b.scratch(workspace_bytes(
                    freq_elems(p, isz),
                    complex_bytes,
                    "fft freq intermediate",
                )?))
            })
            .collect::<Result<_, String>>()?;
        let xdom_ws: Vec<crate::execution_plan::WorkspaceId> = p
            .output_inner_sizes
            .iter()
            .map(|&isz| {
                Ok(b.scratch(workspace_bytes(
                    freq_elems(p, isz),
                    complex_bytes,
                    "fft x-domain buffer",
                )?))
            })
            .collect::<Result<_, String>>()?;

        for g in &self.fwd_zy {
            let k = g.grid_indices.len();
            let formal_args: Vec<crate::execution_plan::Arg> = (0..k)
                .map(crate::execution_plan::Arg::input)
                .chain((0..k).map(crate::execution_plan::Arg::output))
                .collect();
            let reads = g
                .grid_indices
                .iter()
                .map(|&j| crate::execution_plan::Buf::Input(j as usize));
            let writes = g
                .grid_indices
                .iter()
                .map(|&j| crate::execution_plan::Buf::Workspace(freq_ws[j as usize]));
            let candidates = stage_candidates(&g.variants, formal_args);
            b.choice(candidates).reads(reads).writes(writes).add();
        }

        let n_freq = freq_ws.len();
        let n_xdom = xdom_ws.len();
        let formal_args: Vec<crate::execution_plan::Arg> = (0..n_freq + n_aux)
            .map(crate::execution_plan::Arg::input)
            .chain((0..n_xdom + n_aux_out).map(crate::execution_plan::Arg::output))
            .collect();
        let mut reads: Vec<crate::execution_plan::Buf> = freq_ws
            .iter()
            .map(|&w| crate::execution_plan::Buf::Workspace(w))
            .collect();
        for k in 0..n_aux {
            reads.push(crate::execution_plan::Buf::Input(n_grid_in + k));
        }
        let mut writes: Vec<crate::execution_plan::Buf> = xdom_ws
            .iter()
            .map(|&w| crate::execution_plan::Buf::Workspace(w))
            .collect();
        for j in 0..n_aux_out {
            writes.push(crate::execution_plan::Buf::Output(n_grid_out + j));
        }
        let mut candidates = stage_candidates(&self.fused_x, formal_args);
        candidates.push(self.defused_candidate());
        b.choice(candidates).reads(reads).writes(writes).add();

        for g in &self.inv_yz {
            let j = g.grid_output_index as usize;
            let candidates = stage_candidates(
                &g.variants,
                vec![
                    crate::execution_plan::Arg::input(0),
                    crate::execution_plan::Arg::output(0),
                ],
            );
            b.choice(candidates)
                .reads([crate::execution_plan::Buf::Workspace(xdom_ws[j])])
                .writes([crate::execution_plan::Buf::Output(j)])
                .add();
        }

        let plan = b.finish().map_err(|e| format!("{e:?}"))?;
        plan.validate(n_inputs, n_outputs)
            .map_err(|e| format!("{e:?}"))?;
        Ok(plan)
    }

    fn defused_candidate(&self) -> crate::execution_plan::ExecutionPlan {
        use crate::execution_plan::Buf;
        let p = &self.problem;
        let n_grid_in = p.n_grid_in as usize;
        let n_grid_out = p.n_grid_out as usize;
        let n_aux = p.n_aux as usize;
        let n_aux_out = p.n_aux_out as usize;
        let total_x_lines = p.total_x_lines();
        let nx = p.nx() as i64;
        let cbytes = specification::complex_bytes(p.precision) as usize;

        let mut b = crate::execution_plan::PlanBuilder::new();
        let m_fwd = b.module(self.defused_x.fwd.0.cubin.clone());
        let m_con = b.module(self.defused_x.contract.0.cubin.clone());
        let m_inv = self
            .defused_x
            .inv
            .as_ref()
            .map(|(k, _)| b.module(k.cubin.clone()));

        let xfreq: Vec<crate::execution_plan::WorkspaceId> = (0..n_grid_in)
            .map(|j| {
                let isz = p.grid_inner_sizes[j] as i64;
                b.scratch((total_x_lines * nx * isz) as usize * cbytes)
            })
            .collect();
        let xfreq_out: Vec<crate::execution_plan::WorkspaceId> = (0..n_grid_out)
            .map(|j| {
                let oisz = p.output_inner_sizes[j] as i64;
                b.scratch((total_x_lines * nx * oisz) as usize * cbytes)
            })
            .collect();

        for j in 0..n_aux_out {
            let aoisz = p.aux_output_inner_shapes[j].iter().product::<u32>().max(1) as i64;
            let bytes = (total_x_lines * aoisz) as usize * p.aux_output_dtypes[j].size();
            b.memset(Buf::Output(n_grid_out + j), 0, bytes);
        }

        let (fk, fl) = &self.defused_x.fwd;
        let mut fwd = b
            .kernel(m_fwd, &fk.entry)
            .grid(fl.grid)
            .block(fl.block)
            .shmem(fl.shmem);
        for j in 0..n_grid_in {
            fwd = fwd.read(Buf::Input(j));
        }
        for &w in &xfreq {
            fwd = fwd.write(w);
        }
        fwd.add();

        let (ck, cl) = &self.defused_x.contract;
        let mut con = b
            .kernel(m_con, &ck.entry)
            .grid(cl.grid)
            .block(cl.block)
            .shmem(cl.shmem);
        for &w in &xfreq {
            con = con.read(w);
        }
        for k in 0..n_aux {
            con = con.read(Buf::Input(n_grid_in + k));
        }
        for &w in &xfreq_out {
            con = con.write(w);
        }
        for j in 0..n_aux_out {
            con = con.read_write(Buf::Output(n_grid_out + j));
        }
        con.add();

        if let (Some(mi), Some((ik, il))) = (m_inv, self.defused_x.inv.as_ref()) {
            let mut inv = b
                .kernel(mi, &ik.entry)
                .grid(il.grid)
                .block(il.block)
                .shmem(il.shmem);
            for &w in &xfreq_out {
                inv = inv.read(w);
            }
            for j in 0..n_grid_out {
                inv = inv.write(Buf::Output(j));
            }
            inv.add();
        }

        b.finish().expect("defused candidate builds")
    }
}

fn stage_candidates(
    variants: &[(CompiledKernel, Launch)],
    formal_args: Vec<crate::execution_plan::Arg>,
) -> Vec<crate::execution_plan::ExecutionPlan> {
    variants
        .iter()
        .map(|(kernel, launch)| crate::execution_plan::ExecutionPlan {
            modules: vec![crate::execution_plan::KernelModule {
                cubin: kernel.cubin.clone().into(),
            }],
            workspace: vec![],
            nodes: vec![crate::execution_plan::Node {
                op: crate::execution_plan::Op::KernelLaunch {
                    module: 0,
                    entry: kernel.entry.clone(),
                    grid: launch.grid,
                    block: launch.block,
                    shmem: launch.shmem,
                    args: formal_args.clone(),
                },
                deps: vec![],
            }],
        })
        .collect()
}

fn workspace_bytes(elems: i64, bytes_per: i64, what: &str) -> Result<usize, String> {
    let n = elems
        .checked_mul(bytes_per)
        .ok_or_else(|| format!("{what} byte size overflow"))?;
    if n < 0 {
        return Err(format!("{what} has negative byte size {n}"));
    }
    Ok(n as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_compiler::buffer::Buffer;

    fn libmathdx_available() -> bool {
        crate::kernel_compiler::cuda_toolchain::populate_from_python_for_tests();
        crate::kernel_compiler::libmathdx::CufftdxFft::build(
            &crate::kernel_compiler::libmathdx::FftSpec::r2c_f32(8, 120),
        )
        .is_ok()
    }

    fn cuda_available() -> bool {
        crate::cuda_driver::ensure_context().is_ok()
            && crate::cuda_driver::Device::current().sm_arch().unwrap_or(0) > 0
    }

    fn passthrough_ir() -> &'static str {
        r#"; ModuleID = "fn"
target triple = "nvptx64-nvidia-cuda"
target datalayout = ""

define void @"fn"(i32* %"_idx", float* %"g0", float* %"out0")
{
entry:
  %"r0" = getelementptr inbounds float, float* %"g0", i32 0
  %"r1" = getelementptr inbounds float, float* %"g0", i32 1
  %"v0" = load float, float* %"r0"
  %"v1" = load float, float* %"r1"
  %"w0" = getelementptr inbounds float, float* %"out0", i32 0
  %"w1" = getelementptr inbounds float, float* %"out0", i32 1
  store float %"v0", float* %"w0"
  store float %"v1", float* %"w1"
  ret void
}
"#
    }

    fn smoke_problem(sm: u32) -> SpectralMapSpec {
        let grid_buf = Buffer {
            name: "ignored".into(),
            dtype: Dtype::F32,
            ic: vec![],
            extents: vec![],
            elem_size: 0,
        };
        SpectralMapSpec {
            fft_lengths: [16, 16, 16],
            precision: Dtype::F32,
            sm,
            n_grid_in: 1,
            n_grid_out: 1,
            n_aux: 0,
            n_aux_out: 0,
            batch_shape: vec![],
            grid_inner_sizes: vec![1],
            output_inner_sizes: vec![1],
            input_signs: vec![-1],
            output_signs: vec![-1],
            grid_in_bufs: vec![grid_buf],
            aux_bufs: vec![],
            aux_inner_shapes: vec![],
            aux_dtypes: vec![],
            aux_output_inner_shapes: vec![],
            aux_output_dtypes: vec![],
        }
    }

    #[test]
    fn emit_pipeline_smoke() {
        if !libmathdx_available() || !cuda_available() {
            return;
        }
        let sm = crate::cuda_driver::Device::current().sm_arch().unwrap_or(0) as u32;
        let problem = smoke_problem(sm);
        let pipe = SpectralMapPipeline::emit(&problem, passthrough_ir(), "pipe_smoke", &[], None)
            .expect("emit");
        assert_eq!(pipe.fwd_zy.len(), 1);
        assert_eq!(pipe.inv_yz.len(), 1);

        let plan = pipe.to_plan().expect("to_plan");
        assert!(plan.modules.is_empty());
        assert_eq!(plan.workspace.len(), 2);
        assert_eq!(plan.nodes.len(), 3);
        for node in &plan.nodes {
            let crate::execution_plan::Op::Choice { candidates, .. } = &node.op else {
                panic!("expected Choice node, got {node:?}");
            };
            assert!(!candidates.is_empty());
            for c in candidates {
                assert!(!c.modules.is_empty());
            }
        }
        let crate::execution_plan::Op::Choice { candidates, .. } = &plan.nodes[1].op else {
            panic!("fused-X node must be a Choice");
        };
        let defused = candidates.last().expect("defused candidate present");
        assert_eq!(defused.modules.len(), 3);
        assert_eq!(defused.nodes.len(), 3);
        assert_eq!(defused.workspace.len(), 2);
        plan.validate(1, 1).expect("validate");
    }
}
