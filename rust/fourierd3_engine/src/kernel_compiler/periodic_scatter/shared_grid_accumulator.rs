// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::cuda;
use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::Stmt;

use super::accumulator::{Accumulator, AccumulatorEntry, HookCtx, ScatterCtx};
use super::kernel_body::{
    emit_dense_cube_flush, emit_global_atomic_scatter, emit_smem_atomic_scatter,
    emit_zero_fill_smem, full_cache_side, zero_lit_for,
};
use super::specification::{ScatterSpec, smem_strategy_applicable};

pub(crate) struct SmemFullGrid;

impl Accumulator for SmemFullGrid {
    fn name(&self) -> &'static str {
        "p9"
    }

    fn uses_smem(&self) -> bool {
        true
    }

    fn min_blocks(&self) -> i32 {
        4
    }

    fn is_applicable(&self, problem: &ScatterSpec) -> bool {
        if !smem_strategy_applicable(problem) {
            return false;
        }
        let cube_side = cube_side_for(&problem.grid_out_inner_sizes, &problem.grid_out_dtypes);
        problem.grid_shape.iter().all(|&d| cube_side >= d)
    }

    fn ps_init(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let cube_side = cube_side_for(ctx.grid_out_inner_sizes, ctx.grid_out_dtypes);
        let vol = cube_side.pow(3);

        cuda! { stmts => i32 CUBE_SIDE = #cube_side; }

        if !ctx.single_batch {
            stmts.push(Stmt::shared_decl(
                String::from("int"),
                String::from("_cube_batch"),
                None,
            ));
            let first = ctx.out_batch_vars[0].clone();
            cuda! { stmts =>
                if (threadIdx.x == 0) {
                    _cube_batch = #first;
                }
            }
        }

        for j in 0..ctx.n_grid_out {
            let isz = ctx.grid_out_inner_sizes[j];
            let dt = ctx.grid_out_dtypes[j];
            let zero = zero_lit_for(dt);
            let total = vol * isz;
            let name = cube_name(j);
            stmts.push(Stmt::shared_decl(dt, name.clone(), Some(total)));
            emit_zero_fill_smem(stmts, &name, total, zero);
        }

        stmts.push(Stmt::Eval(Expr::call(
            String::from("__syncthreads"),
            vec![],
        )));
    }

    fn ps_scatter(&self, stmts: &mut Vec<Stmt>, ctx: &ScatterCtx<'_>) {
        let cube = cube_name(ctx.j);
        let lc = String::from("_lc");
        let mut smem_scatter: Vec<Stmt> = Vec::new();
        emit_smem_atomic_scatter(&mut smem_scatter, &cube, &lc, ctx.j, ctx.isz);

        // `if (1) { ... }` bounds `_lc` so multiple grid outputs (each
        // running ps_scatter once per j) don't collide on the name.
        if ctx.single_batch {
            cuda! { stmts =>
                if (1) {
                    i32 _lc = (ix * CUBE_SIDE + iy) * CUBE_SIDE + iz;
                    splice!(smem_scatter);
                }
            }
        } else {
            let mut global_branch: Vec<Stmt> = Vec::new();
            emit_global_atomic_scatter(&mut global_branch, ctx.j, ctx.isz, ctx.batch_var);
            let batch_var = ctx.batch_var;
            cuda! { stmts =>
                if (1) {
                    i32 _lc = (ix * CUBE_SIDE + iy) * CUBE_SIDE + iz;
                    if (#batch_var == _cube_batch) {
                        splice!(smem_scatter);
                    } else {
                        splice!(global_branch);
                    }
                }
            }
        }
    }

    fn ps_flush(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let cube_side = cube_side_for(ctx.grid_out_inner_sizes, ctx.grid_out_dtypes);
        let vol = cube_side.pow(3);
        let gidx = gidx_for_full_grid(ctx.single_batch);

        emit_dense_cube_flush(
            stmts,
            ctx.n_grid_out,
            ctx.grid_out_inner_sizes,
            ctx.grid_out_dtypes,
            cube_name,
            &String::from("CUBE_SIDE"),
            &String::from("CUBE_SIDE"),
            vol,
            None,
            &gidx,
        );
    }
}

inventory::submit! {
    AccumulatorEntry { make: || Box::new(SmemFullGrid) }
}

fn cube_side_for(grid_out_inner_sizes: &[i64], grid_out_dtypes: &[Dtype]) -> i64 {
    let bytes_per_cell: i64 = grid_out_inner_sizes
        .iter()
        .zip(grid_out_dtypes)
        .map(|(sz, dt)| sz * dt.size() as i64)
        .sum();
    full_cache_side(bytes_per_cell)
}

fn cube_name(j: usize) -> String {
    format!("{}{j}", String::from("_cube_"))
}

fn gidx_for_full_grid(single_batch: bool) -> String {
    let mut s = String::new();
    if !single_batch {
        s.push_str(&String::from(
            "((long long)(_cube_batch) * GX * GY * GZ) + ",
        ));
    }
    s.push_str(&String::from("((long long)(_lx) * GY + _ly) * GZ + _lz"));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use fourierd3_engine::dtype::Dtype;

    use crate::kernel_compiler::periodic_scatter::{IndexLayout, ScatterSpec};

    fn smem_problem(n: i64) -> ScatterSpec {
        let ic = vec![vec![-1], vec![-1], vec![-1]];
        let extents = vec![vec![n], vec![n], vec![n]];
        ScatterSpec {
            batch_sizes: vec![n],
            layout: IndexLayout {
                ic,
                n_grid_in: 1,
                n_nongrid_in: 0,
                n_grid_out: 1,
                n_nongrid_out: 0,
                n_index: 0,
            },
            buf_batch_extents: extents,
            n_backend_arrays: 0,
            nongrid_in_sizes: vec![],
            nongrid_in_dtypes: vec![],
            grid_in_inner_sizes: vec![1],
            grid_in_dtypes: vec![Dtype::F32],
            grid_out_inner_sizes: vec![1],
            grid_out_dtypes: vec![Dtype::F32],
            nongrid_out_sizes: vec![],
            nongrid_out_dtypes: vec![],
            nongrid_scatter_flags: vec![],
            n_state: 0,
            state_sizes: vec![],
            state_dtypes: vec![],
            pre_ngin_indices: vec![],
            direct_ngin_indices: vec![],
            k: 1,
            s_support: 27,
            cartesian: None,
            grid_shape: [32, 32, 32],
            cell_grid_shape: None,
        }
    }

    fn render(p: &ScatterSpec) -> String {
        fourierd3_engine::ir::stmt::render_module_string(&SmemFullGrid.emit_body(p))
    }

    #[test]
    fn single_batch_smoke() {
        let s = render(&smem_problem(1));
        assert!(s.contains("int CUBE_SIDE = "), "got:\n{s}");
        assert!(s.contains("__shared__ float _cube_0["), "got:\n{s}");
        assert!(s.contains("_cube_0[_ci] = 0.0f"), "got:\n{s}");
        assert!(s.contains("__syncthreads();"), "got:\n{s}");
        assert!(!s.contains("_cube_batch"), "leaked multi-batch:\n{s}");
        assert!(
            s.contains("int _lc = (ix * CUBE_SIDE + iy) * CUBE_SIDE + iz;"),
            "got:\n{s}"
        );
        assert!(
            s.contains("atomicAdd(&_cube_0[_lc], _gout0[0]);"),
            "got:\n{s}"
        );
        assert!(s.contains("if (_cube_0[_ci] != 0.0f)"), "got:\n{s}");
        assert!(
            s.contains(
                "atomicAdd(&grid_out_0[((long long)(_lx) * GY + _ly) * GZ + _lz], float(_cube_0[_ci]));"
            ),
            "got:\n{s}"
        );
    }

    #[test]
    fn multi_batch_emits_fallback() {
        let mut p = smem_problem(2);
        p.buf_batch_extents = vec![vec![2], vec![2], vec![2]];
        p.batch_sizes = vec![2];
        let s = render(&p);
        assert!(s.contains("__shared__ int _cube_batch;"), "got:\n{s}");
        assert!(s.contains("_cube_batch = b_0;"), "got:\n{s}");
        assert!(s.contains("if (b_0 == _cube_batch)"), "got:\n{s}");
        assert!(
            s.contains("atomicAdd(&grid_out_0[(((long long)(b_0) * GX + ix) * GY + iy) * GZ + iz]"),
            "fallback scatter missing:\n{s}"
        );
        assert!(
            s.contains(
                "((long long)(_cube_batch) * GX * GY * GZ) + ((long long)(_lx) * GY + _ly) * GZ + _lz"
            ),
            "got:\n{s}"
        );
    }

    #[test]
    fn inventory_registers_smem_full_grid() {
        let s = super::super::accumulator::all_accumulators()
            .into_iter()
            .find(|s| s.name() == "p9")
            .expect("SmemFullGrid should be registered");
        assert_eq!(s.name(), "p9");
    }
}
