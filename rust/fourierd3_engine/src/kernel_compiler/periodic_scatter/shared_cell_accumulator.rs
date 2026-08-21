// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::cuda;
use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::Stmt;

use super::accumulator::{Accumulator, AccumulatorEntry, HookCtx, ScatterCtx};
use super::kernel_body::{
    CACHE_BUDGET, cell_buffer_spans, emit_dense_cube_flush, emit_global_atomic_scatter,
    emit_smem_atomic_scatter, emit_zero_fill_smem, zero_lit_for,
};
use super::specification::ScatterSpec;

pub(crate) struct SmemCellBuffer;

impl Accumulator for SmemCellBuffer {
    fn name(&self) -> &'static str {
        "b8"
    }

    fn uses_smem(&self) -> bool {
        true
    }

    fn is_applicable(&self, problem: &ScatterSpec) -> bool {
        if problem.n_backend_arrays == 0 || problem.layout.n_index != 0 {
            return false;
        }
        let Some(cell_grid) = problem.cell_grid_shape else {
            return false;
        };
        if cell_grid.iter().any(|&nc| nc <= 0) {
            return false;
        }
        let Some((order, _)) = problem.cartesian else {
            return false;
        };
        let spans = cell_buffer_spans(problem.grid_shape, cell_grid, order as i64);
        spans.iter().product::<i64>() * problem.bytes_per_cell() <= CACHE_BUDGET
    }

    fn ps_init(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let (spans, cell_grid) = sizing(ctx);
        let (sx, sy, sz) = (spans[0], spans[1], spans[2]);
        let vol = sx * sy * sz;
        let (ncx, ncy, ncz) = (cell_grid[0], cell_grid[1], cell_grid[2]);

        cuda! { stmts =>
            i32 SPAN_X = #sx;
            i32 SPAN_Y = #sy;
            i32 SPAN_Z = #sz;
            i32 SPAN_VOL = #vol;
        }

        for n in [
            String::from("_cell_ox"),
            String::from("_cell_oy"),
            String::from("_cell_oz"),
        ] {
            stmts.push(Stmt::shared_decl(String::from("int"), n, None));
        }
        if !ctx.single_batch {
            stmts.push(Stmt::shared_decl(
                String::from("int"),
                String::from("_cell_batch"),
                None,
            ));
        }

        let ncell = ncx * ncy * ncz;
        let nxy = ncx * ncy;
        let cell_batch_assign: Vec<Stmt> = if !ctx.single_batch {
            let first = ctx.out_batch_vars[0].clone();
            let mut v = Vec::new();
            cuda! { v => _cell_batch = #first; }
            v
        } else {
            Vec::new()
        };
        cuda! { stmts =>
            if (threadIdx.x == 0) {
                i32 _block_first = blockIdx.x * blockDim.x;
                i32 _ncell = #ncell;
                i32 _lo = 0;
                i32 _hi = _ncell - 1;
                i32 _cell_id = 0;
                while (_lo <= _hi) {
                    i32 _mid = (_lo + _hi) >> 1;
                    if (cell_starts_ends[_mid * 2] <= (u32)_block_first) {
                        _cell_id = _mid;
                        _lo = _mid + 1;
                    } else {
                        _hi = _mid - 1;
                    }
                }
                i32 _cx = _cell_id % #ncx;
                i32 _cy = (_cell_id / #ncx) % #ncy;
                i32 _cz = _cell_id / #nxy;
                _cell_ox = (_cx * GX) / #ncx;
                _cell_oy = (_cy * GY) / #ncy;
                _cell_oz = (_cz * GZ) / #ncz;
                splice!(cell_batch_assign);
            }
            __syncthreads();
        }

        for j in 0..ctx.n_grid_out {
            let isz = ctx.grid_out_inner_sizes[j];
            let dt = ctx.grid_out_dtypes[j];
            let zero = zero_lit_for(dt);
            let total = vol * isz;
            let name = cbuf_name(j);
            stmts.push(Stmt::shared_decl(dt, name.clone(), Some(total)));
            emit_zero_fill_smem(stmts, &name, total, zero);
        }
        stmts.push(Stmt::Eval(Expr::call(
            String::from("__syncthreads"),
            vec![],
        )));
    }

    fn ps_scatter(&self, stmts: &mut Vec<Stmt>, ctx: &ScatterCtx<'_>) {
        let cbuf = cbuf_name(ctx.j);
        let lc = String::from("_lc");
        let mut smem_branch: Vec<Stmt> = Vec::new();
        cuda! { smem_branch => i32 _lc = (_lx * SPAN_Y + _ly) * SPAN_Z + _lz; }
        emit_smem_atomic_scatter(&mut smem_branch, &cbuf, &lc, ctx.j, ctx.isz);

        let mut global_branch: Vec<Stmt> = Vec::new();
        emit_global_atomic_scatter(&mut global_branch, ctx.j, ctx.isz, ctx.batch_var);

        // `if (1) { ... }` bounds `_lx/_ly/_lz` so multiple grid outputs
        // (each running ps_scatter once per j) don't collide on the names.
        cuda! { stmts =>
            if (1) {
                i32 _lx = (ix - _cell_ox + GX) % GX;
                i32 _ly = (iy - _cell_oy + GY) % GY;
                i32 _lz = (iz - _cell_oz + GZ) % GZ;
                if (_lx < SPAN_X && _ly < SPAN_Y && _lz < SPAN_Z) {
                    splice!(smem_branch);
                } else {
                    splice!(global_branch);
                }
            }
        }
    }

    fn ps_flush(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let (spans, _) = sizing(ctx);
        let vol = spans[0] * spans[1] * spans[2];
        let gidx = gidx_for_cell_buffer(ctx.single_batch);
        let wrap_x = String::from("(_cell_ox + _lx) % GX");
        let wrap_y = String::from("(_cell_oy + _ly) % GY");
        let wrap_z = String::from("(_cell_oz + _lz) % GZ");

        emit_dense_cube_flush(
            stmts,
            ctx.n_grid_out,
            ctx.grid_out_inner_sizes,
            ctx.grid_out_dtypes,
            cbuf_name,
            &String::from("SPAN_Z"),
            &String::from("SPAN_Y"),
            vol,
            Some((&wrap_x, &wrap_y, &wrap_z)),
            &gidx,
        );
    }
}

inventory::submit! {
    AccumulatorEntry { make: || Box::new(SmemCellBuffer) }
}

fn sizing(ctx: &HookCtx<'_>) -> ([i64; 3], [i64; 3]) {
    let cell_grid = ctx
        .problem
        .cell_grid_shape
        .expect("SmemCellBuffer requires cell_grid_shape");
    let order = ctx
        .problem
        .cartesian
        .map(|(o, _)| o as i64)
        .expect("SmemCellBuffer requires Cartesian support_order");
    let spans = cell_buffer_spans(ctx.problem.grid_shape, cell_grid, order);
    (spans, cell_grid)
}

fn cbuf_name(j: usize) -> String {
    format!("{}{j}", String::from("_cbuf_"))
}

fn gidx_for_cell_buffer(single_batch: bool) -> String {
    let mut s = String::new();
    if !single_batch {
        s.push_str(&String::from(
            "((long long)(_cell_batch) * GX * GY * GZ) + ",
        ));
    }
    s.push_str(&String::from("((long long)(gx) * GY + gy) * GZ + gz"));
    s
}
