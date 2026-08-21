// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::cuda;
use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::Stmt;

use super::accumulator::{Accumulator, AccumulatorEntry, HookCtx, ScatterCtx};
use super::kernel_body::{
    cube_cache_side, emit_dense_cube_flush, emit_global_atomic_scatter, emit_smem_atomic_scatter,
    emit_zero_fill_smem, zero_lit_for,
};
use super::specification::{ScatterSpec, smem_strategy_applicable};

pub(crate) struct SmemLocalCube;

impl Accumulator for SmemLocalCube {
    fn name(&self) -> &'static str {
        "h4"
    }

    fn uses_smem(&self) -> bool {
        true
    }

    fn min_blocks(&self) -> i32 {
        4
    }

    fn is_applicable(&self, problem: &ScatterSpec) -> bool {
        smem_strategy_applicable(problem) && problem.single_batch()
    }

    fn ps_init(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let cube_side = local_cube_side(
            ctx.problem.grid_shape,
            ctx.grid_out_inner_sizes,
            ctx.grid_out_dtypes,
        );
        let vol = cube_side.pow(3);

        let scratch_names = [
            String::from("_ref_bx"),
            String::from("_ref_by"),
            String::from("_ref_bz"),
            String::from("_com_x"),
            String::from("_com_y"),
            String::from("_com_z"),
            String::from("_cube_ox"),
            String::from("_cube_oy"),
            String::from("_cube_oz"),
        ];
        for name in scratch_names {
            stmts.push(Stmt::shared_decl(String::from("int"), name, None));
        }

        cuda! { stmts =>
            if (threadIdx.x == 0) {
                _ref_bx = bx;
                _ref_by = by;
                _ref_bz = bz;
            }
            __syncthreads();
        }

        emit_periodic_delta(
            stmts,
            String::from("_dx"),
            String::from("bx"),
            String::from("_ref_bx"),
            String::from("GX"),
        );
        emit_periodic_delta(
            stmts,
            String::from("_dy"),
            String::from("by"),
            String::from("_ref_by"),
            String::from("GY"),
        );
        emit_periodic_delta(
            stmts,
            String::from("_dz"),
            String::from("bz"),
            String::from("_ref_bz"),
            String::from("GZ"),
        );

        cuda! { stmts =>
            i32 _sum_dx = _dx;
            i32 _sum_dy = _dy;
            i32 _sum_dz = _dz;
            i32 _cnt = 1;
            for (i32 _off = 16; _off >= 1; _off >>= 1) {
                _sum_dx += __shfl_xor_sync(0xFFFFFFFFu, _sum_dx, _off);
                _sum_dy += __shfl_xor_sync(0xFFFFFFFFu, _sum_dy, _off);
                _sum_dz += __shfl_xor_sync(0xFFFFFFFFu, _sum_dz, _off);
                _cnt += __shfl_xor_sync(0xFFFFFFFFu, _cnt, _off);
            }
        }

        for name in [
            String::from("_warp_dx"),
            String::from("_warp_dy"),
            String::from("_warp_dz"),
            String::from("_warp_cnt"),
        ] {
            stmts.push(Stmt::shared_decl(String::from("int"), name, Some(32)));
        }
        cuda! { stmts =>
            i32 _wid = threadIdx.x >> 5;
            if ((threadIdx.x & 31) == 0) {
                _warp_dx[_wid] = _sum_dx;
                _warp_dy[_wid] = _sum_dy;
                _warp_dz[_wid] = _sum_dz;
                _warp_cnt[_wid] = _cnt;
            }
            __syncthreads();
        }

        let half = cube_side / 2;
        cuda! { stmts =>
            if (threadIdx.x == 0) {
                i32 _nw = blockDim.x >> 5;
                i32 _tdx = 0;
                i32 _tdy = 0;
                i32 _tdz = 0;
                i32 _tc = 0;
                for (i32 _w = 0; _w < _nw; _w++) {
                    _tdx += _warp_dx[_w];
                    _tdy += _warp_dy[_w];
                    _tdz += _warp_dz[_w];
                    _tc  += _warp_cnt[_w];
                }
                _com_x = (_ref_bx + _tdx / _tc + GX) % GX;
                _com_y = (_ref_by + _tdy / _tc + GY) % GY;
                _com_z = (_ref_bz + _tdz / _tc + GZ) % GZ;
                _cube_ox = (_com_x - #half + GX) % GX;
                _cube_oy = (_com_y - #half + GY) % GY;
                _cube_oz = (_com_z - #half + GZ) % GZ;
            }
            __syncthreads();
            i32 CUBE_SIDE = #cube_side;
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
        let mut smem_branch: Vec<Stmt> = Vec::new();
        cuda! { smem_branch => i32 _lc = (_lx * CUBE_SIDE + _ly) * CUBE_SIDE + _lz; }
        emit_smem_atomic_scatter(&mut smem_branch, &cube, &lc, ctx.j, ctx.isz);

        let mut global_branch: Vec<Stmt> = Vec::new();
        emit_global_atomic_scatter(&mut global_branch, ctx.j, ctx.isz, ctx.batch_var);

        // `if (1) { ... }` bounds `_lx/_ly/_lz` so multiple grid outputs
        // (each running ps_scatter once per j) don't collide on the names.
        cuda! { stmts =>
            if (1) {
                i32 _lx = (ix - _cube_ox + GX) % GX;
                i32 _ly = (iy - _cube_oy + GY) % GY;
                i32 _lz = (iz - _cube_oz + GZ) % GZ;
                if (_lx < CUBE_SIDE && _ly < CUBE_SIDE && _lz < CUBE_SIDE) {
                    splice!(smem_branch);
                } else {
                    splice!(global_branch);
                }
            }
        }
    }

    fn ps_flush(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let cube_side = local_cube_side(
            ctx.problem.grid_shape,
            ctx.grid_out_inner_sizes,
            ctx.grid_out_dtypes,
        );
        let vol = cube_side.pow(3);
        let gidx = String::from("((long long)(gx) * GY + gy) * GZ + gz");
        let wrap_x = String::from("(_cube_ox + _lx) % GX");
        let wrap_y = String::from("(_cube_oy + _ly) % GY");
        let wrap_z = String::from("(_cube_oz + _lz) % GZ");

        emit_dense_cube_flush(
            stmts,
            ctx.n_grid_out,
            ctx.grid_out_inner_sizes,
            ctx.grid_out_dtypes,
            cube_name,
            &String::from("CUBE_SIDE"),
            &String::from("CUBE_SIDE"),
            vol,
            Some((&wrap_x, &wrap_y, &wrap_z)),
            &gidx,
        );
    }
}

inventory::submit! {
    AccumulatorEntry { make: || Box::new(SmemLocalCube) }
}

/// The local cube's per-axis side, capped so the cube never exceeds the grid.
///
/// The cube is a window of the periodic grid: scatter maps a global cell to a
/// local slot through `(cell - origin + G) % G`, and flush maps it back through
/// `(origin + slot) % G`. Both round-trips are bijective only while the window
/// fits inside one grid period — `CUBE_SIDE <= min(GX, GY, GZ)`. If the cube
/// were larger, the origin `(com - CUBE_SIDE/2 + G) % G` can land below zero
/// (one `+G` no longer lifts it non-negative once `CUBE_SIDE/2 > G`), and the
/// flush index `(origin + slot) % G` then addresses cells outside the grid.
/// Capping to the smallest grid extent keeps the window a true subset and the
/// mapping exact for any grid size.
fn local_cube_side(
    grid_shape: [i64; 3],
    grid_out_inner_sizes: &[i64],
    grid_out_dtypes: &[Dtype],
) -> i64 {
    let bytes_per_cell: i64 = grid_out_inner_sizes
        .iter()
        .zip(grid_out_dtypes)
        .map(|(sz, dt)| sz * dt.size() as i64)
        .sum();
    let cache_side = cube_cache_side(bytes_per_cell);
    grid_shape.iter().copied().fold(cache_side, i64::min)
}

fn cube_name(j: usize) -> String {
    format!("{}{j}", String::from("_cube_"))
}

fn emit_periodic_delta(stmts: &mut Vec<Stmt>, delta: String, b: String, ref_: String, g: String) {
    cuda! { stmts =>
        i32 #delta = #b - #ref_;
        if (#delta > #g / 2) {
            #delta -= #g;
        }
        if (#delta < 0 - #g / 2) {
            #delta += #g;
        }
    }
}
