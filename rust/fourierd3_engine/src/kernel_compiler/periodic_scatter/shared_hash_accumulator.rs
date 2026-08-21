// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::cuda;
use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::{Param, Stmt};

use super::accumulator::{Accumulator, AccumulatorEntry, HookCtx, ScatterCtx};
use super::kernel_body::{
    cas_cache_store_fn, grid_cell_expr, hash_cache_size, va_off_name, zero_lit_for,
};
use super::specification::{ScatterSpec, smem_strategy_applicable};

const ANCHOR_SMEM: i64 = 16;

#[derive(Clone, Copy)]
pub(crate) enum HashMode {
    Plain,
    Anchored,
}

pub(crate) struct SmemHash {
    mode: HashMode,
}

impl Accumulator for SmemHash {
    fn name(&self) -> &'static str {
        match self.mode {
            HashMode::Plain => "m7",
            HashMode::Anchored => "t3",
        }
    }

    fn uses_smem(&self) -> bool {
        true
    }

    fn is_applicable(&self, problem: &ScatterSpec) -> bool {
        smem_strategy_applicable(problem) && problem.single_batch()
    }

    fn pre_kernel_source(&self, problem: &ScatterSpec) -> Vec<Stmt> {
        let ct = problem.grid_out_dtypes[0].ctype();
        let mask = self.cache_size(&problem.grid_out_dtypes) - 1;
        let store_fn = match self.mode {
            HashMode::Plain => {
                let mut slot = Vec::new();
                cuda! { slot => i32 _slot = _va & #mask; }
                cas_cache_store_fn(ct, slot, vec![])
            }
            HashMode::Anchored => {
                let mut slot = Vec::new();
                cuda! { slot =>
                    i32 _slot = (((_ix - _ax) & 0xF) << 8)
                        | (((_iy - _ay) & 0xF) << 4)
                        | ((_iz - _az) & 0xF);
                    _slot &= #mask;
                }
                let coord = |name: String| Param::Scalar {
                    ctype: String::from("int"),
                    name,
                };
                let extra = vec![
                    coord(String::from("_ix")),
                    coord(String::from("_iy")),
                    coord(String::from("_iz")),
                    coord(String::from("_ax")),
                    coord(String::from("_ay")),
                    coord(String::from("_az")),
                ];
                cas_cache_store_fn(ct, slot, extra)
            }
        };
        vec![store_fn]
    }

    fn ps_init(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let cache_size = self.cache_size(ctx.grid_out_dtypes);
        let dt = ctx.grid_out_dtypes[0];
        let zero = zero_lit_for(dt);

        if let HashMode::Anchored = self.mode {
            for n in [
                String::from("_anchor_x"),
                String::from("_anchor_y"),
                String::from("_anchor_z"),
            ] {
                stmts.push(Stmt::shared_decl(String::from("int"), n, None));
            }
            cuda! { stmts =>
                if (threadIdx.x == 0) {
                    _anchor_x = bx;
                    _anchor_y = by;
                    _anchor_z = bz;
                }
            }
        }

        if ctx.n_grid_out > 1 {
            let va_stride = *ctx.grid_out_inner_sizes.iter().max().unwrap_or(&1);
            for j in 1..ctx.n_grid_out {
                let name = va_off_name(j);
                let j_lit = j as i64;
                cuda! { stmts =>
                    i64 #name = (i64)#j_lit * GX * GY * GZ * #va_stride;
                }
            }
        }

        stmts.push(Stmt::shared_decl(
            dt,
            String::from("cache_val"),
            Some(cache_size),
        ));
        stmts.push(Stmt::shared_decl(
            String::from("int"),
            String::from("cache_idx"),
            Some(cache_size),
        ));

        let zero_e = Expr::var(zero);
        cuda! { stmts =>
            for (i32 _ci = threadIdx.x; _ci < #cache_size; _ci += blockDim.x) {
                cache_idx[_ci] = -1;
                cache_val[_ci] = #zero_e;
            }
            __syncthreads();
        }
    }

    fn ps_scatter(&self, stmts: &mut Vec<Stmt>, ctx: &ScatterCtx<'_>) {
        let cell = grid_cell_expr(Expr::var(ctx.batch_var.to_string()));
        let buf = format!("{}{}", String::from("grid_out_"), ctx.j);
        let gout = format!("{}{}", String::from("_gout"), ctx.j);

        if ctx.isz == 1 {
            let va = if ctx.j > 0 {
                Expr::add(cell.clone(), Expr::var(va_off_name(ctx.j)))
            } else {
                cell.clone()
            };
            stmts.push(Stmt::Eval(Expr::call(
                String::from("_cache_store"),
                self.cache_store_args(&buf, cell, va, Expr::index(gout, Expr::lit(0))),
            )));
            return;
        }

        let cell_var = format!("{}{}", String::from("_cell_"), ctx.j);
        let a = String::from("_a");
        let cell_lin = Expr::add(
            Expr::mul(Expr::var(&cell_var), Expr::lit(ctx.isz)),
            Expr::var(&a),
        );
        let va = if ctx.j > 0 {
            Expr::add(cell_lin.clone(), Expr::var(va_off_name(ctx.j)))
        } else {
            cell_lin.clone()
        };
        let call_body: Vec<Stmt> = vec![Stmt::Eval(Expr::call(
            String::from("_cache_store"),
            self.cache_store_args(&buf, cell_lin, va, Expr::index(gout, Expr::var(&a))),
        ))];
        let isz = ctx.isz;
        cuda! { stmts =>
            i64 #cell_var = #cell;
            for (i32 _a = 0; _a < #isz; _a++) {
                splice!(call_body);
            }
        }
    }

    fn ps_flush(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let cache_size = self.cache_size(ctx.grid_out_dtypes);
        let ct = ctx.grid_out_dtypes[0].ctype();
        let ci = String::from("_ci");
        let mut dispatch: Vec<Stmt> = Vec::new();
        emit_hash_dispatch(&mut dispatch, ctx.n_grid_out, ct, &ci);
        cuda! { stmts =>
            __syncthreads();
            for (i32 _ci = threadIdx.x; _ci < #cache_size; _ci += blockDim.x) {
                if (cache_idx[_ci] >= 0) {
                    splice!(dispatch);
                }
            }
        }
    }
}

inventory::submit! {
    AccumulatorEntry {
        make: || Box::new(SmemHash { mode: HashMode::Plain }),
    }
}

inventory::submit! {
    AccumulatorEntry {
        make: || Box::new(SmemHash { mode: HashMode::Anchored }),
    }
}

impl SmemHash {
    fn cache_size(&self, grid_out_dtypes: &[Dtype]) -> i64 {
        let reserved = match self.mode {
            HashMode::Plain => 0,
            HashMode::Anchored => ANCHOR_SMEM,
        };
        let max_elem_bytes = grid_out_dtypes
            .iter()
            .map(|dt| dt.size() as i64)
            .max()
            .unwrap_or(4);
        hash_cache_size(max_elem_bytes, reserved)
    }

    fn cache_store_args(&self, buf: &str, cell: Expr, va: Expr, gout_elem: Expr) -> Vec<Expr> {
        let mut args = vec![
            Expr::var(String::from("cache_val")),
            Expr::var(String::from("cache_idx")),
            Expr::var(buf.to_string()),
            cell,
            va,
            gout_elem,
        ];
        if let HashMode::Anchored = self.mode {
            args.extend([
                Expr::var(String::from("ix")),
                Expr::var(String::from("iy")),
                Expr::var(String::from("iz")),
                Expr::var(String::from("_anchor_x")),
                Expr::var(String::from("_anchor_y")),
                Expr::var(String::from("_anchor_z")),
            ]);
        }
        args
    }
}

fn emit_hash_dispatch(stmts: &mut Vec<Stmt>, n_grid_out: usize, ct: &str, ci: &str) {
    let claimed = Expr::index(String::from("cache_idx"), Expr::var(ci));
    let val = Expr::Call(
        ct.to_string(),
        vec![Expr::index(String::from("cache_val"), Expr::var(ci))],
    );
    if n_grid_out == 1 {
        stmts.push(Stmt::Eval(Expr::call(
            String::from("atomicAdd"),
            vec![
                Expr::addr(Expr::index(
                    format!("{}0", String::from("grid_out_")),
                    claimed,
                )),
                val,
            ],
        )));
        return;
    }

    // Build chained if/else if/else from the back so we can wrap each
    // arm as the else branch of the previous one.
    let mut tail: Option<Vec<Stmt>> = None;
    for j in (0..n_grid_out).rev() {
        let buf = format!("{}{j}", String::from("grid_out_"));
        let idx = if j == 0 {
            claimed.clone()
        } else {
            Expr::sub(claimed.clone(), Expr::var(va_off_name(j)))
        };
        let arm = vec![Stmt::Eval(Expr::call(
            String::from("atomicAdd"),
            vec![Expr::addr(Expr::index(buf, idx)), val.clone()],
        ))];
        if j == n_grid_out - 1 {
            tail = Some(arm);
            continue;
        }
        let cond = Expr::lt(claimed.clone(), Expr::var(va_off_name(j + 1)));
        tail = Some(vec![Stmt::If {
            cond,
            then_: arm,
            else_: tail,
        }]);
    }
    if let Some(t) = tail {
        stmts.extend(t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fourierd3_engine::dtype::Dtype;

    use crate::kernel_compiler::periodic_scatter::{IndexLayout, ScatterSpec};

    fn hash_problem() -> ScatterSpec {
        let ic = vec![vec![-1], vec![-1], vec![-1]];
        let extents = vec![vec![1], vec![1], vec![1]];
        ScatterSpec {
            batch_sizes: vec![1],
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

    fn render(accumulator: &SmemHash, p: &ScatterSpec) -> String {
        fourierd3_engine::ir::stmt::render_module_string(&accumulator.emit_body(p))
    }

    #[test]
    fn single_output_smoke() {
        let s = render(
            &SmemHash {
                mode: HashMode::Plain,
            },
            &hash_problem(),
        );
        assert!(s.contains("__shared__ float cache_val["), "got:\n{s}");
        assert!(s.contains("__shared__ int cache_idx["), "got:\n{s}");
        assert!(s.contains("cache_idx[_ci] = -1;"), "got:\n{s}");
        assert!(s.contains("cache_val[_ci] = 0.0f;"), "got:\n{s}");
        assert!(
            s.contains("_cache_store(cache_val, cache_idx,"),
            "got:\n{s}"
        );
        assert!(
            s.contains("if (cache_idx[_ci] >= 0)"),
            "flush missing:\n{s}"
        );
        assert!(
            s.contains("atomicAdd(&grid_out_0[cache_idx[_ci]], float(cache_val[_ci]));"),
            "got:\n{s}"
        );
        assert!(!s.contains("_anchor_x"), "anchor leak:\n{s}");
    }

    #[test]
    fn anchored_emits_anchor_scaffolding() {
        let s = render(
            &SmemHash {
                mode: HashMode::Anchored,
            },
            &hash_problem(),
        );
        assert!(s.contains("__shared__ int _anchor_x;"), "got:\n{s}");
        assert!(s.contains("_anchor_x = bx;"), "got:\n{s}");
    }

    #[test]
    fn inventory_registers_smem_hash() {
        let s = super::super::accumulator::all_accumulators()
            .into_iter()
            .find(|s| s.name() == "m7")
            .expect("SmemHashTable should be registered");
        assert_eq!(s.name(), "m7");
    }

    #[test]
    fn inventory_registers_smem_anchored_hash() {
        let s = super::super::accumulator::all_accumulators()
            .into_iter()
            .find(|s| s.name() == "t3")
            .expect("SmemAnchoredHash should be registered");
        assert_eq!(s.name(), "t3");
    }
}
