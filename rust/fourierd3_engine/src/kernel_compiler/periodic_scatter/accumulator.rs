// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::stmt::Stmt;

use super::kernel_body;
use super::specification::ScatterSpec;

pub(crate) struct HookCtx<'a> {
    pub n_grid_out: usize,
    pub grid_out_inner_sizes: &'a [i64],
    pub grid_out_dtypes: &'a [Dtype],
    pub out_batch_vars: &'a [String],
    pub single_batch: bool,
    pub problem: &'a ScatterSpec,
}

pub(crate) struct ScatterCtx<'a> {
    pub j: usize,
    pub isz: i64,
    pub batch_var: &'a str,
    pub single_batch: bool,
}

pub(crate) trait Accumulator: Send + Sync {
    fn name(&self) -> &'static str;

    fn is_applicable(&self, problem: &ScatterSpec) -> bool {
        let _ = problem;
        true
    }

    fn k_candidates(&self, problem: &ScatterSpec) -> Vec<i32> {
        default_k_candidates(problem)
    }

    fn block_size(&self, problem: &ScatterSpec) -> i32 {
        let _ = problem;
        128
    }

    fn uses_ps_init(&self) -> bool {
        true
    }

    fn ps_init(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let _ = (stmts, ctx);
    }

    fn ps_scatter(&self, stmts: &mut Vec<Stmt>, ctx: &ScatterCtx<'_>) {
        kernel_body::emit_global_atomic_scatter(stmts, ctx.j, ctx.isz, ctx.batch_var);
    }

    fn ps_flush(&self, stmts: &mut Vec<Stmt>, ctx: &HookCtx<'_>) {
        let _ = (stmts, ctx);
    }

    fn emit_body(&self, problem: &ScatterSpec) -> Vec<Stmt> {
        kernel_body::emit_standard_body(self, problem)
    }

    fn uses_smem(&self) -> bool {
        false
    }

    fn min_blocks(&self) -> i32 {
        2
    }

    fn pre_kernel_source(&self, problem: &ScatterSpec) -> Vec<Stmt> {
        let _ = problem;
        Vec::new()
    }

    fn launch_bounds(&self, block_size: i32, k: i32) -> String {
        if self.uses_smem() {
            format!(" __launch_bounds__({block_size}, {})", self.min_blocks())
        } else if k > 1 {
            format!(" __launch_bounds__({block_size}, 4)")
        } else {
            String::new()
        }
    }
}

pub(crate) fn default_k_candidates(problem: &ScatterSpec) -> Vec<i32> {
    let s = problem.s_support;
    match problem.cartesian {
        Some((order, _)) if order > 0 && s % order == 0 => vec![order, 1],
        Some(_) => vec![1],
        None => (1..=32).filter(|k| s % k == 0).collect(),
    }
}

pub(crate) fn atoms_per_block(block_size: i32, k: i32) -> i32 {
    if k == 1 {
        block_size
    } else {
        let warps_per_block = block_size / 32;
        warps_per_block * (32 / k)
    }
}

pub(crate) struct AccumulatorEntry {
    pub make: fn() -> Box<dyn Accumulator>,
}

inventory::collect!(AccumulatorEntry);

pub(crate) fn all_accumulators() -> Vec<Box<dyn Accumulator>> {
    inventory::iter::<AccumulatorEntry>()
        .map(|e| (e.make)())
        .collect()
}
