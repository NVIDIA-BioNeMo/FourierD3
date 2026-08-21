// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::accumulator::{Accumulator, AccumulatorEntry};
use super::specification::ScatterSpec;

pub(crate) struct UncachedAccumulator;

impl Accumulator for UncachedAccumulator {
    fn name(&self) -> &'static str {
        "z5"
    }

    fn is_applicable(&self, problem: &ScatterSpec) -> bool {
        problem.layout.n_grid_out == 0
    }

    fn uses_ps_init(&self) -> bool {
        false
    }
}

inventory::submit! {
    AccumulatorEntry { make: || Box::new(UncachedAccumulator) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fourierd3_engine::dtype::Dtype;

    use crate::kernel_compiler::periodic_scatter::{IndexLayout, ScatterSpec};

    fn interp_problem(n: i64) -> ScatterSpec {
        let ic = vec![vec![-1], vec![-1], vec![-1]];
        let extents = vec![vec![n], vec![n], vec![n]];
        ScatterSpec {
            batch_sizes: vec![n],
            layout: IndexLayout {
                ic,
                n_grid_in: 1,
                n_nongrid_in: 0,
                n_grid_out: 0,
                n_nongrid_out: 1,
                n_index: 0,
            },
            buf_batch_extents: extents,
            n_backend_arrays: 0,
            nongrid_in_sizes: vec![],
            nongrid_in_dtypes: vec![],
            grid_in_inner_sizes: vec![1],
            grid_in_dtypes: vec![Dtype::F32],
            grid_out_inner_sizes: vec![],
            grid_out_dtypes: vec![],
            nongrid_out_sizes: vec![1],
            nongrid_out_dtypes: vec![Dtype::F32],
            nongrid_scatter_flags: vec![false],
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

    #[test]
    fn no_cache_emits_no_scatter() {
        let p = interp_problem(1024);
        let s =
            fourierd3_engine::ir::stmt::render_module_string(&UncachedAccumulator.emit_body(&p));
        assert!(s.contains("V_step("), "got:\n{s}");
        assert!(!s.contains("grid_out_"), "scatter leak:\n{s}");
        assert!(!s.contains("atomicAdd(&grid_out_"), "scatter leak:\n{s}");
        assert!(s.contains("out_0["), "got:\n{s}");
    }

    #[test]
    fn inventory_registers_no_cache() {
        let found = super::super::accumulator::all_accumulators()
            .into_iter()
            .any(|s| s.name() == "z5");
        assert!(
            found,
            "UncachedAccumulator should be registered via inventory"
        );
    }

    #[test]
    fn is_applicable_requires_no_grid_outputs() {
        let mut p = interp_problem(1024);
        assert!(UncachedAccumulator.is_applicable(&p));
        p.layout.n_grid_out = 1;
        assert!(!UncachedAccumulator.is_applicable(&p));
    }
}
