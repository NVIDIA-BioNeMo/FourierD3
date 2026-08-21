// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::dtype::Dtype;

#[derive(Clone, Debug)]
pub(crate) struct IndexLayout {
    pub ic: Vec<Vec<i32>>,
    pub n_grid_in: usize,
    pub n_nongrid_in: usize,
    pub n_grid_out: usize,
    pub n_nongrid_out: usize,
    pub n_index: usize,
}

impl IndexLayout {
    pub(crate) fn grid_in_offset(&self) -> usize {
        1
    }
    pub(crate) fn nongrid_in_offset(&self) -> usize {
        1 + self.n_grid_in
    }
    pub(crate) fn grid_out_offset(&self) -> usize {
        1 + self.n_grid_in + self.n_nongrid_in
    }
    pub(crate) fn nongrid_out_offset(&self) -> usize {
        self.grid_out_offset() + self.n_grid_out
    }
    pub(crate) fn idx_offset(&self) -> usize {
        self.nongrid_out_offset() + self.n_nongrid_out
    }

    pub(crate) fn cell_idx(&self) -> &[i32] {
        &self.ic[0]
    }
    pub(crate) fn grid_in(&self) -> &[Vec<i32>] {
        &self.ic[self.grid_in_offset()..self.nongrid_in_offset()]
    }
    pub(crate) fn nongrid_in(&self) -> &[Vec<i32>] {
        &self.ic[self.nongrid_in_offset()..self.grid_out_offset()]
    }
    pub(crate) fn grid_out(&self) -> &[Vec<i32>] {
        &self.ic[self.grid_out_offset()..self.nongrid_out_offset()]
    }
    pub(crate) fn nongrid_out(&self) -> &[Vec<i32>] {
        &self.ic[self.nongrid_out_offset()..self.idx_offset()]
    }
    pub(crate) fn idx(&self) -> &[Vec<i32>] {
        &self.ic[self.idx_offset()..]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScatterSpec {
    pub batch_sizes: Vec<i64>,
    pub layout: IndexLayout,
    pub buf_batch_extents: Vec<Vec<i64>>,
    pub n_backend_arrays: usize,

    pub nongrid_in_sizes: Vec<i64>,
    pub nongrid_in_dtypes: Vec<Dtype>,
    pub grid_in_inner_sizes: Vec<i64>,
    pub grid_in_dtypes: Vec<Dtype>,
    pub grid_out_inner_sizes: Vec<i64>,
    pub grid_out_dtypes: Vec<Dtype>,
    pub nongrid_out_sizes: Vec<i64>,
    pub nongrid_out_dtypes: Vec<Dtype>,
    pub nongrid_scatter_flags: Vec<bool>,

    pub n_state: usize,
    pub state_sizes: Vec<i64>,
    pub state_dtypes: Vec<Dtype>,
    pub pre_ngin_indices: Vec<usize>,
    pub direct_ngin_indices: Vec<usize>,

    pub k: i32,
    pub s_support: i32,
    pub cartesian: Option<(i32, i32)>,

    pub grid_shape: [i64; 3],
    pub cell_grid_shape: Option<[i64; 3]>,
}

impl ScatterSpec {
    pub(crate) fn uses_atom_map(&self) -> bool {
        self.n_backend_arrays > 0 && self.layout.n_index == 0
    }

    pub(crate) fn bytes_per_cell(&self) -> i64 {
        self.grid_out_inner_sizes
            .iter()
            .zip(&self.grid_out_dtypes)
            .map(|(sz, dt)| sz * dt.size() as i64)
            .sum()
    }

    pub(crate) fn single_batch(&self) -> bool {
        if self.layout.n_grid_out == 0 {
            return true;
        }
        self.buf_batch_extents[self.layout.grid_out_offset()]
            .iter()
            .product::<i64>()
            == 1
    }

    pub(crate) fn grid_out_extents(&self) -> &[Vec<i64>] {
        &self.buf_batch_extents[self.layout.grid_out_offset()..self.layout.nongrid_out_offset()]
    }
}

pub(crate) fn smem_strategy_applicable(problem: &ScatterSpec) -> bool {
    if problem.layout.n_grid_out == 0 {
        return false;
    }
    if problem.batch_sizes.len() > 1 {
        return false;
    }
    let outs = problem.grid_out_extents();
    let first = &outs[0];
    outs.iter().all(|e| e == first)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_compiler::periodic_scatter::accumulator::all_accumulators;

    fn two_output_problem(slots: i64) -> ScatterSpec {
        let ic = vec![vec![-1], vec![-1], vec![-1], vec![-1]];
        let extents = vec![vec![slots]; 4];
        ScatterSpec {
            batch_sizes: vec![slots],
            layout: IndexLayout {
                ic,
                n_grid_in: 1,
                n_nongrid_in: 0,
                n_grid_out: 2,
                n_nongrid_out: 0,
                n_index: 0,
            },
            buf_batch_extents: extents,
            n_backend_arrays: 0,
            nongrid_in_sizes: vec![],
            nongrid_in_dtypes: vec![],
            grid_in_inner_sizes: vec![1],
            grid_in_dtypes: vec![Dtype::F32],
            grid_out_inner_sizes: vec![1, 1],
            grid_out_dtypes: vec![Dtype::F32, Dtype::F32],
            nongrid_out_sizes: vec![],
            nongrid_out_dtypes: vec![],
            nongrid_scatter_flags: vec![],
            n_state: 0,
            state_sizes: vec![],
            state_dtypes: vec![],
            pre_ngin_indices: vec![],
            direct_ngin_indices: vec![],
            k: 1,
            s_support: 32,
            cartesian: None,
            grid_shape: [8, 8, 8],
            cell_grid_shape: None,
        }
    }

    fn applicable_names(problem: &ScatterSpec) -> Vec<&'static str> {
        all_accumulators()
            .into_iter()
            .filter(|s| s.is_applicable(problem))
            .map(|s| s.name())
            .collect()
    }

    #[test]
    fn multi_slot_outputs_exclude_cell_keyed_caches() {
        let names = applicable_names(&two_output_problem(10));
        for forbidden in ["m7", "t3", "h4"] {
            assert!(
                !names.contains(&forbidden),
                "{forbidden} must not apply to a multi-slot grid output; got {names:?}"
            );
        }
        assert!(names.contains(&"c2"), "got {names:?}");
        assert!(names.contains(&"p9"), "got {names:?}");
    }

    #[test]
    fn single_slot_outputs_admit_cell_keyed_caches() {
        let names = applicable_names(&two_output_problem(1));
        for expected in ["m7", "t3", "h4", "p9", "c2"] {
            assert!(
                names.contains(&expected),
                "missing {expected}; got {names:?}"
            );
        }
    }
}
