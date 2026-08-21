// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::accumulator::{Accumulator, AccumulatorEntry};
use super::specification::ScatterSpec;

pub(crate) struct GlobalAtomic;

impl Accumulator for GlobalAtomic {
    fn name(&self) -> &'static str {
        "c2"
    }

    // Atomic-scatter-to-global only makes sense when there is at least
    // one grid output to scatter into. The `n_grid_out == 0` case
    // (pure interpolation) is `UncachedAccumulator`'s territory.
    fn is_applicable(&self, problem: &ScatterSpec) -> bool {
        problem.layout.n_grid_out > 0
    }
}

inventory::submit! {
    AccumulatorEntry { make: || Box::new(GlobalAtomic) }
}

#[cfg(test)]
mod tests {
    use fourierd3_engine::dtype::Dtype;

    use super::*;
    use crate::kernel_compiler::periodic_scatter::{
        IndexLayout, ScatterSpec, build_param_list, build_preamble, emit_full_grid_kernel,
    };

    fn minimal_problem(n: i64) -> ScatterSpec {
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

    fn render_body(problem: &ScatterSpec) -> String {
        fourierd3_engine::ir::stmt::render_module_string(&GlobalAtomic.emit_body(problem))
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn minimal_emit_smoke() {
        let s = render_body(&minimal_problem(1024));
        assert!(s.contains("int TOTAL = n_params[0];"), "got:\n{s}");
        assert!(
            s.contains("long long _raw = (long long)(blockIdx.x) * blockDim.x + threadIdx.x;"),
            "got:\n{s}"
        );
        assert!(s.contains("bool _atom_active = _raw < TOTAL;"), "got:\n{s}");
        assert!(s.contains("int batch_idx_0 = linear_idx;"), "got:\n{s}");
        assert!(
            s.contains("long long ci_off = (long long)(batch_idx_0);"),
            "got:\n{s}"
        );
        assert!(s.contains("int bx = cell_idx[ci_off * 3];"), "got:\n{s}");
        assert!(
            s.contains("int by = cell_idx[ci_off * 3 + 1];"),
            "got:\n{s}"
        );
        assert!(
            s.contains("int bz = cell_idx[ci_off * 3 + 2];"),
            "got:\n{s}"
        );
        assert!(
            s.contains("long long off_grid_0 = (long long)(batch_idx_0);"),
            "got:\n{s}"
        );
        assert!(
            s.contains("long long b_0 = (long long)(batch_idx_0);"),
            "got:\n{s}"
        );
        assert!(s.contains("V_pre();"), "got:\n{s}");
        assert!(
            s.contains("for (int s = 0; s < S_SUPPORT; s++)"),
            "got:\n{s}"
        );
        assert!(s.contains("int ix = bx + SUPPORT[s * 3];"), "got:\n{s}");
        assert!(s.contains("ix -= (ix >= GX) * GX;"), "got:\n{s}");
        assert!(s.contains("ix += (ix < 0) * GX;"), "got:\n{s}");
        assert!(s.contains("int iy = by + SUPPORT[s * 3 + 1];"), "got:\n{s}");
        assert!(s.contains("int iz = bz + SUPPORT[s * 3 + 2];"), "got:\n{s}");
        assert!(
            s.contains("int _sup[] = {SUPPORT[s * 3], SUPPORT[s * 3 + 1], SUPPORT[s * 3 + 2]};"),
            "got:\n{s}"
        );
        assert!(
            s.contains(
                "long long cell_in_0 = (((long long)(off_grid_0) * GX + ix) * GY + iy) * GZ + iz;"
            ),
            "got:\n{s}"
        );
        assert!(
            s.contains("float _gval_0 = __ldg(&grid_in_0[cell_in_0]);"),
            "got:\n{s}"
        );
        assert!(
            s.contains("V_step(&_sidx, _sup, &_gval_0, _gout0);"),
            "got:\n{s}"
        );
        assert!(
            s.contains(
                "atomicAdd(&grid_out_0[(((long long)(b_0) * GX + ix) * GY + iy) * GZ + iz], _gout0[0]);"
            ),
            "got:\n{s}"
        );
    }

    #[test]
    fn k_greater_than_one_emits_warp_cooperation_prologue() {
        let mut p = minimal_problem(1024);
        p.k = 4;
        let s = render_body(&p);
        assert!(s.contains("int _K = 4;"), "got:\n{s}");
        assert!(s.contains("int _APW = 8;"), "got:\n{s}");
        assert!(s.contains("int _WPB = blockDim.x >> 5;"), "got:\n{s}");
        assert!(s.contains("int _APB = _WPB * _APW;"), "got:\n{s}");
        assert!(s.contains("int _warp = threadIdx.x >> 5;"), "got:\n{s}");
        assert!(s.contains("int _lane = threadIdx.x & 31;"), "got:\n{s}");
        assert!(s.contains("int _atom_in_warp = _lane / _K;"), "got:\n{s}");
        assert!(s.contains("int _sub = _lane % _K;"), "got:\n{s}");
        assert!(
            s.contains("bool _lane_active = _lane < _APW * _K;"),
            "got:\n{s}"
        );
        assert!(
            s.contains(
                "long long _raw = (long long)(blockIdx.x) * _APB + _warp * _APW + _atom_in_warp;"
            ),
            "got:\n{s}"
        );
        assert!(
            s.contains("bool _atom_active = _lane_active && _raw < TOTAL;"),
            "got:\n{s}"
        );
        let mut p = minimal_problem(1024);
        p.k = 3;
        let s = render_body(&p);
        assert!(s.contains("int _s0 = _sub * 9;"), "got:\n{s}");
        assert!(
            s.contains("for (int s = _s0; s < _s0 + 9; s++)"),
            "got:\n{s}"
        );
    }

    #[test]
    fn cartesian_nested_loop_when_k_equals_order() {
        let mut p = minimal_problem(1024);
        p.k = 4;
        p.s_support = 64;
        p.cartesian = Some((4, -1));
        let s = render_body(&p);

        assert!(s.contains("int _sdz = _sub + -1;"), "got:\n{s}");
        assert!(s.contains("int iz = bz + _sdz;"), "got:\n{s}");
        assert!(s.contains("iz -= (iz >= GZ) * GZ;"), "got:\n{s}");
        assert!(s.contains("#pragma unroll"), "got:\n{s}");
        assert!(s.contains("for (int _ix = 0; _ix < 4; _ix++)"), "got:\n{s}");
        assert!(s.contains("int _sdx = _ix + -1;"), "got:\n{s}");
        assert!(s.contains("for (int _iy = 0; _iy < 4; _iy++)"), "got:\n{s}");
        assert!(s.contains("int _sdy = _iy + -1;"), "got:\n{s}");
        assert!(
            s.contains("int _sidx = _ix * 16 + _iy * 4 + _sub;"),
            "got:\n{s}"
        );
        assert!(s.contains("int _sup[] = {_sdx, _sdy, _sdz};"), "got:\n{s}");
        assert!(!s.contains("S_SUPPORT"), "flat-fallback leak:\n{s}");
        assert!(!s.contains("SUPPORT[s "), "flat-fallback leak:\n{s}");
    }

    #[test]
    fn cartesian_with_k_neq_order_falls_back_to_flat() {
        let mut p = minimal_problem(1024);
        p.k = 1;
        p.s_support = 64;
        p.cartesian = Some((4, -1));
        let s = render_body(&p);
        assert!(
            s.contains("for (int s = 0; s < S_SUPPORT; s++)"),
            "got:\n{s}"
        );
        assert!(!s.contains("_sdz"), "should not nest:\n{s}");
        assert!(!s.contains("for (int _ix ="), "should not nest:\n{s}");
    }

    #[test]
    fn full_kernel_assembly_smoke() {
        let problem = minimal_problem(1024);
        let body = GlobalAtomic.emit_body(&problem);
        let params = build_param_list(&problem);
        let preamble = build_preamble([32, 32, 32], 27, &[-1, -1, -1]);
        let device_fns = "__device__ __forceinline__ void V_pre() {}\n\
                          __device__ __forceinline__ void V_step(\
                          int* sidx, int* sup, float* g, float* gout) {}\n";
        let module =
            emit_full_grid_kernel(&preamble, device_fns, &[], "", &params, &body, "kernel");
        let src = fourierd3_engine::ir::stmt::render_module_string(&module);
        assert!(src.contains("#define GX 32"), "{src}");
        assert!(src.contains("__constant__ short SUPPORT[]"), "{src}");
        assert!(src.contains("V_step"), "{src}");
        assert!(
            src.contains("extern \"C\" __global__ void kernel("),
            "{src}"
        );
        assert!(src.contains("float* __restrict__ grid_out_0"), "{src}");
        assert!(src.contains("for (int s = 0; s < S_SUPPORT; s++)"), "{src}");
    }

    #[test]
    fn inventory_registers_global_atomic() {
        let found = super::super::accumulator::all_accumulators()
            .into_iter()
            .any(|s| s.name() == "c2");
        assert!(found, "GlobalAtomic should be registered via inventory");
    }
}
