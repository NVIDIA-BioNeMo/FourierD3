// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Accumulation of per-atom contributions onto a periodic mesh: the kernel
//! shapes, the accumulators they can use, and the candidate set the
//! executor autotunes over.

pub(crate) mod accumulator;
pub(crate) mod candidate_plans;
pub(crate) mod global_atomic_accumulator;
pub(crate) mod kernel_body;
pub(crate) mod plan_request;
pub(crate) mod shared_cell_accumulator;
pub(crate) mod shared_cube_accumulator;
pub(crate) mod shared_grid_accumulator;
pub(crate) mod shared_hash_accumulator;
pub(crate) mod specification;
pub(crate) mod uncached_accumulator;

pub(crate) use candidate_plans::{CompileRequest, compile_candidates};
pub(crate) use plan_request::{ScatterPlanRequest, compile_scatter_plan};
pub(crate) use specification::{IndexLayout, ScatterSpec};

use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::{Param, Stmt};

pub(crate) fn build_preamble(grid_shape: [i64; 3], s_support: i32, support: &[i16]) -> Vec<Stmt> {
    let define = |name: String, value: i64| Stmt::Define {
        name,
        value: value.to_string(),
    };
    vec![
        define(String::from("GX"), grid_shape[0]),
        define(String::from("GY"), grid_shape[1]),
        define(String::from("GZ"), grid_shape[2]),
        define(String::from("S_SUPPORT"), s_support as i64),
        Stmt::constant_array_decl(
            String::from("short"),
            String::from("SUPPORT"),
            None,
            support.iter().map(|&v| Expr::lit(v as i64)).collect(),
        ),
    ]
}

pub(crate) fn emit_full_grid_kernel(
    preamble: &[Stmt],
    device_fn_src: &str,
    pre_kernel_src: &[Stmt],
    launch_bounds: &str,
    params: &[Param],
    body: &[Stmt],
    kernel_name: &str,
) -> Vec<Stmt> {
    let mut module: Vec<Stmt> = preamble.to_vec();
    module.push(Stmt::Raw(device_fn_src.to_string()));
    module.extend(pre_kernel_src.iter().cloned());
    module.push(Stmt::Kernel {
        name: kernel_name.to_string(),
        launch_bounds: launch_bounds.to_string(),
        params: params.to_vec(),
        body: body.to_vec(),
    });
    module
}

pub(crate) fn build_param_list(problem: &ScatterSpec) -> Vec<Param> {
    let mut params: Vec<Param> = Vec::new();
    params.push(Param::Pointer {
        const_: true,
        restrict: true,
        ctype: String::from("int"),
        name: String::from("cell_idx"),
    });
    let grid_in_prefix = String::from("grid_in_");
    for (g, dt) in problem.grid_in_dtypes.iter().enumerate() {
        params.push(Param::Pointer {
            const_: true,
            restrict: true,
            ctype: dt.into(),
            name: format!("{grid_in_prefix}{g}"),
        });
    }
    let ngin_prefix = String::from("ngin_");
    for (k, dt) in problem.nongrid_in_dtypes.iter().enumerate() {
        params.push(Param::Pointer {
            const_: true,
            restrict: true,
            ctype: dt.into(),
            name: format!("{ngin_prefix}{k}"),
        });
    }
    let idx_prefix = String::from("idx_");
    for j in 0..problem.layout.n_index {
        params.push(Param::Pointer {
            const_: true,
            restrict: true,
            ctype: String::from("int"),
            name: format!("{idx_prefix}{j}"),
        });
    }
    params.push(Param::Pointer {
        const_: true,
        restrict: true,
        ctype: String::from("int"),
        name: String::from("n_params"),
    });
    if problem.n_backend_arrays > 0 {
        params.push(Param::Pointer {
            const_: true,
            restrict: true,
            ctype: String::from("int"),
            name: String::from("atom_map"),
        });
        params.push(Param::Pointer {
            const_: true,
            restrict: true,
            ctype: String::from("unsigned int"),
            name: String::from("cell_starts_ends"),
        });
    }
    let grid_out_prefix = String::from("grid_out_");
    for (j, dt) in problem.grid_out_dtypes.iter().enumerate() {
        params.push(Param::Pointer {
            const_: false,
            restrict: true,
            ctype: dt.into(),
            name: format!("{grid_out_prefix}{j}"),
        });
    }
    let result_prefix = String::from("result_");
    let out_prefix = String::from("out_");
    for (j, dt) in problem.nongrid_out_dtypes.iter().enumerate() {
        let name = if problem.nongrid_scatter_flags[j] {
            format!("{result_prefix}{j}")
        } else {
            format!("{out_prefix}{j}")
        };
        params.push(Param::Pointer {
            const_: false,
            restrict: true,
            ctype: dt.into(),
            name,
        });
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use fourierd3_engine::ir::stmt::render_module_string;

    #[test]
    fn cas_cache_store_fn_is_a_typed_device_fn() {
        use crate::kernel_compiler::periodic_scatter::kernel_body::cas_cache_store_fn;
        use fourierd3_engine::cuda;
        use fourierd3_engine::ir::stmt::Param;

        let mut slot = Vec::new();
        cuda! { slot => i32 _slot = _va & 255; }
        let f = cas_cache_store_fn("float", slot, vec![]);
        assert!(matches!(
            f,
            Stmt::DeviceFn {
                forceinline: true,
                ..
            }
        ));
        let s = render_module_string(&[f]);
        assert!(
            s.contains(
                "__device__ __forceinline__ void _cache_store(\
                 float* cache_val, int* cache_idx, float* grid, \
                 int _cell, int _va, float _v) {"
            ),
            "{s}"
        );
        assert!(s.contains("int _slot = _va & 255;"), "{s}");
        assert!(
            s.contains("int _old = atomicCAS(&cache_idx[_slot], -1, _va);"),
            "{s}"
        );
        assert!(s.contains("atomicAdd(&cache_val[_slot], _v);"), "{s}");
        assert!(s.contains("atomicAdd(&grid[_cell], _v);"), "{s}");

        let coord = |name: &str| Param::Scalar {
            ctype: "int".into(),
            name: name.into(),
        };
        let extra = ["_ix", "_iy", "_iz", "_ax", "_ay", "_az"]
            .map(coord)
            .to_vec();
        let anchored = render_module_string(&[cas_cache_store_fn("double", vec![], extra)]);
        assert!(
            anchored.contains("double _v, int _ix, int _iy, int _iz, int _ax, int _ay, int _az)"),
            "{anchored}"
        );
    }

    #[test]
    fn preamble_is_typed_defines_plus_constant_array() {
        let p = build_preamble([32, 24, 16], 27, &[-1, 0, 1]);
        assert!(matches!(p[0], Stmt::Define { .. }));
        assert!(matches!(p[3], Stmt::Define { .. }));
        assert!(matches!(
            p[4],
            Stmt::Decl(fourierd3_engine::ir::stmt::Decl {
                storage: fourierd3_engine::ir::stmt::Storage::Constant,
                ..
            })
        ));
        assert_eq!(p.len(), 5);
        assert_eq!(
            render_module_string(&p),
            "#define GX 32\n\
             #define GY 24\n\
             #define GZ 16\n\
             #define S_SUPPORT 27\n\
             __constant__ short SUPPORT[] = {-1, 0, 1};\n"
        );
    }
}
