// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use fourierd3_engine::cuda;
use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::{Param, Stmt};

use super::accumulator::{Accumulator, HookCtx, ScatterCtx};
use super::specification::{IndexLayout, ScatterSpec};

pub(crate) const CACHE_BUDGET: i64 = 48 * 1024;

pub(crate) fn full_cache_side(bytes_per_cell: i64) -> i64 {
    let available = CACHE_BUDGET - 512;
    let mut side = 16i64;
    while side > 4 && side.pow(3) * bytes_per_cell > available {
        side -= 1;
    }
    side
}

pub(crate) fn cube_cache_side(bytes_per_cell: i64) -> i64 {
    let available = CACHE_BUDGET - 1024;
    let mut side = 16i64;
    while side > 4 && side.pow(3) * bytes_per_cell > available {
        side -= 1;
    }
    side
}

pub(crate) fn hash_cache_size(max_elem_bytes: i64, reserved: i64) -> i64 {
    let entry_bytes = 4 + max_elem_bytes;
    let budget = CACHE_BUDGET - reserved;
    let mut n = 1i64;
    while n * 2 <= budget / entry_bytes {
        n *= 2;
    }
    n.max(64)
}

pub(crate) fn cell_buffer_spans(
    grid_shape: [i64; 3],
    cell_grid_shape: [i64; 3],
    order: i64,
) -> [i64; 3] {
    let mut spans = [0i64; 3];
    for axis in 0..3 {
        let g = grid_shape[axis];
        let nc = cell_grid_shape[axis];
        spans[axis] = (g + nc - 1) / nc + order - 1;
    }
    spans
}

pub(crate) fn zero_lit_for(dt: Dtype) -> &'static str {
    match dt {
        Dtype::F64 => "0.0",
        Dtype::F32 => "0.0f",
        _ => "0",
    }
}

pub(crate) fn cas_cache_store_fn(ct: &str, slot_decl: Vec<Stmt>, extra_params: Vec<Param>) -> Stmt {
    let ptr = |ctype: String, name: String| Param::Pointer {
        const_: false,
        restrict: false,
        ctype,
        name,
    };
    let scalar = |ctype: String, name: String| Param::Scalar { ctype, name };
    let int = || String::from("int");

    let mut params = vec![
        ptr(ct.to_string(), String::from("cache_val")),
        ptr(int(), String::from("cache_idx")),
        ptr(ct.to_string(), String::from("grid")),
        scalar(int(), String::from("_cell")),
        scalar(int(), String::from("_va")),
        scalar(ct.to_string(), String::from("_v")),
    ];
    params.extend(extra_params);

    let mut body = slot_decl;
    cuda! { body =>
        i32 _claimed = cache_idx[_slot];
        if (_claimed == _va) {
            atomicAdd(&cache_val[_slot], _v);
        } else if (_claimed == -1) {
            i32 _old = atomicCAS(&cache_idx[_slot], -1, _va);
            if (_old == -1 || _old == _va) {
                atomicAdd(&cache_val[_slot], _v);
            } else {
                atomicAdd(&grid[_cell], _v);
            }
        } else {
            atomicAdd(&grid[_cell], _v);
        }
    }

    Stmt::DeviceFn {
        name: String::from("_cache_store"),
        ret_ty: String::from("void"),
        params,
        body,
        forceinline: true,
    }
}

pub(crate) fn emit_buffer_offset(
    stmts: &mut Vec<Stmt>,
    name: &str,
    ic_entry: &[i32],
    buf_extents: &[i64],
    idx_offset_vars: &HashMap<usize, String>,
) {
    crate::kernel_compiler::batch_indexing::push_row_major_offset_decl(
        stmts,
        name,
        ic_entry,
        buf_extents,
        idx_offset_vars,
    );
}

pub(crate) fn emit_thread_assignment(stmts: &mut Vec<Stmt>, k: i32, use_atom_map: bool) {
    // No early `return`: smem accumulators issue `__syncthreads()` in
    // their init / flush hooks, and a barrier hit by only some threads
    // in the block is undefined behaviour. Instead, set `_atom_active`
    // and clamp the per-atom index into range so every thread runs the
    // collective code; the scatter loop is then gated on `_atom_active`.
    if k > 1 {
        let atoms_per_warp = 32 / k;
        let mut tail: Vec<Stmt> = Vec::new();
        if use_atom_map {
            cuda! { tail =>
                i64 _raw_sorted = (i64)blockIdx.x * _APB + _warp * _APW + _atom_in_warp;
                bool _atom_active = _lane_active && (_raw_sorted < TOTAL);
                i64 sorted_idx = _raw_sorted < TOTAL ? _raw_sorted : (TOTAL > 0 ? TOTAL - 1 : 0);
                i64 linear_idx = atom_map[sorted_idx];
            }
        } else {
            cuda! { tail =>
                i64 _raw = (i64)blockIdx.x * _APB + _warp * _APW + _atom_in_warp;
                bool _atom_active = _lane_active && (_raw < TOTAL);
                i64 linear_idx = _raw < TOTAL ? _raw : (TOTAL > 0 ? TOTAL - 1 : 0);
            }
        }
        cuda! { stmts =>
            i32 TOTAL = n_params[0];
            i32 _K = #k;
            i32 _APW = #atoms_per_warp;
            i32 _WPB = blockDim.x >> 5;
            i32 _APB = _WPB * _APW;
            i32 _warp = threadIdx.x >> 5;
            i32 _lane = threadIdx.x & 31;
            i32 _atom_in_warp = _lane / _K;
            i32 _sub = _lane % _K;
            bool _lane_active = _lane < _APW * _K;
            splice!(tail);
        }
    } else if use_atom_map {
        cuda! { stmts =>
            i32 TOTAL = n_params[0];
            i64 _raw_sorted = (i64)blockIdx.x * blockDim.x + threadIdx.x;
            bool _atom_active = _raw_sorted < TOTAL;
            i64 sorted_idx = _raw_sorted < TOTAL ? _raw_sorted : (TOTAL > 0 ? TOTAL - 1 : 0);
            i64 linear_idx = atom_map[sorted_idx];
        }
    } else {
        cuda! { stmts =>
            i32 TOTAL = n_params[0];
            i64 _raw = (i64)blockIdx.x * blockDim.x + threadIdx.x;
            bool _atom_active = _raw < TOTAL;
            i64 linear_idx = _raw < TOTAL ? _raw : (TOTAL > 0 ? TOTAL - 1 : 0);
        }
    }
}

pub(crate) fn emit_batch_decode(stmts: &mut Vec<Stmt>, batch_sizes: &[i64]) {
    crate::kernel_compiler::batch_indexing::push_batch_decode(stmts, batch_sizes, "linear_idx");
}

pub(crate) fn emit_index_buffer_offsets(
    stmts: &mut Vec<Stmt>,
    layout: &IndexLayout,
    buf_batch_extents: &[Vec<i64>],
) -> HashMap<usize, String> {
    let mut vars = HashMap::new();
    let off_idx_prefix = String::from("off_idx_");
    for j in 0..layout.n_index {
        let var = format!("{off_idx_prefix}{j}");
        let ic_entry = &layout.idx()[j];
        let extents = &buf_batch_extents[layout.idx_offset() + j];
        emit_buffer_offset(stmts, &var, ic_entry, extents, &vars);
        vars.insert(j, var);
    }
    vars
}

pub(crate) fn emit_cell_index_load(
    stmts: &mut Vec<Stmt>,
    layout: &IndexLayout,
    buf_batch_extents: &[Vec<i64>],
    idx_offset_vars: &HashMap<usize, String>,
) {
    // CUDA `%` is sign-of-dividend so `bx % GX` can be negative when bx < 0.
    // The `+GX` correction normalises into [0, G) before the per-support
    // wrap, which only spans one period and would write OOB otherwise.
    let ci_off_name = String::from("ci_off");
    emit_buffer_offset(
        stmts,
        &ci_off_name,
        layout.cell_idx(),
        &buf_batch_extents[0],
        idx_offset_vars,
    );
    cuda! { stmts =>
        i32 bx = cell_idx[#ci_off_name * 3];
        i32 by = cell_idx[#ci_off_name * 3 + 1];
        i32 bz = cell_idx[#ci_off_name * 3 + 2];
        bx = bx % GX;
        by = by % GY;
        bz = bz % GZ;
        bx += (bx < 0) * GX;
        by += (by < 0) * GY;
        bz += (bz < 0) * GZ;
    }
}

pub(crate) fn emit_grid_input_batch_offsets(
    stmts: &mut Vec<Stmt>,
    layout: &IndexLayout,
    buf_batch_extents: &[Vec<i64>],
    idx_offset_vars: &HashMap<usize, String>,
) -> Vec<String> {
    let mut vars = Vec::with_capacity(layout.n_grid_in);
    let off_grid_prefix = String::from("off_grid_");
    for g in 0..layout.n_grid_in {
        let var = format!("{off_grid_prefix}{g}");
        let ic_entry = &layout.grid_in()[g];
        let extents = &buf_batch_extents[layout.grid_in_offset() + g];
        emit_buffer_offset(stmts, &var, ic_entry, extents, idx_offset_vars);
        vars.push(var);
    }
    vars
}

pub(crate) fn emit_nongrid_input_ptrs(
    stmts: &mut Vec<Stmt>,
    layout: &IndexLayout,
    buf_batch_extents: &[Vec<i64>],
    nongrid_in_sizes: &[i64],
    idx_offset_vars: &HashMap<usize, String>,
) -> Vec<Expr> {
    let mut parts = Vec::with_capacity(layout.n_nongrid_in);
    let ngin_prefix = String::from("ngin_");
    let off_ngin_prefix = String::from("off_ngin_");
    for k in 0..layout.n_nongrid_in {
        let extents = &buf_batch_extents[layout.nongrid_in_offset() + k];
        let is_shared = extents.iter().all(|&e| e == 1);
        if is_shared {
            parts.push(Expr::var(format!("{ngin_prefix}{k}")));
        } else {
            let var = format!("{off_ngin_prefix}{k}");
            let ic_entry = &layout.nongrid_in()[k];
            emit_buffer_offset(stmts, &var, ic_entry, extents, idx_offset_vars);
            parts.push(Expr::addr(Expr::index(
                format!("{ngin_prefix}{k}"),
                Expr::mul(Expr::var(var), Expr::lit(nongrid_in_sizes[k])),
            )));
        }
    }
    parts
}

pub(crate) fn emit_grid_output_batch_offsets(
    stmts: &mut Vec<Stmt>,
    layout: &IndexLayout,
    buf_batch_extents: &[Vec<i64>],
    idx_offset_vars: &HashMap<usize, String>,
) -> Vec<String> {
    let mut vars = Vec::with_capacity(layout.n_grid_out);
    let mut seen: HashMap<(Vec<i32>, Vec<i64>), String> = HashMap::new();
    let b_prefix = String::from("b_");
    for j in 0..layout.n_grid_out {
        let ic_entry = layout.grid_out()[j].clone();
        let extents = buf_batch_extents[layout.grid_out_offset() + j].clone();
        let key = (ic_entry.clone(), extents.clone());
        if let Some(v) = seen.get(&key) {
            vars.push(v.clone());
            continue;
        }
        let var = format!("{b_prefix}{j}");
        emit_buffer_offset(stmts, &var, &ic_entry, &extents, idx_offset_vars);
        seen.insert(key, var.clone());
        vars.push(var);
    }
    vars
}

pub(crate) fn emit_v_pre_call(
    stmts: &mut Vec<Stmt>,
    state_sizes: &[i64],
    state_dtypes: &[Dtype],
    pre_ngin_indices: &[usize],
    ngin_call_parts: &[Expr],
) {
    let state_prefix = String::from("_state");
    for (k, (sz, dt)) in state_sizes.iter().zip(state_dtypes).enumerate() {
        let name = format!("{state_prefix}{k}");
        let sz = *sz;
        let ct = dt.ctype();
        cuda! { stmts => #ct #name[#sz]; }
    }
    let mut args: Vec<Expr> = pre_ngin_indices
        .iter()
        .map(|&i| ngin_call_parts[i].clone())
        .collect();
    for k in 0..state_sizes.len() {
        args.push(Expr::var(format!("{state_prefix}{k}")));
    }
    stmts.push(Stmt::Eval(Expr::call(String::from("V_pre"), args)));
}

pub(crate) fn emit_nongrid_accumulator_init(
    stmts: &mut Vec<Stmt>,
    nongrid_out_sizes: &[i64],
    nongrid_out_dtypes: &[Dtype],
) {
    let nout_prefix = String::from("_nout");
    for (j, (sz, dt)) in nongrid_out_sizes.iter().zip(nongrid_out_dtypes).enumerate() {
        let name = format!("{nout_prefix}{j}");
        stmts.push(Stmt::array_decl(dt, name.clone(), vec![*sz]));
        if *sz == 1 {
            cuda! { stmts => #name[0] = 0; }
        } else {
            cuda! { stmts =>
                for (i32 _a = 0; _a < #sz; _a++) {
                    #name[_a] = 0;
                }
            }
        }
    }
}

pub(crate) fn emit_cell_coords_branchless(stmts: &mut Vec<Stmt>) {
    cuda! { stmts =>
        i32 ix = bx + SUPPORT[s * 3];
        ix -= (ix >= GX) * GX;
        ix += (ix < 0) * GX;
        i32 iy = by + SUPPORT[s * 3 + 1];
        iy -= (iy >= GY) * GY;
        iy += (iy < 0) * GY;
        i32 iz = bz + SUPPORT[s * 3 + 2];
        iz -= (iz >= GZ) * GZ;
        iz += (iz < 0) * GZ;
    }
}

pub(crate) fn emit_grid_input_loads(
    stmts: &mut Vec<Stmt>,
    grid_in_inner_sizes: &[i64],
    grid_in_dtypes: &[Dtype],
    grid_off_vars: &[String],
) {
    let cell_in_prefix = String::from("cell_in_");
    let grid_in_prefix = String::from("grid_in_");
    let gval_prefix = String::from("_gval_");
    for (g, ((gsz, dt), off)) in grid_in_inner_sizes
        .iter()
        .zip(grid_in_dtypes)
        .zip(grid_off_vars)
        .enumerate()
    {
        let gsz = *gsz;
        let ct = dt.ctype();
        let cell_name = format!("{cell_in_prefix}{g}");
        let cell = grid_cell_expr(Expr::var(off.clone()));
        let buf = format!("{grid_in_prefix}{g}");
        let val = format!("{gval_prefix}{g}");
        if gsz == 1 {
            cuda! { stmts =>
                i64 #cell_name = #cell;
                #ct #val = __ldg(&#buf[#cell_name]);
            }
        } else {
            cuda! { stmts =>
                i64 #cell_name = #cell;
                #ct #val[#gsz];
                for (i32 _a = 0; _a < #gsz; _a++) {
                    #val[_a] = __ldg(&#buf[#cell_name * #gsz + _a]);
                }
            }
        }
    }
}

pub(crate) fn grid_cell_expr(batch: Expr) -> Expr {
    // Cast to (long long) so every `* G_ + _` step is 64-bit and
    // `cell * inner_size` pointer bases cannot overflow past 2^31.
    Expr::add(
        Expr::mul(
            Expr::add(
                Expr::mul(
                    Expr::add(
                        Expr::mul(
                            Expr::cast(String::from("long long"), batch),
                            Expr::var(String::from("GX")),
                        ),
                        Expr::var(String::from("ix")),
                    ),
                    Expr::var(String::from("GY")),
                ),
                Expr::var(String::from("iy")),
            ),
            Expr::var(String::from("GZ")),
        ),
        Expr::var(String::from("iz")),
    )
}

pub(crate) fn emit_grid_out_scratch(
    stmts: &mut Vec<Stmt>,
    grid_out_inner_sizes: &[i64],
    grid_out_dtypes: &[Dtype],
) {
    let gout_prefix = String::from("_gout");
    for (j, (sz, dt)) in grid_out_inner_sizes.iter().zip(grid_out_dtypes).enumerate() {
        let name = format!("{gout_prefix}{j}");
        let sz = *sz;
        let ct = dt.ctype();
        cuda! { stmts => #ct #name[#sz]; }
    }
}

pub(crate) fn emit_nongrid_out_scratch(
    stmts: &mut Vec<Stmt>,
    nongrid_out_sizes: &[i64],
    nongrid_out_dtypes: &[Dtype],
) {
    let cno_prefix = String::from("_cno");
    for (j, (sz, dt)) in nongrid_out_sizes.iter().zip(nongrid_out_dtypes).enumerate() {
        let name = format!("{cno_prefix}{j}");
        let sz = *sz;
        let ct = dt.ctype();
        cuda! { stmts => #ct #name[#sz]; }
    }
}

pub(crate) fn emit_global_atomic_scatter(
    stmts: &mut Vec<Stmt>,
    j: usize,
    isz: i64,
    batch_var: &str,
) {
    let cell = grid_cell_expr(Expr::var(batch_var.to_string()));
    let buf = format!("{}{j}", String::from("grid_out_"));
    let out = format!("{}{j}", String::from("_gout"));
    if isz == 1 {
        cuda! { stmts =>
            atomicAdd(&#buf[#cell], #out[0]);
        }
    } else {
        let cell_var = format!("{}{j}", String::from("_cell_"));
        cuda! { stmts =>
            i64 #cell_var = #cell;
            for (i32 _a = 0; _a < #isz; _a++) {
                atomicAdd(&#buf[#cell_var * #isz + _a], #out[_a]);
            }
        }
    }
}

pub(crate) fn va_off_name(j: usize) -> String {
    format!("{}{j}", String::from("VA_OFF_"))
}

pub(crate) fn emit_smem_atomic_scatter(
    stmts: &mut Vec<Stmt>,
    smem: &str,
    lc_var: &str,
    j: usize,
    isz: i64,
) {
    let gout = format!("{}{j}", String::from("_gout"));
    if isz == 1 {
        cuda! { stmts =>
            atomicAdd(&#smem[#lc_var], #gout[0]);
        }
    } else {
        cuda! { stmts =>
            for (i32 _a = 0; _a < #isz; _a++) {
                atomicAdd(&#smem[#lc_var * #isz + _a], #gout[_a]);
            }
        }
    }
}

pub(crate) fn emit_zero_fill_smem(stmts: &mut Vec<Stmt>, name: &str, total: i64, zero: &str) {
    let zero_e = Expr::var(zero);
    cuda! { stmts =>
        for (i32 _ci = threadIdx.x; _ci < #total; _ci += blockDim.x) {
            #name[_ci] = #zero_e;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dense_cube_flush(
    stmts: &mut Vec<Stmt>,
    n_grid_out: usize,
    grid_out_inner_sizes: &[i64],
    grid_out_dtypes: &[Dtype],
    smem_name: impl Fn(usize) -> String,
    side_z: &str,
    side_y: &str,
    volume: i64,
    global_wrap: Option<(&str, &str, &str)>,
    gidx: &str,
) {
    let ci_var = String::from("_ci");
    let grid_out_prefix = String::from("grid_out_");
    let wrap_decls: Vec<Stmt> = if let Some((gx, gy, gz)) = global_wrap {
        let mut v = Vec::new();
        cuda! { v =>
            i32 gx = #gx;
            i32 gy = #gy;
            i32 gz = #gz;
        }
        v
    } else {
        Vec::new()
    };

    cuda! { stmts => __syncthreads(); }
    for j in 0..n_grid_out {
        let isz = grid_out_inner_sizes[j];
        let dt = grid_out_dtypes[j];
        let zero = zero_lit_for(dt);
        let total = volume * isz;
        let smem = smem_name(j);
        let gout = format!("{grid_out_prefix}{j}");
        let cast = Expr::Call(
            dt.into(),
            vec![Expr::index(smem.clone(), Expr::var(&ci_var))],
        );

        if isz == 1 {
            cuda! { stmts =>
                for (i32 _ci = threadIdx.x; _ci < #total; _ci += blockDim.x) {
                    if (#smem[_ci] != #zero) {
                        i32 _lc = _ci;
                        i32 _lz = _lc % #side_z;
                        i32 _ly = _lc / #side_z % #side_y;
                        i32 _lx = _lc / (#side_z * #side_y);
                        splice!(wrap_decls.clone());
                        atomicAdd(&#gout[#gidx], #cast);
                    }
                }
            }
        } else {
            cuda! { stmts =>
                for (i32 _ci = threadIdx.x; _ci < #total; _ci += blockDim.x) {
                    if (#smem[_ci] != #zero) {
                        i32 _lc = _ci / #isz;
                        i32 _la = _ci % #isz;
                        i32 _lz = _lc % #side_z;
                        i32 _ly = _lc / #side_z % #side_y;
                        i32 _lx = _lc / (#side_z * #side_y);
                        splice!(wrap_decls.clone());
                        atomicAdd(&#gout[#gidx * #isz + _la], #cast);
                    }
                }
            }
        }
    }
}

pub(crate) fn emit_nongrid_accumulate(stmts: &mut Vec<Stmt>, nongrid_out_sizes: &[i64]) {
    let nout_prefix = String::from("_nout");
    let cno_prefix = String::from("_cno");
    for (j, osz) in nongrid_out_sizes.iter().enumerate() {
        let nout = format!("{nout_prefix}{j}");
        let cno = format!("{cno_prefix}{j}");
        if *osz == 1 {
            cuda! { stmts => #nout[0] += #cno[0]; }
        } else {
            cuda! { stmts =>
                for (i32 _a = 0; _a < #osz; _a++) {
                    #nout[_a] += #cno[_a];
                }
            }
        }
    }
}

pub(crate) fn emit_per_support_loop_shell<F>(
    stmts: &mut Vec<Stmt>,
    cartesian: Option<(i32, i32)>,
    k: i32,
    s_support: i32,
    inner: F,
) where
    F: FnOnce(&mut Vec<Stmt>),
{
    let use_nested = cartesian
        .map(|(order, _mo)| k > 1 && k == order)
        .unwrap_or(false);
    if use_nested {
        let (order, mo) = cartesian.unwrap();
        emit_per_support_loop_shell_cartesian(stmts, order, mo, inner);
    } else {
        emit_per_support_loop_shell_flat(stmts, k, s_support, inner);
    }
}

fn emit_per_support_loop_shell_flat<F>(stmts: &mut Vec<Stmt>, k: i32, s_support: i32, inner: F)
where
    F: FnOnce(&mut Vec<Stmt>),
{
    let mut body: Vec<Stmt> = Vec::new();
    emit_cell_coords_branchless(&mut body);
    cuda! { body =>
        i32 _sidx = s;
        i32 _sup[] = {SUPPORT[s * 3], SUPPORT[s * 3 + 1], SUPPORT[s * 3 + 2]};
    }
    inner(&mut body);

    let mut loop_stmts: Vec<Stmt> = Vec::new();
    if k > 1 {
        let spk = s_support / k;
        cuda! { loop_stmts =>
            i32 _s0 = _sub * #spk;
            for (i32 s = _s0; s < _s0 + #spk; s++) {
                splice!(body);
            }
        }
    } else {
        cuda! { loop_stmts =>
            for (i32 s = 0; s < S_SUPPORT; s++) {
                splice!(body);
            }
        }
    }
    cuda! { stmts =>
        if (_atom_active) {
            splice!(loop_stmts);
        }
    }
}

fn emit_per_support_loop_shell_cartesian<F>(stmts: &mut Vec<Stmt>, order: i32, mo: i32, inner: F)
where
    F: FnOnce(&mut Vec<Stmt>),
{
    cuda! { stmts =>
        i32 _sdz = _sub + #mo;
        i32 iz = bz + _sdz;
        iz -= (iz >= GZ) * GZ;
        iz += (iz < 0) * GZ;
    }

    let order2 = (order * order) as i64;
    let mut y_body: Vec<Stmt> = Vec::new();
    cuda! { y_body =>
        i32 _sdy = _iy + #mo;
        i32 iy = by + _sdy;
        iy -= (iy >= GY) * GY;
        iy += (iy < 0) * GY;
        i32 _sidx = _ix * #order2 + _iy * #order + _sub;
        i32 _sup[] = {_sdx, _sdy, _sdz};
    }
    inner(&mut y_body);

    let mut x_body: Vec<Stmt> = Vec::new();
    cuda! { x_body =>
        i32 _sdx = _ix + #mo;
        i32 ix = bx + _sdx;
        ix -= (ix >= GX) * GX;
        ix += (ix < 0) * GX;
        unroll for (i32 _iy = 0; _iy < #order; _iy++) {
            splice!(y_body);
        }
    }

    let mut outer: Vec<Stmt> = Vec::new();
    cuda! { outer =>
        unroll for (i32 _ix = 0; _ix < #order; _ix++) {
            splice!(x_body);
        }
    }
    cuda! { stmts =>
        if (_atom_active) {
            splice!(outer);
        }
    }
}

pub(crate) fn emit_warp_shuffle_reduction(
    stmts: &mut Vec<Stmt>,
    k: i32,
    nongrid_out_sizes: &[i64],
    nongrid_out_dtypes: &[Dtype],
) {
    cuda! { stmts =>
        i32 _base = _atom_in_warp * _K;
    }
    let nout_prefix = String::from("_nout");
    let red_prefix = String::from("_red");
    let k_bound = k as i64;
    for (j, (osz, dt)) in nongrid_out_sizes.iter().zip(nongrid_out_dtypes).enumerate() {
        let nout = format!("{nout_prefix}{j}");
        let ct = dt.ctype();
        for a in 0..*osz {
            let acc = format!("{red_prefix}{j}_{a}");
            cuda! { stmts =>
                #ct #acc = #nout[#a];
                for (i32 _src = 1; _src < #k_bound; _src++) {
                    #acc += __shfl_sync(0xFFFFFFFFu, #nout[#a], _base + _src);
                }
                #nout[#a] = #acc;
            }
        }
    }
}

pub(crate) fn emit_nongrid_writeback(
    stmts: &mut Vec<Stmt>,
    layout: &IndexLayout,
    buf_batch_extents: &[Vec<i64>],
    nongrid_out_sizes: &[i64],
    nongrid_scatter_flags: &[bool],
    idx_offset_vars: &HashMap<usize, String>,
    k: i32,
) {
    let off_nout_prefix = String::from("off_nout_");
    let nout_prefix = String::from("_nout");
    let result_prefix = String::from("result_");
    let out_prefix = String::from("out_");
    let mut writeback: Vec<Stmt> = Vec::new();
    for j in 0..layout.n_nongrid_out {
        let var = format!("{off_nout_prefix}{j}");
        let ic_entry = &layout.nongrid_out()[j];
        let extents = &buf_batch_extents[layout.nongrid_out_offset() + j];
        emit_buffer_offset(&mut writeback, &var, ic_entry, extents, idx_offset_vars);
        let osz = nongrid_out_sizes[j];
        let is_scatter = nongrid_scatter_flags[j];
        let prefix = if is_scatter {
            &result_prefix
        } else {
            &out_prefix
        };
        let ptr = format!("{prefix}{j}");
        let nout = format!("{nout_prefix}{j}");
        match (osz == 1, is_scatter) {
            (true, true) => cuda! { writeback =>
                atomicAdd(&#ptr[#var], #nout[0]);
            },
            (true, false) => cuda! { writeback =>
                #ptr[#var] = #nout[0];
            },
            (false, true) => cuda! { writeback =>
                for (i32 _a = 0; _a < #osz; _a++) {
                    atomicAdd(&#ptr[#var * #osz + _a], #nout[_a]);
                }
            },
            (false, false) => cuda! { writeback =>
                for (i32 _a = 0; _a < #osz; _a++) {
                    #ptr[#var * #osz + _a] = #nout[_a];
                }
            },
        }
    }
    // Inactive threads share a clamped `linear_idx` with the last active
    // thread — writing their (zero) `_nout` would overwrite that thread's
    // result. Gate writeback on `_atom_active` for the same reason the
    // scatter loop is gated.
    if k > 1 {
        cuda! { stmts =>
            if (_atom_active && _sub == 0) {
                splice!(writeback);
            }
        }
    } else {
        cuda! { stmts =>
            if (_atom_active) {
                splice!(writeback);
            }
        }
    }
}

pub(crate) fn build_v_step_args(problem: &ScatterSpec, ngin_call_parts: &[Expr]) -> Vec<Expr> {
    let gval_prefix = String::from("_gval_");
    let state_prefix = String::from("_state");
    let gout_prefix = String::from("_gout");
    let cno_prefix = String::from("_cno");
    let mut args: Vec<Expr> = vec![
        Expr::addr(Expr::var(String::from("_sidx"))),
        Expr::var(String::from("_sup")),
    ];
    for (g, &gsz) in problem.grid_in_inner_sizes.iter().enumerate() {
        let name = format!("{gval_prefix}{g}");
        if gsz == 1 {
            args.push(Expr::addr(Expr::var(name)));
        } else {
            args.push(Expr::var(name));
        }
    }
    for k in 0..problem.n_state {
        args.push(Expr::var(format!("{state_prefix}{k}")));
    }
    for &i in &problem.direct_ngin_indices {
        args.push(ngin_call_parts[i].clone());
    }
    for j in 0..problem.layout.n_grid_out {
        args.push(Expr::var(format!("{gout_prefix}{j}")));
    }
    for j in 0..problem.layout.n_nongrid_out {
        args.push(Expr::var(format!("{cno_prefix}{j}")));
    }
    args
}

pub(crate) fn emit_standard_body<S: Accumulator + ?Sized>(
    accumulator: &S,
    problem: &ScatterSpec,
) -> Vec<Stmt> {
    let mut body: Vec<Stmt> = Vec::new();

    emit_thread_assignment(&mut body, problem.k, problem.uses_atom_map());
    body.push(Stmt::Blank);

    emit_batch_decode(&mut body, &problem.batch_sizes);
    body.push(Stmt::Blank);

    let idx_offset_vars =
        emit_index_buffer_offsets(&mut body, &problem.layout, &problem.buf_batch_extents);

    emit_cell_index_load(
        &mut body,
        &problem.layout,
        &problem.buf_batch_extents,
        &idx_offset_vars,
    );
    body.push(Stmt::Blank);

    let grid_off_vars = emit_grid_input_batch_offsets(
        &mut body,
        &problem.layout,
        &problem.buf_batch_extents,
        &idx_offset_vars,
    );

    let ngin_call_parts = emit_nongrid_input_ptrs(
        &mut body,
        &problem.layout,
        &problem.buf_batch_extents,
        &problem.nongrid_in_sizes,
        &idx_offset_vars,
    );

    let out_batch_vars = emit_grid_output_batch_offsets(
        &mut body,
        &problem.layout,
        &problem.buf_batch_extents,
        &idx_offset_vars,
    );

    body.push(Stmt::Blank);

    emit_v_pre_call(
        &mut body,
        &problem.state_sizes,
        &problem.state_dtypes,
        &problem.pre_ngin_indices,
        &ngin_call_parts,
    );
    body.push(Stmt::Blank);

    emit_nongrid_accumulator_init(
        &mut body,
        &problem.nongrid_out_sizes,
        &problem.nongrid_out_dtypes,
    );
    body.push(Stmt::Blank);

    let single_batch = problem.single_batch();
    let hook_ctx = HookCtx {
        n_grid_out: problem.layout.n_grid_out,
        grid_out_inner_sizes: &problem.grid_out_inner_sizes,
        grid_out_dtypes: &problem.grid_out_dtypes,
        out_batch_vars: &out_batch_vars,
        single_batch,
        problem,
    };

    if accumulator.uses_ps_init() {
        accumulator.ps_init(&mut body, &hook_ctx);
        body.push(Stmt::Blank);
    }

    let v_step_args = build_v_step_args(problem, &ngin_call_parts);
    let grid_in_inner_sizes = problem.grid_in_inner_sizes.clone();
    let grid_in_dtypes = problem.grid_in_dtypes.clone();
    let grid_out_inner_sizes = problem.grid_out_inner_sizes.clone();
    let grid_out_dtypes = problem.grid_out_dtypes.clone();
    let nongrid_out_sizes = problem.nongrid_out_sizes.clone();
    let nongrid_out_dtypes = problem.nongrid_out_dtypes.clone();
    let n_grid_out = problem.layout.n_grid_out;

    emit_per_support_loop_shell(
        &mut body,
        problem.cartesian,
        problem.k,
        problem.s_support,
        |inner| {
            emit_grid_input_loads(inner, &grid_in_inner_sizes, &grid_in_dtypes, &grid_off_vars);
            emit_grid_out_scratch(inner, &grid_out_inner_sizes, &grid_out_dtypes);
            emit_nongrid_out_scratch(inner, &nongrid_out_sizes, &nongrid_out_dtypes);
            inner.push(Stmt::Eval(Expr::call(String::from("V_step"), v_step_args)));
            for j in 0..n_grid_out {
                let scatter_ctx = ScatterCtx {
                    j,
                    isz: grid_out_inner_sizes[j],
                    batch_var: &out_batch_vars[j],
                    single_batch,
                };
                accumulator.ps_scatter(inner, &scatter_ctx);
            }
            emit_nongrid_accumulate(inner, &nongrid_out_sizes);
        },
    );

    accumulator.ps_flush(&mut body, &hook_ctx);
    body.push(Stmt::Blank);

    if problem.k > 1 {
        emit_warp_shuffle_reduction(
            &mut body,
            problem.k,
            &problem.nongrid_out_sizes,
            &problem.nongrid_out_dtypes,
        );
        body.push(Stmt::Blank);
    }

    emit_nongrid_writeback(
        &mut body,
        &problem.layout,
        &problem.buf_batch_extents,
        &problem.nongrid_out_sizes,
        &problem.nongrid_scatter_flags,
        &idx_offset_vars,
        problem.k,
    );

    body
}
