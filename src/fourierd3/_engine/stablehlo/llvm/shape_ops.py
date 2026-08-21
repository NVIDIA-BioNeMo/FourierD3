# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handlers for operations that rearrange, select, or contract elements.

Because every tensor is a flat list of scalars, these handlers are index
arithmetic: they compute which source slot feeds which destination slot and
emit no LLVM instruction of their own where a pure permutation suffices."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from jaxlib.mlir import ir as mlir_ir
from llvmlite import ir as ll_ir

from fourierd3._engine.stablehlo.attributes import (
    get_dot_dimension_numbers,
    get_i64,
    get_i64_array,
    parse_attr_list,
    parse_attr_scalar,
    result_shape_dtype,
    value_shape_dtype,
)
from fourierd3._engine.stablehlo.llvm.control_flow import _reducer_combine
from fourierd3._engine.stablehlo.llvm.module import _Emitter, _register, _walk_ops
from fourierd3._engine.stablehlo.llvm.values import (
    Vals,
    _dtype_to_ir,
    _is_complex,
    _is_float,
    _row_major_iter,
    _row_major_strides,
    _scalars_per_elem,
)


@_register("stablehlo.broadcast_in_dim")
def _h_broadcast(em: _Emitter, op, ins, env) -> Vals:
    in_shape, dt = value_shape_dtype(op.operands[0])
    out_shape, _ = result_shape_dtype(op)
    dims = get_i64_array(op, "broadcast_dimensions")
    spe = _scalars_per_elem(dt)
    in_strides = _row_major_strides(in_shape) if in_shape else []
    out: Vals = []
    n_out = int(math.prod(out_shape)) if out_shape else 1
    for flat_out in _row_major_iter(out_shape):
        in_flat = 0
        for k, d in enumerate(in_shape):
            out_axis = dims[k]
            idx = 0 if d == 1 else flat_out[out_axis]
            in_flat += idx * in_strides[k]
        for s in range(spe):
            out.append(ins[0][in_flat * spe + s])
        _ = n_out
    return out


@_register("stablehlo.reshape")
def _h_reshape(em: _Emitter, op, ins, env) -> Vals:
    # Row-major reshape doesn't move scalars in the flat IR list.
    return list(ins[0])


@_register("stablehlo.transpose")
def _h_transpose(em: _Emitter, op, ins, env) -> Vals:
    in_shape, dt = value_shape_dtype(op.operands[0])
    perm = get_i64_array(op, "permutation")
    out_shape = tuple(in_shape[p] for p in perm)
    in_strides = _row_major_strides(in_shape)
    spe = _scalars_per_elem(dt)
    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        # out[out_idx] = in[in_idx] where in_idx[perm[k]] = out_idx[k].
        in_idx = [0] * len(in_shape)
        for k, p in enumerate(perm):
            in_idx[p] = out_idx[k]
        in_flat = sum(i * s for i, s in zip(in_idx, in_strides, strict=True))
        for s in range(spe):
            out.append(ins[0][in_flat * spe + s])
    return out


@_register("stablehlo.slice")
def _h_slice(em: _Emitter, op, ins, env) -> Vals:
    in_shape, dt = value_shape_dtype(op.operands[0])
    start = get_i64_array(op, "start_indices")
    limit = get_i64_array(op, "limit_indices")
    strides = get_i64_array(op, "strides")
    in_strides = _row_major_strides(in_shape)
    spe = _scalars_per_elem(dt)
    out_shape = tuple(
        (lim - st + step - 1) // step for st, lim, step in zip(start, limit, strides, strict=True)
    )
    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        in_idx = [st + idx * step for st, step, idx in zip(start, strides, out_idx, strict=True)]
        in_flat = sum(i * s for i, s in zip(in_idx, in_strides, strict=True))
        for s in range(spe):
            out.append(ins[0][in_flat * spe + s])
    return out


@_register("stablehlo.concatenate")
def _h_concatenate(em: _Emitter, op, ins, env) -> Vals:
    dim = get_i64(op, "dimension")
    in_shapes = [value_shape_dtype(o)[0] for o in op.operands]
    out_shape, dt = result_shape_dtype(op)
    spe = _scalars_per_elem(dt)
    in_strides_list = [_row_major_strides(s) for s in in_shapes]
    offsets = [0]
    for s in in_shapes[:-1]:
        offsets.append(offsets[-1] + s[dim])
    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        which = 0
        coord_on_dim = out_idx[dim]
        for i, off in enumerate(offsets):
            end = off + in_shapes[i][dim]
            if coord_on_dim < end:
                which = i
                in_axis_idx = coord_on_dim - off
                break
        in_idx = list(out_idx)
        in_idx[dim] = in_axis_idx
        in_flat = sum(i * s for i, s in zip(in_idx, in_strides_list[which], strict=True))
        for s in range(spe):
            out.append(ins[which][in_flat * spe + s])
    return out


@_register("stablehlo.reverse")
def _h_reverse(em: _Emitter, op, ins, env) -> Vals:
    in_shape, dt = value_shape_dtype(op.operands[0])
    out_shape, _ = result_shape_dtype(op)
    dims = set(get_i64_array(op, "dimensions"))
    in_strides = _row_major_strides(in_shape)
    spe = _scalars_per_elem(dt)
    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        in_idx = tuple((in_shape[k] - 1 - i) if k in dims else i for k, i in enumerate(out_idx))
        in_flat = sum(i * s for i, s in zip(in_idx, in_strides, strict=True))
        for s in range(spe):
            out.append(ins[0][in_flat * spe + s])
    return out


@_register("stablehlo.pad")
def _h_pad(em: _Emitter, op, ins, env) -> Vals:
    in_shape, dt = value_shape_dtype(op.operands[0])
    out_shape, _ = result_shape_dtype(op)
    lo = get_i64_array(op, "edge_padding_low")
    # `edge_padding_high` is implicit in `out_shape`; we don't need it
    # to derive the source index — high pad maps the same way as low.
    inter = get_i64_array(op, "interior_padding")
    in_strides = _row_major_strides(in_shape)
    spe = _scalars_per_elem(dt)
    pad_val_list = ins[1]
    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        in_idx: list[int] = []
        is_pad = False
        for k, oi in enumerate(out_idx):
            adj = oi - lo[k]
            stride = inter[k] + 1
            if adj < 0 or adj >= in_shape[k] * stride or adj % stride != 0:
                is_pad = True
                break
            in_idx.append(adj // stride)
        if is_pad:
            for s in range(spe):
                out.append(pad_val_list[s])
        else:
            in_flat = sum(i * s for i, s in zip(in_idx, in_strides, strict=True))
            for s in range(spe):
                out.append(ins[0][in_flat * spe + s])
    return out


@_register("stablehlo.sort")
def _h_sort(em: _Emitter, op, ins, env) -> Vals:
    dimension = get_i64(op, "dimension")
    n_keys = len(ins)
    shape, _ = value_shape_dtype(op.operands[0])
    if any(value_shape_dtype(op.operands[k])[0] != shape for k in range(n_keys)):
        raise NotImplementedError("sort: all keys must share one shape")

    sort_size = shape[dimension]
    if sort_size <= 1:
        return list(ins) if n_keys > 1 else ins[0]

    total = int(math.prod(shape))
    sort_stride = _row_major_strides(shape)[dimension]
    num_lanes = total // sort_size

    comparator = op.regions[0].blocks[0]

    bufs: list[list[ll_ir.Value]] = [list(v) for v in ins]

    def should_swap(p_j: int, p_jm1: int) -> ll_ir.Value:
        # comparator block_args are interleaved: (k0_arg1, k0_arg2, k1_arg1, k1_arg2, ...)
        sub_env: dict[Any, Vals] = {}
        for k in range(n_keys):
            sub_env[comparator.arguments[2 * k]] = [bufs[k][p_j]]
            sub_env[comparator.arguments[2 * k + 1]] = [bufs[k][p_jm1]]
        _walk_ops(em, comparator.operations, sub_env)
        ret_op = list(comparator.operations)[-1]
        assert str(ret_op.name) == "stablehlo.return"
        return sub_env[ret_op.operands[0]][0]

    for lane in range(num_lanes):
        if sort_stride == 1:
            base = lane * sort_size
        else:
            lane_axes = [shape[k] for k in range(len(shape)) if k != dimension]
            lane_strides = [
                _row_major_strides(shape)[k] for k in range(len(shape)) if k != dimension
            ]
            lane_idx = []
            rem = lane
            for d in lane_axes:
                lane_idx.append(rem % d)
                rem //= d
            lane_idx = list(reversed(lane_idx))
            base = sum(i * s for i, s in zip(lane_idx, lane_strides, strict=True))

        for i in range(1, sort_size):
            for j in range(i, 0, -1):
                p_j = base + j * sort_stride
                p_jm1 = base + (j - 1) * sort_stride
                swap = should_swap(p_j, p_jm1)
                for k in range(n_keys):
                    a = bufs[k][p_jm1]
                    b = bufs[k][p_j]
                    bufs[k][p_jm1] = em.b.select(swap, b, a)
                    bufs[k][p_j] = em.b.select(swap, a, b)

    return bufs if n_keys > 1 else bufs[0]


@_register("stablehlo.dynamic_update_slice")
def _h_dynamic_update_slice(em: _Emitter, op, ins, env) -> Vals:
    op_shape, dt = value_shape_dtype(op.operands[0])
    upd_shape, _ = value_shape_dtype(op.operands[1])
    n_axes = len(op_shape)
    starts = [ins[2 + k][0] for k in range(n_axes)]
    spe = _scalars_per_elem(dt)
    op_strides = _row_major_strides(op_shape)
    upd_strides = _row_major_strides(upd_shape)

    idx_t = ll_ir.IntType(32)
    clamped_starts: list[ll_ir.Value] = []
    for k in range(n_axes):
        s = starts[k]
        if s.type != idx_t:
            if hasattr(s.type, "width") and s.type.width > 32:
                s = em.b.trunc(s, idx_t)
            else:
                s = em.b.sext(s, idx_t)
        zero = ll_ir.Constant(idx_t, 0)
        lim = ll_ir.Constant(idx_t, op_shape[k] - upd_shape[k])
        s = em.b.select(em.b.icmp_signed("<", s, zero), zero, s)
        s = em.b.select(em.b.icmp_signed(">", s, lim), lim, s)
        clamped_starts.append(s)

    out: Vals = []
    for op_idx in _row_major_iter(op_shape):
        in_update = ll_ir.Constant(ll_ir.IntType(1), 1)
        upd_local: list[ll_ir.Value] = []
        for k in range(n_axes):
            s = clamped_starts[k]
            coord = ll_ir.Constant(idx_t, op_idx[k])
            ge_lo = em.b.icmp_signed(">=", coord, s)
            hi = em.b.add(s, ll_ir.Constant(idx_t, upd_shape[k]))
            lt_hi = em.b.icmp_signed("<", coord, hi)
            in_update = em.b.and_(em.b.and_(in_update, ge_lo), lt_hi)
            upd_local.append(em.b.sub(coord, s))

        upd_flat = ll_ir.Constant(idx_t, 0)
        for k in range(n_axes):
            upd_flat = em.b.add(
                upd_flat,
                em.b.mul(upd_local[k], ll_ir.Constant(idx_t, upd_strides[k])),
            )

        op_flat = sum(op_idx[k] * op_strides[k] for k in range(n_axes))
        for sl in range(spe):
            update_pick = ins[1][sl]
            upd_total = int(math.prod(upd_shape)) if upd_shape else 1
            for cand in range(1, upd_total):
                pred = em.b.icmp_signed("==", upd_flat, ll_ir.Constant(idx_t, cand))
                update_pick = em.b.select(pred, ins[1][cand * spe + sl], update_pick)
            operand_v = ins[0][op_flat * spe + sl]
            out.append(em.b.select(in_update, update_pick, operand_v))
    return out


@_register("stablehlo.dot_general")
def _h_dot_general(em: _Emitter, op, ins, env) -> Vals:

    out_shape, dt = result_shape_dtype(op)
    lhs_shape, _ = value_shape_dtype(op.operands[0])
    rhs_shape, _ = value_shape_dtype(op.operands[1])
    (clhs, crhs), (blhs, brhs) = get_dot_dimension_numbers(op)
    lhs_free = [d for d in range(len(lhs_shape)) if d not in clhs and d not in blhs]
    rhs_free = [d for d in range(len(rhs_shape)) if d not in crhs and d not in brhs]
    nb, nl = len(blhs), len(lhs_free)
    lhs_strides = _row_major_strides(lhs_shape)
    rhs_strides = _row_major_strides(rhs_shape)
    spe = _scalars_per_elem(dt)
    c_ranges = [lhs_shape[d] for d in clhs]
    n_contraction = int(math.prod(c_ranges)) if c_ranges else 1

    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        batch_idx = out_idx[:nb]
        lhs_free_idx = out_idx[nb : nb + nl]
        rhs_free_idx = out_idx[nb + nl :]

        acc: list[ll_ir.Value] = [ll_ir.Constant(_dtype_to_ir(dt), 0)] * spe
        for c_flat in range(n_contraction):
            cvals = []
            rem = c_flat
            for d in c_ranges:
                cvals.append(rem % d)
                rem //= d
            li = [0] * len(lhs_shape)
            ri = [0] * len(rhs_shape)
            for k, d in enumerate(blhs):
                li[d] = batch_idx[k]
            for k, d in enumerate(lhs_free):
                li[d] = lhs_free_idx[k]
            for k, d in enumerate(clhs):
                li[d] = cvals[k]
            for k, d in enumerate(brhs):
                ri[d] = batch_idx[k]
            for k, d in enumerate(rhs_free):
                ri[d] = rhs_free_idx[k]
            for k, d in enumerate(crhs):
                ri[d] = cvals[k]
            lhs_flat = sum(i * s for i, s in zip(li, lhs_strides, strict=True))
            rhs_flat = sum(i * s for i, s in zip(ri, rhs_strides, strict=True))
            if _is_complex(dt):
                ar = ins[0][lhs_flat * 2]
                ai = ins[0][lhs_flat * 2 + 1]
                br = ins[1][rhs_flat * 2]
                bi = ins[1][rhs_flat * 2 + 1]
                pre = em.b.fsub(em.b.fmul(ar, br), em.b.fmul(ai, bi))
                pim = em.b.fadd(em.b.fmul(ar, bi), em.b.fmul(ai, br))
                acc[0] = em.b.fadd(acc[0], pre)
                acc[1] = em.b.fadd(acc[1], pim)
            else:
                a = ins[0][lhs_flat]
                b = ins[1][rhs_flat]
                if _is_float(dt):
                    acc[0] = em.b.fadd(acc[0], em.b.fmul(a, b))
                else:
                    acc[0] = em.b.add(acc[0], em.b.mul(a, b))
        out.extend(acc)
    return out


@_register("stablehlo.reduce")
def _h_reduce(em: _Emitter, op, ins, env):
    n_inputs = len(op.operands) // 2
    input_dts = [value_shape_dtype(op.operands[k])[1] for k in range(n_inputs)]
    in_shape = value_shape_dtype(op.operands[0])[0]
    out_shape, _ = result_shape_dtype(op)
    axes = set(get_i64_array(op, "dimensions"))
    inits = [ins[n_inputs + k] for k in range(n_inputs)]
    combine = _reducer_combine(em, op.regions[0], n_inputs=n_inputs)

    in_strides = _row_major_strides(in_shape)
    spes = [_scalars_per_elem(dt) for dt in input_dts]

    kept_axes = [k for k in range(len(in_shape)) if k not in axes]
    if tuple(in_shape[k] for k in kept_axes) != tuple(out_shape):
        raise AssertionError(
            f"reduce out_shape {out_shape} doesn't match kept axes "
            f"{tuple(in_shape[k] for k in kept_axes)}"
        )

    out_per_input: list[Vals] = [[] for _ in range(n_inputs)]
    for out_idx in _row_major_iter(out_shape):
        accs = [list(inits[k]) for k in range(n_inputs)]
        red_shapes = [in_shape[k] for k in range(len(in_shape)) if k in axes]
        for red_idx in _row_major_iter(tuple(red_shapes)):
            full_idx = [0] * len(in_shape)
            ki, ri = 0, 0
            for k in range(len(in_shape)):
                if k in axes:
                    full_idx[k] = red_idx[ri]
                    ri += 1
                else:
                    full_idx[k] = out_idx[ki]
                    ki += 1
            in_flat = sum(i * s for i, s in zip(full_idx, in_strides, strict=True))
            curs = [
                [ins[k][in_flat * spes[k] + s] for s in range(spes[k])] for k in range(n_inputs)
            ]
            accs = combine(accs, curs)
        for k in range(n_inputs):
            out_per_input[k].extend(accs[k])
    return out_per_input if n_inputs > 1 else out_per_input[0]


@_register("stablehlo.reduce_window")
def _h_reduce_window(em: _Emitter, op, ins, env) -> Vals:
    in_shape, dt = value_shape_dtype(op.operands[0])
    out_shape, _ = result_shape_dtype(op)
    init = ins[1]
    combine = _reducer_combine(em, op.regions[0], n_inputs=1)
    spe = _scalars_per_elem(dt)
    n_axes = len(in_shape)

    window_dims = get_i64_array(op, "window_dimensions")
    window_strides = get_i64_array(op, "window_strides")
    window_dilations = get_i64_array(op, "window_dilations")
    base_dilations = get_i64_array(op, "base_dilations")
    if any(d != 1 for d in base_dilations):
        raise NotImplementedError("reduce_window base_dilations != 1")
    padding_attr = mlir_ir.DenseIntElementsAttr(op.attributes["padding"])
    padding = np.array(padding_attr).reshape(n_axes, 2)

    in_strides = _row_major_strides(in_shape)

    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        starts = [out_idx[k] * window_strides[k] for k in range(n_axes)]
        acc = list(init)
        for w in _row_major_iter(tuple(window_dims)):
            in_pos = [starts[k] + w[k] * window_dilations[k] - padding[k, 0] for k in range(n_axes)]
            in_bounds = all(0 <= in_pos[k] < in_shape[k] for k in range(n_axes))
            if not in_bounds:
                continue  # padding contributes the reducer's identity — acc unchanged.
            in_flat = sum(in_pos[k] * in_strides[k] for k in range(n_axes))
            cur = [ins[0][in_flat * spe + s] for s in range(spe)]
            acc = combine([acc], [cur])[0]
        out.extend(acc)
    return out


@_register("stablehlo.dynamic_slice")
def _h_dynamic_slice(em: _Emitter, op, ins, env) -> Vals:
    in_shape, dt = value_shape_dtype(op.operands[0])
    out_shape, _ = result_shape_dtype(op)
    n_axes = len(in_shape)
    # start_indices are operands[1..n_axes+1], each scalar tensor.
    starts = [ins[1 + k] for k in range(n_axes)]
    spe = _scalars_per_elem(dt)
    in_strides = _row_major_strides(in_shape)

    in_total = int(math.prod(in_shape))
    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        idx_t = ll_ir.IntType(32)
        flat = ll_ir.Constant(idx_t, 0)
        for k in range(n_axes):
            s = starts[k][0]
            if s.type != idx_t:
                if s.type.width > 32:
                    s = em.b.trunc(s, idx_t)
                else:
                    s = em.b.sext(s, idx_t)
            max_start = in_shape[k] - out_shape[k]
            zero = ll_ir.Constant(idx_t, 0)
            lim = ll_ir.Constant(idx_t, max_start)
            s = em.b.select(em.b.icmp_signed("<", s, zero), zero, s)
            s = em.b.select(em.b.icmp_signed(">", s, lim), lim, s)
            term = em.b.add(s, ll_ir.Constant(idx_t, out_idx[k]))
            flat = em.b.add(flat, em.b.mul(term, ll_ir.Constant(idx_t, in_strides[k])))
        for sl in range(spe):
            result = ins[0][sl]
            for cand in range(1, in_total):
                pred = em.b.icmp_signed("==", flat, ll_ir.Constant(idx_t, cand))
                result = em.b.select(pred, ins[0][cand * spe + sl], result)
            out.append(result)
    return out


@_register("stablehlo.scatter")
def _h_scatter(em: _Emitter, op, ins, env) -> Vals:
    operand_shape, dt = value_shape_dtype(op.operands[0])
    si_shape, _ = value_shape_dtype(op.operands[1])
    upd_shape, _ = value_shape_dtype(op.operands[2])

    dn = str(op.attributes["scatter_dimension_numbers"])
    update_window_dims = parse_attr_list(dn, "update_window_dims")
    inserted_window_dims = parse_attr_list(dn, "inserted_window_dims")
    scatter_to_operand = parse_attr_list(dn, "scatter_dims_to_operand_dims")
    # StableHLO printer elides `index_vector_dim` when it equals
    # `rank(scatter_indices) - 1` (the last axis of the indices buffer).
    index_vector_dim = parse_attr_scalar(dn, "index_vector_dim", default=max(len(si_shape) - 1, 0))

    combine = _reducer_combine(em, op.regions[0], n_inputs=1)
    spe = _scalars_per_elem(dt)
    n_axes = len(operand_shape)
    operand_strides = _row_major_strides(operand_shape) if operand_shape else []
    operand_total = int(math.prod(operand_shape)) if operand_shape else 1

    si_rank = len(si_shape)
    si_strides = _row_major_strides(si_shape) if si_shape else []
    upd_strides = _row_major_strides(upd_shape) if upd_shape else []
    n_upd_axes = len(upd_shape)

    upd_batch_dims = tuple(d for d in range(n_upd_axes) if d not in update_window_dims)
    upd_batch_shape = tuple(upd_shape[d] for d in upd_batch_dims)
    upd_window_shape = tuple(upd_shape[d] for d in update_window_dims)
    operand_window_dims = tuple(a for a in range(n_axes) if a not in inserted_window_dims)

    idx_t = ll_ir.IntType(32)

    def to_i32(v: ll_ir.Value) -> ll_ir.Value:
        if v.type == idx_t:
            return v
        if v.type.width > 32:
            return em.b.trunc(v, idx_t)
        return em.b.sext(v, idx_t)

    # Start with output == operand; mutate in place via select-chain.
    cur_out: list[ll_ir.Value] = list(ins[0])

    K = len(scatter_to_operand)
    for batch_idx in _row_major_iter(upd_batch_shape):
        scatter_starts: list[ll_ir.Value] = []
        for k in range(K):
            if index_vector_dim == si_rank:
                si_pos = batch_idx
            else:
                tmp = list(batch_idx)
                tmp.insert(index_vector_dim, k)
                si_pos = tuple(tmp)
            flat_si = sum(si_pos[i] * si_strides[i] for i in range(si_rank))
            scatter_starts.append(to_i32(ins[1][flat_si]))

        for window_pos in _row_major_iter(upd_window_shape):
            operand_idx: list[ll_ir.Value] = []
            for axis in range(n_axes):
                if axis in scatter_to_operand:
                    j = scatter_to_operand.index(axis)
                    base = scatter_starts[j]
                else:
                    base = ll_ir.Constant(idx_t, 0)
                if axis in inserted_window_dims:
                    operand_idx.append(base)
                else:
                    nc_k = operand_window_dims.index(axis)
                    off = window_pos[nc_k]
                    operand_idx.append(em.b.add(base, ll_ir.Constant(idx_t, off)))

            flat = ll_ir.Constant(idx_t, 0)
            for k in range(n_axes):
                flat = em.b.add(
                    flat,
                    em.b.mul(operand_idx[k], ll_ir.Constant(idx_t, operand_strides[k])),
                )

            upd_full_idx = [0] * n_upd_axes
            for ui, d in enumerate(upd_batch_dims):
                upd_full_idx[d] = batch_idx[ui]
            for ui, d in enumerate(update_window_dims):
                upd_full_idx[d] = window_pos[ui]
            upd_flat = sum(upd_full_idx[k] * upd_strides[k] for k in range(n_upd_axes))
            update_vals = [ins[2][upd_flat * spe + s] for s in range(spe)]

            for cand in range(operand_total):
                pred = em.b.icmp_signed("==", flat, ll_ir.Constant(idx_t, cand))
                current = [cur_out[cand * spe + s] for s in range(spe)]
                new = combine([current], [update_vals])[0]
                for s in range(spe):
                    cur_out[cand * spe + s] = em.b.select(pred, new[s], cur_out[cand * spe + s])
    return cur_out


@_register("stablehlo.gather")
def _h_gather(em: _Emitter, op, ins, env) -> Vals:
    operand_shape, operand_dt = value_shape_dtype(op.operands[0])
    si_shape, _ = value_shape_dtype(op.operands[1])
    out_shape, _ = result_shape_dtype(op)

    dn = str(op.attributes["dimension_numbers"])
    offset_dims = parse_attr_list(dn, "offset_dims")
    collapsed = parse_attr_list(dn, "collapsed_slice_dims")
    start_index_map = parse_attr_list(dn, "start_index_map")
    # StableHLO printer elides `index_vector_dim` when it equals the
    # last axis of `start_indices`.
    index_vector_dim = parse_attr_scalar(dn, "index_vector_dim", default=max(len(si_shape) - 1, 0))
    slice_sizes = get_i64_array(op, "slice_sizes")

    spe = _scalars_per_elem(operand_dt)
    n_axes = len(operand_shape)
    operand_strides = _row_major_strides(operand_shape) if operand_shape else []
    in_total = int(math.prod(operand_shape)) if operand_shape else 1
    idx_t = ll_ir.IntType(32)

    si_strides = _row_major_strides(si_shape) if si_shape else []
    si_rank = len(si_shape)

    batch_dims_out = tuple(d for d in range(len(out_shape)) if d not in offset_dims)
    noncollapsed = tuple(a for a in range(n_axes) if a not in collapsed)

    def to_i32(v: ll_ir.Value) -> ll_ir.Value:
        if v.type == idx_t:
            return v
        if v.type.width > 32:
            return em.b.trunc(v, idx_t)
        return em.b.sext(v, idx_t)

    out: Vals = []
    for out_idx in _row_major_iter(out_shape):
        K = len(start_index_map)
        starts: list[ll_ir.Value] = []
        for k in range(K):
            if index_vector_dim == si_rank:
                si_pos = tuple(out_idx[d] for d in batch_dims_out)
            else:
                si_pos = [out_idx[d] for d in batch_dims_out]
                si_pos.insert(index_vector_dim, k)
                si_pos = tuple(si_pos)
            flat_si = sum(si_pos[i] * si_strides[i] for i in range(si_rank))
            starts.append(to_i32(ins[1][flat_si]))

        operand_idx_runtime: list[ll_ir.Value] = []
        for axis in range(n_axes):
            if axis in start_index_map:
                j = start_index_map.index(axis)
                max_start = operand_shape[axis] - slice_sizes[axis]
                s = starts[j]
                zero = ll_ir.Constant(idx_t, 0)
                lim = ll_ir.Constant(idx_t, max_start)
                s = em.b.select(em.b.icmp_signed("<", s, zero), zero, s)
                s = em.b.select(em.b.icmp_signed(">", s, lim), lim, s)
                base = s
            else:
                base = ll_ir.Constant(idx_t, 0)
            if axis in collapsed:
                operand_idx_runtime.append(base)
            else:
                nc_k = noncollapsed.index(axis)
                off = out_idx[offset_dims[nc_k]]
                operand_idx_runtime.append(em.b.add(base, ll_ir.Constant(idx_t, off)))

        flat = ll_ir.Constant(idx_t, 0)
        for k in range(n_axes):
            flat = em.b.add(
                flat,
                em.b.mul(operand_idx_runtime[k], ll_ir.Constant(idx_t, operand_strides[k])),
            )

        for sl in range(spe):
            result = ins[0][sl]
            for cand in range(1, in_total):
                pred = em.b.icmp_signed("==", flat, ll_ir.Constant(idx_t, cand))
                result = em.b.select(pred, ins[0][cand * spe + sl], result)
            out.append(result)
    return out
