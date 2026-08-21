# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handlers for operations whose operands include a region.

The emitted device function is straight-line code, so each region is
inlined: calls are walked into, `case` selects a branch at compile time
when the predicate is constant, and `while` is unrolled to its trip count."""

from __future__ import annotations

from typing import Any

from llvmlite import ir as ll_ir

from fourierd3._engine.stablehlo.attributes import (
    get_comparison_direction,
    get_dense_value,
)
from fourierd3._engine.stablehlo.llvm.module import _Emitter, _register, _walk_ops
from fourierd3._engine.stablehlo.llvm.values import Vals


@_register("func.call")
def _h_func_call(em: _Emitter, op, ins, env) -> Vals:
    from fourierd3._engine.stablehlo.attributes import get_callee_name

    callee_name = get_callee_name(op)
    func_op = em.func_table.get(callee_name)
    if func_op is None:
        raise NotImplementedError(f"func.call to unknown function {callee_name!r}")

    func_block = next(iter(func_op.regions[0].blocks))
    sub_env: dict[Any, Vals] = {}
    for k, block_arg in enumerate(func_block.arguments):
        sub_env[block_arg] = ins[k]

    _walk_ops(em, func_block.operations, sub_env)

    ret_op = list(func_block.operations)[-1]
    assert str(ret_op.name) in ("func.return", "stablehlo.return")
    if len(ret_op.operands) == 1:
        return sub_env[ret_op.operands[0]]
    return [sub_env[r] for r in ret_op.operands]


def _reducer_combine(em: _Emitter, region, n_inputs: int = 1):
    body = region.blocks[0]

    def combine(accs, curs):
        assert len(accs) == n_inputs and len(curs) == n_inputs
        sub_env: dict[Any, Vals] = {}
        for k in range(n_inputs):
            sub_env[body.arguments[k]] = accs[k]
            sub_env[body.arguments[n_inputs + k]] = curs[k]
        _walk_ops(em, body.operations, sub_env)
        ret_op = list(body.operations)[-1]
        assert str(ret_op.name) == "stablehlo.return"
        return [list(sub_env[r]) for r in ret_op.operands]

    return combine


@_register("stablehlo.case")
def _h_case(em: _Emitter, op, ins, env) -> Vals:
    selector = ins[0][0]
    n_results = len(op.results)
    n_branches = len(op.regions)

    branch_outs: list[list[Vals]] = []
    for region in op.regions:
        block = region.blocks[0]
        _walk_ops(em, block.operations, env)
        ret_op = list(block.operations)[-1]
        assert str(ret_op.name) == "stablehlo.return"
        if n_results == 1:
            branch_outs.append([env[ret_op.operands[0]]])
        else:
            branch_outs.append([env[r] for r in ret_op.operands])

    # selector is clamped to [0, n_branches - 1] by stablehlo semantics.
    idx_t = ll_ir.IntType(32)
    if selector.type != idx_t:
        if selector.type.width > 32:
            selector = em.b.trunc(selector, idx_t)
        else:
            selector = em.b.sext(selector, idx_t)

    out_results: list[Vals] = []
    for r in range(n_results):
        nelem = len(branch_outs[0][r])
        merged: Vals = []
        for i in range(nelem):
            cur = branch_outs[n_branches - 1][r][i]
            for k in range(n_branches - 1):
                pred = em.b.icmp_signed("==", selector, ll_ir.Constant(idx_t, k))
                cur = em.b.select(pred, branch_outs[k][r][i], cur)
            merged.append(cur)
        out_results.append(merged)
    return out_results if n_results > 1 else out_results[0]


@_register("stablehlo.while")
def _h_while(em: _Emitter, op, ins, env) -> Vals:
    cond_block = op.regions[0].blocks[0]
    body_block = op.regions[1].blocks[0]
    n_carry = len(op.operands)

    trip = _extract_while_trip_count(cond_block, ins)

    carry: list[Vals] = list(ins)
    for _ in range(trip):
        sub_env: dict[Any, Vals] = {}
        for k, ba in enumerate(body_block.arguments):
            sub_env[ba] = carry[k]
        _walk_ops(em, body_block.operations, sub_env)
        ret_op = list(body_block.operations)[-1]
        assert str(ret_op.name) == "stablehlo.return"
        carry = [sub_env[r] for r in ret_op.operands]

    return carry if n_carry > 1 else carry[0]


def _extract_while_trip_count(cond_block, init_carry: list) -> int:
    constants: dict[Any, int] = {}
    compares: dict[Any, tuple[int, int, str]] = {}
    arg_list = list(cond_block.arguments)
    ret_op = list(cond_block.operations)[-1]
    assert str(ret_op.name) == "stablehlo.return"

    for o in cond_block.operations:
        n = str(o.name)
        if n == "stablehlo.return":
            break
        if n == "stablehlo.constant":
            arr = get_dense_value(o)
            if arr.size == 1:
                constants[o.results[0]] = int(arr.reshape(()).item())
            continue
        if n == "stablehlo.compare":
            direction = get_comparison_direction(o)
            if direction not in ("LT", "LE"):
                continue  # unrelated comparison; ignored
            lhs, rhs = o.operands[0], o.operands[1]
            if lhs in arg_list and rhs in constants:
                k = arg_list.index(lhs)
                compares[o.results[0]] = (k, constants[rhs], direction)
            continue
        # Other cond-region ops are tolerated — we only care about what
        # the return value depends on.

    ret_v = ret_op.operands[0]
    if ret_v in constants:
        val = constants[ret_v]
        if val == 0:
            return 0
        raise NotImplementedError("while cond returns constant True → unbounded loop, can't unroll")
    if ret_v in compares:
        k, bound, kind = compares[ret_v]
        init_counter = _extract_python_int(init_carry[k])
        trip = bound - init_counter if kind == "LT" else bound - init_counter + 1
        return max(trip, 0)
    raise NotImplementedError("while cond doesn't reduce to a known trip count")


def _extract_python_int(vals: Vals) -> int:
    if len(vals) != 1:
        raise NotImplementedError("counter carry must be a scalar")
    v = vals[0]
    if not isinstance(v, ll_ir.Constant):
        raise NotImplementedError("initial counter must be a compile-time constant")
    return int(v.constant)
