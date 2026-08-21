# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handlers for operations whose result is fixed at compile time."""

from __future__ import annotations

import math

from llvmlite import ir as ll_ir

from fourierd3._engine.stablehlo.attributes import (
    get_dense_value,
    get_i64,
    result_shape_dtype,
)
from fourierd3._engine.stablehlo.llvm.module import _Emitter, _register
from fourierd3._engine.stablehlo.llvm.values import (
    Vals,
    _constant_vals,
    _dtype_to_ir,
    _is_float,
    _row_major_strides,
)


@_register("stablehlo.constant")
def _h_constant(em: _Emitter, op, ins, env) -> Vals:
    shape, dt = result_shape_dtype(op)
    arr = get_dense_value(op)
    return _constant_vals(arr, dt)


@_register("stablehlo.iota")
def _h_iota(em: _Emitter, op, ins, env) -> Vals:
    shape, dt = result_shape_dtype(op)
    axis = get_i64(op, "iota_dimension")
    strides = _row_major_strides(shape)
    out: Vals = []
    ir_t = _dtype_to_ir(dt)
    n = int(math.prod(shape))
    for flat in range(n):
        coord = (flat // strides[axis]) % shape[axis]
        if _is_float(dt):
            out.append(ll_ir.Constant(ir_t, float(coord)))
        else:
            out.append(ll_ir.Constant(ir_t, int(coord)))
    return out
