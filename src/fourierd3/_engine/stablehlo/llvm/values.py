# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a StableHLO tensor value is represented as a flat list of LLVM scalars.

Every tensor in the emitted device function is fully unrolled: an array of
shape `s` and dtype `d` becomes `prod(s) * scalars_per_elem(d)` LLVM values,
in row-major order, with a complex element occupying two consecutive slots."""

from __future__ import annotations

import math
import struct

import numpy as np
from llvmlite import ir as ll_ir
from llvmlite.ir import types as _ll_types

from fourierd3._engine.stablehlo.attributes import result_shape_dtype, value_shape_dtype


class _BFloatType(_ll_types._BaseFloatType):
    null = "0.0"
    intrinsic_name = "bf16"

    def __str__(self) -> str:
        return "bfloat"

    def format_constant(self, value: float) -> str:
        # Round-to-zero f32 → bf16 by clearing the low 16 mantissa bits,
        # then format as a regular f64 literal — llvmlite's text format
        # accepts any decimal that the assembler can re-round.
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        truncated = struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]
        return _ll_types._format_double(truncated)


_BFloatType._create_instance()

Vals = list[ll_ir.Value]


def _is_float(dtype) -> bool:
    dt = np.dtype(dtype)
    # numpy classifies bf16 as kind 'V' (void), not floating; pull it
    # into the float family by name so reducer / constant paths route
    # through the float branch.
    return np.issubdtype(dt, np.floating) or dt.name == "bfloat16"


def _is_complex(dtype) -> bool:
    return np.issubdtype(np.dtype(dtype), np.complexfloating)


def _is_int(dtype) -> bool:
    return np.issubdtype(np.dtype(dtype), np.integer) or np.dtype(dtype) == np.bool_


def _dtype_to_ir(dtype) -> ll_ir.Type:
    dt = np.dtype(dtype)
    if dt == np.float32 or dt == np.complex64:
        return ll_ir.FloatType()
    if dt == np.float64 or dt == np.complex128:
        return ll_ir.DoubleType()
    if dt == np.float16:
        return ll_ir.HalfType()
    if dt.name == "bfloat16":
        return _BFloatType()
    if dt == np.int32:
        return ll_ir.IntType(32)
    if dt == np.int64:
        return ll_ir.IntType(64)
    if dt == np.int16 or dt == np.uint16:
        return ll_ir.IntType(16)
    if dt == np.int8 or dt == np.uint8:
        return ll_ir.IntType(8)
    if dt == np.uint32:
        return ll_ir.IntType(32)
    if dt == np.uint64:
        return ll_ir.IntType(64)
    if dt == np.bool_:
        return ll_ir.IntType(1)
    raise NotImplementedError(f"unsupported dtype {dtype}")


def _scalars_per_elem(dtype) -> int:
    return 2 if _is_complex(dtype) else 1


def _aval_size(shape, dtype) -> int:
    return int(math.prod(shape)) * _scalars_per_elem(dtype)


def _f_prefix(dtype) -> str:
    dt = np.dtype(dtype)
    if dt == np.float32:
        return "f"
    if dt == np.float64:
        return ""
    raise NotImplementedError(f"libdevice has no entry for dtype {dtype}")


def _constant_vals(arr: np.ndarray, dtype) -> Vals:
    ir_t = _dtype_to_ir(dtype)
    flat = np.asarray(arr).ravel()
    if _is_complex(dtype):
        out: Vals = []
        for v in flat:
            c = complex(v)
            out.append(ll_ir.Constant(ir_t, c.real))
            out.append(ll_ir.Constant(ir_t, c.imag))
        return out
    if _is_float(dtype):
        return [ll_ir.Constant(ir_t, float(v)) for v in flat]
    return [ll_ir.Constant(ir_t, int(v)) for v in flat]


def _broadcast_flat_indices(shape: tuple, out_shape: tuple) -> list[int]:
    pad = len(out_shape) - len(shape)
    out: list[int] = []

    def walk(out_idx: list[int]):
        if len(out_idx) == len(out_shape):
            flat = 0
            for k, d in enumerate(shape):
                i = out_idx[pad + k]
                if d == 1:
                    i = 0
                flat = flat * d + i
            out.append(flat)
            return
        axis = len(out_idx)
        for j in range(out_shape[axis]):
            out_idx.append(j)
            walk(out_idx)
            out_idx.pop()

    walk([])
    return out


def _broadcast_pair(a: Vals, c: Vals, a_shape, c_shape, out_shape) -> tuple[Vals, Vals]:
    a_idx = _broadcast_flat_indices(a_shape, out_shape)
    c_idx = _broadcast_flat_indices(c_shape, out_shape)
    return [a[i] for i in a_idx], [c[i] for i in c_idx]


def _broadcast_pair_complex(a: Vals, c: Vals, a_shape, c_shape, out_shape) -> tuple[Vals, Vals]:
    a_idx = _broadcast_flat_indices(a_shape, out_shape)
    c_idx = _broadcast_flat_indices(c_shape, out_shape)
    a_out: Vals = []
    c_out: Vals = []
    for i in a_idx:
        a_out.append(a[2 * i])
        a_out.append(a[2 * i + 1])
    for i in c_idx:
        c_out.append(c[2 * i])
        c_out.append(c[2 * i + 1])
    return a_out, c_out


def _binop_args(op, ins: list[Vals]):
    a_shape, _ = value_shape_dtype(op.operands[0])
    c_shape, _ = value_shape_dtype(op.operands[1])
    out_shape, out_dt = result_shape_dtype(op)
    if _is_complex(out_dt):
        a, c = _broadcast_pair_complex(ins[0], ins[1], a_shape, c_shape, out_shape)
    else:
        a, c = _broadcast_pair(ins[0], ins[1], a_shape, c_shape, out_shape)
    return a, c, out_dt


def _zip_binop(em, a: Vals, b: Vals, dtype, op: str) -> Vals:
    if _is_float(dtype):
        if op == "add":
            return [em.b.fadd(x, y) for x, y in zip(a, b, strict=True)]
        if op == "sub":
            return [em.b.fsub(x, y) for x, y in zip(a, b, strict=True)]
        if op == "mul":
            return [em.b.fmul(x, y) for x, y in zip(a, b, strict=True)]
        if op == "div":
            return [em.b.fdiv(x, y) for x, y in zip(a, b, strict=True)]
        if op == "rem":
            return [em.b.frem(x, y) for x, y in zip(a, b, strict=True)]
    else:
        if op == "add":
            return [em.b.add(x, y) for x, y in zip(a, b, strict=True)]
        if op == "sub":
            return [em.b.sub(x, y) for x, y in zip(a, b, strict=True)]
        if op == "mul":
            return [em.b.mul(x, y) for x, y in zip(a, b, strict=True)]
        if op == "div":
            return [em.b.sdiv(x, y) for x, y in zip(a, b, strict=True)]
        if op == "rem":
            return [em.b.srem(x, y) for x, y in zip(a, b, strict=True)]
    raise NotImplementedError(f"{op} on {dtype}")


def _float_bits(dtype) -> int:
    return 64 if np.dtype(dtype) in (np.float64, np.complex128) else 32


def _int_bits(dtype) -> int:
    return np.dtype(dtype).itemsize * 8


def _row_major_strides(shape) -> list[int]:
    n = len(shape)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _row_major_iter(shape):
    if not shape:
        yield ()
        return
    n = int(math.prod(shape))
    strides = _row_major_strides(shape)
    for flat in range(n):
        idx = tuple((flat // strides[k]) % shape[k] for k in range(len(shape)))
        yield idx
