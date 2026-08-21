# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import struct

import numpy as np
from jaxlib.mlir import ir

_MLIR_ELEM_TO_DTYPE: dict[str, np.dtype] = {
    "i1": np.dtype("bool"),
    "i8": np.dtype("int8"),
    "i16": np.dtype("int16"),
    "i32": np.dtype("int32"),
    "i64": np.dtype("int64"),
    "ui8": np.dtype("uint8"),
    "ui16": np.dtype("uint16"),
    "ui32": np.dtype("uint32"),
    "ui64": np.dtype("uint64"),
    "f16": np.dtype("float16"),
    "bf16": np.dtype("bfloat16"),
    "f32": np.dtype("float32"),
    "f64": np.dtype("float64"),
    "complex<f32>": np.dtype("complex64"),
    "complex<f64>": np.dtype("complex128"),
}


def mlir_elem_to_dtype(elem_type: ir.Type) -> np.dtype:
    s = str(elem_type)
    dt = _MLIR_ELEM_TO_DTYPE.get(s)
    if dt is None:
        raise ValueError(f"Unsupported MLIR element type: {elem_type}")
    return dt


def _parse_tensor_type(mlir_type):
    t = ir.RankedTensorType(mlir_type)
    return tuple(t.shape), mlir_elem_to_dtype(t.element_type)


def result_shape_dtype(op: ir.Operation, idx: int = 0):
    return _parse_tensor_type(op.results[idx].type)


def value_shape_dtype(value: ir.Value):
    return _parse_tensor_type(value.type)


_FLOAT_OR_HEX = r"[-+]?(?:0x[0-9A-Fa-f]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
_COMPLEX_PAIR_RE = re.compile(r"\(\s*(" + _FLOAT_OR_HEX + r")\s*,\s*(" + _FLOAT_OR_HEX + r")\s*\)")


def _mlir_literal_to_float(s: str, base_dtype: np.dtype) -> float:
    if "0x" in s:
        bits = int(s, 16)
        if base_dtype == np.dtype("float32"):
            return struct.unpack("<f", struct.pack("<I", bits))[0]
        return struct.unpack("<d", struct.pack("<Q", bits))[0]
    return float(base_dtype.type(s))


def _parse_dense_complex(op_str: str, dtype: np.dtype) -> np.ndarray:
    m = re.search(r"dense<(.+?)>\s*:", op_str)
    if not m:
        raise ValueError(f"Cannot parse dense complex value from: {op_str!r}")
    body = m.group(1).strip()

    pairs = _COMPLEX_PAIR_RE.findall(body)
    if not pairs:
        raise ValueError(f"No complex pairs found in dense value: {body!r}")

    base = np.dtype("float32") if dtype == np.dtype("complex64") else np.dtype("float64")
    values = [
        dtype.type(
            complex(
                _mlir_literal_to_float(re_s, base),
                _mlir_literal_to_float(im_s, base),
            )
        )
        for re_s, im_s in pairs
    ]
    return np.array(values, dtype=dtype)


def get_dense_value(op: ir.Operation) -> np.ndarray:
    shape, dtype = result_shape_dtype(op)
    if np.issubdtype(dtype, np.complexfloating):
        return _parse_dense_complex(str(op), dtype).reshape(shape)
    if np.dtype(dtype).name == "bfloat16":
        # numpy can't unbox a `bfloat16` DenseElementsAttr; round-trip
        # via the MLIR attribute API (splat) or textual form (non-splat).
        return _parse_dense_bfloat16(op, shape)
    attr = op.attributes["value"]
    da = ir.DenseElementsAttr(attr)
    return np.array(da)


def _parse_dense_bfloat16(op: ir.Operation, shape: tuple) -> np.ndarray:
    # Returns f64 intentionally — every caller goes through float() and bf16's range fits.
    da = ir.DenseElementsAttr(op.attributes["value"])
    if da.is_splat:
        val = float(ir.FloatAttr(da.get_splat_value()).value)
        return np.full(shape, val, dtype=np.float64)
    # Brackets may be nested for multi-dim; flat order is row-major.
    m = re.search(r"dense<(.+?)>\s*:\s*tensor<", str(op), re.DOTALL)
    if m is None:
        raise ValueError(f"cannot find dense<...> in {op}")
    body = m.group(1)
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", body)
    return np.array([float(n) for n in nums], dtype=np.float64).reshape(shape)


def get_i64_array(op: ir.Operation, name: str) -> tuple[int, ...]:
    return tuple(ir.DenseI64ArrayAttr(op.attributes[name]))


def get_i64(op: ir.Operation, name: str) -> int:
    return ir.IntegerAttr(op.attributes[name]).value


def get_comparison_direction(op: ir.Operation) -> str:
    s = str(op.attributes["comparison_direction"])
    m = re.search(r"comparison_direction (\w+)", s)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse comparison_direction: {s}")


def parse_attr_list(attr_str: str, key: str) -> tuple[int, ...]:
    m = re.search(key + r"\s*=\s*\[([^\]]*)\]", attr_str)
    if m:
        t = m.group(1).strip()
        return tuple(int(x) for x in t.split(",") if x.strip()) if t else ()
    return ()


def parse_attr_scalar(attr_str: str, key: str, default: int = 0) -> int:
    m = re.search(key + r"\s*=\s*(\d+)", attr_str)
    return int(m.group(1)) if m else default


def get_dot_dimension_numbers(op: ir.Operation):
    s = str(op.attributes["dot_dimension_numbers"])
    return (
        (
            parse_attr_list(s, "lhs_contracting_dimensions"),
            parse_attr_list(s, "rhs_contracting_dimensions"),
        ),
        (
            parse_attr_list(s, "lhs_batching_dimensions"),
            parse_attr_list(s, "rhs_batching_dimensions"),
        ),
    )


def get_callee_name(op: ir.Operation) -> str:
    return ir.FlatSymbolRefAttr(op.attributes["callee"]).value


def get_func_name(func_op: ir.Operation) -> str:
    return ir.StringAttr(func_op.attributes["sym_name"]).value


def build_func_table(module: ir.Module) -> dict[str, ir.Operation]:
    table: dict[str, ir.Operation] = {}
    for func_op in module.body.operations:
        name = get_func_name(func_op)
        table[name] = func_op
    return table
