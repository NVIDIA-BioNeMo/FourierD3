# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handlers for operations that act on each element independently."""

from __future__ import annotations

import numpy as np
from llvmlite import ir as ll_ir

from fourierd3._engine.stablehlo.attributes import (
    get_comparison_direction,
    result_shape_dtype,
    value_shape_dtype,
)
from fourierd3._engine.stablehlo.llvm.module import (
    _HANDLERS,
    _Emitter,
    _register,
)
from fourierd3._engine.stablehlo.llvm.values import (
    Vals,
    _binop_args,
    _broadcast_flat_indices,
    _broadcast_pair,
    _broadcast_pair_complex,
    _dtype_to_ir,
    _float_bits,
    _int_bits,
    _is_complex,
    _is_float,
    _zip_binop,
)


@_register("stablehlo.add")
def _h_add(em: _Emitter, op, ins, env) -> Vals:
    a, c, dt = _binop_args(op, ins)
    if _is_complex(dt):
        out: Vals = []
        for i in range(0, len(a), 2):
            out.append(em.b.fadd(a[i], c[i]))
            out.append(em.b.fadd(a[i + 1], c[i + 1]))
        return out
    return _zip_binop(em, a, c, dt, "add")


@_register("stablehlo.subtract")
def _h_sub(em: _Emitter, op, ins, env) -> Vals:
    a, c, dt = _binop_args(op, ins)
    if _is_complex(dt):
        out: Vals = []
        for i in range(0, len(a), 2):
            out.append(em.b.fsub(a[i], c[i]))
            out.append(em.b.fsub(a[i + 1], c[i + 1]))
        return out
    return _zip_binop(em, a, c, dt, "sub")


@_register("stablehlo.multiply")
def _h_mul(em: _Emitter, op, ins, env) -> Vals:
    a, c, dt = _binop_args(op, ins)
    if _is_complex(dt):
        # (a+bi)(c+di) = (ac - bd) + (ad + bc)i
        out: Vals = []
        for i in range(0, len(a), 2):
            ar, ai = a[i], a[i + 1]
            br, bi = c[i], c[i + 1]
            out.append(em.b.fsub(em.b.fmul(ar, br), em.b.fmul(ai, bi)))
            out.append(em.b.fadd(em.b.fmul(ar, bi), em.b.fmul(ai, br)))
        return out
    return _zip_binop(em, a, c, dt, "mul")


@_register("stablehlo.divide")
def _h_div(em: _Emitter, op, ins, env) -> Vals:
    a, c, dt = _binop_args(op, ins)
    if _is_complex(dt):
        # (a+bi) / (c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
        out: Vals = []
        for i in range(0, len(a), 2):
            ar, ai = a[i], a[i + 1]
            br, bi = c[i], c[i + 1]
            denom = em.b.fadd(em.b.fmul(br, br), em.b.fmul(bi, bi))
            re_num = em.b.fadd(em.b.fmul(ar, br), em.b.fmul(ai, bi))
            im_num = em.b.fsub(em.b.fmul(ai, br), em.b.fmul(ar, bi))
            out.append(em.b.fdiv(re_num, denom))
            out.append(em.b.fdiv(im_num, denom))
        return out
    return _zip_binop(em, a, c, dt, "div")


@_register("stablehlo.remainder")
def _h_rem(em: _Emitter, op, ins, env) -> Vals:
    a, c, dt = _binop_args(op, ins)
    return _zip_binop(em, a, c, dt, "rem")


@_register("stablehlo.negate")
def _h_neg(em: _Emitter, op, ins, env) -> Vals:
    _, dt = result_shape_dtype(op)
    if _is_float(dt) or _is_complex(dt):
        return [em.b.fneg(v) for v in ins[0]]
    zero = ll_ir.Constant(_dtype_to_ir(dt), 0)
    return [em.b.sub(zero, v) for v in ins[0]]


@_register("stablehlo.abs")
def _h_abs(em: _Emitter, op, ins, env) -> Vals:
    in_shape, in_dt = value_shape_dtype(op.operands[0])
    _, out_dt = result_shape_dtype(op)
    if _is_complex(in_dt):
        # |a+bi| = sqrt(a² + b²) — real output.
        out: Vals = []
        for i in range(0, len(ins[0]), 2):
            re, im = ins[0][i], ins[0][i + 1]
            mag2 = em.b.fadd(em.b.fmul(re, re), em.b.fmul(im, im))
            sqrt_fn = em.libdevice("sqrt", 1, out_dt)
            out.append(em.b.call(sqrt_fn, [mag2]))
        return out
    if _is_float(in_dt):
        fabs_fn = em.libdevice("fabs", 1, in_dt)
        return [em.b.call(fabs_fn, [v]) for v in ins[0]]
    out2: Vals = []
    zero = ll_ir.Constant(_dtype_to_ir(in_dt), 0)
    for v in ins[0]:
        is_neg = em.b.icmp_signed("<", v, zero)
        neg_v = em.b.sub(zero, v)
        out2.append(em.b.select(is_neg, neg_v, v))
    return out2


@_register("chlo.square")
def _h_square(em: _Emitter, op, ins, env) -> Vals:
    _, dt = result_shape_dtype(op)
    if _is_complex(dt):
        out: Vals = []
        for i in range(0, len(ins[0]), 2):
            re, im = ins[0][i], ins[0][i + 1]
            out.append(em.b.fsub(em.b.fmul(re, re), em.b.fmul(im, im)))
            out.append(em.b.fmul(em.b.fmul(re, im), ll_ir.Constant(re.type, 2.0)))
        return out
    if _is_float(dt):
        return [em.b.fmul(v, v) for v in ins[0]]
    return [em.b.mul(v, v) for v in ins[0]]


_LIBDEVICE_UNARY: dict[str, str] = {
    "stablehlo.exponential": "exp",
    "stablehlo.log": "log",
    "stablehlo.sqrt": "sqrt",
    "stablehlo.rsqrt": "rsqrt",
    "stablehlo.sine": "sin",
    "stablehlo.cosine": "cos",
    "stablehlo.tanh": "tanh",
    "stablehlo.tan": "tan",
    "stablehlo.floor": "floor",
    "stablehlo.ceil": "ceil",
    "stablehlo.exponential_minus_one": "expm1",
    "stablehlo.log_plus_one": "log1p",
    "stablehlo.round_nearest_even": "rint",
    "stablehlo.round_nearest_afz": "round",
    "chlo.erf": "erf",
    "chlo.erfc": "erfc",
    "chlo.tan": "tan",
    "chlo.asin": "asin",
    "chlo.acos": "acos",
    "chlo.atan": "atan",
    "chlo.sinh": "sinh",
    "chlo.cosh": "cosh",
    "chlo.atanh": "atanh",
    "chlo.asinh": "asinh",
    "chlo.acosh": "acosh",
}


def _make_unary_libdevice(intrinsic: str):
    def handler(em: _Emitter, op, ins, env) -> Vals:
        _, dt = result_shape_dtype(op)
        fn = em.libdevice(intrinsic, 1, dt)
        return [em.b.call(fn, [v]) for v in ins[0]]

    return handler


for _hlo_name, _intr in _LIBDEVICE_UNARY.items():
    _HANDLERS[_hlo_name] = _make_unary_libdevice(_intr)


def _declare_cuda_intrinsic(em: _Emitter, name: str, arity: int, ret_t, arg_t) -> ll_ir.Function:
    if name in em.libdevice_decls:
        return em.libdevice_decls[name]
    fn = ll_ir.Function(em.module, ll_ir.FunctionType(ret_t, [arg_t] * arity), name=name)
    em.libdevice_decls[name] = fn
    return fn


_CUDA_INT_WIDTH_INTRINSIC: dict[str, tuple[str, str]] = {
    "stablehlo.popcnt": ("__popc", "__popcll"),
    "stablehlo.count_leading_zeros": ("__clz", "__clzll"),
}


def _make_cuda_int_width_intrinsic(name32: str, name64: str):
    def handler(em: _Emitter, op, ins, env) -> Vals:
        _, dt = result_shape_dtype(op)
        t = _dtype_to_ir(dt)
        fn_name = name32 if _int_bits(dt) == 32 else name64
        fn = _declare_cuda_intrinsic(em, fn_name, 1, t, t)
        return [em.b.call(fn, [v]) for v in ins[0]]

    return handler


for _hlo, (_n32, _n64) in _CUDA_INT_WIDTH_INTRINSIC.items():
    _HANDLERS[_hlo] = _make_cuda_int_width_intrinsic(_n32, _n64)


_CUDA_FP_PREDICATE: dict[str, str] = {
    "stablehlo.is_finite": "isfinite",
}


def _make_cuda_fp_predicate(name: str):
    def handler(em: _Emitter, op, ins, env) -> Vals:
        _, dt = value_shape_dtype(op.operands[0])
        in_t = _dtype_to_ir(dt)
        fn = _declare_cuda_intrinsic(em, name, 1, ll_ir.IntType(1), in_t)
        return [em.b.call(fn, [v]) for v in ins[0]]

    return handler


for _hlo, _name in _CUDA_FP_PREDICATE.items():
    _HANDLERS[_hlo] = _make_cuda_fp_predicate(_name)


@_register("stablehlo.maximum")
def _h_max(em: _Emitter, op, ins, env) -> Vals:
    a, c, dt = _binop_args(op, ins)
    if _is_float(dt):
        fn = em.libdevice("fmax", 2, dt)
        return [em.b.call(fn, [x, y]) for x, y in zip(a, c, strict=True)]
    out: Vals = []
    for x, y in zip(a, c, strict=True):
        out.append(em.b.select(em.b.icmp_signed(">", x, y), x, y))
    return out


@_register("stablehlo.minimum")
def _h_min(em: _Emitter, op, ins, env) -> Vals:
    a, c, dt = _binop_args(op, ins)
    if _is_float(dt):
        fn = em.libdevice("fmin", 2, dt)
        return [em.b.call(fn, [x, y]) for x, y in zip(a, c, strict=True)]
    out: Vals = []
    for x, y in zip(a, c, strict=True):
        out.append(em.b.select(em.b.icmp_signed("<", x, y), x, y))
    return out


@_register("stablehlo.power")
def _h_pow(em: _Emitter, op, ins, env) -> Vals:
    a, c, dt = _binop_args(op, ins)
    fn = em.libdevice("pow", 2, dt)
    return [em.b.call(fn, [x, y]) for x, y in zip(a, c, strict=True)]


@_register("stablehlo.clamp")
def _h_clamp(em: _Emitter, op, ins, env) -> Vals:
    out_shape, dt = result_shape_dtype(op)
    lo_shape, _ = value_shape_dtype(op.operands[0])
    val_shape, _ = value_shape_dtype(op.operands[1])
    hi_shape, _ = value_shape_dtype(op.operands[2])
    lo, val = _broadcast_pair(ins[0], ins[1], lo_shape, val_shape, out_shape)
    hi, val = _broadcast_pair(ins[2], val, hi_shape, out_shape, out_shape)
    if _is_float(dt):
        fmax = em.libdevice("fmax", 2, dt)
        fmin = em.libdevice("fmin", 2, dt)
        return [
            em.b.call(fmin, [em.b.call(fmax, [v, lo_v]), hi_v])
            for lo_v, v, hi_v in zip(lo, val, hi, strict=True)
        ]
    out: Vals = []
    for lo_v, v, hi_v in zip(lo, val, hi, strict=True):
        v = em.b.select(em.b.icmp_signed("<", v, lo_v), lo_v, v)
        v = em.b.select(em.b.icmp_signed(">", v, hi_v), hi_v, v)
        out.append(v)
    return out


@_register("stablehlo.sign")
def _h_sign(em: _Emitter, op, ins, env) -> Vals:
    _, dt = result_shape_dtype(op)
    if _is_float(dt):
        ir_t = _dtype_to_ir(dt)
        zero = ll_ir.Constant(ir_t, 0.0)
        one = ll_ir.Constant(ir_t, 1.0)
        neg_one = ll_ir.Constant(ir_t, -1.0)
        out: Vals = []
        for v in ins[0]:
            pos = em.b.fcmp_ordered(">", v, zero)
            neg = em.b.fcmp_ordered("<", v, zero)
            r = em.b.select(pos, one, zero)
            out.append(em.b.select(neg, neg_one, r))
        return out
    ir_t = _dtype_to_ir(dt)
    zero = ll_ir.Constant(ir_t, 0)
    one = ll_ir.Constant(ir_t, 1)
    neg_one = ll_ir.Constant(ir_t, -1)
    out2: Vals = []
    for v in ins[0]:
        pos = em.b.icmp_signed(">", v, zero)
        neg = em.b.icmp_signed("<", v, zero)
        r = em.b.select(pos, one, zero)
        out2.append(em.b.select(neg, neg_one, r))
    return out2


@_register("stablehlo.logistic")
def _h_logistic(em: _Emitter, op, ins, env) -> Vals:
    _, dt = result_shape_dtype(op)
    # 1 / (1 + exp(-x))
    exp_fn = em.libdevice("exp", 1, dt)
    one = ll_ir.Constant(_dtype_to_ir(dt), 1.0)
    out: Vals = []
    for v in ins[0]:
        neg = em.b.fneg(v)
        e = em.b.call(exp_fn, [neg])
        denom = em.b.fadd(one, e)
        out.append(em.b.fdiv(one, denom))
    return out


_FCMP_DIR = {
    "EQ": "oeq",
    "NE": "one",
    "LT": "olt",
    "LE": "ole",
    "GT": "ogt",
    "GE": "oge",
}

_ICMP_DIR = {
    "EQ": "==",
    "NE": "!=",
    "LT": "<",
    "LE": "<=",
    "GT": ">",
    "GE": ">=",
}


@_register("stablehlo.compare")
def _h_compare(em: _Emitter, op, ins, env) -> Vals:
    a_shape, a_dt = value_shape_dtype(op.operands[0])
    c_shape, _ = value_shape_dtype(op.operands[1])
    out_shape, _ = result_shape_dtype(op)
    if _is_complex(a_dt):
        a, c = _broadcast_pair_complex(ins[0], ins[1], a_shape, c_shape, out_shape)
    else:
        a, c = _broadcast_pair(ins[0], ins[1], a_shape, c_shape, out_shape)
    direction = get_comparison_direction(op)
    out: Vals = []
    if _is_complex(a_dt):
        if direction not in ("EQ", "NE"):
            raise NotImplementedError(f"complex compare {direction}")
        for i in range(0, len(a), 2):
            re_eq = em.b.fcmp_ordered("oeq", a[i], c[i])
            im_eq = em.b.fcmp_ordered("oeq", a[i + 1], c[i + 1])
            eq = em.b.and_(re_eq, im_eq)
            out.append(eq if direction == "EQ" else em.b.not_(eq))
        return out
    if _is_float(a_dt):
        pred = _FCMP_DIR[direction]
        return [em.b.fcmp_ordered(pred, x, y) for x, y in zip(a, c, strict=True)]
    pred = _ICMP_DIR[direction]
    return [em.b.icmp_signed(pred, x, y) for x, y in zip(a, c, strict=True)]


@_register("stablehlo.select")
def _h_select(em: _Emitter, op, ins, env) -> Vals:
    pred_shape, _ = value_shape_dtype(op.operands[0])
    t_shape, _ = value_shape_dtype(op.operands[1])
    f_shape, _ = value_shape_dtype(op.operands[2])
    out_shape, dt = result_shape_dtype(op)
    if _is_complex(dt):
        # Broadcast pred to one-bool-per-element, then duplicate per RIRI slot.
        pred_idx = _broadcast_flat_indices(pred_shape, out_shape)
        t, f = _broadcast_pair_complex(ins[1], ins[2], t_shape, f_shape, out_shape)
        out: Vals = []
        for k, pi in enumerate(pred_idx):
            p = ins[0][pi]
            out.append(em.b.select(p, t[2 * k], f[2 * k]))
            out.append(em.b.select(p, t[2 * k + 1], f[2 * k + 1]))
        return out
    pred_idx = _broadcast_flat_indices(pred_shape, out_shape)
    t, f = _broadcast_pair(ins[1], ins[2], t_shape, f_shape, out_shape)
    return [em.b.select(ins[0][pi], t[k], f[k]) for k, pi in enumerate(pred_idx)]


@_register("stablehlo.bitcast_convert")
def _h_bitcast_convert(em: _Emitter, op, ins, env) -> Vals:
    _, in_dt = value_shape_dtype(op.operands[0])
    _, out_dt = result_shape_dtype(op)
    if _is_complex(in_dt) or _is_complex(out_dt):
        raise NotImplementedError("bitcast_convert across complex dtypes")
    dst_t = _dtype_to_ir(out_dt)
    return [em.b.bitcast(v, dst_t) for v in ins[0]]


@_register("stablehlo.convert")
def _h_convert(em: _Emitter, op, ins, env) -> Vals:
    _, in_dt = value_shape_dtype(op.operands[0])
    _, out_dt = result_shape_dtype(op)
    if in_dt == out_dt:
        return list(ins[0])
    in_is_complex = _is_complex(in_dt)
    out_is_complex = _is_complex(out_dt)
    if not in_is_complex and out_is_complex:
        real_out_dt = np.float32 if np.dtype(out_dt) == np.complex64 else np.float64
        real_t = _dtype_to_ir(real_out_dt)
        in_t = _dtype_to_ir(in_dt)
        zero = ll_ir.Constant(real_t, 0.0)
        out: Vals = []
        for v in ins[0]:
            re = _convert_scalar(em.b, v, in_dt, real_out_dt, in_t, real_t)
            out.append(re)
            out.append(zero)
        return out
    if in_is_complex and not out_is_complex:
        in_real_dt = np.float32 if np.dtype(in_dt) == np.complex64 else np.float64
        in_real_t = _dtype_to_ir(in_real_dt)
        out_t = _dtype_to_ir(out_dt)
        out2: Vals = []
        for i in range(0, len(ins[0]), 2):
            re = ins[0][i]
            out2.append(_convert_scalar(em.b, re, in_real_dt, out_dt, in_real_t, out_t))
        return out2
    if in_is_complex and out_is_complex:
        in_real_dt = np.float32 if np.dtype(in_dt) == np.complex64 else np.float64
        out_real_dt = np.float32 if np.dtype(out_dt) == np.complex64 else np.float64
        in_real_t = _dtype_to_ir(in_real_dt)
        out_real_t = _dtype_to_ir(out_real_dt)
        return [
            _convert_scalar(em.b, v, in_real_dt, out_real_dt, in_real_t, out_real_t) for v in ins[0]
        ]
    src_t = _dtype_to_ir(in_dt)
    dst_t = _dtype_to_ir(out_dt)
    return [_convert_scalar(em.b, v, in_dt, out_dt, src_t, dst_t) for v in ins[0]]


def _convert_scalar(b, v, in_dt, out_dt, src_t, dst_t):
    in_is_float = _is_float(in_dt)
    out_is_float = _is_float(out_dt)
    if in_is_float and out_is_float:
        return (
            b.fpext(v, dst_t) if _float_bits(out_dt) > _float_bits(in_dt) else b.fptrunc(v, dst_t)
        )
    if in_is_float and not out_is_float:
        if np.dtype(out_dt) == np.bool_:
            zero = ll_ir.Constant(src_t, 0.0)
            return b.fcmp_ordered("one", v, zero)
        return b.fptosi(v, dst_t)
    if not in_is_float and out_is_float:
        if np.dtype(in_dt) == np.bool_:
            return b.uitofp(v, dst_t)
        return b.sitofp(v, dst_t)
    if np.dtype(in_dt) == np.bool_:
        return b.zext(v, dst_t)
    sw = _int_bits(in_dt)
    dw = _int_bits(out_dt)
    if sw == dw:
        return v
    if sw < dw:
        return b.sext(v, dst_t)
    return b.trunc(v, dst_t)


_BUILDER_BINOPS: dict[str, str] = {
    "stablehlo.and": "and_",
    "stablehlo.or": "or_",
    "stablehlo.xor": "xor",
    "stablehlo.shift_left": "shl",
    "stablehlo.shift_right_arithmetic": "ashr",
    "stablehlo.shift_right_logical": "lshr",
}


def _make_builder_binop(method: str):
    def handler(em: _Emitter, op, ins, env) -> Vals:
        a, c, _ = _binop_args(op, ins)
        emit = getattr(em.b, method)
        return [emit(x, y) for x, y in zip(a, c, strict=True)]

    return handler


for _hlo, _method in _BUILDER_BINOPS.items():
    _HANDLERS[_hlo] = _make_builder_binop(_method)


@_register("stablehlo.not")
def _h_not(em: _Emitter, op, ins, env) -> Vals:
    return [em.b.not_(v) for v in ins[0]]


@_register("stablehlo.complex")
def _h_complex(em: _Emitter, op, ins, env) -> Vals:
    out: Vals = []
    for re, im in zip(ins[0], ins[1], strict=True):
        out.append(re)
        out.append(im)
    return out


@_register("stablehlo.real")
def _h_real(em: _Emitter, op, ins, env) -> Vals:
    return [ins[0][2 * i] for i in range(len(ins[0]) // 2)]


@_register("stablehlo.imag")
def _h_imag(em: _Emitter, op, ins, env) -> Vals:
    return [ins[0][2 * i + 1] for i in range(len(ins[0]) // 2)]
