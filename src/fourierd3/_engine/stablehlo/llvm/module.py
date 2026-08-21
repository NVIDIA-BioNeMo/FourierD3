# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LLVM module the device function is emitted into.

`walk_main` turns a StableHLO module into a single LLVM function taking one
pointer per input and output, dispatching each operation to the handler
registered for its name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from jaxlib.mlir import ir as mlir_ir
from llvmlite import ir as ll_ir

from fourierd3._engine.stablehlo.attributes import (
    build_func_table,
    value_shape_dtype,
)
from fourierd3._engine.stablehlo.llvm.values import (
    Vals,
    _aval_size,
    _constant_vals,
    _dtype_to_ir,
    _f_prefix,
    _is_complex,
    _is_float,
    _scalars_per_elem,
)


class _NamingBuilder:
    _NO_NAME = frozenset({"store", "ret", "ret_void", "branch", "cbranch", "switch"})

    def __init__(self, builder: ll_ir.IRBuilder, em: _Emitter):
        self._b = builder
        self._em = em

    def __getattr__(self, name: str):
        attr = getattr(self._b, name)
        if not callable(attr) or name in self._NO_NAME:
            return attr
        em = self._em

        def wrapper(*args, **kwargs):
            if "name" not in kwargs:
                kwargs["name"] = em.fresh()
            return attr(*args, **kwargs)

        return wrapper


class _Emitter:
    def __init__(self, module: ll_ir.Module, builder: ll_ir.IRBuilder):
        self.module = module
        self._raw = builder
        self._name_counter = 0
        self.b = _NamingBuilder(builder, self)
        self.libdevice_decls: dict[str, ll_ir.Function] = {}
        # Populated by walk_main so func.call can recurse into the callee's body.
        self.func_table: dict[str, mlir_ir.Operation] = {}

    def fresh(self) -> str:
        n = self._name_counter
        self._name_counter += 1
        return f"v{n}"

    def libdevice(self, intrinsic: str, arity: int, dtype) -> ll_ir.Function:
        suffix = _f_prefix(dtype)
        sym = f"__nv_{intrinsic}{suffix}"
        if sym in self.libdevice_decls:
            return self.libdevice_decls[sym]
        ir_t = _dtype_to_ir(dtype)
        fn = ll_ir.Function(self.module, ll_ir.FunctionType(ir_t, [ir_t] * arity), name=sym)
        self.libdevice_decls[sym] = fn
        return fn


Handler = Callable[["_Emitter", "mlir_ir.Operation", list[Vals], dict], Vals]

_HANDLERS: dict[str, Handler] = {}


def _register(*names: str):
    def deco(fn: Handler) -> Handler:
        for n in names:
            _HANDLERS[n] = fn
        return fn

    return deco


def walk_main(
    module: mlir_ir.Module,
    closed_consts: list,
    arg_names: list[str] | None,
    name: str,
) -> str:
    func_table = build_func_table(module)
    main_op = func_table["main"]
    main_block = next(iter(main_op.regions[0].blocks))

    n_consts = len(closed_consts)
    n_total_args = len(main_block.arguments)
    n_invars = n_total_args - n_consts

    ret_op = list(main_block.operations)[-1]
    assert str(ret_op.name) == "func.return"
    n_outvars = len(ret_op.operands)

    ll_module = ll_ir.Module(name=name)
    ll_module.triple = "nvptx64-nvidia-cuda"

    param_types: list[ll_ir.Type] = []
    invar_shape_dtypes: list[tuple[tuple[int, ...], np.dtype]] = []
    for i in range(n_invars):
        ba = main_block.arguments[n_consts + i]
        shape, dt = value_shape_dtype(ba)
        param_types.append(_dtype_to_ir(dt).as_pointer())
        invar_shape_dtypes.append((shape, dt))

    outvar_shape_dtypes: list[tuple[tuple[int, ...], np.dtype]] = []
    for k in range(n_outvars):
        shape, dt = value_shape_dtype(ret_op.operands[k])
        param_types.append(_dtype_to_ir(dt).as_pointer())
        outvar_shape_dtypes.append((shape, dt))

    fn = ll_ir.Function(ll_module, ll_ir.FunctionType(ll_ir.VoidType(), param_types), name=name)
    for i in range(n_invars):
        fn.args[i].name = arg_names[i] if arg_names and i < len(arg_names) else f"arg{i}"
    for k in range(n_outvars):
        fn.args[n_invars + k].name = f"out{k}"

    entry = fn.append_basic_block("entry")
    b = ll_ir.IRBuilder(entry)
    em = _Emitter(ll_module, b)
    em.func_table = func_table

    env: dict[Any, Vals] = {}

    # Non-scalar constants land in addrspace(4) globals (the LLVM IR analogue
    # of CUDA __constant__ memory); scalar consts inline as IR literals.
    for i in range(n_consts):
        block_arg = main_block.arguments[i]
        shape, dt = value_shape_dtype(block_arg)
        arr = np.asarray(closed_consts[i])
        flat = arr.ravel()
        if flat.size <= 1:
            env[block_arg] = _constant_vals(arr, dt)
        else:
            env[block_arg] = _emit_const_global(em, ll_module, arr, dt, i)

    for i in range(n_invars):
        block_arg = main_block.arguments[n_consts + i]
        shape, dt = value_shape_dtype(block_arg)
        env[block_arg] = _load_param(em, fn.args[i], shape, dt)

    _walk_ops(em, main_block.operations, env)

    for k in range(n_outvars):
        operand = ret_op.operands[k]
        vals = env[operand]
        shape, dt = outvar_shape_dtypes[k]
        _store_param(em, fn.args[n_invars + k], vals, shape, dt)
    em._raw.ret_void()
    return str(ll_module)


def _walk_ops(em: _Emitter, ops, env: dict) -> None:
    for op in ops:
        name = str(op.name)
        if name in ("func.return", "stablehlo.return"):
            break
        ins = [env[operand] for operand in op.operands]
        handler = _HANDLERS.get(name)
        if handler is None:
            raise NotImplementedError(f"StableHLO op {name!r} not supported")
        out_vals = handler(em, op, ins, env)
        if len(op.results) == 1:
            env[op.results[0]] = out_vals
        else:
            for res, vals in zip(op.results, out_vals, strict=True):
                env[res] = vals


def _load_param(em: _Emitter, ptr: ll_ir.Argument, shape, dtype) -> Vals:
    n = _aval_size(shape, dtype)
    if n == 1:
        return [em.b.load(ptr, name=em.fresh())]
    out: Vals = []
    for j in range(n):
        gep = em.b.gep(ptr, [ll_ir.Constant(ll_ir.IntType(32), j)], inbounds=True)
        out.append(em.b.load(gep, name=em.fresh()))
    return out


def _store_param(em: _Emitter, ptr: ll_ir.Argument, vals: Vals, shape, dtype) -> None:
    n = _aval_size(shape, dtype)
    if len(vals) != n:
        raise AssertionError(f"outvar size mismatch: got {len(vals)}, expected {n}")
    for j, v in enumerate(vals):
        gep = em.b.gep(ptr, [ll_ir.Constant(ll_ir.IntType(32), j)], inbounds=True)
        em.b.store(v, gep)


def _emit_const_global(
    em: _Emitter, module: ll_ir.Module, arr: np.ndarray, dtype, idx: int
) -> Vals:
    ir_t = _dtype_to_ir(dtype)
    flat = arr.ravel()
    spe = _scalars_per_elem(dtype)
    n_slots = flat.size * spe
    if _is_complex(dtype):
        scalars: list[ll_ir.Constant] = []
        for v in flat:
            c = complex(v)
            scalars.append(ll_ir.Constant(ir_t, c.real))
            scalars.append(ll_ir.Constant(ir_t, c.imag))
    elif _is_float(dtype):
        scalars = [ll_ir.Constant(ir_t, float(v)) for v in flat]
    else:
        scalars = [ll_ir.Constant(ir_t, int(v)) for v in flat]

    arr_t = ll_ir.ArrayType(ir_t, n_slots)
    gv = ll_ir.GlobalVariable(module, arr_t, name=f"_cc{idx}", addrspace=4)
    gv.global_constant = True
    gv.initializer = ll_ir.Constant(arr_t, scalars)
    gv.linkage = "internal"

    out: Vals = []
    for j in range(n_slots):
        gep = em._raw.gep(
            gv,
            [
                ll_ir.Constant(ll_ir.IntType(32), 0),
                ll_ir.Constant(ll_ir.IntType(32), j),
            ],
            inbounds=True,
            name=em.fresh(),
        )
        out.append(em._raw.load(gep, name=em.fresh()))
    return out


def generate_device_ir(closed_jaxpr, *, name: str, arg_names: list[str] | None) -> str:
    """Lower a closed jaxpr to StableHLO, then to the LLVM IR of one device function."""
    from fourierd3._engine.tracing.capture import _lower_to_stablehlo

    module = _lower_to_stablehlo(closed_jaxpr)
    return walk_main(module, list(closed_jaxpr.consts), arg_names, name)


# Imported for their registration side effect: each module fills `_HANDLERS`
# with the StableHLO operations it knows how to emit. They import names defined
# above, so this stays at the bottom.
from fourierd3._engine.stablehlo.llvm import (  # noqa: E402,F401
    constants,
    control_flow,
    elementwise,
    shape_ops,
)
