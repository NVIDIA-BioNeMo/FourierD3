# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import numpy as np
from jax._src import core, sharding_impls
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir


def _try_concretize_tracer(tracer):
    try:
        v = tracer.val
        if isinstance(v, core.Literal):
            return np.asarray(v.val)
        frame = tracer._trace.frame
        cobj = frame.constvar_to_val.get(v)
        if cobj is not None:
            return np.asarray(cobj.original)
    except (AttributeError, TypeError):
        pass
    return None


def extract_captures(
    closed_jaxpr: core.ClosedJaxpr,
    on_the_right: bool = False,
) -> tuple[core.ClosedJaxpr, list]:
    jaxpr = closed_jaxpr.jaxpr
    consts = list(closed_jaxpr.consts)

    concrete_idx: list[int] = []
    concrete_consts: list = []
    captured_idx: list[int] = []
    captured_values: list = []

    for i, c in enumerate(consts):
        if not isinstance(c, core.Tracer):
            concrete_idx.append(i)
            concrete_consts.append(c)
            continue
        concrete_val = _try_concretize_tracer(c)
        if concrete_val is not None:
            concrete_idx.append(i)
            concrete_consts.append(concrete_val)
        else:
            captured_idx.append(i)
            captured_values.append(c)

    if not captured_idx:
        return core.ClosedJaxpr(jaxpr, concrete_consts), []

    captured_vars = [jaxpr.constvars[i] for i in captured_idx]
    orig_invars = list(jaxpr.invars)
    new_jaxpr = jaxpr.replace(
        constvars=[jaxpr.constvars[i] for i in concrete_idx],
        invars=orig_invars + captured_vars if on_the_right else captured_vars + orig_invars,
    )
    return core.ClosedJaxpr(new_jaxpr, concrete_consts), captured_values


def _lower_to_stablehlo(closed_jaxpr: core.ClosedJaxpr):
    jaxpr = closed_jaxpr.jaxpr
    n_consts = len(closed_jaxpr.consts)
    if n_consts > 0:
        flat_jaxpr = jaxpr.replace(
            constvars=[],
            invars=list(jaxpr.constvars) + list(jaxpr.invars),
        )
        flat_closed = core.ClosedJaxpr(flat_jaxpr, [])
    else:
        flat_closed = closed_jaxpr

    all_avals = [v.aval for v in flat_closed.jaxpr.invars]
    backend = xb.get_backend()
    lowering_args = {
        "num_const_args": 0,
        "in_avals": all_avals,
        "ordered_effects": [],
        "platforms": [backend.platform],
        "backend": backend,
        "axis_context": sharding_impls.ShardingContext(num_devices=1),
        "donated_args": [False] * len(all_avals),
        "lowering_parameters": mlir.LoweringParameters(),
    }
    if jax.__version_info__ >= (0, 11):
        result = mlir.lower_jaxpr_to_module(
            "fn",
            flat_closed.jaxpr,
            out_avals=[v.aval for v in flat_closed.jaxpr.outvars],
            **lowering_args,
        )
    else:
        result = mlir.lower_jaxpr_to_module("fn", flat_closed, **lowering_args)
    _legalize_chlo(result.module)
    return result.module


def _legalize_chlo(module) -> None:
    import jaxlib.mlir._mlir_libs._stablehlo as sh
    from jaxlib.mlir.passmanager import PassManager

    sh.register_stablehlo_passes()
    pm = PassManager.parse(
        "builtin.module(func.func(chlo-legalize-to-stablehlo))",
        context=module.context,
    )
    pm.run(module.operation)
