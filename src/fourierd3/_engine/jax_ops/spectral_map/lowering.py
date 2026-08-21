# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from fourierd3._engine import _extension
from fourierd3._engine.dtypes import dtype_id
from fourierd3._engine.indexing import _compute_batch_sizes
from fourierd3._engine.runtime.compile_plans import compile_plan
from fourierd3._engine.runtime.execute_plans import run_plan
from fourierd3._engine.stablehlo.llvm import generate_device_ir
from fourierd3._engine.tracing.capture import extract_captures
from fourierd3._engine.tracing.optimize import optimize

_PRECISIONS = (np.dtype("float32"), np.dtype("float64"))


@functools.lru_cache(maxsize=1)
def _engine_sm() -> int:
    return int(round(float(jax.devices("gpu")[0].compute_capability) * 10))


def _buf(name, dtype, ic, batch_shape):
    return (name, dtype_id(dtype), list(ic), [int(s) for s in batch_shape], 0)


def lower_spectral_map(
    *args,
    jaxpr,
    n_grid_in,
    n_grid_out,
    fft_lengths,
    is_hermitian,
    input_signs,
    output_signs,
    index_configuration,
    outputs_shape_dtype,
):
    nx, ny, nz = map(int, fft_lengths)
    if not is_hermitian or ny != nz:
        raise NotImplementedError(
            f"spectral_map CUDA needs is_hermitian and ny==nz "
            f"(got is_hermitian={is_hermitian}, ny={ny}, nz={nz})"
        )
    grids, auxs = list(args[:n_grid_in]), list(args[n_grid_in:])
    real_dtype = grids[0].dtype
    if real_dtype not in _PRECISIONS or any(g.dtype != real_dtype for g in grids):
        raise NotImplementedError(
            f"spectral_map CUDA: grids must share one of "
            f"{set(_PRECISIONS)}, got {[g.dtype for g in grids]}"
        )

    n_aux = len(auxs)
    n_aux_out = len(outputs_shape_dtype) - n_grid_out
    nz_half = nz // 2 + 1
    ic = index_configuration
    nba = len(ic[0])

    def buf_shape(i):
        host = args[i] if i < n_grid_in + n_aux else outputs_shape_dtype[i - n_grid_in - n_aux]
        return tuple(int(s) for s in host.shape[:nba])

    batch_shape = tuple(_compute_batch_sizes(ic[: n_grid_in + n_aux], buf_shape, nba))
    total_batch = max(1, math.prod(batch_shape))

    def inner_of(shape):
        return max(1, math.prod(int(d) for d in shape[nba + 3 :]))

    grid_inner_sizes = [inner_of(g.shape) for g in grids]
    output_inner_sizes = [inner_of(o.shape) for o in outputs_shape_dtype[:n_grid_out]]

    grid_in_bufs = [
        _buf(f"in_{j}", real_dtype, ic[j], grids[j].shape[:nba]) for j in range(n_grid_in)
    ]

    invars, outvars = jaxpr.jaxpr.invars, jaxpr.jaxpr.outvars
    aux_avals = [invars[1 + n_grid_in + k].aval for k in range(n_aux)]
    aux_dtype_ids = [dtype_id(a.dtype) for a in aux_avals]
    aux_inner_shapes = [[int(d) for d in a.shape] for a in aux_avals]
    aux_bufs = [
        _buf(f"aux_{k}", aux_avals[k].dtype, ic[n_grid_in + k], auxs[k].shape[:nba])
        for k in range(n_aux)
    ]

    aout_avals = [outvars[n_grid_out + j].aval for j in range(n_aux_out)]
    aux_output_dtype_ids = [dtype_id(a.dtype) for a in aout_avals]
    aux_output_inner_shapes = [[int(d) for d in a.shape] for a in aout_avals]

    closed, _ = extract_captures(jaxpr)
    closed = optimize(closed, fast_math=(real_dtype == jnp.float32))
    arg_names = ["i"] + [f"g{j}" for j in range(n_grid_in)] + [f"a{k}" for k in range(n_aux)]
    device_fn_ir = generate_device_ir(closed, name="fn", arg_names=arg_names)

    plan_bytes, pending = compile_plan(
        _extension.compile_spectral_map_to_bytes,
        fft_lengths=(nx, ny, nz),
        precision=dtype_id(real_dtype),
        sm=_engine_sm(),
        batch_shape=[int(s) for s in batch_shape],
        grid_in_bufs=grid_in_bufs,
        grid_inner_sizes=grid_inner_sizes,
        input_signs=[int(s) for s in input_signs],
        n_grid_out=n_grid_out,
        output_inner_sizes=output_inner_sizes,
        output_signs=[int(s) for s in output_signs],
        aux_bufs=aux_bufs,
        aux_dtypes=aux_dtype_ids,
        aux_inner_shapes=aux_inner_shapes,
        aux_output_dtypes=aux_output_dtype_ids,
        aux_output_inner_shapes=aux_output_inner_shapes,
        device_fn_ir=device_fn_ir,
    )
    grid_out_specs = [
        jax.ShapeDtypeStruct(
            batch_shape + tuple(int(s) for s in outputs_shape_dtype[j].shape[nba:]),
            real_dtype,
        )
        for j in range(n_grid_out)
    ]
    aux_out_specs = [
        jax.ShapeDtypeStruct(
            (total_batch * nz_half * ny, max(1, math.prod(aux_output_inner_shapes[j]))),
            np.dtype(aout_avals[j].dtype),
        )
        for j in range(n_aux_out)
    ]
    raw = run_plan(
        plan_bytes,
        [*grids, *auxs],
        grid_out_specs + aux_out_specs,
        pending=pending,
    )
    grid_raw = raw[:n_grid_out]
    aux_raw = raw[n_grid_out:]

    def collapse(arr, osd):
        axes = tuple(a for a in range(nba) if osd.shape[a] < batch_shape[a])
        return arr.sum(axis=axes, keepdims=True) if axes else arr

    real_results = [collapse(grid_raw[j], outputs_shape_dtype[j]) for j in range(n_grid_out)]

    aux_results = []
    for j in range(n_aux_out):
        osd = outputs_shape_dtype[n_grid_out + j]
        summed = aux_raw[j].reshape(total_batch, -1, aux_raw[j].shape[-1]).sum(axis=1)
        target = batch_shape + tuple(aux_output_inner_shapes[j])
        aux_results.append(collapse(summed.reshape(target).astype(osd.dtype), osd))

    return real_results + aux_results
