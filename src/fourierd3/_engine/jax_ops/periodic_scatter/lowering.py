# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import numpy as np

from fourierd3._engine import _extension
from fourierd3._engine.dtypes import dtype_id
from fourierd3._engine.jax_ops.periodic_scatter.derived_jaxprs import extract_meta
from fourierd3._engine.jax_ops.periodic_scatter.layout import (
    detect_cartesian_support,
    support_3d,
    support_is_separable,
)
from fourierd3._engine.runtime.compile_plans import compile_plan
from fourierd3._engine.runtime.execute_plans import run_plan
from fourierd3._engine.stablehlo.llvm import generate_device_ir
from fourierd3._engine.tracing.capture import extract_captures
from fourierd3._engine.tracing.optimize import optimize


def _buf(name, dtype, ic, ext, elem_size):
    return (name, dtype_id(dtype), list(ic), list(ext), int(elem_size))


def lower_periodic_scatter(
    parsed,
    layout,
    jaxpr,
    support,
    *,
    batch_sizes,
    buf_batch_extents,
    grid_shape,
    outputs_shape_dtype,
    n_backend_arrays,
    cuda_opts=None,
):
    cuda_opts = cuda_opts or {}
    sup3 = support_3d(support)
    ext = layout.split_buf_extents(buf_batch_extents)

    meta = extract_meta(jaxpr, layout.n_grid_in, layout.n_grid_out)

    if support_is_separable(support):
        cart = (len(support), support[0])
    else:
        cart = detect_cartesian_support(sup3)

    v_arg_names = (
        ["_sidx", "_sup"]
        + [f"_g{g}" for g in range(layout.n_grid_in)]
        + [f"_ni{k}" for k in range(layout.n_nongrid_in)]
    )
    closed, _ = extract_captures(jaxpr)
    closed = optimize(closed, fast_math=True)
    device_fn_ir = generate_device_ir(closed, name="V", arg_names=v_arg_names)

    opts = ["-lineinfo", "--use_fast_math", *cuda_opts.get("opts", "").split()]
    cart_order, cart_mo = cart or (0, 0)
    cell_grid_shape = cuda_opts.get("cell_grid_shape")

    plan_bytes, pending = compile_plan(
        _extension.compile_periodic_scatter_to_bytes,
        device_fn_ir,
        [int(c) for triple in sup3 for c in triple],
        _buf("cell_idx", np.int32, layout.cell_idx, ext["cell_idx"], 3),
        [
            _buf(
                f"grid_in_{g}",
                meta["grid_in_dtypes"][g],
                layout.grid_in[g],
                ext["grid_in"][g],
                meta["grid_in_inner_sizes"][g],
            )
            for g in range(layout.n_grid_in)
        ],
        [
            _buf(
                f"ngin_{k}",
                meta["nongrid_in_dtypes"][k],
                layout.nongrid_in[k],
                ext["nongrid_in"][k],
                meta["nongrid_in_sizes"][k],
            )
            for k in range(layout.n_nongrid_in)
        ],
        [
            _buf(f"idx_{j}", np.int32, layout.idx[j], ext["idx"][j], 1)
            for j in range(layout.n_index)
        ],
        [
            _buf(
                f"grid_out_{j}",
                meta["grid_out_dtypes"][j],
                layout.grid_out[j],
                ext["grid_out"][j],
                meta["grid_out_inner_sizes"][j],
            )
            for j in range(layout.n_grid_out)
        ],
        [
            _buf(
                f"out_{j}",
                meta["nongrid_out_dtypes"][j],
                layout.nongrid_out[j],
                ext["nongrid_out"][j],
                meta["nongrid_out_sizes"][j],
            )
            for j in range(layout.n_nongrid_out)
        ],
        list(batch_sizes),
        n_backend_arrays,
        cart_order,
        cart_mo,
        [int(d) for d in grid_shape],
        [int(d) for d in cell_grid_shape] if cell_grid_shape is not None else None,
        opts,
    )
    inputs = [
        parsed.cell_idx,
        *parsed.grid_in,
        *parsed.nongrid_in,
        *parsed.index_buffers,
        *parsed.backend_arrays,
    ]
    out_specs = [jax.ShapeDtypeStruct(o.shape, o.dtype) for o in outputs_shape_dtype]
    return run_plan(plan_bytes, inputs, out_specs, pending=pending)
