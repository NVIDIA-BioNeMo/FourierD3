# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import ClosedJaxpr as _ClosedJaxpr  # noqa: F401

from fourierd3._engine.tracing.differentiate import _make_partial, _sds, jaxpr_to_fn


def extract_meta(jaxpr, n_grid_in, n_grid_out):
    invars = jaxpr.jaxpr.invars
    outvars = jaxpr.jaxpr.outvars

    grid_in_iv = invars[2 : 2 + n_grid_in]
    nongrid_in_iv = invars[2 + n_grid_in :]
    grid_out_ov = outvars[:n_grid_out]
    nongrid_out_ov = outvars[n_grid_out:]

    grid_in_inner_shapes = tuple(v.aval.shape for v in grid_in_iv)
    grid_in_inner_sizes = tuple(math.prod(v.aval.shape) for v in grid_in_iv)
    grid_out_inner_shapes = tuple(v.aval.shape for v in grid_out_ov)
    grid_out_inner_sizes = tuple(math.prod(v.aval.shape) for v in grid_out_ov)
    grid_out_sizes = grid_out_inner_sizes

    return {
        "grid_in_inner_shapes": grid_in_inner_shapes,
        "grid_in_dtypes": tuple(v.aval.dtype for v in grid_in_iv),
        "grid_in_inner_sizes": grid_in_inner_sizes,
        "nongrid_in_shapes": tuple(v.aval.shape for v in nongrid_in_iv),
        "nongrid_in_dtypes": tuple(v.aval.dtype for v in nongrid_in_iv),
        "nongrid_in_sizes": tuple(math.prod(v.aval.shape) for v in nongrid_in_iv),
        "grid_out_inner_shapes": grid_out_inner_shapes,
        "grid_out_dtypes": tuple(v.aval.dtype for v in grid_out_ov),
        "grid_out_inner_sizes": grid_out_inner_sizes,
        "grid_out_sizes": grid_out_sizes,
        "nongrid_out_shapes": tuple(v.aval.shape for v in nongrid_out_ov),
        "nongrid_out_dtypes": tuple(v.aval.dtype for v in nongrid_out_ov),
        "nongrid_out_sizes": tuple(math.prod(v.aval.shape) for v in nongrid_out_ov),
    }


def derive_jvp_jaxpr(jaxpr, n_grid_in, nonzero_grid_idx=None, nonzero_nongrid_in_idx=None):
    invars = jaxpr.jaxpr.invars
    n_nongrid_in = len(invars) - 2 - n_grid_in

    if nonzero_grid_idx is None:
        nonzero_grid_idx = list(range(n_grid_in))
    if nonzero_nongrid_in_idx is None:
        nonzero_nongrid_in_idx = list(range(n_nongrid_in))

    n_nz_grid = len(nonzero_grid_idx)
    fn = jaxpr_to_fn(jaxpr)

    def jvp_fn(support_index, support_offset, *all_args):
        o = 0
        primals_gathered = all_args[o : o + n_grid_in]
        o += n_grid_in
        t_gathered_nz = all_args[o : o + n_nz_grid]
        o += n_nz_grid
        primals_nongrid_in = all_args[o : o + n_nongrid_in]
        o += n_nongrid_in
        t_nongrid_in_nz = all_args[o:]

        # support_index and support_offset are non-differentiable (int32)
        all_primals = (
            (support_index, support_offset) + tuple(primals_gathered) + tuple(primals_nongrid_in)
        )
        active_idx = [2 + g for g in nonzero_grid_idx] + [
            2 + n_grid_in + k for k in nonzero_nongrid_in_idx
        ]
        fn_partial, diff_primals = _make_partial(fn, all_primals, active_idx)
        tangents = tuple(t_gathered_nz) + tuple(t_nongrid_in_nz)
        _, t_out = jax.jvp(fn_partial, diff_primals, tangents)
        return t_out

    sup_idx_ex = _sds(invars[0])
    sup_ex = _sds(invars[1])
    gathered_exs = [_sds(invars[2 + g]) for g in range(n_grid_in)]
    t_gathered_exs = [_sds(invars[2 + g]) for g in nonzero_grid_idx]
    nongrid_in_exs = [_sds(invars[2 + n_grid_in + k]) for k in range(n_nongrid_in)]
    t_nongrid_in_exs = [_sds(invars[2 + n_grid_in + k]) for k in nonzero_nongrid_in_idx]
    return jax.make_jaxpr(jvp_fn)(
        sup_idx_ex,
        sup_ex,
        *gathered_exs,
        *t_gathered_exs,
        *nongrid_in_exs,
        *t_nongrid_in_exs,
    )


def derive_transpose_jaxpr(
    jaxpr,
    n_grid_in,
    n_grid_out,
    needed_grid_idx=None,
    needed_nongrid_in_idx=None,
    nonzero_ct_idx=None,
):
    invars = jaxpr.jaxpr.invars
    outvars = jaxpr.jaxpr.outvars
    n_nongrid_in = len(invars) - 2 - n_grid_in
    n_outputs = len(outvars)

    if needed_grid_idx is None:
        needed_grid_idx = list(range(n_grid_in))
    if needed_nongrid_in_idx is None:
        needed_nongrid_in_idx = list(range(n_nongrid_in))
    if nonzero_ct_idx is None:
        nonzero_ct_idx = list(range(n_outputs))

    defined_grid_idx = [g for g in range(n_grid_in) if g not in needed_grid_idx]
    defined_nongrid_in_idx = [k for k in range(n_nongrid_in) if k not in needed_nongrid_in_idx]

    nonzero_ct_grid_idx = [j for j in nonzero_ct_idx if j < n_grid_out]
    nonzero_ct_nongrid_idx = [j for j in nonzero_ct_idx if j >= n_grid_out]

    n_ct_grid = len(nonzero_ct_grid_idx)
    n_def_grid_in = len(defined_grid_idx)
    n_ct_nongrid = len(nonzero_ct_nongrid_idx)

    all_needed = [2 + g for g in needed_grid_idx] + [
        2 + n_grid_in + k for k in needed_nongrid_in_idx
    ]
    fn = jaxpr_to_fn(jaxpr)

    zero_consts = tuple(
        np.zeros(invars[k].aval.shape, dtype=invars[k].aval.dtype) for k in all_needed
    )

    def transpose_fn(support_index, support_offset, *args):
        o = 0
        ct_grid = args[o : o + n_ct_grid]
        o += n_ct_grid
        def_grid = args[o : o + n_def_grid_in]
        o += n_def_grid_in
        ct_nongrid = args[o : o + n_ct_nongrid]
        o += n_ct_nongrid
        def_nongrid_in = args[o:]

        all_inputs = [support_index, support_offset] + [None] * (n_grid_in + n_nongrid_in)
        for i, g in enumerate(defined_grid_idx):
            all_inputs[2 + g] = def_grid[i]
        for i, k in enumerate(defined_nongrid_in_idx):
            all_inputs[2 + n_grid_in + k] = def_nongrid_in[i]
        for i, k in enumerate(all_needed):
            all_inputs[k] = zero_consts[i]

        all_cts = [jnp.zeros(v.aval.shape, v.aval.dtype) for v in outvars]
        for i, j in enumerate(nonzero_ct_grid_idx):
            all_cts[j] = ct_grid[i]
        for i, j in enumerate(nonzero_ct_nongrid_idx):
            all_cts[j] = ct_nongrid[i]

        fn_partial, diff_args = _make_partial(fn, all_inputs, all_needed)
        ct_result = jax.linear_transpose(fn_partial, *diff_args)(tuple(all_cts))

        n_needed_grid_in = len(needed_grid_idx)
        return tuple(ct_result[:n_needed_grid_in]) + tuple(ct_result[n_needed_grid_in:])

    sup_idx_ex = _sds(invars[0])
    sup_ex = _sds(invars[1])
    ct_grid_exs = [_sds(outvars[j]) for j in nonzero_ct_grid_idx]
    def_grid_exs = [_sds(invars[2 + g]) for g in defined_grid_idx]
    ct_nongrid_exs = [_sds(outvars[j]) for j in nonzero_ct_nongrid_idx]
    def_nongrid_in_exs = [_sds(invars[2 + n_grid_in + k]) for k in defined_nongrid_in_idx]
    return jax.make_jaxpr(transpose_fn)(
        sup_idx_ex,
        sup_ex,
        *ct_grid_exs,
        *def_grid_exs,
        *ct_nongrid_exs,
        *def_nongrid_in_exs,
    )
