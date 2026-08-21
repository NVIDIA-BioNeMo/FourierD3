# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from fourierd3._engine.tracing.differentiate import _make_partial, _sds, jaxpr_to_fn


def derive_jvp_jaxpr(jaxpr, n_grid_in, nonzero_grid_idx=None, nonzero_aux_idx=None):
    invars = jaxpr.jaxpr.invars
    n_aux = len(invars) - 1 - n_grid_in

    if nonzero_grid_idx is None:
        nonzero_grid_idx = list(range(n_grid_in))
    if nonzero_aux_idx is None:
        nonzero_aux_idx = list(range(n_aux))

    n_nz_grid = len(nonzero_grid_idx)
    fn = jaxpr_to_fn(jaxpr)

    active_jaxpr_idx = [1 + g for g in nonzero_grid_idx] + [
        1 + n_grid_in + k for k in nonzero_aux_idx
    ]

    def jvp_fn(*all_args):
        o = 1
        i_arg = all_args[:1]
        primals_grid = all_args[o : o + n_grid_in]
        o += n_grid_in
        t_grid_nz = all_args[o : o + n_nz_grid]
        o += n_nz_grid
        primals_aux = all_args[o : o + n_aux]
        o += n_aux
        t_aux_nz = all_args[o:]

        fn_partial, diff_primals = _make_partial(
            fn, list(i_arg) + list(primals_grid) + list(primals_aux), active_jaxpr_idx
        )
        _, t_out = jax.jvp(fn_partial, diff_primals, tuple(t_grid_nz) + tuple(t_aux_nz))
        return t_out

    exs = (
        [_sds(invars[0])]
        + [_sds(invars[1 + g]) for g in range(n_grid_in)]
        + [_sds(invars[1 + g]) for g in nonzero_grid_idx]
        + [_sds(invars[1 + n_grid_in + k]) for k in range(n_aux)]
        + [_sds(invars[1 + n_grid_in + k]) for k in nonzero_aux_idx]
    )
    return jax.make_jaxpr(jvp_fn)(*exs)


def derive_transpose_jaxpr(
    jaxpr,
    n_grid_in,
    n_grid_out,
    fft_lengths,
    *,
    needed_grid_idx=None,
    needed_aux_idx=None,
    nonzero_ct_idx=None,
):
    # DFT normalisation: divide cotangents by N before transposing, multiply results by N
    # so that the unitary DFT convention is preserved through the transpose bind.
    invars = jaxpr.jaxpr.invars
    outvars = jaxpr.jaxpr.outvars
    n_aux = len(invars) - 1 - n_grid_in
    n_outputs = len(outvars)

    if needed_grid_idx is None:
        needed_grid_idx = list(range(n_grid_in))
    if needed_aux_idx is None:
        needed_aux_idx = list(range(n_aux))
    if nonzero_ct_idx is None:
        nonzero_ct_idx = list(range(n_outputs))

    defined_grid_idx = [g for g in range(n_grid_in) if g not in needed_grid_idx]
    defined_aux_idx = [k for k in range(n_aux) if k not in needed_aux_idx]
    nz_ct_grid = [j for j in nonzero_ct_idx if j < n_grid_out]
    nz_ct_aux = [j for j in nonzero_ct_idx if j >= n_grid_out]

    all_needed_jaxpr = [1 + g for g in needed_grid_idx] + [
        1 + n_grid_in + k for k in needed_aux_idx
    ]
    fn = jaxpr_to_fn(jaxpr)
    N = float(math.prod(fft_lengths))

    zero_consts = tuple(
        np.zeros(invars[k].aval.shape, dtype=invars[k].aval.dtype) for k in all_needed_jaxpr
    )

    def transpose_fn(*all_args):
        o = 1
        i_arg = all_args[0]
        ct_grid = all_args[o : o + len(nz_ct_grid)]
        o += len(nz_ct_grid)
        def_grid = all_args[o : o + len(defined_grid_idx)]
        o += len(defined_grid_idx)
        ct_aux = all_args[o : o + len(nz_ct_aux)]
        o += len(nz_ct_aux)
        def_aux = all_args[o:]

        ct_grid_pre = tuple(ct / N for ct in ct_grid)

        all_inputs = [None] * (1 + n_grid_in + n_aux)
        all_inputs[0] = i_arg
        for ii, g in enumerate(defined_grid_idx):
            all_inputs[1 + g] = def_grid[ii]
        for ii, k in enumerate(defined_aux_idx):
            all_inputs[1 + n_grid_in + k] = def_aux[ii]
        for ii, k in enumerate(all_needed_jaxpr):
            all_inputs[k] = zero_consts[ii]

        all_cts = [jnp.zeros(v.aval.shape, v.aval.dtype) for v in outvars]
        for ii, j in enumerate(nz_ct_grid):
            all_cts[j] = ct_grid_pre[ii]
        for ii, j in enumerate(nz_ct_aux):
            all_cts[j] = ct_aux[ii]

        fn_partial, diff_args = _make_partial(fn, all_inputs, all_needed_jaxpr)
        ct_result = jax.linear_transpose(fn_partial, *diff_args)(tuple(all_cts))

        n_ng = len(needed_grid_idx)
        ct_grid_out = tuple(ct * N for ct in ct_result[:n_ng])

        return ct_grid_out + tuple(ct_result[n_ng:])

    exs = (
        [_sds(invars[0])]
        + [_sds(outvars[j]) for j in nz_ct_grid]
        + [_sds(invars[1 + g]) for g in defined_grid_idx]
        + [_sds(outvars[j]) for j in nz_ct_aux]
        + [_sds(invars[1 + n_grid_in + k]) for k in defined_aux_idx]
    )
    return jax.make_jaxpr(transpose_fn)(*exs)
