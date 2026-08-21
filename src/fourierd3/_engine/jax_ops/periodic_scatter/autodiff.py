# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

import jax
from jax.interpreters import ad

from fourierd3._engine.indexing import _assemble_cotangents
from fourierd3._engine.jax_ops.periodic_scatter.derived_jaxprs import (
    derive_jvp_jaxpr,
    derive_transpose_jaxpr,
)
from fourierd3._engine.jax_ops.periodic_scatter.layout import (
    IndexLayout,
    ScatterOperands,
    ScatterTangents,
)
from fourierd3._engine.jax_ops.periodic_scatter.primitive import periodic_scatter_p


def _jvp(
    primals,
    tangents,
    *,
    jaxpr,
    support,
    grid_shape,
    n_grid_in,
    n_grid_out,
    index_configuration,
    outputs_shape_dtype,
    n_backend_arrays,
    cuda_opts=(),
    _force_reference=False,
):
    parsed = ScatterOperands.parse(
        primals[0],
        *primals[1:],
        jaxpr=jaxpr,
        n_grid_in=n_grid_in,
        n_backend_arrays=n_backend_arrays,
    )
    n_total_outputs = len(outputs_shape_dtype)
    layout = IndexLayout(
        ic=tuple(index_configuration),
        n_grid_in=n_grid_in,
        n_nongrid_in=parsed.n_nongrid_in,
        n_grid_out=n_grid_out,
        n_nongrid_out=n_total_outputs - n_grid_out,
        n_index=parsed.n_index,
    )
    tans = ScatterTangents(
        d_grid_in=tuple(tangents[layout.grid_in_offset : layout.nongrid_in_offset]),
        d_nongrid_in=tuple(tangents[layout.nongrid_in_offset : layout.grid_out_offset]),
    )
    return _jvp_impl(
        parsed,
        layout,
        tans,
        jaxpr=jaxpr,
        support=support,
        grid_shape=grid_shape,
        outputs_shape_dtype=outputs_shape_dtype,
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )


def _jvp_impl(
    parsed: ScatterOperands,
    layout: IndexLayout,
    tans: ScatterTangents,
    *,
    jaxpr,
    support,
    grid_shape,
    outputs_shape_dtype,
    cuda_opts=(),
    _force_reference=False,
):
    n_grid_in = layout.n_grid_in
    n_nongrid_in = layout.n_nongrid_in
    n_grid_out = layout.n_grid_out

    primal_out = periodic_scatter_p.bind(
        *parsed.as_positional(),
        jaxpr=jaxpr,
        support=support,
        grid_shape=grid_shape,
        n_grid_in=n_grid_in,
        n_grid_out=n_grid_out,
        index_configuration=layout.ic,
        outputs_shape_dtype=outputs_shape_dtype,
        n_backend_arrays=parsed.n_backend_arrays,
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )

    nz_grid_in = [g for g in range(n_grid_in) if not isinstance(tans.d_grid_in[g], ad.Zero)]
    nz_nongrid_in = [
        k for k in range(n_nongrid_in) if not isinstance(tans.d_nongrid_in[k], ad.Zero)
    ]

    if not nz_grid_in and not nz_nongrid_in:
        return primal_out, [ad.Zero.from_primal_value(p) for p in primal_out]

    jvp_jaxpr = derive_jvp_jaxpr(jaxpr, n_grid_in, nz_grid_in, nz_nongrid_in)
    n_grid_in_jvp = n_grid_in + len(nz_grid_in)

    t_grid_in = tuple(tans.d_grid_in[g] for g in nz_grid_in)
    t_nongrid_in = tuple(tans.d_nongrid_in[k] for k in nz_nongrid_in)

    jvp_ic = tuple(
        [layout.cell_idx]
        + list(layout.grid_in)
        + [layout.grid_in[g] for g in nz_grid_in]
        + list(layout.nongrid_in)
        + [layout.nongrid_in[k] for k in nz_nongrid_in]
        + list(layout.out)
        + list(layout.idx)
    )

    jvp_args = dataclasses.replace(
        parsed,
        grid_in=parsed.grid_in + t_grid_in,
        nongrid_in=parsed.nongrid_in + t_nongrid_in,
        n_grid_in=n_grid_in_jvp,
        n_nongrid_in=n_nongrid_in + len(nz_nongrid_in),
    )

    tangent_out = periodic_scatter_p.bind(
        *jvp_args.as_positional(),
        jaxpr=jvp_jaxpr,
        support=support,
        grid_shape=grid_shape,
        n_grid_in=n_grid_in_jvp,
        n_grid_out=n_grid_out,
        index_configuration=jvp_ic,
        outputs_shape_dtype=outputs_shape_dtype,
        n_backend_arrays=parsed.n_backend_arrays,
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )
    return primal_out, tangent_out


def _build_transpose_ic(
    layout: IndexLayout,
    nz_ct_grid,
    nz_ct_nongrid,
    def_grid_in,
    def_nongrid_in,
    undef_grid_in,
    undef_nongrid_in,
):
    # Cotangent inputs come from the primal's output slots (they share layout).
    ct_grid_in_ics = [layout.out[j] for j in nz_ct_grid]
    def_grid_in_ics = [layout.grid_in[g] for g in def_grid_in]
    ct_nongrid_in_ics = [layout.out[j] for j in nz_ct_nongrid]
    def_nongrid_in_ics = [layout.nongrid_in[k] for k in def_nongrid_in]
    needed_grid_in_ics = [layout.grid_in[g] for g in undef_grid_in]
    needed_nongrid_in_ics = [layout.nongrid_in[k] for k in undef_nongrid_in]
    idx_ics = list(layout.idx)
    return tuple(
        [layout.cell_idx]
        + ct_grid_in_ics
        + def_grid_in_ics
        + ct_nongrid_in_ics
        + def_nongrid_in_ics
        + needed_grid_in_ics
        + needed_nongrid_in_ics
        + idx_ics
    )


def _build_transpose_osd(grid_in_or_undef, nongrid_in_or_undef, undef_grid_in, undef_nongrid_in):
    grid_in_ct_osd = [
        jax.ShapeDtypeStruct(grid_in_or_undef[g].aval.shape, grid_in_or_undef[g].aval.dtype)
        for g in undef_grid_in
    ]
    nongrid_in_ct_osd = [
        jax.ShapeDtypeStruct(nongrid_in_or_undef[k].aval.shape, nongrid_in_or_undef[k].aval.dtype)
        for k in undef_nongrid_in
    ]
    return tuple(grid_in_ct_osd + nongrid_in_ct_osd)


def _transpose(
    cts,
    cell_idx,
    *args,
    jaxpr,
    support,
    grid_shape,
    n_grid_in,
    n_grid_out,
    index_configuration,
    outputs_shape_dtype,
    n_backend_arrays,
    cuda_opts,
    _force_reference=False,
):
    assert not ad.is_undefined_primal(cell_idx)
    parsed = ScatterOperands.parse(
        cell_idx,
        *args,
        jaxpr=jaxpr,
        n_grid_in=n_grid_in,
        n_backend_arrays=n_backend_arrays,
    )
    n_total_outputs = len(outputs_shape_dtype)
    layout = IndexLayout(
        ic=tuple(index_configuration),
        n_grid_in=n_grid_in,
        n_nongrid_in=parsed.n_nongrid_in,
        n_grid_out=n_grid_out,
        n_nongrid_out=n_total_outputs - n_grid_out,
        n_index=parsed.n_index,
    )
    del outputs_shape_dtype  # only needed to derive layout above
    return _transpose_impl(
        cts,
        parsed,
        layout,
        jaxpr=jaxpr,
        support=support,
        grid_shape=grid_shape,
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )


def _transpose_impl(
    cts,
    parsed: ScatterOperands,
    layout: IndexLayout,
    *,
    jaxpr,
    support,
    grid_shape,
    cuda_opts,
    _force_reference=False,
):
    n_grid_in = layout.n_grid_in
    n_nongrid_in = layout.n_nongrid_in
    n_grid_out = layout.n_grid_out
    n_index = layout.n_index
    n_backend_arrays = parsed.n_backend_arrays
    grid_in_or_undef = list(parsed.grid_in)
    nongrid_in_or_undef = list(parsed.nongrid_in)

    undef_grid_in = [g for g in range(n_grid_in) if ad.is_undefined_primal(grid_in_or_undef[g])]
    undef_nongrid_in = [
        k for k in range(n_nongrid_in) if ad.is_undefined_primal(nongrid_in_or_undef[k])
    ]
    def_grid_in = [g for g in range(n_grid_in) if g not in undef_grid_in]
    def_nongrid_in = [k for k in range(n_nongrid_in) if k not in undef_nongrid_in]

    if (not undef_grid_in and not undef_nongrid_in) or all(isinstance(c, ad.Zero) for c in cts):
        result = [None]  # cell_idx
        for g in range(n_grid_in):
            result.append(ad.Zero(grid_in_or_undef[g].aval) if g in undef_grid_in else None)
        for k in range(n_nongrid_in):
            result.append(ad.Zero(nongrid_in_or_undef[k].aval) if k in undef_nongrid_in else None)
        result.extend([None] * n_index)
        result.extend([None] * n_backend_arrays)
        return tuple(result)

    nonzero_ct_idx = [j for j, c in enumerate(cts) if not isinstance(c, ad.Zero)]
    nz_ct_grid = [j for j in nonzero_ct_idx if j < n_grid_out]
    nz_ct_nongrid = [j for j in nonzero_ct_idx if j >= n_grid_out]

    transpose_jaxpr = derive_transpose_jaxpr(
        jaxpr,
        n_grid_in,
        n_grid_out,
        needed_grid_idx=undef_grid_in,
        needed_nongrid_in_idx=undef_nongrid_in,
        nonzero_ct_idx=nonzero_ct_idx,
    )

    ct_grid_arrays = tuple(cts[j] for j in nz_ct_grid)
    def_grid_in_arrays = tuple(grid_in_or_undef[g] for g in def_grid_in)
    ct_nongrid_arrays = tuple(cts[j] for j in nz_ct_nongrid)
    def_nongrid_in_arrays = tuple(nongrid_in_or_undef[k] for k in def_nongrid_in)

    n_grid_in_tr = len(ct_grid_arrays) + len(def_grid_in_arrays)
    n_nongrid_in_tr = len(ct_nongrid_arrays) + len(def_nongrid_in_arrays)
    n_grid_out_tr = len(undef_grid_in)

    transpose_ic = _build_transpose_ic(
        layout,
        nz_ct_grid,
        nz_ct_nongrid,
        def_grid_in,
        def_nongrid_in,
        undef_grid_in,
        undef_nongrid_in,
    )
    transpose_osd = _build_transpose_osd(
        grid_in_or_undef,
        nongrid_in_or_undef,
        undef_grid_in,
        undef_nongrid_in,
    )

    tr_args = dataclasses.replace(
        parsed,
        grid_in=ct_grid_arrays + def_grid_in_arrays,
        nongrid_in=ct_nongrid_arrays + def_nongrid_in_arrays,
        n_grid_in=n_grid_in_tr,
        n_nongrid_in=n_nongrid_in_tr,
    )

    tr_results = periodic_scatter_p.bind(
        *tr_args.as_positional(),
        jaxpr=transpose_jaxpr,
        support=support,
        grid_shape=grid_shape,
        n_grid_in=n_grid_in_tr,
        n_grid_out=n_grid_out_tr,
        index_configuration=transpose_ic,
        outputs_shape_dtype=transpose_osd,
        n_backend_arrays=n_backend_arrays,
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )

    grid_in_cts = dict(zip(undef_grid_in, tr_results[:n_grid_out_tr], strict=False))
    nongrid_in_cts = dict(zip(undef_nongrid_in, tr_results[n_grid_out_tr:], strict=False))

    result = [None]  # cell_idx
    result.extend(_assemble_cotangents(grid_in_or_undef, grid_in_cts))
    result.extend(_assemble_cotangents(nongrid_in_or_undef, nongrid_in_cts))
    result.extend([None] * n_index)
    result.extend([None] * n_backend_arrays)
    return tuple(result)
