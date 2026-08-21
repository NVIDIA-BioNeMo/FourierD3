# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax._src.core import ClosedJaxpr as _ClosedJaxpr
from jax.extend import core as jex_core
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe

from fourierd3._engine.indexing import (
    _batch_rule,
    _compute_batch_sizes,
    _make_indexing_fn,
)
from fourierd3._engine.jax_ops.periodic_scatter.derived_jaxprs import jaxpr_to_fn
from fourierd3._engine.jax_ops.periodic_scatter.layout import (
    IndexLayout,
    ScatterOperands,
    support_3d,
)
from fourierd3._engine.jax_ops.periodic_scatter.lowering import lower_periodic_scatter

periodic_scatter_p = jex_core.Primitive("periodic_scatter")
periodic_scatter_p.multiple_results = True


def _assert_index_dtypes(cell_idx, index_buffers):
    assert cell_idx.dtype == jnp.int32, f"cell_idx must be int32, got {cell_idx.dtype}"
    for i, ib in enumerate(index_buffers):
        assert ib.dtype == jnp.int32, f"index_buffers[{i}] must be int32, got {ib.dtype}"


def _buf_batch_extents(parsed: ScatterOperands, outputs_shape_dtype, num_batch_axes):
    ext = []
    ext.append(tuple(int(s) for s in parsed.cell_idx.shape[:num_batch_axes]))
    for g in parsed.grid_in:
        ext.append(tuple(int(s) for s in g.shape[:num_batch_axes]))
    for f in parsed.nongrid_in:
        ext.append(tuple(int(s) for s in f.shape[:num_batch_axes]))
    for o in outputs_shape_dtype:
        ext.append(tuple(int(s) for s in o.shape[:num_batch_axes]))
    for ib in parsed.index_buffers:
        ext.append(tuple(int(s) for s in ib.shape[:num_batch_axes]))
    return tuple(ext)


def _abstract_eval(cell_idx, *args, n_backend_arrays, **kwargs):
    outputs_shape_dtype = kwargs["outputs_shape_dtype"]
    return [jax_core.ShapedArray(o.shape, o.dtype) for o in outputs_shape_dtype]


periodic_scatter_p.def_abstract_eval(_abstract_eval)


def _ref_impl(
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
    cuda_opts=(),
    _force_reference=False,
):
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
    return _ref_impl_impl(
        parsed,
        layout,
        jaxpr=jaxpr,
        support=support,
        grid_shape=grid_shape,
        outputs_shape_dtype=outputs_shape_dtype,
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )


def _ref_impl_impl(
    parsed: ScatterOperands,
    layout: IndexLayout,
    *,
    jaxpr,
    support,
    grid_shape,
    outputs_shape_dtype,
    cuda_opts=(),
    _force_reference=False,
):
    _assert_index_dtypes(parsed.cell_idx, parsed.index_buffers)
    support_pts = support_3d(support)  # expand 1D separable -> 3D
    S = len(support_pts)
    support_arr = jnp.array(support_pts, dtype=jnp.int32)
    grid_sizes = jnp.array(grid_shape, dtype=jnp.int32)
    n_gin = layout.n_grid_in
    n_gout = layout.n_grid_out
    n_ngout = layout.n_nongrid_out
    nba = len(layout.ic[0])  # num_batch_axes

    def buf_shape(buf_idx):
        if buf_idx == 0:
            return parsed.cell_idx.shape[:nba]
        if buf_idx < layout.nongrid_in_offset:
            return parsed.grid_in[buf_idx - layout.grid_in_offset].shape[:nba]
        if buf_idx < layout.grid_out_offset:
            return parsed.nongrid_in[buf_idx - layout.nongrid_in_offset].shape[:nba]
        if buf_idx < layout.idx_offset:
            return outputs_shape_dtype[buf_idx - layout.grid_out_offset].shape[:nba]
        return parsed.index_buffers[buf_idx - layout.idx_offset].shape[:nba]

    batch_shape = _compute_batch_sizes(layout.ic, buf_shape, nba)
    total_batch = math.prod(batch_shape)
    indexing = _make_indexing_fn(
        layout.ic, buf_shape, list(parsed.index_buffers), layout.idx_offset, nba
    )

    def bc_flat(x):
        inner = x.shape[nba:]
        return jnp.broadcast_to(x, batch_shape + inner).reshape((total_batch,) + inner)

    def bc_idx(buf_idx):
        return tuple(
            jnp.broadcast_to(i, batch_shape).reshape((total_batch,)) for i in indexing(buf_idx)
        )

    ci_flat = bc_flat(parsed.cell_idx[indexing(0)])

    V_fn = jaxpr_to_fn(jaxpr)
    V_vmapped = jax.vmap(V_fn, in_axes=(None, None) + (0,) * (n_gin + layout.n_nongrid_in))

    results = [jnp.zeros(o.shape, o.dtype) for o in outputs_shape_dtype]

    for s in range(S):
        sup = support_arr[s]
        ix = (ci_flat[:, 0] + sup[0]) % grid_sizes[0]
        iy = (ci_flat[:, 1] + sup[1]) % grid_sizes[1]
        iz = (ci_flat[:, 2] + sup[2]) % grid_sizes[2]

        grid_vals = [
            parsed.grid_in[g][bc_idx(layout.grid_in_offset + g) + (ix, iy, iz)]
            for g in range(n_gin)
        ]

        ng_vals = [
            bc_flat(parsed.nongrid_in[k][indexing(layout.nongrid_in_offset + k)])
            for k in range(layout.n_nongrid_in)
        ]

        step = V_vmapped(jnp.int32(s), sup, *grid_vals, *ng_vals)

        for j in range(n_gout):
            out_idx = bc_idx(layout.grid_out_offset + j) + (ix, iy, iz)
            results[j] = results[j].at[out_idx].add(step[j])

        for k in range(n_ngout):
            j_out = n_gout + k
            out_idx = bc_idx(layout.nongrid_out_offset + k)
            results[j_out] = results[j_out].at[out_idx].add(step[j_out])

    return results


periodic_scatter_p.def_impl(_ref_impl)

mlir.register_lowering(periodic_scatter_p, mlir.lower_fun(_ref_impl, multiple_results=True), None)


def _cuda_lowering(
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
    cuda_opts=(),
):
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
    num_batch_axes = len(layout.ic[0])
    _assert_index_dtypes(parsed.cell_idx, parsed.index_buffers)
    buf_ext = _buf_batch_extents(parsed, outputs_shape_dtype, num_batch_axes)
    batch_sizes = _compute_batch_sizes(layout.ic, lambda i: buf_ext[i], num_batch_axes)

    return lower_periodic_scatter(
        parsed,
        layout,
        jaxpr,
        support,
        batch_sizes=batch_sizes,
        buf_batch_extents=buf_ext,
        grid_shape=grid_shape,
        outputs_shape_dtype=outputs_shape_dtype,
        n_backend_arrays=parsed.n_backend_arrays,
        cuda_opts=dict(cuda_opts) if cuda_opts else {},
    )


def _cuda_or_ref(cell_idx, *args, **kw):
    if kw.pop("_force_reference", False):
        return _ref_impl(cell_idx, *args, **kw)
    return _cuda_lowering(cell_idx, *args, **kw)


mlir.register_lowering(
    periodic_scatter_p,
    mlir.lower_fun(_cuda_or_ref, multiple_results=True),
    "cuda",
)


def _grid_batch_rule(batched_args, batch_dims, *, n_backend_arrays, **kw):
    if n_backend_arrays > 0:
        backend_dims = batch_dims[len(batched_args) - n_backend_arrays :]
        if any(d is not None for d in backend_dims):
            raise NotImplementedError("vmap over backend arrays is not supported")
    return _batch_rule(
        periodic_scatter_p, batched_args, batch_dims, n_backend_arrays=n_backend_arrays, **kw
    )


batching.primitive_batchers[periodic_scatter_p] = _grid_batch_rule

from fourierd3._engine.jax_ops.periodic_scatter.autodiff import _jvp, _transpose  # noqa: E402

ad.primitive_jvps[periodic_scatter_p] = _jvp
ad.primitive_transposes[periodic_scatter_p] = _transpose


def _dce(used_outs, eqn):
    if not any(used_outs):
        return [False] * len(eqn.invars), None
    if all(used_outs):
        return [True] * len(eqn.invars), eqn

    params = eqn.params
    jaxpr = params["jaxpr"]
    n_grid_in = params["n_grid_in"]
    n_grid_out = params["n_grid_out"]
    n_backend_arrays = params["n_backend_arrays"]
    n_total_outputs = len(params["outputs_shape_dtype"])
    # First two jaxpr invars are support_index + support_offset (not positional args)
    n_nongrid_in = len(jaxpr.jaxpr.invars) - 2 - n_grid_in
    n_index = len(eqn.invars) - 1 - n_grid_in - n_nongrid_in - n_backend_arrays
    layout = IndexLayout(
        ic=tuple(params["index_configuration"]),
        n_grid_in=n_grid_in,
        n_nongrid_in=n_nongrid_in,
        n_grid_out=n_grid_out,
        n_nongrid_out=n_total_outputs - n_grid_out,
        n_index=n_index,
    )
    return _dce_impl(used_outs, layout, eqn.invars, eqn.outvars, params, eqn)


def _dce_impl(used_outs, layout: IndexLayout, eqn_invars, eqn_outvars, params, eqn):
    jaxpr = params["jaxpr"]
    old_inner = jaxpr.jaxpr

    new_inner, used = pe.dce_jaxpr(old_inner, list(used_outs))
    n_cv = len(old_inner.constvars)

    # Preserve support_index and support_offset (first two jaxpr invars): the
    # primitive interface always requires them even if the kernel body doesn't
    # reference them.
    used = list(used)
    new_invars_inner = list(new_inner.invars)
    sup_vars = [old_inner.invars[0], old_inner.invars[1]]
    sup_ids = {id(v) for v in sup_vars}
    remaining = [v for v in new_invars_inner if id(v) not in sup_ids]
    for k in range(2):
        used[n_cv + k] = True
    new_inner = new_inner.replace(invars=sup_vars + remaining)

    new_consts = [c for c, u in zip(jaxpr.consts, used[:n_cv], strict=False) if u]
    new_jaxpr = _ClosedJaxpr(new_inner, new_consts)

    used_jaxpr_invars = used[n_cv:]

    n_grid_in = layout.n_grid_in
    n_grid_out = layout.n_grid_out
    n_index = layout.n_index
    n_backend_arrays = params["n_backend_arrays"]
    osd = params["outputs_shape_dtype"]

    used_grid_in = list(used_jaxpr_invars[2 : 2 + n_grid_in])
    used_nongrid_in = list(used_jaxpr_invars[2 + n_grid_in :])

    used_positional = (
        [True] + used_grid_in + used_nongrid_in + [True] * n_index + [True] * n_backend_arrays
    )

    kept_grid_in_ics = [ics for ics, u in zip(layout.grid_in, used_grid_in, strict=True) if u]
    kept_nongrid_in_ics = [
        ics for ics, u in zip(layout.nongrid_in, used_nongrid_in, strict=True) if u
    ]
    kept_grid_out_ics = [
        ics for ics, u in zip(layout.grid_out, used_outs[:n_grid_out], strict=True) if u
    ]
    kept_nongrid_out_ics = [
        ics for ics, u in zip(layout.nongrid_out, used_outs[n_grid_out:], strict=True) if u
    ]
    idx_ics = list(layout.idx)

    new_n_grid_in = sum(used_grid_in)
    new_n_grid_out = sum(used_outs[:n_grid_out])

    new_ic = tuple(
        [layout.cell_idx]
        + kept_grid_in_ics
        + kept_nongrid_in_ics
        + kept_grid_out_ics
        + kept_nongrid_out_ics
        + idx_ics
    )
    new_osd = tuple(o for o, u in zip(osd, used_outs, strict=False) if u)

    new_params = {
        **params,
        "jaxpr": new_jaxpr,
        "n_grid_in": new_n_grid_in,
        "n_grid_out": new_n_grid_out,
        "index_configuration": new_ic,
        "outputs_shape_dtype": new_osd,
    }

    new_invars = [v for v, u in zip(eqn_invars, used_positional, strict=False) if u]
    new_outvars = [v for v, u in zip(eqn_outvars, used_outs, strict=False) if u]
    new_eqn = eqn.replace(invars=new_invars, outvars=new_outvars, params=new_params)

    return used_positional, new_eqn


pe.dce_rules[periodic_scatter_p] = _dce
