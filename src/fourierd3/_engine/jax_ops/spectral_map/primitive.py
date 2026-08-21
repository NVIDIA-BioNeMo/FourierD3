# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax._src.core import ClosedJaxpr as _ClosedJaxpr
from jax.extend import core as jex_core
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe

from fourierd3._engine import _extension
from fourierd3._engine.indexing import _batch_rule, _compute_batch_sizes
from fourierd3._engine.jax_ops.spectral_map.lowering import lower_spectral_map
from fourierd3._engine.tracing.differentiate import jaxpr_to_fn

spectral_map_p = jex_core.Primitive("spectral_map")
spectral_map_p.multiple_results = True


def _abstract_eval(*args, outputs_shape_dtype, **kw):
    return [jax_core.ShapedArray(o.shape, o.dtype) for o in outputs_shape_dtype]


spectral_map_p.def_abstract_eval(_abstract_eval)


def _fft_fns(sign, fft_lengths, spatial_axes=(-3, -2, -1)):
    N = math.prod(fft_lengths)
    axes = tuple(spatial_axes)
    if sign == 1:
        return (
            lambda g: jnp.fft.ifftn(g, axes=axes) * N,
            lambda g: jnp.fft.fftn(g, axes=axes) / N,
        )
    return (
        lambda g: jnp.fft.fftn(g, axes=axes),
        lambda g: jnp.fft.ifftn(g, axes=axes),
    )


def _rfft_fns(sign, fft_lengths, spatial_axes=(-3, -2, -1)):
    axes = tuple(spatial_axes)
    if sign == 1:
        return (
            lambda g: jnp.conj(jnp.fft.rfftn(g, axes=axes)),
            lambda g: jnp.fft.irfftn(jnp.conj(g), s=fft_lengths, axes=axes),
        )
    return (
        lambda g: jnp.fft.rfftn(g, axes=axes),
        lambda g: jnp.fft.irfftn(g, s=fft_lengths, axes=axes),
    )


def _ref_impl(
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
    _force_reference,
):
    ic = index_configuration
    num_batch_axes = len(ic[0])
    grids = list(args[:n_grid_in])
    auxs = list(args[n_grid_in:])
    n_aux = len(auxs)
    n_outputs = len(outputs_shape_dtype)

    nx, ny, nz = fft_lengths
    spatial_axes = tuple(range(num_batch_axes, num_batch_axes + 3))

    use_rfft = is_hermitian and all(
        jnp.issubdtype(grids[j].dtype, jnp.floating) for j in range(n_grid_in)
    )

    if use_rfft:
        nzh = nz // 2 + 1
        freq_shape = (nx, ny, nzh)
    else:
        freq_shape = (nx, ny, nz)
    total_freq = math.prod(freq_shape)

    def buf_shape(buf_idx):
        if buf_idx < n_grid_in:
            return grids[buf_idx].shape[:num_batch_axes]
        elif buf_idx < n_grid_in + n_aux:
            return auxs[buf_idx - n_grid_in].shape[:num_batch_axes]
        else:
            return outputs_shape_dtype[buf_idx - n_grid_in - n_aux].shape[:num_batch_axes]

    batch_sizes = _compute_batch_sizes(ic, buf_shape, num_batch_axes)
    total_batch = math.prod(batch_sizes) if batch_sizes else 1
    n_flat = total_batch * total_freq

    grids_ft = [None] * n_grid_in
    fwd_groups: dict[int, list[int]] = {}
    for j in range(n_grid_in):
        fwd_groups.setdefault(input_signs[j], []).append(j)
    for sign, idxs in fwd_groups.items():
        fft_fn = (_rfft_fns if use_rfft else _fft_fns)(sign, fft_lengths, spatial_axes)[0]
        inner_shapes = [grids[j].shape[num_batch_axes + 3 :] for j in idxs]
        inner_sizes = [math.prod(s) for s in inner_shapes]
        parts = [
            jnp.broadcast_to(
                grids[j].reshape(*grids[j].shape[:num_batch_axes], *fft_lengths, sz),
                batch_sizes + fft_lengths + (sz,),
            )
            for j, sz in zip(idxs, inner_sizes, strict=True)
        ]
        cat_ft = fft_fn(jnp.concatenate(parts, axis=-1))
        off = 0
        for j, inner_j, sz in zip(idxs, inner_shapes, inner_sizes, strict=True):
            g_ft = cat_ft[..., off : off + sz].reshape(*batch_sizes, *freq_shape, *inner_j)
            grids_ft[j] = g_ft.reshape(n_flat, *inner_j)
            off += sz

    ix = jnp.round(jnp.fft.fftfreq(nx) * nx).astype(jnp.int32)
    iy = jnp.round(jnp.fft.fftfreq(ny) * ny).astype(jnp.int32)
    if use_rfft:
        iz = jnp.round(jnp.fft.rfftfreq(nz) * nz).astype(jnp.int32)
    else:
        iz = jnp.round(jnp.fft.fftfreq(nz) * nz).astype(jnp.int32)
    i_mesh = jnp.stack(jnp.meshgrid(ix, iy, iz, indexing="ij"), axis=-1)
    i_flat = jnp.broadcast_to(i_mesh, batch_sizes + i_mesh.shape).reshape(n_flat, 3)

    auxs_flat = []
    for a in auxs:
        a_bc = jnp.broadcast_to(a, batch_sizes + a.shape[num_batch_axes:])
        inner = a_bc.shape[len(batch_sizes) :]
        auxs_flat.append(jnp.repeat(a_bc.reshape(total_batch, *inner), total_freq, axis=0))

    fn = jaxpr_to_fn(jaxpr)
    outs_flat = jax.vmap(lambda *a: fn(*a))(i_flat, *grids_ft, *auxs_flat)

    grid_ifft: dict[int, jax.Array] = {}
    if n_grid_out > 0:
        inv_groups: dict[int, list[int]] = {}
        for j in range(n_grid_out):
            inv_groups.setdefault(output_signs[j], []).append(j)
        for sign, idxs in inv_groups.items():
            ifft_fn = (_rfft_fns if use_rfft else _fft_fns)(sign, fft_lengths, spatial_axes)[1]
            inner_shapes = [outs_flat[j].shape[1:] for j in idxs]
            inner_sizes = [math.prod(s) for s in inner_shapes]
            parts = [
                outs_flat[j].reshape(*batch_sizes, *freq_shape, sz)
                for j, sz in zip(idxs, inner_sizes, strict=True)
            ]
            cat_ifft = ifft_fn(jnp.concatenate(parts, axis=-1))
            off = 0
            for j, inner_j, sz in zip(idxs, inner_shapes, inner_sizes, strict=True):
                grid_ifft[j] = cat_ifft[..., off : off + sz].reshape(
                    *batch_sizes, *fft_lengths, *inner_j
                )
                off += sz

    if use_rfft:
        rfft_w = jnp.ones(nzh)
        if nz > 1:
            end = nzh if nz % 2 == 1 else nzh - 1
            rfft_w = rfft_w.at[1:end].set(2.0)

    results = []
    for j in range(n_outputs):
        osd = outputs_shape_dtype[j]
        if j < n_grid_out:
            out_j = grid_ifft[j]
            if jnp.issubdtype(osd.dtype, jnp.floating):
                out_j = out_j.real
        else:
            out_j = outs_flat[j]
            inner = out_j.shape[1:]
            out_j = out_j.reshape(*batch_sizes, *freq_shape, *inner)
            freq_axes = tuple(range(len(batch_sizes), len(batch_sizes) + 3))
            if use_rfft:
                w_shape = [1] * out_j.ndim
                w_shape[len(batch_sizes) + 2] = nzh
                w = rfft_w.reshape(w_shape)
                out_j = (out_j.real * w).sum(axis=freq_axes).astype(osd.dtype)
            else:
                out_j = out_j.sum(axis=freq_axes)
        reduce = tuple(a for a in range(num_batch_axes) if osd.shape[a] < batch_sizes[a])
        if reduce:
            out_j = out_j.sum(axis=reduce, keepdims=True)
        results.append(out_j)

    return results


spectral_map_p.def_impl(_ref_impl)

mlir.register_lowering(spectral_map_p, mlir.lower_fun(_ref_impl, multiple_results=True), None)


def _cuda_with_fallback(*args, **kw):
    kw = dict(kw)
    force = kw.pop("_force_reference")
    if force:
        return _ref_impl(*args, **kw, _force_reference=force)
    try:
        return lower_spectral_map(*args, **kw)
    except (NotImplementedError, _extension.KernelInfeasibleError) as exc:
        import warnings

        warnings.warn(
            f"spectral_map CUDA lowering unavailable, falling back to reference "
            f"implementation: {exc}",
            stacklevel=2,
        )
        return _ref_impl(*args, **kw, _force_reference=force)


mlir.register_lowering(
    spectral_map_p,
    mlir.lower_fun(_cuda_with_fallback, multiple_results=True),
    "cuda",
)


batching.primitive_batchers[spectral_map_p] = functools.partial(_batch_rule, spectral_map_p)


from fourierd3._engine.jax_ops.spectral_map.autodiff import _jvp, _transpose  # noqa: E402

ad.primitive_jvps[spectral_map_p] = _jvp
ad.primitive_transposes[spectral_map_p] = _transpose


def _dce(used_outs, eqn):
    if not any(used_outs):
        return [False] * len(eqn.invars), None

    jaxpr = eqn.params["jaxpr"]
    old_inner = jaxpr.jaxpr

    new_inner, used_invars = pe.dce_jaxpr(old_inner, list(used_outs))
    used_invars = list(used_invars)

    # freq_idx (first invar) must always be kept — downstream AD rules and
    # CUDA lowering assume invars[0] is the frequency index.
    if not used_invars[0]:
        freq_var = old_inner.invars[0]
        new_inner = new_inner.replace(invars=[freq_var, *new_inner.invars])
        used_invars[0] = True

    old_cv_map = {id(v): c for v, c in zip(old_inner.constvars, jaxpr.consts, strict=True)}
    new_consts = [old_cv_map[id(v)] for v in new_inner.constvars]

    if all(used_invars) and all(used_outs):
        new_jaxpr = _ClosedJaxpr(new_inner, new_consts)
        new_eqn = eqn.replace(params={**eqn.params, "jaxpr": new_jaxpr})
        return [True] * len(eqn.invars), new_eqn

    new_jaxpr = _ClosedJaxpr(new_inner, new_consts)

    n_grid_in = eqn.params["n_grid_in"]
    n_grid_out = eqn.params["n_grid_out"]
    ic = eqn.params["index_configuration"]
    osd = eqn.params["outputs_shape_dtype"]
    input_signs = eqn.params["input_signs"]
    output_signs = eqn.params["output_signs"]
    n_total_outputs = len(osd)
    n_aux = len(eqn.invars) - n_grid_in

    used_grid = list(used_invars[1 : 1 + n_grid_in])
    used_aux = list(used_invars[1 + n_grid_in :])
    used_positional = used_grid + used_aux

    new_n_grid_in = sum(used_grid)
    new_n_grid_out = sum(used_outs[:n_grid_out])

    kept_grid_ics = [ic[g] for g in range(n_grid_in) if used_grid[g]]
    kept_aux_ics = [ic[n_grid_in + k] for k in range(n_aux) if used_aux[k]]
    kept_out_ics = [ic[n_grid_in + n_aux + j] for j in range(n_total_outputs) if used_outs[j]]

    new_ic = tuple(kept_grid_ics + kept_aux_ics + kept_out_ics)
    new_osd = tuple(o for o, u in zip(osd, used_outs, strict=False) if u)
    new_input_signs = tuple(s for s, u in zip(input_signs, used_grid, strict=False) if u)
    new_output_signs = tuple(
        s for s, u in zip(output_signs, used_outs[:n_grid_out], strict=False) if u
    )

    new_params = {
        **eqn.params,
        "jaxpr": new_jaxpr,
        "n_grid_in": new_n_grid_in,
        "n_grid_out": new_n_grid_out,
        "input_signs": new_input_signs,
        "output_signs": new_output_signs,
        "index_configuration": new_ic,
        "outputs_shape_dtype": new_osd,
    }

    new_invars = [v for v, u in zip(eqn.invars, used_positional, strict=False) if u]
    new_outvars = [v for v, u in zip(eqn.outvars, used_outs, strict=False) if u]
    new_eqn = eqn.replace(invars=new_invars, outvars=new_outvars, params=new_params)

    return used_positional, new_eqn


pe.dce_rules[spectral_map_p] = _dce
