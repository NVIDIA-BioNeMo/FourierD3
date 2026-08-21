# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from fourierd3._engine.jax_ops.spectral_map.primitive import spectral_map_p
from fourierd3._engine.tracing.capture import extract_captures


def _is_batch_shape(x):
    return isinstance(x, tuple) and all(isinstance(e, int) for e in x)


@eqx.filter_jit
def _bind(grid_leaves, aux_leaves, **static_kw):
    return spectral_map_p.bind(*grid_leaves, *aux_leaves, **static_kw)


def spectral_map(
    fn,
    grid,
    aux=None,
    *,
    is_hermitian=False,
    sign=-1,
    num_batch_axes=None,
    output_shape=None,
    _force_reference=False,
):
    for s in jax.tree.leaves(sign):
        if s not in (-1, 1):
            raise ValueError(f"sign leaves must be -1 or +1, got {s}")

    grid_leaves, grid_treedef = jax.tree.flatten(grid)
    n_grid = len(grid_leaves)

    min_ndim = min(leaf.ndim for leaf in grid_leaves)
    assert min_ndim >= 3, f"grid leaves must be at least 3D, got {min_ndim}D"
    if num_batch_axes is None:
        num_batch_axes = min_ndim - 3
    spatial_slice = slice(num_batch_axes, num_batch_axes + 3)
    fft_lengths = tuple(int(s) for s in grid_leaves[0].shape[spatial_slice])
    grid_inner_shapes = tuple(
        tuple(int(s) for s in leaf.shape[num_batch_axes + 3 :]) for leaf in grid_leaves
    )
    for leaf in grid_leaves:
        assert leaf.ndim >= num_batch_axes + 3, (
            f"grid leaf ndim {leaf.ndim} < num_batch_axes({num_batch_axes}) + 3"
        )
        assert tuple(int(s) for s in leaf.shape[spatial_slice]) == fft_lengths, (
            "all grid leaves must have the same spatial shape"
        )

    aux_leaves, aux_treedef = jax.tree.flatten(aux)

    i_ex = jax.ShapeDtypeStruct((3,), jnp.int32)
    grid_ft_exs = [
        jax.ShapeDtypeStruct(grid_inner_shapes[j], jnp.result_type(leaf, 1j))
        for j, leaf in enumerate(grid_leaves)
    ]
    aux_inner_exs = [
        jax.ShapeDtypeStruct(leaf.shape[num_batch_axes:], leaf.dtype) for leaf in aux_leaves
    ]

    grid_ft_tree = jax.tree.unflatten(grid_treedef, grid_ft_exs)
    aux_tree = jax.tree.unflatten(aux_treedef, aux_inner_exs)
    grid_result_ex, aux_result_ex = jax.eval_shape(fn, i_ex, grid_ft_tree, aux_tree)

    grid_result_leaves_ex = jax.tree.leaves(grid_result_ex)
    aux_result_leaves_ex = jax.tree.leaves(aux_result_ex)
    grid_result_treedef = jax.tree.structure(grid_result_ex)
    aux_result_treedef = jax.tree.structure(aux_result_ex)
    n_grid_out = len(grid_result_leaves_ex)
    n_aux_out = len(aux_result_leaves_ex)

    def fn_flat(i, *flat_args):
        gft = jax.tree.unflatten(grid_treedef, flat_args[:n_grid])
        aux_t = jax.tree.unflatten(aux_treedef, flat_args[n_grid:])
        grid_res, aux_res = fn(i, gft, aux_t)
        grid_out = tuple(jax.tree.leaves(grid_res))
        aux_out = tuple(jax.tree.leaves(aux_res))
        return grid_out + aux_out

    closed_jaxpr = jax.make_jaxpr(fn_flat)(i_ex, *grid_ft_exs, *aux_inner_exs)

    closed_jaxpr, captures = extract_captures(closed_jaxpr, on_the_right=True)
    cap_leaves = jax.tree.leaves(captures)
    if cap_leaves:
        aux_leaves = list(aux_leaves) + [
            c.reshape((1,) * num_batch_axes + c.shape) for c in cap_leaves
        ]
    n_aux = len(aux_leaves)

    batch_shape = (1,) * num_batch_axes
    for leaf in grid_leaves + aux_leaves:
        bs = tuple(int(s) for s in leaf.shape[:num_batch_axes])
        batch_shape = tuple(max(a, b) for a, b in zip(batch_shape, bs, strict=True))

    ic = tuple([(-1,) * num_batch_axes] * (n_grid + n_aux + n_grid_out + n_aux_out))

    if output_shape is not None:
        broadcast_tree = jax.tree.broadcast(
            output_shape, (grid_result_ex, aux_result_ex), is_leaf=_is_batch_shape
        )
        all_batch_shapes = jax.tree.leaves(broadcast_tree, is_leaf=_is_batch_shape)
        grid_batch_shapes = all_batch_shapes[:n_grid_out]
        aux_batch_shapes = all_batch_shapes[n_grid_out:]
    else:
        grid_batch_shapes = [batch_shape] * n_grid_out
        aux_batch_shapes = [batch_shape] * n_aux_out

    all_real = all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in grid_leaves)

    def _grid_out_dtype(ex):
        return jnp.finfo(ex.dtype).dtype if is_hermitian and all_real else ex.dtype

    outputs_shape_dtype = tuple(
        [
            jax.ShapeDtypeStruct(
                grid_batch_shapes[j] + fft_lengths + grid_result_leaves_ex[j].shape,
                _grid_out_dtype(ex),
            )
            for j, ex in enumerate(grid_result_leaves_ex)
        ]
        + [
            jax.ShapeDtypeStruct(aux_batch_shapes[j] + ex.shape, ex.dtype)
            for j, ex in enumerate(aux_result_leaves_ex)
        ]
    )

    _sign_struct = (tuple(range(n_grid)), tuple(range(n_grid_out)))
    input_signs, output_signs = jax.tree.broadcast(sign, _sign_struct)

    flat_results = _bind(
        grid_leaves,
        aux_leaves,
        jaxpr=closed_jaxpr,
        n_grid_in=n_grid,
        n_grid_out=n_grid_out,
        fft_lengths=fft_lengths,
        is_hermitian=is_hermitian,
        input_signs=input_signs,
        output_signs=output_signs,
        index_configuration=ic,
        outputs_shape_dtype=outputs_shape_dtype,
        _force_reference=_force_reference,
    )

    grid_out = jax.tree.unflatten(grid_result_treedef, flat_results[:n_grid_out])
    aux_out = jax.tree.unflatten(aux_result_treedef, flat_results[n_grid_out:])
    return grid_out, aux_out
