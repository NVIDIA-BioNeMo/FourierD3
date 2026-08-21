# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import ClosedJaxpr as _ClosedJaxpr

from fourierd3._engine.indexing import MultiAxisIndex
from fourierd3._engine.jax_ops.periodic_scatter.primitive import (
    _compute_batch_sizes,
    periodic_scatter_p,
)
from fourierd3._engine.tracing.capture import extract_captures

IndexSpec = MultiAxisIndex | jax.Array | np.ndarray | None


def _is_index_leaf(x):
    return x is None or isinstance(x, (MultiAxisIndex, jax.Array, np.ndarray))


def _is_batch_shape_leaf(x):
    return isinstance(x, tuple) and all(isinstance(e, int) for e in x)


def _broadcast_index(index, tree):
    if index is None:
        return [None] * len(jax.tree.leaves(tree))
    broadcasted = jax.tree.broadcast(index, tree, is_leaf=_is_index_leaf)
    return jax.tree.leaves(broadcasted, is_leaf=_is_index_leaf)


def _sanitize_index(idx, num_batch_axes):
    if idx is None:
        return (slice(None),) * num_batch_axes
    if not isinstance(idx, tuple):
        idx = (idx,)
    ellipsis_pos = [k for k, e in enumerate(idx) if e is Ellipsis]
    if ellipsis_pos:
        i = ellipsis_pos[0]
        idx = idx[:i] + (slice(None),) * (num_batch_axes - len(idx) + 1) + idx[i + 1 :]
    idx = idx + (slice(None),) * (num_batch_axes - len(idx))
    return idx[:num_batch_axes]


def _index_spec_to_ic(idx_spec, num_batch_axes, unique_indices, seen_ids, *, name="index"):
    sanitized = _sanitize_index(idx_spec, num_batch_axes)
    result = []
    for entry in sanitized:
        if isinstance(entry, slice):
            result.append(-1)
        else:
            arr = jnp.asarray(entry, dtype=jnp.int32)
            if arr.ndim < num_batch_axes:
                raise ValueError(
                    f"{name} array has shape {arr.shape} ({arr.ndim}D) but "
                    f"{num_batch_axes} batch axes are required. "
                    f"Reshape to {arr.shape + (1,) * (num_batch_axes - arr.ndim)} "
                    f"to broadcast, or use fourierd3.s_[idx, :] for explicit axis selection."
                )
            obj_id = id(entry)
            if obj_id in seen_ids:
                result.append(seen_ids[obj_id])
            else:
                k = len(unique_indices)
                seen_ids[obj_id] = k
                unique_indices.append(arr)
                result.append(k)
    return tuple(result)


def _resolve_batch_shape(batch_shape, out_tree, num_batch_axes):
    if _is_batch_shape_leaf(batch_shape):
        if len(batch_shape) != num_batch_axes:
            raise ValueError(
                f"batch_shape has {len(batch_shape)} elements but cell_idx "
                f"implies num_batch_axes={num_batch_axes} "
                f"(cell_idx.ndim - 1). Use batch_shape with exactly "
                f"{num_batch_axes} element(s), or reshape cell_idx."
            )
        n_leaves = len(jax.tree.leaves(out_tree))
        return [batch_shape] * n_leaves
    broadcasted = jax.tree.broadcast(batch_shape, out_tree, is_leaf=_is_batch_shape_leaf)
    leaves = jax.tree.leaves(broadcasted, is_leaf=_is_batch_shape_leaf)
    for bs in leaves:
        if len(bs) != num_batch_axes:
            raise ValueError(
                f"Per-leaf batch_shape {bs} has {len(bs)} elements but "
                f"cell_idx implies num_batch_axes={num_batch_axes}."
            )
    return leaves


def _parse_args(support, grid_shape, cuda_opts, cell_idx):
    support_arr = np.asarray(support, dtype=np.int32)
    if support_arr.ndim == 1:
        support_tuple = tuple(int(x) for x in support_arr)
    elif support_arr.ndim == 2 and support_arr.shape[1] == 3:
        support_tuple = tuple(tuple(int(x) for x in row) for row in support_arr)
    else:
        raise ValueError(
            f"support must be a 1D tuple of ints (separable) or an Nx3 array "
            f"of 3D offsets, got shape {support_arr.shape}"
        )
    grid_shape = tuple(int(x) for x in grid_shape)
    if len(grid_shape) != 3:
        raise ValueError(f"grid_shape must be (gx, gy, gz), got {grid_shape}")
    opts = cuda_opts or {}
    cuda_opts = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in opts.items()))
    cell_idx = jnp.asarray(cell_idx)
    return support_tuple, grid_shape, cuda_opts, cell_idx


@eqx.filter_jit
def _scatter_to_mesh_jit(
    kernel,
    cell_idx,
    grid_leaves,
    input,
    grid_shape,
    support_tuple,
    *,
    n_grid_in=0,
    n_grid_out=0,
    grid_treedef=None,
    index_buffers=(),
    backend_arrays=(),
    cell_idx_ic=(-1,),
    grid_ics=(),
    input_ics=(),
    out_ics=(),
    idx_ics=(),
    outputs_shape_dtype=(),
    out_treedef=None,
    cuda_opts=(),
    _force_reference=False,
):
    grid_leaves = list(grid_leaves)
    input_leaves, input_treedef = jax.tree.flatten(input)
    num_batch_axes = len(cell_idx_ic)

    def kernel_flat(support_index, support_offset, *grid_and_inputs):
        grid_flat = grid_and_inputs[:n_grid_in]
        input_flat = grid_and_inputs[n_grid_in:]
        grid_pytree = jax.tree.unflatten(grid_treedef, grid_flat)
        input_pytree = jax.tree.unflatten(input_treedef, input_flat)
        return tuple(
            jax.tree.leaves(kernel(support_index, support_offset, grid_pytree, input_pytree))
        )

    support_index_ex = jnp.int32(0)
    support_offset_ex = jnp.zeros(3, dtype=jnp.int32)
    grid_value_exs = [jnp.zeros(g.shape[num_batch_axes + 3 :], dtype=g.dtype) for g in grid_leaves]
    input_exs = [jnp.zeros(f.shape[num_batch_axes:], dtype=f.dtype) for f in input_leaves]
    jaxpr = jax.make_jaxpr(kernel_flat)(
        support_index_ex, support_offset_ex, *grid_value_exs, *input_exs
    )

    jaxpr, captures = extract_captures(jaxpr)
    if captures:
        jaxpr_inner = jaxpr.jaxpr
        n_dyn = len(captures)
        jaxpr_inner = jaxpr_inner.replace(
            invars=jaxpr_inner.invars[n_dyn:] + jaxpr_inner.invars[:n_dyn]
        )
        jaxpr = _ClosedJaxpr(jaxpr_inner, list(jaxpr.consts))

    all_inputs = list(input_leaves) + [c.reshape((1,) * num_batch_axes + c.shape) for c in captures]

    capture_ics = [(-1,) * num_batch_axes] * len(captures)
    ic = [cell_idx_ic, *grid_ics, *input_ics, *capture_ics, *out_ics, *idx_ics]

    results = periodic_scatter_p.bind(
        cell_idx,
        *grid_leaves,
        *all_inputs,
        *index_buffers,
        *backend_arrays,
        jaxpr=jaxpr,
        support=support_tuple,
        grid_shape=grid_shape,
        n_grid_in=n_grid_in,
        n_grid_out=n_grid_out,
        index_configuration=tuple(ic),
        outputs_shape_dtype=outputs_shape_dtype,
        n_backend_arrays=len(backend_arrays),
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )
    return jax.tree.unflatten(out_treedef, results)


def scatter_over_support(
    support: tuple[int, ...] | tuple[tuple[int, int, int], ...],
    kernel: Callable[[Any, Any, Any], tuple[Any, Any]],
    cell_idx: jax.Array,
    grid_input: Any,
    input: Any,
    grid_shape: tuple[int, int, int],
    grid_output_batch_shape: tuple[int, ...] | Any | None = None,
    output_batch_shape: tuple[int, ...] | None = None,
    *,
    cell_idx_index: IndexSpec = None,
    grid_input_index: IndexSpec | Any = None,
    input_index: IndexSpec | Any = None,
    grid_output_index: IndexSpec | Any = None,
    output_index: IndexSpec | Any = None,
    cell_meta: tuple[jax.Array, jax.Array, tuple[int, int, int]] | None = None,
    cuda_opts: dict[str, Any] | None = None,
    _force_reference=False,
) -> tuple[Any, Any]:
    # The public `cell_meta` arg is the only place in the codebase that knows
    # what "cell meta" means; from here on it's an opaque tuple of backend
    # arrays that the primitive layer just threads through.
    if cell_meta is not None:
        atom_map, cell_starts_ends, cell_grid_shape = cell_meta
        backend_arrays: tuple = (atom_map, cell_starts_ends)
        cuda_opts = dict(cuda_opts) if cuda_opts else {}
        cuda_opts["cell_grid_shape"] = tuple(int(x) for x in cell_grid_shape)
    else:
        backend_arrays = ()

    support_tuple, grid_shape, cuda_opts, cell_idx = _parse_args(
        support, grid_shape, cuda_opts, cell_idx
    )

    if cell_idx.shape[-1] != 3:
        raise ValueError(f"cell_idx must have shape (*batch, 3), got {cell_idx.shape}.")

    num_batch_axes = cell_idx.ndim - 1

    grid_leaves, grid_treedef = jax.tree.flatten(grid_input)
    n_grid_in = len(grid_leaves)

    grid_inner_shapes: list[tuple[int, ...]] = []
    for i, g in enumerate(grid_leaves):
        if g.ndim < num_batch_axes + 3:
            raise ValueError(
                f"grid_input leaf {i} has ndim={g.ndim}, expected >= "
                f"{num_batch_axes + 3} (num_batch_axes={num_batch_axes} + 3 spatial)."
            )
        spatial = g.shape[num_batch_axes : num_batch_axes + 3]
        if spatial != grid_shape:
            raise ValueError(
                f"grid_input leaf {i} spatial dims {spatial} != grid_shape {grid_shape}."
            )
        grid_inner_shapes.append(g.shape[num_batch_axes + 3 :])

    input_leaves, input_treedef = jax.tree.flatten(input)
    for i, f in enumerate(input_leaves):
        if f.ndim < num_batch_axes:
            raise ValueError(f"input leaf {i} has ndim={f.ndim}, expected >= {num_batch_axes}.")

    sup_idx_st = jax.ShapeDtypeStruct((), jnp.int32)
    sup_st = jax.ShapeDtypeStruct((3,), jnp.int32)
    grid_val_sts = [
        jax.ShapeDtypeStruct(s, g.dtype)
        for s, g in zip(grid_inner_shapes, grid_leaves, strict=False)
    ]
    grid_val_pytree_st = jax.tree.unflatten(grid_treedef, grid_val_sts)
    input_sts = [jax.ShapeDtypeStruct(f.shape[num_batch_axes:], f.dtype) for f in input_leaves]
    input_pytree_st = jax.tree.unflatten(input_treedef, input_sts)
    raw_out = jax.eval_shape(kernel, sup_idx_st, sup_st, grid_val_pytree_st, input_pytree_st)

    if not isinstance(raw_out, tuple) or len(raw_out) != 2:
        raise ValueError(
            f"Kernel must return a 2-tuple (grid_output, output). Got {type(raw_out)}."
        )

    grid_out_raw, output_raw = raw_out
    grid_out_sd_leaves = jax.tree.leaves(grid_out_raw)
    output_sd_leaves = jax.tree.leaves(output_raw)
    n_grid_out = len(grid_out_sd_leaves)
    out_treedef = jax.tree.structure(raw_out)

    if grid_output_batch_shape is None:
        grid_output_batch_shape = (1,) * num_batch_axes
    grid_out_batch_leaves = _resolve_batch_shape(
        grid_output_batch_shape, grid_out_raw, num_batch_axes
    )

    unique_indices: list = []
    seen_ids: dict = {}

    cell_idx_ic = _index_spec_to_ic(
        cell_idx_index, num_batch_axes, unique_indices, seen_ids, name="cell_idx_index"
    )
    grid_in_idx_leaves = _broadcast_index(grid_input_index, grid_input)
    grid_ics = [
        _index_spec_to_ic(idx, num_batch_axes, unique_indices, seen_ids, name="grid_input_index")
        for idx in grid_in_idx_leaves
    ]
    input_idx_leaves = _broadcast_index(input_index, input)
    input_ics = [
        _index_spec_to_ic(idx, num_batch_axes, unique_indices, seen_ids, name="input_index")
        for idx in input_idx_leaves
    ]
    grid_out_idx_leaves = _broadcast_index(grid_output_index, grid_out_raw)
    grid_out_ics = [
        _index_spec_to_ic(idx, num_batch_axes, unique_indices, seen_ids, name="grid_output_index")
        for idx in grid_out_idx_leaves
    ]
    output_idx_leaves_spec = _broadcast_index(output_index, output_raw)
    if any(idx is not None for idx in output_idx_leaves_spec):
        raise NotImplementedError(
            "output_index for non-grid outputs is not yet supported. "
            "Use grid outputs for scatter-add routing."
        )
    output_ics = [(-1,) * num_batch_axes] * len(output_sd_leaves)
    idx_ics = [(-1,) * num_batch_axes] * len(unique_indices)

    if output_batch_shape is None:
        ic_for_infer = [
            cell_idx_ic,
            *grid_ics,
            *input_ics,
            *grid_out_ics,
            *output_ics,
            *idx_ics,
        ]

        def buf_batch_shape_fn(buf_idx):
            if buf_idx == 0:
                return cell_idx.shape[:num_batch_axes]
            current = 1
            if buf_idx < current + n_grid_in:
                return grid_leaves[buf_idx - current].shape[:num_batch_axes]
            current += n_grid_in
            if buf_idx < current + len(input_leaves):
                return input_leaves[buf_idx - current].shape[:num_batch_axes]
            current += len(input_leaves)
            if buf_idx < current + n_grid_out:
                return grid_out_batch_leaves[buf_idx - current]
            current += n_grid_out
            if buf_idx < current + len(output_sd_leaves):
                return (1,) * num_batch_axes
            current += len(output_sd_leaves)
            idx_buf_idx = buf_idx - current
            if idx_buf_idx < len(unique_indices):
                return unique_indices[idx_buf_idx].shape[:num_batch_axes]
            return (1,) * num_batch_axes

        output_batch_shape = _compute_batch_sizes(ic_for_infer, buf_batch_shape_fn, num_batch_axes)

    gx, gy, gz = grid_shape
    grid_osd = [
        jax.ShapeDtypeStruct(grid_out_batch_leaves[j] + (gx, gy, gz) + o.shape, o.dtype)
        for j, o in enumerate(grid_out_sd_leaves)
    ]
    out_osd = [
        jax.ShapeDtypeStruct(output_batch_shape + o.shape, o.dtype) for o in output_sd_leaves
    ]
    outputs_shape_dtype = tuple(grid_osd + out_osd)
    out_ics = tuple(grid_out_ics + output_ics)

    return _scatter_to_mesh_jit(
        kernel,
        cell_idx,
        tuple(grid_leaves),
        input,
        grid_shape,
        support_tuple,
        n_grid_in=n_grid_in,
        n_grid_out=n_grid_out,
        grid_treedef=grid_treedef,
        index_buffers=tuple(unique_indices),
        backend_arrays=backend_arrays,
        cell_idx_ic=cell_idx_ic,
        grid_ics=tuple(grid_ics),
        input_ics=tuple(input_ics),
        out_ics=out_ics,
        idx_ics=tuple(idx_ics),
        outputs_shape_dtype=outputs_shape_dtype,
        out_treedef=out_treedef,
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )


def scatter_to_mesh(
    support: tuple[int, ...] | tuple[tuple[int, int, int], ...],
    kernel: Callable[[Any, Any], Any],
    cell_idx: jax.Array,
    features: Any,
    grid_shape: tuple[int, int, int],
    batch_shape: tuple[int, ...] | Any | None = None,
    *,
    cell_idx_index: IndexSpec = None,
    features_index: IndexSpec | Any = None,
    output_index: IndexSpec | Any = None,
    cell_meta: tuple[jax.Array, jax.Array, tuple[int, int, int]] | None = None,
    cuda_opts: dict[str, Any] | None = None,
    _force_reference=False,
) -> Any:
    def spreading_kernel(support_index, support_offset, grid_input, input):
        return kernel(support_index, support_offset, input), None

    grid_out, _ = scatter_over_support(
        support,
        spreading_kernel,
        cell_idx,
        None,
        features,
        grid_shape,
        grid_output_batch_shape=batch_shape,
        cell_idx_index=cell_idx_index,
        input_index=features_index,
        grid_output_index=output_index,
        cell_meta=cell_meta,
        cuda_opts=cuda_opts,
        _force_reference=_force_reference,
    )
    return grid_out
