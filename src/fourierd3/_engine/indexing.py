# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.interpreters import ad


class MultiAxisIndex(tuple):
    pass


class _IndexHelper:
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return MultiAxisIndex(key)


s_ = _IndexHelper()


def _compute_batch_sizes(index_configuration, buf_batch_shape_fn, num_batch_axis):
    batch_sizes = []
    for axis in range(num_batch_axis):
        mx = 1
        for buf_idx, ic in enumerate(index_configuration):
            if ic[axis] < 0:
                s = buf_batch_shape_fn(buf_idx)[axis]
                if s != 1:
                    assert mx in (1, s), f"batch size mismatch on axis {axis}: {mx} vs {s}"
                    mx = s
        batch_sizes.append(mx)
    return tuple(batch_sizes)


def _iota(shape, axis):
    i = jnp.arange(shape[axis])
    out_shape = [1] * len(shape)
    out_shape[axis] = shape[axis]
    return jnp.reshape(i, out_shape)


def _make_indexing_fn(
    index_configuration, buf_batch_shape_fn, index_buffers, n_data_bufs, num_batch_axis
):
    def indexing(buf_idx):
        ic = index_configuration[buf_idx]
        shape = buf_batch_shape_fn(buf_idx)[:num_batch_axis]
        result = []
        for axis, ref in enumerate(ic):
            if ref < 0:
                result.append(_iota(shape, axis))
            else:
                idx_ic_pos = n_data_bufs + ref
                result.append(index_buffers[ref][indexing(idx_ic_pos)])
        return tuple(result)

    return indexing


def _batch_rule(prim, batched_args, batch_dims, *, index_configuration, outputs_shape_dtype, **kw):
    def align(arr, bdim):
        if bdim is None:
            return jnp.expand_dims(arr, 0)
        return jnp.moveaxis(arr, bdim, 0)

    prepared = [align(a, d) for a, d in zip(batched_args, batch_dims, strict=False)]

    new_dim = 1
    for x in prepared:
        if x.shape[0] != 1:
            assert new_dim in (1, x.shape[0])
            new_dim = x.shape[0]

    new_ic = tuple((-1,) + ic for ic in index_configuration)
    new_osd = tuple(
        jax.ShapeDtypeStruct((new_dim,) + o.shape, o.dtype) for o in outputs_shape_dtype
    )

    results = prim.bind(
        *prepared,
        index_configuration=new_ic,
        outputs_shape_dtype=new_osd,
        **kw,
    )
    return results, (0,) * len(results)


def _assemble_cotangents(args_or_undef, ct_dict):
    result = []
    for k in range(len(args_or_undef)):
        if k in ct_dict:
            result.append(ct_dict[k])
        elif ad.is_undefined_primal(args_or_undef[k]):
            result.append(ad.Zero(args_or_undef[k].aval))
        else:
            result.append(None)
    return result
