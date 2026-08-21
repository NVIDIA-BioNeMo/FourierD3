# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax


def jaxpr_to_fn(closed_jaxpr):
    def fn(*args):
        return tuple(jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args))

    return fn


def _sds(var):
    return jax.ShapeDtypeStruct(var.aval.shape, var.aval.dtype)


def _make_partial(fn, inputs, active_idx):
    def partial_fn(*active):
        all_inputs = list(inputs)
        for i, k in enumerate(active_idx):
            all_inputs[k] = active[i]
        return fn(*all_inputs)

    return partial_fn, tuple(inputs[k] for k in active_idx)
