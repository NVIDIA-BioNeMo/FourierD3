# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpolation of the reference C6 tables at each atom's coordination number."""

import jax
import jax.numpy as jnp

__all__ = ["c6_interpolation_weights"]


def c6_interpolation_weights(cn, cnref_per_atom):
    logits = -4.0 * jnp.square(cn[:, None] - cnref_per_atom)
    logits = jnp.where(cnref_per_atom == -1, -jnp.inf, logits)
    return jax.nn.softmax(logits, axis=1)
