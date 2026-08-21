# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fractional coordination number of every atom from a directed neighbor list."""

import jax
import jax.numpy as jnp

__all__ = ["coordination_numbers"]


def coordination_numbers(strain, positions, rcov, lattice, r_cut, src, dst, shift):
    eps = jnp.eye(3, dtype=positions.dtype) + strain
    # HIGHEST precision: a default f32 matmul lowers to TF32 tensor cores, which
    # truncate the coordinates to a ~10-bit mantissa (~1e-3 relative). The CN
    # switching function below is razor-sharp near the cutoff, so that
    # truncation amplifies into a ~3% force error.
    pos = jnp.matmul(positions, eps, precision=jax.lax.Precision.HIGHEST)
    cell = jnp.matmul(lattice, eps, precision=jax.lax.Precision.HIGHEST)

    shifted = jnp.matmul(
        shift.astype(pos.dtype),
        cell,
        precision=jax.lax.Precision.HIGHEST,
    )
    distance = jnp.linalg.norm(pos[dst] - pos[src] + shifted, axis=1)
    covalent_distance = rcov[src] + rcov[dst]
    transition = 0.5 * (covalent_distance + r_cut)
    stabilized_distance = jnp.maximum(distance, transition)
    steepness = 16.0 + (stabilized_distance - transition) ** 2 / (
        (r_cut - stabilized_distance) ** 2 + 1e-6
    )
    contribution = jax.nn.sigmoid(steepness * ((4.0 / 3.0) * covalent_distance / distance - 1.0))
    return jnp.zeros(positions.shape[0], dtype=positions.dtype).at[src].add(contribution)
