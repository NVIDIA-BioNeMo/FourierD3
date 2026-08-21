# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from fourierd3.dispersion.coordination import coordination_numbers


def test_coordination_numbers_match_direct_edge_sum():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.0],
            [0.3, 1.4, 0.1],
            [1.0, 1.2, 1.3],
        ],
        dtype=jnp.float32,
    )
    lattice = jnp.eye(3, dtype=jnp.float32) * 5.0
    strain = jnp.array(
        [
            [0.01, 0.02, 0.0],
            [0.0, -0.01, 0.015],
            [0.0, 0.0, 0.005],
        ],
        dtype=jnp.float32,
    )
    rcov = jnp.array([0.7, 0.8, 0.75, 0.9], dtype=jnp.float32)
    src = jnp.array([0, 1, 2, 0, 3], dtype=jnp.int32)
    dst = jnp.array([1, 2, 3, 3, 0], dtype=jnp.int32)
    shift = jnp.array(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]],
        dtype=jnp.int32,
    )
    r_cut = 6.0

    eps = jnp.eye(3, dtype=positions.dtype) + strain
    pos = jnp.matmul(positions, eps, precision=jax.lax.Precision.HIGHEST)
    cell = jnp.matmul(lattice, eps, precision=jax.lax.Precision.HIGHEST)
    shifted = jnp.matmul(shift.astype(pos.dtype), cell, precision=jax.lax.Precision.HIGHEST)
    distance = jnp.linalg.norm(pos[dst] - pos[src] + shifted, axis=1)
    covalent_distance = rcov[src] + rcov[dst]
    transition = 0.5 * (covalent_distance + r_cut)
    stabilized_distance = jnp.maximum(distance, transition)
    steepness = 16.0 + (stabilized_distance - transition) ** 2 / (
        (r_cut - stabilized_distance) ** 2 + 1e-6
    )
    contribution = jax.nn.sigmoid(steepness * ((4.0 / 3.0) * covalent_distance / distance - 1.0))
    expected = jnp.zeros(positions.shape[0], dtype=positions.dtype).at[src].add(contribution)

    actual = coordination_numbers(strain, positions, rcov, lattice, r_cut, src, dst, shift)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
