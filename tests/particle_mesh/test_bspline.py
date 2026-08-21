# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cardinal B-spline transfer functions.

The weights a scatter uses must sum to one at every offset, or the total
scattered mass depends on where a point falls inside its cell.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from fourierd3.particle_mesh.bspline import bspline_transfer


@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_weights_are_a_partition_of_unity(order):
    support, weights, _ = bspline_transfer(order)

    assert len(support) == order
    assert support == sorted(support)

    theta = jnp.linspace(0.0, 1.0, 17, dtype=jnp.float32)
    w = np.asarray(weights(theta))

    assert w.shape == (theta.shape[0], order)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=0, atol=1e-6)


@pytest.mark.parametrize("order", [4, 6])
def test_fourier_factor_is_positive_on_the_mesh(order):
    _, _, fourier = bspline_transfer(order)
    grid_size = (8, 8, 8)

    freq = jnp.stack(
        jnp.meshgrid(*[jnp.arange(n) / n for n in grid_size], indexing="ij"),
        axis=-1,
    ).astype(jnp.float32)
    factor = np.asarray(fourier(freq, grid_size=grid_size))

    assert factor.shape == grid_size
    assert np.all(factor > 0.0)
