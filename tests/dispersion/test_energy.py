# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

import fourierd3.dispersion.energy as energy


def test_derivatives_helper_differentiates_the_energy(monkeypatch):
    def fake_energy(strain, positions, lattice, metadata):
        del metadata
        pos = positions + jnp.einsum("ab,ib->ia", strain, positions)
        cell = lattice + jnp.einsum("ab,Ab->Aa", strain, lattice)
        return jnp.sum(pos**2) + 0.5 * jnp.sum(cell**2)

    monkeypatch.setattr(energy, "dispersion_energy", fake_energy)
    positions = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    lattice = jnp.eye(3, dtype=jnp.float32)

    computed_energy, grad_positions, grad_strain = energy.dispersion_energy_and_derivatives(
        positions, lattice, {}
    )

    strain = jnp.zeros((3, 3), dtype=positions.dtype)
    expected_energy = fake_energy(strain, positions, lattice, {})
    expected_strain, expected_positions = jax.grad(
        lambda eps, pos: fake_energy(eps, pos, lattice, {}),
        argnums=(0, 1),
    )(strain, positions)

    np.testing.assert_allclose(computed_energy, expected_energy)
    np.testing.assert_allclose(grad_positions, expected_positions)
    np.testing.assert_allclose(grad_strain, expected_strain)
