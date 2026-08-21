# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fourierd3
from fourierd3._engine.runtime.compile_plans import compilation_context


def test_energy_and_derivatives_on_gpu():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("requires a CUDA-enabled JAX installation")

    rng = np.random.default_rng(0)
    n_atoms, n_species, n_rank, n_edges = 8, 2, 2, 24
    src = rng.integers(0, n_atoms, size=n_edges)
    dst = (src + 1 + rng.integers(0, n_atoms - 1, size=n_edges)) % n_atoms
    metadata = {
        "species": jnp.asarray(rng.integers(0, n_species, size=n_atoms)),
        "n_species": n_species,
        "rcov": jnp.asarray(rng.uniform(0.5, 1.5, size=n_atoms), dtype=jnp.float32),
        "cnref": jnp.asarray(rng.uniform(0.0, 5.0, size=(n_species, 3)), dtype=jnp.float32),
        "v_q": jnp.asarray(rng.normal(size=(n_species, 3, n_rank)), dtype=jnp.float32),
        "eigs": jnp.asarray(rng.uniform(0.1, 1.0, size=n_rank), dtype=jnp.float32),
        "selfcont": jnp.asarray(rng.uniform(0.0, 1.0, size=(n_species, 1)), dtype=jnp.float32),
        "sqrtQz": jnp.asarray(rng.uniform(0.5, 2.0, size=n_species), dtype=jnp.float32),
        "params": jnp.asarray([1.0, 0.7875, 0.4289, 4.4407], dtype=jnp.float32),
        "grid_size": (8, 8, 8),
        "r_cut": 6.0,
        "transfer_fn": fourierd3.bspline_transfer(4),
        "src": jnp.asarray(src),
        "dst": jnp.asarray(dst),
        "shift": jnp.asarray(rng.integers(-1, 2, size=(n_edges, 3))),
    }
    positions = jnp.asarray(rng.uniform(0.0, 10.0, size=(n_atoms, 3)), dtype=jnp.float32)
    lattice = jnp.asarray(10.0 * np.eye(3), dtype=jnp.float32)
    run = jax.jit(
        lambda pos, cell: fourierd3.dispersion_energy_and_derivatives(pos, cell, metadata)
    )

    with compilation_context(compile_budget_ms=1):
        computed_energy, grad_positions, grad_strain = run(positions, lattice)
        jax.block_until_ready((computed_energy, grad_positions, grad_strain))

    assert computed_energy.shape == ()
    assert grad_positions.shape == positions.shape
    assert grad_strain.shape == (3, 3)
    assert bool(jnp.isfinite(computed_energy))
    assert bool(jnp.all(jnp.isfinite(grad_positions)))
    assert bool(jnp.all(jnp.isfinite(grad_strain)))
