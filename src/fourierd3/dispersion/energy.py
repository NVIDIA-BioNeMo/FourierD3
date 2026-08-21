# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic DFT-D3(BJ) dispersion energy and its position and strain derivatives.

The energy assembles the pipeline the package is built around: coordination
numbers, C6 interpolation, particle-mesh scatter of the resulting coefficients,
and one reciprocal-space contraction against the damped-dispersion kernel.
"""

import jax
import jax.numpy as jnp

from fourierd3._engine.indexing import s_
from fourierd3._engine.jax_ops.periodic_scatter.layout import support_is_separable
from fourierd3.dispersion.coefficients import c6_interpolation_weights
from fourierd3.dispersion.coordination import coordination_numbers
from fourierd3.dispersion.reciprocal_kernel import reciprocal_kernel_matrix
from fourierd3.particle_mesh.periodic_scatter import scatter_to_mesh
from fourierd3.particle_mesh.spectral_map import spectral_map

__all__ = ["dispersion_energy", "dispersion_energy_and_derivatives"]


def dispersion_energy(strain, positions, lattice, md):
    prec = jax.lax.Precision.HIGHEST
    pos = positions + jnp.einsum("ab,ib->ia", strain, positions, precision=prec)
    cell = lattice + jnp.einsum("ab,Ab->Aa", strain, lattice, precision=prec)
    volume = jnp.abs(jnp.linalg.det(cell))
    species = md["species"]
    n_species = md["n_species"]

    strain0 = jnp.zeros((3, 3), dtype=positions.dtype)
    cn = coordination_numbers(
        strain0,
        pos,
        md["rcov"],
        cell,
        md["r_cut"],
        md["src"],
        md["dst"],
        md["shift"],
    )

    weights = c6_interpolation_weights(cn, md["cnref"][species])
    c6 = jnp.einsum("np,npr->nr", weights, md["v_q"][species])

    support, w_fn, fourier_fn = md["transfer_fn"]
    grid_size = md["grid_size"]
    lattice_inv = jnp.linalg.inv(cell)
    grid_size_arr = jnp.array(grid_size, dtype=positions.dtype)
    s = jnp.matmul(pos, lattice_inv, precision=jax.lax.Precision.HIGHEST) * grid_size_arr
    ci = jnp.floor(s).astype(jnp.int32)

    if support_is_separable(support):
        mo = support[0]

        def spread_kernel(_i, sup, f):
            w = w_fn(f["s"] - jnp.floor(f["s"]))
            ix, iy, iz = sup[0] - mo, sup[1] - mo, sup[2] - mo
            return w[0][ix] * w[1][iy] * w[2][iz] * f["v"]
    else:

        def spread_kernel(i, _sup, f):
            return w_fn(f["s"] - jnp.floor(f["s"]), i) * f["v"]

    rho = scatter_to_mesh(
        support,
        spread_kernel,
        ci,
        {"v": c6, "s": s},
        grid_size,
        (n_species,),
        output_index=s_[species],
    )
    rho = jnp.moveaxis(rho, 0, -2)

    k_matrix = 2 * jnp.pi * lattice_inv

    def kspace_fn(i, rho_k, _aux):
        freq = i.astype(positions.dtype) / grid_size_arr
        c_hat_k = rho_k / fourier_fn(freq, grid_size=grid_size)
        k_vec = k_matrix @ i.astype(positions.dtype)
        k = jnp.sqrt(jnp.sum(jnp.square(k_vec)) + 1e-36)
        K = reciprocal_kernel_matrix(k, md["sqrtQz"], md["params"])
        per_rank = jnp.sum(jnp.real(c_hat_k * (K @ jnp.conj(c_hat_k))), axis=0)
        return (), jnp.dot(md["eigs"], per_rank)

    _, total = spectral_map(
        kspace_fn,
        rho,
        is_hermitian=True,
        num_batch_axes=0,
    )

    vself = jnp.dot(md["eigs"], jnp.sum(jnp.square(c6) * md["selfcont"][species], axis=0))
    return total / (-2 * volume) + vself / 2


def dispersion_energy_and_derivatives(positions, lattice, md):
    strain0 = jnp.zeros((3, 3), dtype=positions.dtype)

    energy, (grad_strain, grad_positions) = jax.value_and_grad(
        lambda strain, pos: dispersion_energy(strain, pos, lattice, md),
        argnums=(0, 1),
    )(strain0, positions)
    return energy, grad_positions, grad_strain
