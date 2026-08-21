# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fourier transform of the damped r^-6 and r^-8 D3(BJ) interactions.

Both damped power laws have closed-form radial transforms; evaluating them
per reciprocal-lattice point is what replaces the real-space dispersion sum.
"""

import jax.numpy as jnp

__all__ = ["reciprocal_kernel_matrix"]


def reciprocal_kernel_matrix(k_norm, sqrtQz, params):
    s6, s8, a1, a2 = params
    pi = jnp.pi
    QzQz = jnp.outer(sqrtQz, sqrtQz)
    R = a1 * jnp.sqrt(3 * QzQz) + a2
    kR = k_norm * R
    small = kR < 1e-15
    ks = jnp.where(k_norm < 1e-15, 1.0, k_norm)
    kRs, ksq = ks * R, k_norm**2

    ft6_0 = (2 * pi**2) / (3 * R**3) - (2 * pi**2 * R / 9) * ksq
    n6 = jnp.exp(-kRs) - 2 * jnp.exp(-kRs / 2) * jnp.cos(pi / 3 + kRs * jnp.sqrt(3.0) / 2)
    ft6 = s6 * jnp.where(small, ft6_0, (2 * pi**2) / (3 * ks * R**4) * n6)

    s8p, c8p = jnp.sin(pi / 8), jnp.cos(pi / 8)
    ft8_0 = (pi**2 * jnp.sqrt(2.0) * s8p) / R**5
    ft8_0 = ft8_0 - (ft8_0 * R**2 / 6) * ksq
    n8 = jnp.exp(-kRs * s8p) * jnp.cos(pi / 4 + kRs * c8p) + jnp.exp(-kRs * c8p) * jnp.cos(
        3 * pi / 4 + kRs * s8p
    )
    ft8 = 3 * s8 * jnp.where(small, ft8_0, -(pi**2) / (ks * R**6) * n8)

    return ft6 + QzQz * ft8
