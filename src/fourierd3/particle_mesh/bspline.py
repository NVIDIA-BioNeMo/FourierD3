# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from sympy import Rational, Symbol, expand, symbols

from fourierd3.particle_mesh.polynomial_table import weights_to_array

Support1D = list[int]


def _M_n(n: int, k: int, v: Symbol) -> Rational:
    if n == 1:
        return Rational(1) if k == 0 else Rational(0)
    if k < 0 or k >= n:
        return Rational(0)
    return expand(v / (n - 1) * _M_n(n - 1, k, v) + (n - v) / (n - 1) * _M_n(n - 1, k - 1, v - 1))


def cardinal_bspline_weights(order):
    u = symbols("u", real=True)
    return {k: _M_n(order, k, k + 1 - u) for k in range(order)}, u


def cardinal_bspline_fourier_grid_dft(order, size):
    poly_dict, u_sym = cardinal_bspline_weights(order)
    bs = np.zeros(size)
    for k in range(order):
        bs[k] = float(poly_dict[k].subs(u_sym, 0))
    mod = np.abs(np.fft.fft(bs)) ** 2
    if size % 2 == 0:
        mod[size // 2] = 0.5 * (mod[size // 2 - 1] + mod[size // 2 + 1])
    return mod**0.5


def _horner(coeffs, t):
    val = jnp.zeros_like(t) + coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        c = coeffs[i]
        val = t * val + c if abs(c) > 1e-15 else t * val
    return val


class BSplineWeights(eqx.Module):
    poly_coeffs: tuple = eqx.field(static=True)
    order: int = eqx.field(static=True)

    def __call__(self, theta: jax.Array) -> jax.Array:
        return jnp.stack([_horner(self.poly_coeffs[i], theta) for i in range(self.order)], axis=-1)


class BSplineFourierKernel(eqx.Module):
    order: int

    def __call__(self, freq, *, grid_size=None):
        assert grid_size is not None
        nx, ny, nz = grid_size
        dt = freq.dtype
        bx = jnp.array(cardinal_bspline_fourier_grid_dft(self.order, nx), dtype=dt)
        by = jnp.array(cardinal_bspline_fourier_grid_dft(self.order, ny), dtype=dt)
        bz = jnp.array(cardinal_bspline_fourier_grid_dft(self.order, nz), dtype=dt)
        i = jnp.round(freq[..., 0] * nx).astype(jnp.int32)
        j = jnp.round(freq[..., 1] * ny).astype(jnp.int32)
        k = jnp.round(freq[..., 2] * nz).astype(jnp.int32)
        return bx[i] * by[j] * bz[k]


def bspline_transfer(order: int) -> tuple[Support1D, BSplineWeights, BSplineFourierKernel]:
    if order < 2:
        raise ValueError(f"B-spline order must be >= 2, got {order}")
    poly_dict, u_sym = cardinal_bspline_weights(order)
    offsets, poly = weights_to_array(poly_dict, u_sym)
    shift = 1 - order // 2 if order % 2 == 0 else -(order - 1) // 2
    offsets = offsets + shift
    support: Support1D = sorted(int(o) for o in offsets)
    coeffs = tuple(tuple(float(c) for c in row) for row in poly)
    return (
        support,
        BSplineWeights(poly_coeffs=coeffs, order=order),
        BSplineFourierKernel(order=order),
    )
