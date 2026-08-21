# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from sympy import Poly, Rational, Symbol


def weights_to_array(polynomials: dict[int, Rational], u: Symbol) -> tuple[np.ndarray, np.ndarray]:
    offsets = sorted(polynomials.keys())
    support_size = len(offsets)

    max_degree = 0
    for expr in polynomials.values():
        poly = Poly(expr, u)
        if not poly.is_zero:
            d = poly.degree()
            # In sympy, degree() returns an integer or -inf (S.NegativeInfinity)
            if d is not None and d >= 0:
                max_degree = max(max_degree, int(d))

    poly_order = max_degree + 1
    coeffs = np.zeros((support_size, poly_order), dtype=np.float64)

    for offset, expr in polynomials.items():
        poly = Poly(expr, u)
        for (i,), coeff in poly.terms():
            coeffs[offsets.index(offset), i] = float(coeff)

    offsets = np.array(offsets, dtype=np.int32)
    return offsets, coeffs
