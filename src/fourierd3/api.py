# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The supported surface of FourierD3.

Everything else in the package is an implementation detail: the particle-mesh
operations, the kernel compiler, and the plan executor are free to change
shape as long as these entry points keep their meaning.
"""

from fourierd3.dispersion.energy import dispersion_energy, dispersion_energy_and_derivatives
from fourierd3.particle_mesh.bspline import bspline_transfer

__all__ = [
    "bspline_transfer",
    "dispersion_energy",
    "dispersion_energy_and_derivatives",
]
