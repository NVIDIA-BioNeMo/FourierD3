# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fourierd3._engine.runtime.discover_cuda import autodiscover as _autodiscover_cuda
from fourierd3.api import (
    bspline_transfer,
    dispersion_energy,
    dispersion_energy_and_derivatives,
)

__version__ = "0.3.0"

_autodiscover_cuda()

__all__ = [
    "bspline_transfer",
    "dispersion_energy",
    "dispersion_energy_and_derivatives",
]
