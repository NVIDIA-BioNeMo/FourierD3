# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward FFT, a per-frequency map, and inverse FFT as one fused operation."""

from fourierd3._engine.jax_ops.spectral_map.api import spectral_map

__all__ = ["spectral_map"]
