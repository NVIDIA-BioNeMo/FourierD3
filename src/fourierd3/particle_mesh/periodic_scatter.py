# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accumulation of per-point contributions onto a periodic mesh."""

from fourierd3._engine.jax_ops.periodic_scatter.api import scatter_to_mesh

__all__ = ["scatter_to_mesh"]
