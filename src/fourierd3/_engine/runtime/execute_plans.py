# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handing execution-plan bytes to the engine and running them from JAX.

Loading a plan returns an opaque handle plus the workspace it needs; the
handle travels through the XLA FFI call as an attribute, so the executor
keeps the loaded modules resident across replays.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
from jax import ffi

from fourierd3._engine import _extension
from fourierd3._engine.runtime.compile_plans import (
    PendingPlan,
    current_state,
    debug_dump_path,
)

ffi.register_ffi_target(
    "run_plan",
    _extension.run_plan_capsule(),
    platform="CUDA",
)


def run_plan(
    plan_bytes: bytes,
    inputs: Sequence[jax.Array],
    out_specs: Sequence[jax.ShapeDtypeStruct],
    *,
    pending: PendingPlan | None = None,
    vmap_method: str | None = None,
) -> list[jax.Array]:
    handle, workspace_nbytes = _extension.load_plan(
        plan_bytes, tune_report_to=debug_dump_path("tune_report")
    )
    if pending is not None:
        state = current_state()
        if state is not None:
            state.pending.append((pending.key, pending.companions, handle))
    workspace = jax.ShapeDtypeStruct((int(workspace_nbytes),), np.uint8)
    outs = ffi.ffi_call("run_plan", [workspace, *out_specs], vmap_method=vmap_method)(
        *inputs, plan=np.uint64(handle)
    )
    return list(outs[1:])
