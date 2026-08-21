# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiling a lowered operation into execution-plan bytes, once per shape.

A compilation context memoizes plans by (entry point, arguments, device) so a
second trace of the same computation reuses the plan the executor already
tuned, instead of re-enumerating the candidate set.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import os
import pathlib
from collections.abc import Iterator
from contextvars import ContextVar
from typing import NamedTuple

import jax

from fourierd3._engine import _extension


class _CompilationState(NamedTuple):
    cache: dict
    compile_budget_ms: float | None
    pending: list
    debug_dump_dir: str | None
    debug_seq: Iterator[int]


class PendingPlan(NamedTuple):
    """Identifies a freshly compiled plan whose tuned form is worth caching."""

    key: str
    companions: tuple


_compilation: ContextVar = ContextVar("fourierd3_compilation", default=None)


@contextlib.contextmanager
def compilation_context(
    cache: dict | None = None,
    *,
    compile_budget_ms: float | None = None,
    debug_dump_dir: str | None = None,
):
    """Compile inside this context under one set of settings and one cache.

    On exit, every plan the executor tuned during the context is written back
    into `cache`, so a later context with the same cache skips compilation.
    """
    if cache is None:
        cache = {}
    state = _CompilationState(
        cache,
        compile_budget_ms,
        [],
        debug_dump_dir,
        itertools.count(),
    )
    token = _compilation.set(state)
    try:
        yield cache
    finally:
        try:
            for key, companions, handle in state.pending:
                tuned = _extension.tuned_plan_bytes(handle)
                if tuned is not None:
                    cache[key] = (tuned, companions)
        finally:
            _compilation.reset(token)


def current_state():
    return _compilation.get()


def debug_dump_path(prefix: str) -> str | None:
    state = _compilation.get()
    if state is None or state.debug_dump_dir is None:
        return None
    os.makedirs(state.debug_dump_dir, exist_ok=True)
    return os.path.join(state.debug_dump_dir, f"{prefix}_{next(state.debug_seq):03d}.json")


def _device_fingerprint() -> tuple:
    device = jax.devices("gpu")[0]
    so = pathlib.Path(_extension.__file__).stat()
    return (device.device_kind, device.compute_capability, so.st_size, so.st_mtime_ns)


def compile_plan(compile_to_bytes, /, *args, **kwargs) -> tuple:
    state = _compilation.get()
    if state is None:
        result = compile_to_bytes(*args, **kwargs)
        plan_bytes, *companions = result if isinstance(result, tuple) else (result,)
        return (plan_bytes, *companions, None)

    material = repr(
        (compile_to_bytes.__name__, args, sorted(kwargs.items()), _device_fingerprint())
    )
    key = hashlib.sha256(material.encode()).hexdigest()[:32]

    hit = state.cache.get(key)
    if hit is not None:
        plan_bytes, companions = hit
        return (plan_bytes, *companions, None)

    # A re-lowering inside the same compilation context (e.g. a second jit
    # trace of the same computation) reuses the pending handle's tuned winner
    # instead of recompiling the full candidate set.
    for pending_key, companions, handle in state.pending:
        if pending_key == key:
            tuned = _extension.tuned_plan_bytes(handle)
            if tuned is not None:
                state.cache[key] = (tuned, companions)
                return (tuned, *companions, None)
            break

    effort = (
        {} if state.compile_budget_ms is None else {"compile_budget_ms": state.compile_budget_ms}
    )
    result = compile_to_bytes(*args, **kwargs, **effort)
    plan_bytes, *companions = result if isinstance(result, tuple) else (result,)
    return (plan_bytes, *companions, PendingPlan(key, tuple(companions)))
