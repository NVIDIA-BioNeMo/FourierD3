# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScatterOperands:
    cell_idx: Any
    grid_in: tuple
    nongrid_in: tuple
    index_buffers: tuple
    backend_arrays: tuple
    n_grid_in: int
    n_nongrid_in: int
    n_index: int
    n_backend_arrays: int

    @classmethod
    def parse(cls, cell_idx, *args, jaxpr, n_grid_in, n_backend_arrays):
        if n_grid_in < 0:
            raise ValueError(f"n_grid_in must be >= 0, got {n_grid_in}")
        if n_backend_arrays < 0:
            raise ValueError(f"n_backend_arrays must be >= 0, got {n_backend_arrays}")
        n_invars = len(jaxpr.jaxpr.invars)
        if n_invars < 2 + n_grid_in:
            raise ValueError(
                f"jaxpr has {n_invars} invars but n_grid_in={n_grid_in} requires >= {2 + n_grid_in}"
            )
        n_nongrid_in = n_invars - n_grid_in - 2
        assert n_nongrid_in >= 0  # follows from the invar check above
        n_index = len(args) - n_grid_in - n_nongrid_in - n_backend_arrays
        if n_index < 0:
            raise ValueError(
                f"positional arg count {len(args)} is too small for "
                f"n_grid_in={n_grid_in}, n_nongrid_in={n_nongrid_in}, "
                f"n_backend_arrays={n_backend_arrays}"
            )

        grid_in = tuple(args[:n_grid_in])
        nongrid_in = tuple(args[n_grid_in : n_grid_in + n_nongrid_in])
        idx_end = n_grid_in + n_nongrid_in + n_index
        index_buffers = tuple(args[n_grid_in + n_nongrid_in : idx_end])
        backend_arrays = tuple(args[idx_end:])
        assert len(backend_arrays) == n_backend_arrays
        return cls(
            cell_idx=cell_idx,
            grid_in=grid_in,
            nongrid_in=nongrid_in,
            index_buffers=index_buffers,
            backend_arrays=backend_arrays,
            n_grid_in=n_grid_in,
            n_nongrid_in=n_nongrid_in,
            n_index=n_index,
            n_backend_arrays=n_backend_arrays,
        )

    def as_positional(self) -> tuple:
        return (
            self.cell_idx,
            *self.grid_in,
            *self.nongrid_in,
            *self.index_buffers,
            *self.backend_arrays,
        )


@dataclass(frozen=True)
class ScatterTangents:
    d_grid_in: tuple
    d_nongrid_in: tuple


@dataclass(frozen=True)
class IndexLayout:
    ic: tuple
    n_grid_in: int
    n_nongrid_in: int
    n_grid_out: int
    n_nongrid_out: int
    n_index: int

    @property
    def grid_in_offset(self) -> int:
        return 1

    @property
    def nongrid_in_offset(self) -> int:
        return 1 + self.n_grid_in

    @property
    def grid_out_offset(self) -> int:
        return 1 + self.n_grid_in + self.n_nongrid_in

    @property
    def nongrid_out_offset(self) -> int:
        return self.grid_out_offset + self.n_grid_out

    @property
    def idx_offset(self) -> int:
        return self.nongrid_out_offset + self.n_nongrid_out

    @property
    def n_total_outputs(self) -> int:
        return self.n_grid_out + self.n_nongrid_out

    @property
    def cell_idx(self):
        return self.ic[0]

    @property
    def grid_in(self) -> tuple:
        return tuple(self.ic[self.grid_in_offset : self.nongrid_in_offset])

    @property
    def nongrid_in(self) -> tuple:
        return tuple(self.ic[self.nongrid_in_offset : self.grid_out_offset])

    @property
    def grid_out(self) -> tuple:
        return tuple(self.ic[self.grid_out_offset : self.nongrid_out_offset])

    @property
    def nongrid_out(self) -> tuple:
        return tuple(self.ic[self.nongrid_out_offset : self.idx_offset])

    @property
    def out(self) -> tuple:
        return tuple(self.ic[self.grid_out_offset : self.idx_offset])

    @property
    def idx(self) -> tuple:
        return tuple(self.ic[self.idx_offset :])

    def split_buf_extents(self, buf_batch_extents) -> dict:
        return {
            "cell_idx": buf_batch_extents[0],
            "grid_in": tuple(buf_batch_extents[self.grid_in_offset : self.nongrid_in_offset]),
            "nongrid_in": tuple(buf_batch_extents[self.nongrid_in_offset : self.grid_out_offset]),
            "grid_out": tuple(buf_batch_extents[self.grid_out_offset : self.nongrid_out_offset]),
            "nongrid_out": tuple(buf_batch_extents[self.nongrid_out_offset : self.idx_offset]),
            "idx": tuple(buf_batch_extents[self.idx_offset :]),
        }


def support_is_separable(support):
    return len(support) > 0 and isinstance(support[0], int)


def support_3d(support):
    if support_is_separable(support):
        return tuple((dx, dy, dz) for dx in support for dy in support for dz in support)
    return support


def support_count(support):
    if support_is_separable(support):
        return len(support) ** 3
    return len(support)


def detect_cartesian_support(support):
    xs = sorted({s[0] for s in support})
    ys = sorted({s[1] for s in support})
    zs = sorted({s[2] for s in support})
    if len(support) != len(xs) * len(ys) * len(zs):
        return None
    if xs != ys or ys != zs:
        return None
    order = len(xs)
    mo = xs[0]
    if xs != list(range(mo, mo + order)):
        return None
    if set(support) != {(dx, dy, dz) for dx in xs for dy in ys for dz in zs}:
        return None
    return order, mo
