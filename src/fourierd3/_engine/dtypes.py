# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

# ids 0..=3 are fixed because FFI dtype wire ids mirror fourierd3_engine::dtype::Dtype
_DTYPE_ID: dict[np.dtype, int] = {
    np.dtype("float32"): 0,
    np.dtype("float64"): 1,
    np.dtype("float16"): 2,
    np.dtype("bfloat16"): 3,
    np.dtype("bool"): 4,
    np.dtype("int8"): 5,
    np.dtype("int16"): 6,
    np.dtype("int32"): 7,
    np.dtype("int64"): 8,
    np.dtype("uint8"): 9,
    np.dtype("uint16"): 10,
    np.dtype("uint32"): 11,
    np.dtype("uint64"): 12,
    np.dtype("complex64"): 13,
    np.dtype("complex128"): 14,
}


def dtype_id(dtype) -> int:
    key = np.dtype(dtype)
    if key not in _DTYPE_ID:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return _DTYPE_ID[key]
