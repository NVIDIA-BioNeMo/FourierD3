// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::xla_ffi::sys::{
    XLA_FFI_DataType, XLA_FFI_DataType_BF16, XLA_FFI_DataType_C64, XLA_FFI_DataType_C128,
    XLA_FFI_DataType_F16, XLA_FFI_DataType_F32, XLA_FFI_DataType_F64, XLA_FFI_DataType_PRED,
    XLA_FFI_DataType_S8, XLA_FFI_DataType_S16, XLA_FFI_DataType_S32, XLA_FFI_DataType_S64,
    XLA_FFI_DataType_U8, XLA_FFI_DataType_U16, XLA_FFI_DataType_U32, XLA_FFI_DataType_U64,
};

pub(crate) trait Dtype: Copy + 'static {
    const TAG: XLA_FFI_DataType;
    const NAME: &'static str;
}

macro_rules! impl_dtype {
    ($t:ty, $tag:ident, $name:literal) => {
        impl Dtype for $t {
            const TAG: XLA_FFI_DataType = $tag;
            const NAME: &'static str = $name;
        }
    };
}

impl_dtype!(u64, XLA_FFI_DataType_U64, "u64");

pub(crate) fn dtype_size(dt: XLA_FFI_DataType) -> Option<usize> {
    const SIZES: &[(XLA_FFI_DataType, usize)] = &[
        (XLA_FFI_DataType_PRED, 1),
        (XLA_FFI_DataType_S8, 1),
        (XLA_FFI_DataType_U8, 1),
        (XLA_FFI_DataType_S16, 2),
        (XLA_FFI_DataType_U16, 2),
        (XLA_FFI_DataType_F16, 2),
        (XLA_FFI_DataType_BF16, 2),
        (XLA_FFI_DataType_S32, 4),
        (XLA_FFI_DataType_U32, 4),
        (XLA_FFI_DataType_F32, 4),
        (XLA_FFI_DataType_S64, 8),
        (XLA_FFI_DataType_U64, 8),
        (XLA_FFI_DataType_F64, 8),
        (XLA_FFI_DataType_C64, 8),
        (XLA_FFI_DataType_C128, 16),
    ];
    SIZES
        .iter()
        .find_map(|&(tag, size)| (tag == dt).then_some(size))
}
