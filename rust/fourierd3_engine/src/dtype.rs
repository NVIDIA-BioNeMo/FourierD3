// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::ir::expr::{Expr, IntoExpr};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F64,
    F16,
    Bf16,
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Complex64,
    Complex128,
}

impl Dtype {
    pub fn ctype(self) -> &'static str {
        match self {
            Dtype::F32 => "float",
            Dtype::F64 => "double",
            Dtype::F16 => "__half",
            Dtype::Bf16 => "__nv_bfloat16",
            Dtype::Bool => "bool",
            Dtype::I8 => "signed char",
            Dtype::I16 => "short",
            Dtype::I32 => "int",
            Dtype::I64 => "long long",
            Dtype::U8 => "unsigned char",
            Dtype::U16 => "unsigned short",
            Dtype::U32 => "unsigned int",
            Dtype::U64 => "unsigned long long",
            Dtype::Complex64 => "cuFloatComplex",
            Dtype::Complex128 => "cuDoubleComplex",
        }
    }

    pub fn from_id(id: i32) -> Result<Dtype, String> {
        Ok(match id {
            0 => Dtype::F32,
            1 => Dtype::F64,
            2 => Dtype::F16,
            3 => Dtype::Bf16,
            4 => Dtype::Bool,
            5 => Dtype::I8,
            6 => Dtype::I16,
            7 => Dtype::I32,
            8 => Dtype::I64,
            9 => Dtype::U8,
            10 => Dtype::U16,
            11 => Dtype::U32,
            12 => Dtype::U64,
            13 => Dtype::Complex64,
            14 => Dtype::Complex128,
            v => return Err(format!("unknown dtype id {v}")),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::I8 | Dtype::U8 => 1,
            Dtype::F16 | Dtype::Bf16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F64 | Dtype::I64 | Dtype::U64 | Dtype::Complex64 => 8,
            Dtype::Complex128 => 16,
        }
    }

    pub fn device_pointer_element(self) -> Option<&'static str> {
        match self {
            Dtype::Complex64 => Some("float"),
            Dtype::Complex128 => Some("double"),
            // LLVM IR has no unsigned integer types; unsigned buffers are named with their signed C spelling.
            Dtype::U8 => Some("signed char"),
            Dtype::U16 => Some("short"),
            Dtype::U32 => Some("int"),
            Dtype::U64 => Some("long long"),
            _ => None,
        }
    }

    pub fn zero(self) -> Expr {
        match self {
            Dtype::F16 | Dtype::Bf16 => Expr::cast(self.ctype(), (0.0f32).into_expr()),
            _ => Expr::lit(0),
        }
    }

    pub fn lit(self, v: f64) -> Expr {
        match self {
            Dtype::F32 => (v as f32).into_expr(),
            Dtype::F64 => v.into_expr(),
            Dtype::F16 => Expr::cast("__half", (v as f32).into_expr()),
            Dtype::Bf16 => Expr::cast("__nv_bfloat16", (v as f32).into_expr()),
            other => Expr::cast(other.ctype(), (v as f32).into_expr()),
        }
    }
}

impl From<Dtype> for String {
    fn from(d: Dtype) -> Self {
        d.ctype().to_string()
    }
}

impl From<&Dtype> for String {
    fn from(d: &Dtype) -> Self {
        d.ctype().to_string()
    }
}

#[macro_export]
macro_rules! ctype {
    (bool) => {
        $crate::dtype::Dtype::Bool.ctype()
    };
    (i8) => {
        $crate::dtype::Dtype::I8.ctype()
    };
    (i16) => {
        $crate::dtype::Dtype::I16.ctype()
    };
    (i32) => {
        $crate::dtype::Dtype::I32.ctype()
    };
    (i64) => {
        $crate::dtype::Dtype::I64.ctype()
    };
    (u8) => {
        $crate::dtype::Dtype::U8.ctype()
    };
    (u16) => {
        $crate::dtype::Dtype::U16.ctype()
    };
    (u32) => {
        $crate::dtype::Dtype::U32.ctype()
    };
    (u64) => {
        $crate::dtype::Dtype::U64.ctype()
    };
    (f16) => {
        $crate::dtype::Dtype::F16.ctype()
    };
    (bf16) => {
        $crate::dtype::Dtype::Bf16.ctype()
    };
    (f32) => {
        $crate::dtype::Dtype::F32.ctype()
    };
    (f64) => {
        $crate::dtype::Dtype::F64.ctype()
    };
    (c64) => {
        $crate::dtype::Dtype::Complex64.ctype()
    };
    (c128) => {
        $crate::dtype::Dtype::Complex128.ctype()
    };
}
