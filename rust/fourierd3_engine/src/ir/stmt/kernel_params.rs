// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kernel and device-function signatures: the parameter forms, the
//! `kernel_params!` builder macro, and the signature-plus-body render.

use crate::ir::code_builder::CodeBuilder;
use crate::ir::stmt::Stmt;
use crate::{emit, emit_ln};

#[derive(Clone, Debug)]
pub enum Param {
    Pointer {
        const_: bool,
        restrict: bool,
        ctype: String,
        name: String,
    },
    Scalar {
        ctype: String,
        name: String,
    },
}

impl Param {
    pub(crate) fn emit(&self, cb: &mut CodeBuilder) {
        match self {
            Param::Pointer {
                const_,
                restrict,
                ctype,
                name,
            } => {
                if *const_ {
                    emit!(cb, "const ");
                }
                cb.push_str(ctype);
                emit!(cb, "* ");
                if *restrict {
                    emit!(cb, "__restrict__ ");
                }
                cb.push_str(name);
            }
            Param::Scalar { ctype, name } => {
                cb.push_str(ctype);
                emit!(cb, " ");
                cb.push_str(name);
            }
        }
    }
}

pub(crate) fn emit_param_list(cb: &mut CodeBuilder, params: &[Param]) {
    let n = params.len();
    for (i, p) in params.iter().enumerate() {
        p.emit(cb);
        if i + 1 < n {
            emit_ln!(cb, ",");
        }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __kp_ptr {
    ($cty:expr, $const_:expr, $restrict:expr, $name:ident, $($rest:tt)*) => {{
        let mut v = ::std::vec![$crate::ir::stmt::Param::Pointer {
            const_: $const_, restrict: $restrict,
            ctype: $cty.into(), name: ::std::stringify!($name).to_owned(),
        }];
        v.extend($crate::kernel_params!($($rest)*));
        v
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __kp_scalar {
    ($cty:expr, $name:ident, $($rest:tt)*) => {{
        let mut v = ::std::vec![$crate::ir::stmt::Param::Scalar {
            ctype: $cty.into(), name: ::std::stringify!($name).to_owned(),
        }];
        v.extend($crate::kernel_params!($($rest)*));
        v
    }};
}

#[macro_export]
macro_rules! kernel_params {
    ($(,)?) => { ::std::vec::Vec::<$crate::ir::stmt::Param>::new() };

    (const $ty:ident * restrict $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_ptr!($crate::ctype!($ty), true, true, $name, $($($rest)*)?) };
    (const # $ty:tt * restrict $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_ptr!($ty, true, true, $name, $($($rest)*)?) };

    (const $ty:ident * $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_ptr!($crate::ctype!($ty), true, false, $name, $($($rest)*)?) };
    (const # $ty:tt * $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_ptr!($ty, true, false, $name, $($($rest)*)?) };

    ($ty:ident * restrict $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_ptr!($crate::ctype!($ty), false, true, $name, $($($rest)*)?) };
    (# $ty:tt * restrict $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_ptr!($ty, false, true, $name, $($($rest)*)?) };

    ($ty:ident * $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_ptr!($crate::ctype!($ty), false, false, $name, $($($rest)*)?) };
    (# $ty:tt * $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_ptr!($ty, false, false, $name, $($($rest)*)?) };

    ($ty:ident $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_scalar!($crate::ctype!($ty), $name, $($($rest)*)?) };
    (# $ty:tt $name:ident $(, $($rest:tt)*)?) =>
        { $crate::__kp_scalar!($ty, $name, $($($rest)*)?) };
}

pub(crate) fn emit_kernel_into(
    cb: &mut CodeBuilder,
    name: &str,
    launch_bounds: &str,
    params: &[Param],
    body: &[Stmt],
) {
    emit!(cb, "extern \"C\" __global__");
    if !launch_bounds.is_empty() {
        cb.push_str(launch_bounds);
    }
    emit!(cb, " void ");
    cb.push_str(name);
    emit_ln!(cb, "(");
    cb.block(|cb| emit_param_list(cb, params));
    emit_ln!(cb, ") {");
    cb.block(|cb| {
        for s in body {
            s.emit(cb);
        }
    });
    emit!(cb, "}");
    cb.newline();
}
