// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rendering statements and control flow.

use crate::ir::code_builder::CodeBuilder;
use crate::ir::expr::{AssignOp, Expr};
use crate::ir::stmt::{Decl, ForInit, Param, Stmt, Unroll, emit_kernel_into};
use crate::{cb, emit, emit_ln};

impl Stmt {
    pub fn decl(ty: impl Into<String>, name: impl Into<String>, init: Option<Expr>) -> Self {
        Stmt::Decl(Decl::scalar(ty, name, init))
    }

    pub fn decl_multi(ty: impl Into<String>, decls: Vec<(String, Option<Expr>)>) -> Self {
        Stmt::Decl(Decl::multi(ty, decls))
    }

    pub fn array_decl(ty: impl Into<String>, name: impl Into<String>, dims: Vec<i64>) -> Self {
        Stmt::Decl(Decl::array(ty, name, dims))
    }

    pub fn array_init_decl(
        ty: impl Into<String>,
        name: impl Into<String>,
        size: Option<i64>,
        init: Vec<Expr>,
    ) -> Self {
        Stmt::Decl(Decl::array_init(ty, name, size, init))
    }

    pub fn shared_decl(ty: impl Into<String>, name: impl Into<String>, size: Option<i64>) -> Self {
        Stmt::Decl(Decl::shared(ty, name, size))
    }

    pub fn extern_shared_decl(ty: impl Into<String>, name: impl Into<String>) -> Self {
        Stmt::Decl(Decl::extern_shared(ty, name))
    }

    pub fn constant_array_decl(
        ty: impl Into<String>,
        name: impl Into<String>,
        size: Option<i64>,
        init: Vec<Expr>,
    ) -> Self {
        Stmt::Decl(Decl::constant_array(ty, name, size, init))
    }

    pub fn assign(lhs: Expr, op: AssignOp, rhs: Expr) -> Self {
        Stmt::Eval(Expr::assign(op, lhs, rhs))
    }

    pub(crate) fn emit(&self, cb: &mut CodeBuilder) {
        match self {
            Stmt::Decl(d) => {
                d.emit(cb);
                emit_ln!(cb, ";");
            }
            Stmt::Eval(e) => {
                e.emit(cb);
                emit_ln!(cb, ";");
            }
            Stmt::ExternDeviceDecl { name, param_types } => {
                Self::emit_extern_device(cb, name, param_types);
            }
            Stmt::DeviceFn {
                name,
                ret_ty,
                params,
                body,
                forceinline,
            } => {
                Self::emit_device_fn(cb, name, ret_ty, params, body, *forceinline);
            }
            Stmt::Return(value) => Self::emit_return(cb, value.as_ref()),
            Stmt::If { cond, then_, else_ } => Self::emit_if(cb, cond, then_, else_.as_deref()),
            Stmt::For {
                init,
                cond,
                step,
                body,
                unroll,
            } => {
                Self::emit_for(cb, init, cond, step, body, *unroll);
            }
            Stmt::While { cond, body } => Self::emit_while(cb, cond, body),
            Stmt::Continue => emit_ln!(cb, "continue;"),
            Stmt::Define { name, value } => {
                Self::emit_define(cb, name, value);
            }
            Stmt::Raw(s) => Self::emit_raw(cb, s),
            Stmt::Kernel {
                name,
                launch_bounds,
                params,
                body,
            } => {
                emit_kernel_into(cb, name, launch_bounds, params, body);
            }
            Stmt::Blank => {
                cb.newline();
            }
        }
    }

    fn emit_extern_device(cb: &mut CodeBuilder, name: &str, param_types: &[String]) {
        emit!(cb, "extern \"C\" __device__ void ");
        cb.push_str(name);
        emit!(cb, "(");
        for (i, ty) in param_types.iter().enumerate() {
            if i > 0 {
                emit!(cb, ", ");
            }
            cb.push_str(ty);
        }
        emit_ln!(cb, ");");
    }

    fn emit_device_fn(
        cb: &mut CodeBuilder,
        name: &str,
        ret_ty: &str,
        params: &[Param],
        body: &[Stmt],
        forceinline: bool,
    ) {
        emit!(cb, "__device__");
        if forceinline {
            emit!(cb, " __forceinline__");
        }
        emit!(cb, " ");
        cb.push_str(ret_ty);
        emit!(cb, " ");
        cb.push_str(name);
        emit!(cb, "(");
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                emit!(cb, ", ");
            }
            param.emit(cb);
        }
        emit_ln!(cb, ") {");
        Self::emit_block(cb, body);
        emit_ln!(cb, "}");
    }

    fn emit_return(cb: &mut CodeBuilder, value: Option<&Expr>) {
        emit!(cb, "return");
        if let Some(value) = value {
            emit!(cb, " ");
            value.emit(cb);
        }
        emit_ln!(cb, ";");
    }

    fn emit_if(cb: &mut CodeBuilder, cond: &Expr, then_: &[Stmt], else_: Option<&[Stmt]>) {
        emit!(cb, "if (");
        cond.emit(cb);
        emit_ln!(cb, ") {");
        Self::emit_block(cb, then_);
        match else_ {
            None => emit_ln!(cb, "}"),
            Some([nested @ Stmt::If { .. }]) => {
                emit!(cb, "} else ");
                nested.emit(cb);
            }
            Some(stmts) => {
                emit_ln!(cb, "} else {");
                Self::emit_block(cb, stmts);
                emit_ln!(cb, "}");
            }
        }
    }

    fn emit_for(
        cb: &mut CodeBuilder,
        init: &ForInit,
        cond: &Expr,
        step: &Expr,
        body: &[Stmt],
        unroll: Unroll,
    ) {
        Self::emit_unroll(cb, unroll);
        emit!(cb, "for (");
        init.emit(cb);
        emit!(cb, "; ");
        cond.emit(cb);
        emit!(cb, "; ");
        step.emit(cb);
        emit_ln!(cb, ") {");
        Self::emit_block(cb, body);
        emit_ln!(cb, "}");
    }

    fn emit_unroll(cb: &mut CodeBuilder, unroll: Unroll) {
        match unroll {
            Unroll::None => {}
            Unroll::All => emit_ln!(cb, "#pragma unroll"),
            Unroll::Count(n) => cb!(cb, "#pragma unroll {}\n", n),
        }
    }

    fn emit_while(cb: &mut CodeBuilder, cond: &Expr, body: &[Stmt]) {
        emit!(cb, "while (");
        cond.emit(cb);
        emit_ln!(cb, ") {");
        Self::emit_block(cb, body);
        emit_ln!(cb, "}");
    }

    fn emit_define(cb: &mut CodeBuilder, name: &str, value: &str) {
        emit!(cb, "#define ");
        cb.push_str(name);
        if !value.is_empty() {
            emit!(cb, " ");
            cb.push_str(value);
        }
        cb.newline();
    }

    fn emit_raw(cb: &mut CodeBuilder, source: &str) {
        cb.push_str(source);
        if !source.ends_with('\n') {
            cb.newline();
        }
    }

    fn emit_block(cb: &mut CodeBuilder, body: &[Stmt]) {
        cb.block(|cb| {
            for stmt in body {
                stmt.emit(cb);
            }
        });
    }
}
