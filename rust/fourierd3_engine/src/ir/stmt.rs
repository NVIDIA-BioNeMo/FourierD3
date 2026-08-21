// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The statement half of the CUDA IR: what a statement can be, and the entry
//! points that render a module of them into CUDA source.

use crate::emit;
use crate::ir::code_builder::CodeBuilder;
use crate::ir::expr::Expr;

mod emit_decl;
mod emit_stmt;
pub(crate) mod kernel_params;
#[cfg(test)]
mod tests;

pub use kernel_params::Param;
pub(crate) use kernel_params::emit_kernel_into;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Unroll {
    None,
    All,
    Count(i64),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Storage {
    Plain,
    Shared,
    ExternShared,
    Constant,
}

impl Storage {
    fn emit_prefix(self, cb: &mut CodeBuilder) {
        match self {
            Storage::Plain => {}
            Storage::Shared => emit!(cb, "__shared__ "),
            Storage::ExternShared => emit!(cb, "extern __shared__ "),
            Storage::Constant => emit!(cb, "__constant__ "),
        }
    }
}

#[derive(Clone, Debug)]
pub enum DeclInit {
    Scalar(Expr),
    List(Vec<Expr>),
}

#[derive(Clone, Debug)]
pub struct Declarator {
    pub name: String,
    pub dims: Vec<Option<i64>>,
    pub init: Option<DeclInit>,
}

#[derive(Clone, Debug)]
pub struct Decl {
    pub storage: Storage,
    pub ty: String,
    pub declarators: Vec<Declarator>,
}

#[derive(Clone, Debug)]
pub enum ForInit {
    Decl(Decl),
    Expr(Expr),
}

impl ForInit {
    fn emit(&self, cb: &mut CodeBuilder) {
        match self {
            ForInit::Decl(d) => d.emit(cb),
            ForInit::Expr(e) => e.emit(cb),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Decl(Decl),
    Eval(Expr),
    ExternDeviceDecl {
        name: String,
        param_types: Vec<String>,
    },
    DeviceFn {
        name: String,
        ret_ty: String,
        params: Vec<Param>,
        body: Vec<Stmt>,
        forceinline: bool,
    },
    Return(Option<Expr>),
    If {
        cond: Expr,
        then_: Vec<Stmt>,
        else_: Option<Vec<Stmt>>,
    },
    For {
        init: ForInit,
        cond: Expr,
        step: Expr,
        body: Vec<Stmt>,
        unroll: Unroll,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    Continue,
    Define {
        name: String,
        value: String,
    },
    Raw(String),
    Kernel {
        name: String,
        launch_bounds: String,
        params: Vec<Param>,
        body: Vec<Stmt>,
    },
    Blank,
}

pub fn render_module(stmts: &[Stmt]) -> Vec<u8> {
    let mut cb = CodeBuilder::new();
    for s in stmts {
        s.emit(&mut cb);
    }
    cb.finish()
}

pub fn render_module_string(stmts: &[Stmt]) -> String {
    let mut cb = CodeBuilder::new();
    for s in stmts {
        s.emit(&mut cb);
    }
    cb.finish_string()
}
