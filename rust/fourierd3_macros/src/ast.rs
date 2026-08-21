// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The syntax tree the `cuda!` grammar parses into, one step above the
//! `fourierd3_engine` statement IR it is emitted as.

use proc_macro2::TokenStream as TokenStream2;
use syn::Expr as SynExpr;

pub(crate) struct CudaInput {
    pub(crate) target: SynExpr,
    pub(crate) stmts: Vec<CStmt>,
}

pub(crate) enum CType {
    Lit(String),
    Interp(TokenStream2),
}

pub(crate) enum CName {
    Lit(String),
    Interp(TokenStream2),
}

pub(crate) enum CSize {
    Lit(i64),
    Interp(TokenStream2),
}

pub(crate) enum CExpr {
    Var(String),
    Lit(i64),
    Interp(TokenStream2),
    BinOp(COp, Box<CExpr>, Box<CExpr>),
    Index(Box<CExpr>, Box<CExpr>),
    Addr(Box<CExpr>),
    Neg(Box<CExpr>),
    PostInc(Box<CExpr>),
    Ternary {
        cond: Box<CExpr>,
        then_: Box<CExpr>,
        else_: Box<CExpr>,
    },
    Assign {
        op: CAssignOp,
        target: Box<CExpr>,
        value: Box<CExpr>,
    },
    Cast {
        ty: CType,
        expr: Box<CExpr>,
    },
    Call(CName, Vec<CExpr>),
    Member(Box<CExpr>, String),
}

#[derive(Copy, Clone)]
pub(crate) enum COp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    LogicalAnd,
    LogicalOr,
}

impl COp {
    pub(crate) fn precedence(self) -> u8 {
        match self {
            COp::LogicalOr => 0,
            COp::LogicalAnd => 1,
            COp::BitOr => 2,
            COp::BitAnd => 3,
            COp::Eq | COp::Ne => 4,
            COp::Lt | COp::Le | COp::Gt | COp::Ge => 5,
            COp::Shl | COp::Shr => 6,
            COp::Add | COp::Sub => 7,
            COp::Mul | COp::Div | COp::Mod => 8,
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) enum CAssignOp {
    Set,
    AddAssign,
    SubAssign,
    DivAssign,
    BitAndAssign,
    ShrAssign,
}

pub(crate) enum CUnroll {
    None,
    All,
}

pub(crate) enum CStmt {
    Decl {
        ty: CType,
        decls: Vec<(CName, Option<CExpr>)>,
    },
    ArrayDecl {
        ty: CType,
        name: CName,
        dims: Vec<CSize>,
    },
    ArrayInitDecl {
        ty: CType,
        name: CName,
        size: Option<CSize>,
        init: Vec<CExpr>,
    },
    ExternSharedDecl {
        ty: CType,
        name: CName,
    },
    ExternDeviceDecl {
        name: CName,
        param_types: Vec<CType>,
    },
    Eval(CExpr),
    If {
        cond: CExpr,
        then_: Vec<CStmt>,
        else_: Option<Vec<CStmt>>,
    },
    For {
        init: Box<CStmt>,
        cond: CExpr,
        step: CExpr,
        body: Vec<CStmt>,
        unroll: CUnroll,
    },
    While {
        cond: CExpr,
        body: Vec<CStmt>,
    },
    Continue,
    Return(Option<CExpr>),
    Blank,
    Splice(TokenStream2),
}
