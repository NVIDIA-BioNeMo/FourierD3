// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The expression half of the CUDA IR: operators, operands, and the
//! conversions that let plain Rust values stand in for expressions.

mod construct;
mod render;
#[cfg(test)]
mod tests;

use crate::emit;
use crate::ir::code_builder::CodeBuilder;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Op {
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

impl Op {
    fn emit_symbol(self, cb: &mut CodeBuilder) {
        match self {
            Op::Add => emit!(cb, " + "),
            Op::Sub => emit!(cb, " - "),
            Op::Mul => emit!(cb, " * "),
            Op::Div => emit!(cb, " / "),
            Op::Mod => emit!(cb, " % "),
            Op::Eq => emit!(cb, " == "),
            Op::Ne => emit!(cb, " != "),
            Op::Lt => emit!(cb, " < "),
            Op::Le => emit!(cb, " <= "),
            Op::Gt => emit!(cb, " > "),
            Op::Ge => emit!(cb, " >= "),
            Op::Shl => emit!(cb, " << "),
            Op::Shr => emit!(cb, " >> "),
            Op::BitAnd => emit!(cb, " & "),
            Op::BitOr => emit!(cb, " | "),
            Op::LogicalAnd => emit!(cb, " && "),
            Op::LogicalOr => emit!(cb, " || "),
        }
    }

    fn precedence(self) -> u8 {
        match self {
            Op::LogicalOr => 3,
            Op::LogicalAnd => 4,
            Op::BitOr => 5,
            Op::BitAnd => 7,
            Op::Eq | Op::Ne => 8,
            Op::Lt | Op::Le | Op::Gt | Op::Ge => 9,
            Op::Shl | Op::Shr => 10,
            Op::Add | Op::Sub => 11,
            Op::Mul | Op::Div | Op::Mod => 12,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AssignOp {
    Set,
    AddAssign,
    SubAssign,
    DivAssign,
    BitAndAssign,
    ShrAssign,
}

impl AssignOp {
    fn emit_symbol(self, cb: &mut CodeBuilder) {
        match self {
            AssignOp::Set => emit!(cb, " = "),
            AssignOp::AddAssign => emit!(cb, " += "),
            AssignOp::SubAssign => emit!(cb, " -= "),
            AssignOp::DivAssign => emit!(cb, " /= "),
            AssignOp::BitAndAssign => emit!(cb, " &= "),
            AssignOp::ShrAssign => emit!(cb, " >>= "),
        }
    }
}

pub(crate) const POSTFIX_PREC: u8 = 14;
// Cast sits at 13: tighter than every binop (max 12), looser than postfix 14.
// A cast used as a postfix operand therefore parenthesises itself
// (`((float4*)p)[i]`), while one used as a binop operand does not.
pub(crate) const CAST_PREC: u8 = 13;
pub(crate) const TERNARY_PREC: u8 = 2;
pub(crate) const ASSIGN_PREC: u8 = 1;
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnOp {
    Addr,
    Neg,
}

impl UnOp {
    fn emit_symbol(self, cb: &mut CodeBuilder) {
        match self {
            UnOp::Addr => emit!(cb, "&"),
            UnOp::Neg => emit!(cb, "-"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FloatBits {
    F32,
    F64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var(String),
    Lit(i64),
    FloatLit(f64, FloatBits),
    BinOp(Op, Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    PostInc(Box<Expr>),
    Call(String, Vec<Expr>),
    Ternary {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Box<Expr>,
    },
    Assign {
        op: AssignOp,
        target: Box<Expr>,
        value: Box<Expr>,
    },
    Cast {
        ty: String,
        expr: Box<Expr>,
    },
    Member {
        base: Box<Expr>,
        name: String,
    },
}

pub trait IntoExpr {
    fn into_expr(self) -> Expr;
}

impl IntoExpr for Expr {
    fn into_expr(self) -> Expr {
        self
    }
}
impl IntoExpr for &Expr {
    fn into_expr(self) -> Expr {
        self.clone()
    }
}
impl IntoExpr for &str {
    fn into_expr(self) -> Expr {
        Expr::var(self)
    }
}
impl IntoExpr for String {
    fn into_expr(self) -> Expr {
        Expr::var(self)
    }
}
impl IntoExpr for &String {
    fn into_expr(self) -> Expr {
        Expr::var(self.as_str())
    }
}
impl IntoExpr for f32 {
    fn into_expr(self) -> Expr {
        Expr::FloatLit(self as f64, FloatBits::F32)
    }
}
impl IntoExpr for f64 {
    fn into_expr(self) -> Expr {
        Expr::FloatLit(self, FloatBits::F64)
    }
}
impl IntoExpr for i32 {
    fn into_expr(self) -> Expr {
        Expr::lit(self as i64)
    }
}
impl IntoExpr for i64 {
    fn into_expr(self) -> Expr {
        Expr::lit(self)
    }
}
impl IntoExpr for usize {
    fn into_expr(self) -> Expr {
        Expr::lit(self as i64)
    }
}
