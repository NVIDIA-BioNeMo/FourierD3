// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rendering an expression to CUDA source, parenthesizing only where
//! the operator precedence of the parent demands it.

use crate::ir::code_builder::CodeBuilder;
use crate::ir::expr::{ASSIGN_PREC, CAST_PREC, Expr, FloatBits, POSTFIX_PREC, TERNARY_PREC};
use crate::{cb, emit};

impl Expr {
    pub fn render(&self) -> String {
        let mut cb = CodeBuilder::new();
        self.emit(&mut cb);
        cb.finish_string()
    }

    pub(crate) fn emit(&self, cb: &mut CodeBuilder) {
        self.emit_into(cb, 0);
    }

    fn emit_into(&self, cb: &mut CodeBuilder, parent_prec: u8) {
        match self {
            Expr::Var(name) => cb.push_str(name),
            Expr::Lit(v) => cb!(cb, "{}", v),
            Expr::FloatLit(v, bits) => {
                cb!(cb, "{:e}", v);
                if matches!(bits, FloatBits::F32) {
                    emit!(cb, "f");
                }
            }
            Expr::Index(arr, idx) => {
                arr.emit_into(cb, POSTFIX_PREC);
                emit!(cb, "[");
                idx.emit_into(cb, 0);
                emit!(cb, "]");
            }
            Expr::Unary(op, e) => {
                op.emit_symbol(cb);
                e.emit_into(cb, POSTFIX_PREC);
            }
            Expr::PostInc(target) => {
                target.emit_into(cb, POSTFIX_PREC);
                emit!(cb, "++");
            }
            Expr::Call(name, args) => {
                Self::emit_call(cb, name, args);
            }
            Expr::BinOp(op, lhs, rhs) => {
                Self::emit_binary(cb, parent_prec, *op, lhs, rhs);
            }
            Expr::Ternary { cond, then_, else_ } => {
                Self::emit_ternary(cb, parent_prec, cond, then_, else_);
            }
            Expr::Assign { op, target, value } => {
                Self::emit_assignment(cb, parent_prec, *op, target, value);
            }
            Expr::Cast { ty, expr } => {
                Self::emit_cast(cb, parent_prec, ty, expr);
            }
            Expr::Member { base, name } => {
                base.emit_into(cb, POSTFIX_PREC);
                emit!(cb, ".");
                cb.push_str(name);
            }
        }
    }

    fn emit_call(cb: &mut CodeBuilder, name: &str, args: &[Expr]) {
        cb.push_str(name);
        emit!(cb, "(");
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                emit!(cb, ", ");
            }
            arg.emit_into(cb, 0);
        }
        emit!(cb, ")");
    }

    fn emit_binary(
        cb: &mut CodeBuilder,
        parent_prec: u8,
        op: crate::ir::expr::Op,
        lhs: &Expr,
        rhs: &Expr,
    ) {
        let prec = op.precedence();
        let needs_parens = prec < parent_prec;
        emit_if(cb, needs_parens, "(");
        lhs.emit_into(cb, prec);
        op.emit_symbol(cb);
        rhs.emit_into(cb, prec + 1);
        emit_if(cb, needs_parens, ")");
    }

    fn emit_ternary(
        cb: &mut CodeBuilder,
        parent_prec: u8,
        cond: &Expr,
        then_: &Expr,
        else_: &Expr,
    ) {
        let needs_parens = TERNARY_PREC < parent_prec;
        emit_if(cb, needs_parens, "(");
        cond.emit_into(cb, TERNARY_PREC + 1);
        emit!(cb, " ? ");
        then_.emit_into(cb, TERNARY_PREC + 1);
        emit!(cb, " : ");
        else_.emit_into(cb, TERNARY_PREC);
        emit_if(cb, needs_parens, ")");
    }

    fn emit_assignment(
        cb: &mut CodeBuilder,
        parent_prec: u8,
        op: crate::ir::expr::AssignOp,
        target: &Expr,
        value: &Expr,
    ) {
        let needs_parens = ASSIGN_PREC < parent_prec;
        emit_if(cb, needs_parens, "(");
        target.emit_into(cb, ASSIGN_PREC + 1);
        op.emit_symbol(cb);
        value.emit_into(cb, ASSIGN_PREC);
        emit_if(cb, needs_parens, ")");
    }

    fn emit_cast(cb: &mut CodeBuilder, parent_prec: u8, ty: &str, expr: &Expr) {
        let needs_parens = CAST_PREC < parent_prec;
        emit_if(cb, needs_parens, "(");
        emit!(cb, "(");
        cb.push_str(ty);
        emit!(cb, ")(");
        expr.emit_into(cb, 0);
        emit!(cb, ")");
        emit_if(cb, needs_parens, ")");
    }

    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Lit(0))
    }
}

fn emit_if(cb: &mut CodeBuilder, condition: bool, text: &str) {
    if condition {
        cb.push_str(text);
    }
}
