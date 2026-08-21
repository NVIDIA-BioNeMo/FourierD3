// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building expressions, with the constant folding that happens at
//! construction time so a folded operand never reaches the renderer.

use crate::ir::expr::{AssignOp, Expr, IntoExpr, Op, UnOp};

#[allow(clippy::should_implement_trait)]
impl Expr {
    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    pub fn lit(v: i64) -> Self {
        Expr::Lit(v)
    }

    pub fn index(arr: impl IntoExpr, idx: Expr) -> Self {
        Expr::Index(Box::new(arr.into_expr()), Box::new(idx))
    }

    pub fn addr(e: Expr) -> Self {
        Expr::Unary(UnOp::Addr, Box::new(e))
    }

    pub fn neg(e: Expr) -> Self {
        match e {
            Expr::Lit(n) => Expr::Lit(-n),
            Expr::FloatLit(v, bits) => Expr::FloatLit(-v, bits),
            _ => Expr::Unary(UnOp::Neg, Box::new(e)),
        }
    }

    pub fn post_inc(target: Expr) -> Self {
        Expr::PostInc(Box::new(target))
    }

    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::Call(name.into(), args)
    }

    pub fn member(base: impl IntoExpr, name: impl Into<String>) -> Self {
        Expr::Member {
            base: Box::new(base.into_expr()),
            name: name.into(),
        }
    }

    pub fn add(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (Expr::Lit(0), _) => b,
            (_, Expr::Lit(0)) => a,
            (Expr::Lit(x), Expr::Lit(y)) => Expr::Lit(x + y),
            _ => Expr::BinOp(Op::Add, Box::new(a), Box::new(b)),
        }
    }

    pub fn sub(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (_, Expr::Lit(0)) => a,
            (Expr::Lit(x), Expr::Lit(y)) => Expr::Lit(x - y),
            (Expr::Var(x), Expr::Var(y)) if x == y => Expr::Lit(0),
            _ => Expr::BinOp(Op::Sub, Box::new(a), Box::new(b)),
        }
    }

    pub fn product<I>(factors: I) -> Self
    where
        I: IntoIterator<Item = Expr>,
    {
        let mut it = factors.into_iter();
        let first = it.next().expect("Expr::product: empty factor list");
        it.fold(first, Expr::mul)
    }

    pub fn mul(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (Expr::Lit(0), _) | (_, Expr::Lit(0)) => Expr::Lit(0),
            (Expr::Lit(1), _) => b,
            (_, Expr::Lit(1)) => a,
            (Expr::Lit(x), Expr::Lit(y)) => Expr::Lit(x * y),
            _ => Expr::BinOp(Op::Mul, Box::new(a), Box::new(b)),
        }
    }

    pub fn div(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (_, Expr::Lit(1)) => a,
            (Expr::Lit(x), Expr::Lit(y)) if *y != 0 => Expr::Lit(x / y),
            _ => Expr::BinOp(Op::Div, Box::new(a), Box::new(b)),
        }
    }

    pub fn rem(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (_, Expr::Lit(1)) => Expr::Lit(0),
            (Expr::Lit(x), Expr::Lit(y)) if *y != 0 => Expr::Lit(x % y),
            _ => Expr::BinOp(Op::Mod, Box::new(a), Box::new(b)),
        }
    }

    pub fn eq(a: Expr, b: Expr) -> Self {
        Expr::BinOp(Op::Eq, Box::new(a), Box::new(b))
    }
    pub fn ne(a: Expr, b: Expr) -> Self {
        Expr::BinOp(Op::Ne, Box::new(a), Box::new(b))
    }
    pub fn lt(a: Expr, b: Expr) -> Self {
        Expr::BinOp(Op::Lt, Box::new(a), Box::new(b))
    }
    pub fn le(a: Expr, b: Expr) -> Self {
        Expr::BinOp(Op::Le, Box::new(a), Box::new(b))
    }
    pub fn gt(a: Expr, b: Expr) -> Self {
        Expr::BinOp(Op::Gt, Box::new(a), Box::new(b))
    }
    pub fn ge(a: Expr, b: Expr) -> Self {
        Expr::BinOp(Op::Ge, Box::new(a), Box::new(b))
    }

    pub fn shr(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (_, Expr::Lit(0)) => a,
            (Expr::Lit(x), Expr::Lit(y)) if *y >= 0 && *y < 64 => Expr::Lit(x >> y),
            _ => Expr::BinOp(Op::Shr, Box::new(a), Box::new(b)),
        }
    }

    pub fn band(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (Expr::Lit(0), _) | (_, Expr::Lit(0)) => Expr::Lit(0),
            (Expr::Lit(x), Expr::Lit(y)) => Expr::Lit(x & y),
            _ => Expr::BinOp(Op::BitAnd, Box::new(a), Box::new(b)),
        }
    }

    pub fn land(a: Expr, b: Expr) -> Self {
        Expr::BinOp(Op::LogicalAnd, Box::new(a), Box::new(b))
    }

    pub fn lor(a: Expr, b: Expr) -> Self {
        Expr::BinOp(Op::LogicalOr, Box::new(a), Box::new(b))
    }

    pub fn bor(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (Expr::Lit(0), _) => b,
            (_, Expr::Lit(0)) => a,
            (Expr::Lit(x), Expr::Lit(y)) => Expr::Lit(x | y),
            _ => Expr::BinOp(Op::BitOr, Box::new(a), Box::new(b)),
        }
    }

    pub fn shl(a: Expr, b: Expr) -> Self {
        match (&a, &b) {
            (_, Expr::Lit(0)) => a,
            (Expr::Lit(x), Expr::Lit(y)) if *y >= 0 && *y < 64 => Expr::Lit(x << y),
            _ => Expr::BinOp(Op::Shl, Box::new(a), Box::new(b)),
        }
    }

    pub fn ternary(cond: Expr, then_: Expr, else_: Expr) -> Self {
        Expr::Ternary {
            cond: Box::new(cond),
            then_: Box::new(then_),
            else_: Box::new(else_),
        }
    }

    pub fn assign(op: AssignOp, target: Expr, value: Expr) -> Self {
        Expr::Assign {
            op,
            target: Box::new(target),
            value: Box::new(value),
        }
    }

    pub fn cast(ty: impl Into<String>, expr: Expr) -> Self {
        Expr::Cast {
            ty: ty.into(),
            expr: Box::new(expr),
        }
    }
}
