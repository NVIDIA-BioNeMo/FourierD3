// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing of expressions: precedence climbing over the operator table,
//! unary forms, calls, indexing, and the assignment operators.

use quote::quote;
use syn::parse::ParseStream;
use syn::parse::discouraged::Speculative;
use syn::{Expr as SynExpr, Ident, LitInt, Result, Token, braced, bracketed, parenthesized};

use crate::ast::{CAssignOp, CExpr, CName, COp, CType};
use crate::keywords::punct::PlusPlus;
use crate::statements::parse_call_args;
use crate::type_names::{apply_ptr, looks_like_unary_start, parse_lit_i64, parse_type};

pub(crate) fn parse_expr(input: ParseStream) -> Result<CExpr> {
    let cond = parse_expr_prec(input, 0)?;
    if input.peek(Token![?]) {
        let _: Token![?] = input.parse()?;
        let then_ = parse_expr(input)?;
        let _: Token![:] = input
            .parse()
            .map_err(|_| input.error("expected `:` in ternary `<cond> ? <then> : <else>`"))?;
        let else_ = parse_expr(input)?;
        return Ok(CExpr::Ternary {
            cond: Box::new(cond),
            then_: Box::new(then_),
            else_: Box::new(else_),
        });
    }
    Ok(cond)
}

pub(crate) fn parse_expr_prec(input: ParseStream, min_prec: u8) -> Result<CExpr> {
    let mut left = parse_unary(input)?;
    while let Some(op) = peek_binop(input) {
        let prec = op.precedence();
        if prec < min_prec {
            break;
        }
        consume_binop(input)?;
        let right = parse_expr_prec(input, prec + 1)?;
        left = CExpr::BinOp(op, Box::new(left), Box::new(right));
    }
    Ok(left)
}

pub(crate) fn peek_binop(input: ParseStream) -> Option<COp> {
    if peek_assign_op(input).is_some() {
        return None;
    }
    parse_binop(&input.fork()).ok()
}

pub(crate) fn consume_binop(input: ParseStream) -> Result<()> {
    parse_binop(input)?;
    Ok(())
}

macro_rules! parse_binary_token {
    ($input:expr; $([$($token:tt)+] => $op:expr),+ $(,)?) => {{
        $(if $input.peek(Token![$($token)+]) {
            let _: Token![$($token)+] = $input.parse()?;
            Ok($op)
        } else)+ {
            Err($input.error("expected binary operator"))
        }
    }};
}

fn parse_binop(input: ParseStream) -> Result<COp> {
    if let Ok(op) = parse_multi_character_binop(input) {
        return Ok(op);
    }
    parse_single_character_binop(input)
}

fn parse_multi_character_binop(input: ParseStream) -> Result<COp> {
    parse_binary_token!(input;
        [==] => COp::Eq,
        [!=] => COp::Ne,
        [<=] => COp::Le,
        [>=] => COp::Ge,
        [<<] => COp::Shl,
        [>>] => COp::Shr,
        [&&] => COp::LogicalAnd,
        [||] => COp::LogicalOr,
    )
}

fn parse_single_character_binop(input: ParseStream) -> Result<COp> {
    parse_binary_token!(input;
        [+] => COp::Add,
        [-] => COp::Sub,
        [*] => COp::Mul,
        [/] => COp::Div,
        [%] => COp::Mod,
        [<] => COp::Lt,
        [>] => COp::Gt,
        [&] => COp::BitAnd,
        [|] => COp::BitOr,
    )
}

pub(crate) fn parse_unary(input: ParseStream) -> Result<CExpr> {
    if input.peek(Token![-]) {
        let _: Token![-] = input.parse()?;
        let inner = parse_unary(input)?;
        return Ok(CExpr::Neg(Box::new(inner)));
    }
    if input.peek(Token![&]) && !input.peek(Token![&&]) {
        let _: Token![&] = input.parse()?;
        let inner = parse_unary(input)?;
        return Ok(CExpr::Addr(Box::new(inner)));
    }
    parse_postfix(input)
}

/// Fold a member chain that bottoms out in an identifier back into the dotted
/// callee spelling `Expr::Call` carries (e.g. `thr_mma.partition_C`).
pub(crate) fn dotted_callee(e: &CExpr) -> Option<String> {
    match e {
        CExpr::Var(name) => Some(name.clone()),
        CExpr::Member(base, name) => Some(format!("{}.{name}", dotted_callee(base)?)),
        _ => None,
    }
}

pub(crate) fn parse_postfix(input: ParseStream) -> Result<CExpr> {
    let mut e = parse_atom(input)?;
    loop {
        if input.peek(syn::token::Bracket) {
            e = parse_index_postfix(input, e)?;
        } else if input.peek(syn::token::Paren) {
            e = parse_call_postfix(input, e)?;
        } else if input.parse::<PlusPlus>().is_ok() {
            e = CExpr::PostInc(Box::new(e));
        } else if input.peek(Token![.]) && input.peek2(Ident) {
            let _: Token![.] = input.parse()?;
            let ident: Ident = input.parse()?;
            e = CExpr::Member(Box::new(e), ident.to_string());
        } else {
            break;
        }
    }
    Ok(e)
}

fn parse_index_postfix(input: ParseStream, base: CExpr) -> Result<CExpr> {
    let content;
    bracketed!(content in input);
    let index = parse_expr(&content)?;
    if !content.is_empty() {
        return Err(content.error("expected `]`"));
    }
    Ok(CExpr::Index(Box::new(base), Box::new(index)))
}

fn callee_name(expr: &CExpr, input: ParseStream) -> Result<CName> {
    match expr {
        CExpr::Var(name) => Ok(CName::Lit(name.clone())),
        CExpr::Interp(tokens) => Ok(CName::Interp(tokens.clone())),
        CExpr::Member(..) => dotted_callee(expr)
            .map(CName::Lit)
            .ok_or_else(|| input.error("member-call callee must be a chain of identifiers")),
        _ => Err(input
            .error("function call requires a bare identifier or `#name` interpolation as callee")),
    }
}

fn parse_call_postfix(input: ParseStream, callee: CExpr) -> Result<CExpr> {
    let name = callee_name(&callee, input)?;
    let content;
    parenthesized!(content in input);
    Ok(CExpr::Call(name, parse_call_args(&content)?))
}

pub(crate) fn parse_atom(input: ParseStream) -> Result<CExpr> {
    if input.peek(Token![#]) {
        return parse_interpolation(input);
    }
    if input.peek(LitInt) {
        let lit: LitInt = input.parse()?;
        return Ok(CExpr::Lit(parse_lit_i64(&lit)?));
    }
    if input.peek(syn::token::Paren) {
        return parse_parenthesized(input);
    }
    if input.peek(Ident) {
        return Ok(CExpr::Var(input.parse::<Ident>()?.to_string()));
    }
    Err(input.error("expected expression"))
}

fn parse_interpolation(input: ParseStream) -> Result<CExpr> {
    let _: Token![#] = input.parse()?;
    if input.peek(syn::token::Brace) {
        let content;
        braced!(content in input);
        let expr: SynExpr = content.parse()?;
        Ok(CExpr::Interp(quote! { #expr }))
    } else {
        let id: Ident = input.parse()?;
        Ok(CExpr::Interp(quote! { #id }))
    }
}

fn parse_cast_type(input: ParseStream) -> Result<CType> {
    let content;
    parenthesized!(content in input);
    let ty = parse_type(&content)?;
    let mut stars = 0usize;
    while content.peek(Token![*]) {
        let _: Token![*] = content.parse()?;
        stars += 1;
    }
    if !content.is_empty() {
        return Err(content.error("trailing tokens in cast type"));
    }
    Ok(apply_ptr(ty, stars))
}

fn parse_parenthesized(input: ParseStream) -> Result<CExpr> {
    let fork = input.fork();
    if let Ok(ty) = parse_cast_type(&fork)
        && looks_like_unary_start(&fork)
    {
        input.advance_to(&fork);
        return Ok(CExpr::Cast {
            ty,
            expr: Box::new(parse_unary(input)?),
        });
    }
    let content;
    parenthesized!(content in input);
    let inner = parse_expr(&content)?;
    if !content.is_empty() {
        return Err(content.error("expected `)`"));
    }
    Ok(inner)
}

pub(crate) fn peek_assign_op(input: ParseStream) -> Option<CAssignOp> {
    if input.peek(Token![+=]) {
        Some(CAssignOp::AddAssign)
    } else if input.peek(Token![-=]) {
        Some(CAssignOp::SubAssign)
    } else if input.peek(Token![/=]) {
        Some(CAssignOp::DivAssign)
    } else if input.peek(Token![&=]) {
        Some(CAssignOp::BitAndAssign)
    } else if input.peek(Token![>>=]) {
        Some(CAssignOp::ShrAssign)
    } else if input.peek(Token![==]) {
        None
    } else if input.peek(Token![=]) {
        Some(CAssignOp::Set)
    } else {
        None
    }
}

pub(crate) fn consume_assign_op(input: ParseStream) -> Result<()> {
    if input.peek(Token![+=]) {
        let _: Token![+=] = input.parse()?;
    } else if input.peek(Token![-=]) {
        let _: Token![-=] = input.parse()?;
    } else if input.peek(Token![/=]) {
        let _: Token![/=] = input.parse()?;
    } else if input.peek(Token![&=]) {
        let _: Token![&=] = input.parse()?;
    } else if input.peek(Token![>>=]) {
        let _: Token![>>=] = input.parse()?;
    } else if input.peek(Token![=]) {
        let _: Token![=] = input.parse()?;
    } else {
        return Err(input.error("expected assignment operator"));
    }
    Ok(())
}
