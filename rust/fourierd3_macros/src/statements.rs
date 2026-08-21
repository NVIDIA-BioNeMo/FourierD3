// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing of the CUDA statement forms used by FourierD3 kernels.

use quote::quote;
use syn::parse::discouraged::Speculative;
use syn::parse::{Parse, ParseStream};
use syn::{Error, Expr as SynExpr, Result, Token, braced, parenthesized};

use crate::ast::{CExpr, CName, CStmt, CUnroll, CudaInput};
use crate::declarations::{
    parse_assign_or_call, parse_atomic_add, parse_decl_body_after_name, parse_extern_decl,
    parse_return_stmt,
};
use crate::expressions::{consume_assign_op, parse_expr, peek_assign_op};
use crate::keywords::kw;
use crate::type_names::{parse_name, parse_type};

impl Parse for CudaInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let target: SynExpr = input.parse()?;
        let _: Token![=>] = input.parse()?;
        let stmts = parse_stmt_list(input)?;
        Ok(CudaInput { target, stmts })
    }
}

pub(crate) fn parse_stmt_list(input: ParseStream) -> Result<Vec<CStmt>> {
    let mut stmts = Vec::new();
    let mut errors: Option<Error> = None;
    while !input.is_empty() {
        match parse_stmt(input) {
            Ok(stmt) => stmts.push(stmt),
            Err(error) => {
                match &mut errors {
                    Some(previous) => previous.combine(error),
                    None => errors = Some(error),
                }
                skip_to_next_statement(input);
            }
        }
    }
    match errors {
        Some(error) => Err(error),
        None => Ok(stmts),
    }
}

fn skip_to_next_statement(input: ParseStream) {
    use proc_macro2::{Delimiter, TokenTree};
    while !input.is_empty() {
        if input.peek(Token![;]) {
            let _: Token![;] = input.parse().expect("peeked");
            return;
        }
        let result = input.step(|cursor| {
            cursor
                .token_tree()
                .map(|(token, next)| {
                    let stop = matches!(
                        &token,
                        TokenTree::Group(group) if group.delimiter() == Delimiter::Brace
                    );
                    (stop, next)
                })
                .ok_or_else(|| cursor.error("unexpected end of input"))
        });
        match result {
            Ok(true) | Err(_) => return,
            Ok(false) => {}
        }
    }
}

fn parse_stmt(input: ParseStream) -> Result<CStmt> {
    if input.peek(Token![;]) {
        let _: Token![;] = input.parse()?;
        return Ok(CStmt::Blank);
    }
    if input.peek(Token![if]) {
        return parse_if(input);
    }
    if input.peek(Token![for]) {
        return parse_for(input, CUnroll::None);
    }
    if input.peek(Token![while]) {
        return parse_while(input);
    }
    if input.peek(Token![continue]) {
        return parse_continue(input);
    }
    if input.peek(kw::unroll) {
        return parse_unrolled_for(input);
    }
    if input.peek(kw::splice) && input.peek2(Token![!]) {
        return parse_splice(input);
    }
    if input.peek(kw::atomicAdd) {
        return parse_atomic_add(input);
    }
    if input.peek(kw::__syncthreads) {
        return parse_syncthreads(input);
    }
    if input.peek(Token![extern]) {
        return parse_extern_decl(input);
    }
    if input.peek(Token![return]) {
        return parse_return_stmt(input);
    }

    let fork = input.fork();
    if let (Ok(ty), Ok(name)) = (parse_type(&fork), parse_name(&fork))
        && (fork.peek(Token![=]) || fork.peek(syn::token::Bracket) || fork.peek(Token![;]))
    {
        input.advance_to(&fork);
        return parse_decl_body_after_name(input, ty, name);
    }

    parse_assign_or_call(input)
}

fn parse_if(input: ParseStream) -> Result<CStmt> {
    let _: Token![if] = input.parse()?;
    let condition;
    parenthesized!(condition in input);
    let cond = parse_expr(&condition)?;
    if !condition.is_empty() {
        return Err(condition.error("expected `)` after `if` condition"));
    }

    if input.peek(Token![return]) {
        let return_token: Token![return] = input.parse()?;
        let _: Token![;] = input.parse().map_err(|_| {
            Error::new(
                return_token.span,
                "`if (...) return` must be terminated with `;`",
            )
        })?;
        return Ok(CStmt::If {
            cond,
            then_: vec![CStmt::Return(None)],
            else_: None,
        });
    }
    if input.peek(Token![continue]) {
        let _: Token![continue] = input.parse()?;
        let _: Token![;] = input
            .parse()
            .map_err(|_| input.error("`if (...) continue` must be terminated with `;`"))?;
        return Ok(CStmt::If {
            cond,
            then_: vec![CStmt::Continue],
            else_: None,
        });
    }
    if !input.peek(syn::token::Brace) {
        return Err(
            input.error("`if (...)` must be followed by `return;`, `continue;`, or `{ ... }`")
        );
    }

    let then_content;
    braced!(then_content in input);
    let then_ = parse_stmt_list(&then_content)?;
    let else_ = if input.peek(Token![else]) {
        let _: Token![else] = input.parse()?;
        if input.peek(Token![if]) {
            Some(vec![parse_if(input)?])
        } else if input.peek(syn::token::Brace) {
            let else_content;
            braced!(else_content in input);
            Some(parse_stmt_list(&else_content)?)
        } else {
            return Err(input.error("`else` must be followed by `if` or `{ ... }`"));
        }
    } else {
        None
    };
    Ok(CStmt::If { cond, then_, else_ })
}

fn parse_syncthreads(input: ParseStream) -> Result<CStmt> {
    let keyword: kw::__syncthreads = input.parse()?;
    let args;
    parenthesized!(args in input);
    if !args.is_empty() {
        return Err(Error::new(
            keyword.span,
            "`__syncthreads()` takes no arguments",
        ));
    }
    let _: Token![;] = input.parse().map_err(|_| {
        Error::new(
            keyword.span,
            "`__syncthreads()` must be terminated with `;`",
        )
    })?;
    Ok(CStmt::Eval(CExpr::Call(
        CName::Lit("__syncthreads".into()),
        vec![],
    )))
}

fn parse_unrolled_for(input: ParseStream) -> Result<CStmt> {
    let unroll: kw::unroll = input.parse()?;
    if !input.peek(Token![for]) {
        return Err(Error::new(
            unroll.span,
            "`unroll` must be immediately followed by a `for` loop",
        ));
    }
    parse_for(input, CUnroll::All)
}

fn parse_for(input: ParseStream, unroll: CUnroll) -> Result<CStmt> {
    let _: Token![for] = input.parse()?;
    let header;
    parenthesized!(header in input);
    let init = Box::new(parse_stmt(&header)?);
    let cond = parse_expr(&header)?;
    let _: Token![;] = header.parse()?;
    let step = parse_for_step(&header)?;
    if !header.is_empty() {
        return Err(header.error("unexpected tokens in for-loop header"));
    }
    let body_content;
    braced!(body_content in input);
    let body = parse_stmt_list(&body_content)?;
    Ok(CStmt::For {
        init,
        cond,
        step,
        body,
        unroll,
    })
}

fn parse_for_step(input: ParseStream) -> Result<CExpr> {
    let lhs = parse_expr(input)?;
    if let Some(op) = peek_assign_op(input) {
        consume_assign_op(input)?;
        let rhs = parse_expr(input)?;
        Ok(CExpr::Assign {
            op,
            target: Box::new(lhs),
            value: Box::new(rhs),
        })
    } else {
        Ok(lhs)
    }
}

fn parse_splice(input: ParseStream) -> Result<CStmt> {
    let keyword: kw::splice = input.parse()?;
    let _: Token![!] = input.parse()?;
    let content;
    parenthesized!(content in input);
    let expression: SynExpr = content.parse()?;
    if !content.is_empty() {
        return Err(content.error("expected `)` after `splice!` argument"));
    }
    let _: Token![;] = input
        .parse()
        .map_err(|_| Error::new(keyword.span, "`splice!(...)` must be terminated with `;`"))?;
    Ok(CStmt::Splice(quote! { #expression }))
}

pub(crate) fn parse_call_args(content: ParseStream) -> Result<Vec<CExpr>> {
    let mut args = Vec::new();
    while !content.is_empty() {
        args.push(parse_expr(content)?);
        if content.peek(Token![,]) {
            let _: Token![,] = content.parse()?;
        } else if !content.is_empty() {
            return Err(content.error("expected `,` between call arguments"));
        }
    }
    Ok(args)
}

fn parse_while(input: ParseStream) -> Result<CStmt> {
    let _: Token![while] = input.parse()?;
    let header;
    parenthesized!(header in input);
    let cond = parse_expr(&header)?;
    if !header.is_empty() {
        return Err(header.error("trailing tokens in while header"));
    }
    let body_content;
    braced!(body_content in input);
    let body = parse_stmt_list(&body_content)?;
    Ok(CStmt::While { cond, body })
}

fn parse_continue(input: ParseStream) -> Result<CStmt> {
    let _: Token![continue] = input.parse()?;
    let _: Token![;] = input
        .parse()
        .map_err(|_| input.error("expected `;` after `continue`"))?;
    Ok(CStmt::Continue)
}
