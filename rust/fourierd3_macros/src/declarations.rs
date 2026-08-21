// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing of declarations, assignment statements, and calls.

use syn::parse::ParseStream;
use syn::{Error, Ident, Result, Token, braced, bracketed, parenthesized};

use crate::ast::{CExpr, CName, CSize, CStmt, CType};
use crate::expressions::{consume_assign_op, parse_expr, parse_postfix, peek_assign_op};
use crate::keywords::kw;
use crate::type_names::{parse_name, parse_size, parse_type};

pub(crate) fn parse_atomic_add(input: ParseStream) -> Result<CStmt> {
    let _: kw::atomicAdd = input.parse()?;
    let args;
    parenthesized!(args in input);
    if !args.peek(Token![&]) || args.peek(Token![&&]) {
        return Err(Error::new(
            args.span(),
            "atomicAdd's first argument must be `&<lvalue>` (the address of the atomic target)",
        ));
    }
    let _: Token![&] = args.parse()?;
    let target = parse_postfix(&args)?;
    let _: Token![,] = args
        .parse()
        .map_err(|_| Error::new(args.span(), "atomicAdd requires exactly two arguments"))?;
    let value = parse_expr(&args)?;
    if !args.is_empty() {
        return Err(Error::new(
            args.span(),
            "atomicAdd takes exactly 2 arguments; got extra tokens",
        ));
    }
    let _: Token![;] = input.parse()?;
    Ok(CStmt::Eval(CExpr::Call(
        CName::Lit("atomicAdd".into()),
        vec![CExpr::Addr(Box::new(target)), value],
    )))
}

pub(crate) fn parse_extern_decl(input: ParseStream) -> Result<CStmt> {
    let extern_token: Token![extern] = input.parse()?;
    if input.peek(kw::__shared__) {
        return parse_extern_shared(input);
    }
    if input.peek(syn::LitStr) {
        return parse_extern_device(input);
    }
    Err(Error::new(
        extern_token.span,
        "expected `__shared__ T name[];` or `\"C\" __device__ void name(types);`",
    ))
}

fn parse_extern_shared(input: ParseStream) -> Result<CStmt> {
    let _: kw::__shared__ = input.parse()?;
    let ty = parse_type(input)?;
    let name = parse_name(input)?;
    let brackets;
    bracketed!(brackets in input);
    if !brackets.is_empty() {
        return Err(brackets.error("`extern __shared__` requires empty brackets `[]`"));
    }
    let _: Token![;] = input.parse()?;
    Ok(CStmt::ExternSharedDecl { ty, name })
}

fn parse_extern_device(input: ParseStream) -> Result<CStmt> {
    let abi: syn::LitStr = input.parse()?;
    if abi.value() != "C" {
        return Err(Error::new(abi.span(), "only `extern \"C\"` is supported"));
    }
    expect_ident(input, "__device__")?;
    expect_ident(input, "void")?;
    let name = parse_name(input)?;
    let params;
    parenthesized!(params in input);
    let param_types = parse_comma_separated_types(&params)?;
    let _: Token![;] = input.parse()?;
    Ok(CStmt::ExternDeviceDecl { name, param_types })
}

fn expect_ident(input: ParseStream, expected: &str) -> Result<()> {
    let actual: Ident = input.parse()?;
    if actual == expected {
        Ok(())
    } else {
        Err(Error::new(actual.span(), format!("expected `{expected}`")))
    }
}

fn parse_comma_separated_types(input: ParseStream) -> Result<Vec<CType>> {
    let mut types = Vec::new();
    while !input.is_empty() {
        types.push(parse_type(input)?);
        if input.is_empty() {
            break;
        }
        let _: Token![,] = input
            .parse()
            .map_err(|_| input.error("expected `,` between parameter types"))?;
    }
    Ok(types)
}

pub(crate) fn parse_return_stmt(input: ParseStream) -> Result<CStmt> {
    let _: Token![return] = input.parse()?;
    if input.peek(Token![;]) {
        let _: Token![;] = input.parse()?;
        return Ok(CStmt::Return(None));
    }
    let value = parse_expr(input)?;
    let _: Token![;] = input
        .parse()
        .map_err(|_| input.error("`return <expr>` must be terminated with `;`"))?;
    Ok(CStmt::Return(Some(value)))
}

pub(crate) fn parse_decl_body_after_name(
    input: ParseStream,
    ty: CType,
    name: CName,
) -> Result<CStmt> {
    if !input.peek(syn::token::Bracket) {
        return parse_scalar_declaration(input, ty, name);
    }

    let (dims, first_empty) = parse_array_dimensions(input)?;
    if input.peek(Token![=]) {
        return parse_array_initializer(input, ty, name, dims, first_empty);
    }
    if dims.is_empty() {
        return Err(input.error(
            "expected `= { ... }` after `[]` (empty brackets only valid for an initializer-list decl)",
        ));
    }
    let _: Token![;] = input.parse()?;
    Ok(CStmt::ArrayDecl { ty, name, dims })
}

fn parse_optional_initializer(input: ParseStream) -> Result<Option<CExpr>> {
    if input.peek(Token![=]) {
        let _: Token![=] = input.parse()?;
        Ok(Some(parse_expr(input)?))
    } else {
        Ok(None)
    }
}

fn parse_scalar_declaration(input: ParseStream, ty: CType, name: CName) -> Result<CStmt> {
    let mut decls = vec![(name, parse_optional_initializer(input)?)];
    while input.peek(Token![,]) {
        let _: Token![,] = input.parse()?;
        decls.push((parse_name(input)?, parse_optional_initializer(input)?));
    }
    let _: Token![;] = input.parse()?;
    Ok(CStmt::Decl { ty, decls })
}

fn parse_array_dimensions(input: ParseStream) -> Result<(Vec<CSize>, bool)> {
    let mut dims = Vec::new();
    let mut first_empty = false;
    while input.peek(syn::token::Bracket) {
        let content;
        bracketed!(content in input);
        if content.is_empty() {
            if !dims.is_empty() {
                return Err(content.error("empty `[]` is valid only for the first dimension"));
            }
            first_empty = true;
        } else {
            let size = parse_size(&content)?;
            if !content.is_empty() {
                return Err(content.error("expected `]` after array size"));
            }
            dims.push(size);
        }
    }
    Ok((dims, first_empty))
}

fn parse_initializer_values(input: ParseStream) -> Result<Vec<CExpr>> {
    let mut values = Vec::new();
    while !input.is_empty() {
        values.push(parse_expr(input)?);
        if input.is_empty() {
            break;
        }
        let _: Token![,] = input.parse()?;
    }
    Ok(values)
}

fn parse_array_initializer(
    input: ParseStream,
    ty: CType,
    name: CName,
    dims: Vec<CSize>,
    first_empty: bool,
) -> Result<CStmt> {
    let _: Token![=] = input.parse()?;
    let values;
    braced!(values in input);
    let init = parse_initializer_values(&values)?;
    let _: Token![;] = input.parse()?;
    if dims.len() > 1 {
        return Err(input.error("initializer-list declarations must be one-dimensional"));
    }
    let size = if first_empty {
        None
    } else {
        Some(dims.into_iter().next().expect("one explicit dimension"))
    };
    Ok(CStmt::ArrayInitDecl {
        ty,
        name,
        size,
        init,
    })
}

pub(crate) fn parse_assign_or_call(input: ParseStream) -> Result<CStmt> {
    let lhs_span = input.span();
    let lhs = parse_expr(input)?;
    if let Some(op) = peek_assign_op(input) {
        consume_assign_op(input)?;
        let rhs = parse_expr(input)?;
        let _: Token![;] = input.parse()?;
        return Ok(CStmt::Eval(CExpr::Assign {
            op,
            target: Box::new(lhs),
            value: Box::new(rhs),
        }));
    }
    match lhs {
        CExpr::Call(callee, args) => {
            let _: Token![;] = input.parse().map_err(|_| {
                Error::new(
                    input.span(),
                    "expected an assignment operator or `;` after a call",
                )
            })?;
            Ok(CStmt::Eval(CExpr::Call(callee, args)))
        }
        _ => Err(Error::new(
            lhs_span,
            "expression statement must be a call; bare expressions are not allowed",
        )),
    }
}
