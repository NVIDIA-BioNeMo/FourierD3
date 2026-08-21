// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Splitting a textual LLVM module into globals, the function header,
//! and the instruction list of its single basic block.

use super::instruction::parse_instr;
use super::lexer::{parse_at_ident, parse_operand, parse_percent_ident, split_token};
use super::{Instr, LlvmFunction, LlvmGlobal, LlvmModule, LlvmParam, LlvmType, Operand};

pub(crate) fn parse_module(text: &str) -> Result<LlvmModule, String> {
    let mut name: Option<String> = None;
    let mut params: Vec<LlvmParam> = Vec::new();
    let mut instrs: Vec<Instr> = Vec::new();
    let mut globals: Vec<LlvmGlobal> = Vec::new();
    let mut in_body = false;

    for raw in text.lines() {
        let line = raw.trim();
        if is_ignored_line(line) {
            continue;
        }
        if !in_body {
            parse_module_header(line, &mut name, &mut params, &mut globals, &mut in_body)?;
            continue;
        }
        if line == "}" {
            in_body = false;
            continue;
        }
        if !line.ends_with(':') {
            instrs.push(parse_instr(line)?);
        }
    }

    Ok(LlvmModule {
        globals,
        function: LlvmFunction {
            name: name.ok_or_else(|| "no `define` line found".to_string())?,
            params,
            instrs,
        },
    })
}

fn is_ignored_line(line: &str) -> bool {
    line.is_empty()
        || line.starts_with(';')
        || line.starts_with("target ")
        || line.starts_with("declare ")
}

fn parse_module_header(
    line: &str,
    name: &mut Option<String>,
    params: &mut Vec<LlvmParam>,
    globals: &mut Vec<LlvmGlobal>,
    in_body: &mut bool,
) -> Result<(), String> {
    if line.starts_with('@') {
        globals.push(parse_global(line)?);
    } else if let Some(rest) = line.strip_prefix("define ") {
        let (function_name, function_params) = parse_define_header(rest)?;
        *name = Some(function_name);
        *params = function_params;
    } else if line == "{" {
        *in_body = true;
    }
    Ok(())
}

fn parse_global(line: &str) -> Result<LlvmGlobal, String> {
    let (name, rest) = parse_at_ident(line)?;
    let rest = rest.trim_start();
    let rest = rest
        .strip_prefix('=')
        .ok_or_else(|| format!("expected `=` after global name in {line:?}"))?
        .trim_start();
    let rest = rest.trim_start_matches("internal").trim_start();
    let rest = rest.trim_start_matches("external").trim_start();
    let rest = rest.trim_start_matches("private").trim_start();
    let (addrspace, rest) = if let Some(after) = rest.strip_prefix("addrspace(") {
        let close = after
            .find(')')
            .ok_or_else(|| format!("malformed addrspace in {line:?}"))?;
        let n: u32 = after[..close]
            .parse()
            .map_err(|e| format!("bad addrspace number: {e}"))?;
        (n, after[close + 1..].trim_start())
    } else {
        (0u32, rest)
    };
    let rest = rest
        .strip_prefix("constant")
        .ok_or_else(|| format!("only `constant` globals supported: {line:?}"))?
        .trim_start();
    let rest = rest
        .strip_prefix('[')
        .ok_or_else(|| format!("expected array type [N x T] in {line:?}"))?;
    let end = rest
        .find(']')
        .ok_or_else(|| format!("unterminated array type in {line:?}"))?;
    let inside = &rest[..end];
    let (count_str, ty_str) = inside
        .split_once(" x ")
        .ok_or_else(|| format!("malformed array type {inside:?}"))?;
    let count: usize = count_str
        .trim()
        .parse()
        .map_err(|e| format!("bad array length: {e}"))?;
    let elem_ty = LlvmType::parse(ty_str.trim())?;
    let rest = rest[end + 1..].trim_start();
    let init = rest
        .strip_prefix('[')
        .and_then(|r| r.rfind(']').map(|i| &r[..i]))
        .ok_or_else(|| format!("missing initialiser list in {line:?}"))?;
    let mut values: Vec<Operand> = Vec::new();
    for chunk in init.split(',') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let (_ty, lit) = split_token(chunk);
        let op = parse_operand(lit.trim(), &elem_ty)?;
        values.push(op);
    }
    if values.len() != count {
        return Err(format!(
            "global {name:?} declared [{count} x ..] but has {} elems",
            values.len()
        ));
    }
    Ok(LlvmGlobal {
        name,
        addrspace,
        elem_ty,
        values,
    })
}

fn parse_define_header(s: &str) -> Result<(String, Vec<LlvmParam>), String> {
    let s = s
        .strip_prefix("void ")
        .ok_or_else(|| format!("only `define void` is supported, got: {s:?}"))?;
    let s = s.trim_start();
    let (fname, rest) = parse_at_ident(s)?;
    let rest = rest.trim_start();
    let inner = rest
        .strip_prefix('(')
        .and_then(|r| r.rfind(')').map(|i| &r[..i]))
        .ok_or_else(|| format!("missing parenthesised param list: {rest:?}"))?;

    let mut params = Vec::new();
    if !inner.trim().is_empty() {
        for chunk in inner.split(',') {
            params.push(parse_param(chunk.trim())?);
        }
    }
    Ok((fname, params))
}

fn parse_param(s: &str) -> Result<LlvmParam, String> {
    let (ty_str, rest) = split_token(s);
    let ty_str = ty_str
        .strip_suffix('*')
        .ok_or_else(|| format!("expected pointer parameter type, got {ty_str:?}"))?;
    let elem_ty = LlvmType::parse(ty_str)?;
    let (name, tail) = parse_percent_ident(rest.trim_start())?;
    if !tail.trim().is_empty() {
        return Err(format!("unexpected trailing tokens in param: {tail:?}"));
    }
    Ok(LlvmParam { name, elem_ty })
}
