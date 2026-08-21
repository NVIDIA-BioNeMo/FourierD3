// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The token-level pieces every instruction grammar shares: operands,
//! identifiers, and comma or whitespace splits.

use super::{LlvmType, Operand};

pub(super) fn parse_operand(s: &str, ty: &LlvmType) -> Result<Operand, String> {
    if let Some(rest) = s.strip_prefix('%') {
        let (name, _) = parse_quoted_or_bare(rest)?;
        return Ok(Operand::Ssa(name));
    }
    if ty.is_float() {
        if let Some(hex) = s.strip_prefix("0x") {
            let bits = u64::from_str_radix(hex, 16)
                .map_err(|e| format!("bad float hex literal {s:?}: {e}"))?;
            return Ok(Operand::FloatHex(bits));
        }
        let f: f64 = s
            .parse()
            .map_err(|e| format!("bad float literal {s:?}: {e}"))?;
        return Ok(Operand::FloatHex(f.to_bits()));
    }
    let v: i64 = s
        .parse()
        .map_err(|e| format!("bad int literal {s:?}: {e}"))?;
    Ok(Operand::IntLit(v))
}

pub(super) fn split_token(s: &str) -> (&str, &str) {
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    }
}

pub(super) fn split_comma(s: &str) -> Result<(&str, &str), String> {
    let i = s
        .find(',')
        .ok_or_else(|| format!("expected `,` in operand list: {s:?}"))?;
    Ok((&s[..i], &s[i + 1..]))
}

pub(super) fn parse_at_ident(s: &str) -> Result<(String, &str), String> {
    let s = s
        .strip_prefix('@')
        .ok_or_else(|| format!("expected `@`-prefixed name: {s:?}"))?;
    parse_quoted_or_bare(s)
}

pub(super) fn parse_percent_ident(s: &str) -> Result<(String, &str), String> {
    let s = s
        .strip_prefix('%')
        .ok_or_else(|| format!("expected `%`-prefixed name: {s:?}"))?;
    parse_quoted_or_bare(s)
}

pub(super) fn parse_quoted_or_bare(s: &str) -> Result<(String, &str), String> {
    if let Some(rest) = s.strip_prefix('"') {
        let end = rest
            .find('"')
            .ok_or_else(|| format!("unterminated quoted identifier: {s:?}"))?;
        Ok((rest[..end].to_string(), &rest[end + 1..]))
    } else {
        let end = s
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '.')
            .unwrap_or(s.len());
        if end == 0 {
            return Err(format!("empty identifier in: {s:?}"));
        }
        Ok((s[..end].to_string(), &s[end..]))
    }
}
