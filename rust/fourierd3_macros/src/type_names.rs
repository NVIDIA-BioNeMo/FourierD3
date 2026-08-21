// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Translation between the Rust type mnemonics the DSL accepts and the CUDA
//! spellings they emit, and parsing of types, names, and array sizes.

use quote::quote;
use syn::parse::ParseStream;
use syn::{Error, Expr as SynExpr, Ident, LitInt, Result, Token, braced};

use crate::ast::{CName, CSize, CType};

// Integer mnemonics use built-in C types (`signed char`, `short`, `int`,
// `long long`) rather than `<cstdint>` aliases — NVRTC does not pull standard
// headers by default, so `int64_t` would not resolve.
// The CUDA column must equal `fourierd3_engine::dtype::Dtype::ctype` row for row;
// `fourierd3_engine::ir::stmt::dsl_type_mnemonics_match_dtype_ctype` pins the two.
pub(crate) const RUST_TO_CUDA_TYPE: &[(&str, &str)] = &[
    ("void", "void"),
    ("bool", "bool"),
    ("i8", "signed char"),
    ("i16", "short"),
    ("i32", "int"),
    ("i64", "long long"),
    ("u8", "unsigned char"),
    ("u16", "unsigned short"),
    ("u32", "unsigned int"),
    ("u64", "unsigned long long"),
    ("f16", "__half"),
    ("bf16", "__nv_bfloat16"),
    ("f32", "float"),
    ("f64", "double"),
    ("c64", "cuFloatComplex"),
    ("c128", "cuDoubleComplex"),
];

pub(crate) fn cuda_spelling_hint(name: &str) -> Option<String> {
    let scalar = match name {
        "int" => Some("i32"),
        "signed char" => Some("i8"),
        "short" => Some("i16"),
        "long" => Some("i64"),
        "long long" => Some("i64"),
        "int8_t" => Some("i8"),
        "int16_t" => Some("i16"),
        "int64_t" => Some("i64"),
        "unsigned char" => Some("u8"),
        "unsigned short" => Some("u16"),
        "unsigned long" => Some("u64"),
        "unsigned long long" => Some("u64"),
        "uint8_t" => Some("u8"),
        "uint16_t" => Some("u16"),
        "uint32_t" => Some("u32"),
        "uint64_t" => Some("u64"),
        "unsigned" => Some("u32"),
        "float" => Some("f32"),
        "double" => Some("f64"),
        "__half" => Some("f16"),
        "__nv_bfloat16" => Some("bf16"),
        "cuFloatComplex" => Some("c64"),
        "cuDoubleComplex" => Some("c128"),
        _ => None,
    };
    scalar
        .map(String::from)
        .or_else(|| cuda_to_rust_vector_type(name))
}

pub(crate) fn rust_to_cuda_type(name: &str) -> Option<&'static str> {
    RUST_TO_CUDA_TYPE
        .iter()
        .find_map(|(rust, cuda)| (*rust == name).then_some(*cuda))
}

pub(crate) const RUST_TO_CUDA_VECTOR_STEM: &[(&str, &str)] = &[
    ("i8", "char"),
    ("u8", "uchar"),
    ("i16", "short"),
    ("u16", "ushort"),
    ("i32", "int"),
    ("u32", "uint"),
    ("i64", "longlong"),
    ("u64", "ulonglong"),
    ("f32", "float"),
    ("f64", "double"),
];

pub(crate) fn rust_to_cuda_vector_type(name: &str) -> Option<String> {
    let (elem, width) = name.split_once('x')?;
    if !matches!(width, "1" | "2" | "3" | "4") {
        return None;
    }
    let stem = RUST_TO_CUDA_VECTOR_STEM
        .iter()
        .find_map(|(rust, cuda)| (*rust == elem).then_some(*cuda))?;
    Some(format!("{stem}{width}"))
}

pub(crate) fn cuda_to_rust_vector_type(name: &str) -> Option<String> {
    let (stem, width) = name.split_at(name.len().saturating_sub(1));
    if !matches!(width, "1" | "2" | "3" | "4") {
        return None;
    }
    let elem = RUST_TO_CUDA_VECTOR_STEM
        .iter()
        .find_map(|(rust, cuda)| (*cuda == stem).then_some(*rust))?;
    Some(format!("{elem}x{width}"))
}

pub(crate) fn apply_ptr(ty: CType, stars: usize) -> CType {
    if stars == 0 {
        return ty;
    }
    let suffix: String = "*".repeat(stars);
    match ty {
        CType::Lit(s) => CType::Lit(format!("{s}{suffix}")),
        CType::Interp(ts) => CType::Interp(quote! { format!("{}{}", #ts, #suffix) }),
    }
}

pub(crate) fn parse_type(input: ParseStream) -> Result<CType> {
    if input.peek(Token![#]) {
        let _: Token![#] = input.parse()?;
        if input.peek(syn::token::Brace) {
            let content;
            braced!(content in input);
            let expr: SynExpr = content.parse()?;
            return Ok(CType::Interp(quote! { #expr }));
        }
        let id: Ident = input.parse()?;
        return Ok(CType::Interp(quote! { #id }));
    }
    let id: Ident = input.parse()?;
    let name = id.to_string();
    if name == "auto" {
        return Ok(CType::Lit("auto".into()));
    }
    if let Some(cuda) = rust_to_cuda_type(&name) {
        return Ok(CType::Lit(cuda.into()));
    }
    if let Some(cuda) = rust_to_cuda_vector_type(&name) {
        return Ok(CType::Lit(cuda));
    }
    let msg = match cuda_spelling_hint(&name) {
        Some(rust) => format!(
            "unknown type `{name}`; use the Rust spelling `{rust}` (cuda! requires Rust-typed declarations)"
        ),
        None => format!(
            "unknown type `{name}`; expected one of bool, i8/i16/i32/i64, u8/u16/u32/u64, f16/bf16/f32/f64, c64/c128"
        ),
    };
    Err(Error::new(id.span(), msg))
}

pub(crate) fn parse_name(input: ParseStream) -> Result<CName> {
    if input.peek(Token![#]) {
        let _: Token![#] = input.parse()?;
        if input.peek(syn::token::Brace) {
            let content;
            braced!(content in input);
            let expr: SynExpr = content.parse()?;
            return Ok(CName::Interp(quote! { #expr }));
        }
        let id: Ident = input.parse()?;
        return Ok(CName::Interp(quote! { #id }));
    }
    let id: Ident = input.parse()?;
    Ok(CName::Lit(id.to_string()))
}

pub(crate) fn parse_size(input: ParseStream) -> Result<CSize> {
    if input.peek(Token![#]) {
        let _: Token![#] = input.parse()?;
        if input.peek(syn::token::Brace) {
            let content;
            braced!(content in input);
            let expr: SynExpr = content.parse()?;
            return Ok(CSize::Interp(quote! { #expr }));
        }
        let id: Ident = input.parse()?;
        return Ok(CSize::Interp(quote! { #id }));
    }
    let lit: LitInt = input.parse()?;
    Ok(CSize::Lit(parse_lit_i64(&lit)?))
}

pub(crate) fn looks_like_unary_start(input: ParseStream) -> bool {
    input.peek(Ident)
        || input.peek(LitInt)
        || input.peek(syn::token::Paren)
        || input.peek(Token![#])
        || input.peek(Token![&])
        || input.peek(Token![-])
}

pub(crate) fn parse_lit_i64(lit: &LitInt) -> Result<i64> {
    let raw = lit.to_string();
    let suffix = lit.suffix();
    let body = if suffix.is_empty() {
        raw.as_str()
    } else {
        raw.strip_suffix(suffix).unwrap_or(&raw)
    };
    let cleaned: String = body.chars().filter(|c| *c != '_').collect();
    let (radix, digits) = if let Some(d) = cleaned
        .strip_prefix("0x")
        .or_else(|| cleaned.strip_prefix("0X"))
    {
        (16u32, d.to_string())
    } else if let Some(d) = cleaned
        .strip_prefix("0b")
        .or_else(|| cleaned.strip_prefix("0B"))
    {
        (2u32, d.to_string())
    } else if let Some(d) = cleaned
        .strip_prefix("0o")
        .or_else(|| cleaned.strip_prefix("0O"))
    {
        (8u32, d.to_string())
    } else {
        (10u32, cleaned)
    };
    if radix == 10 {
        digits
            .parse::<i64>()
            .map_err(|e| Error::new(lit.span(), e.to_string()))
    } else {
        u64::from_str_radix(&digits, radix)
            .map(|v| v as i64)
            .map_err(|e| Error::new(lit.span(), e.to_string()))
    }
}
