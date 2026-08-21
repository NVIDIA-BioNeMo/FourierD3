// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proc-macro front-end for `fourierd3_engine`: a strict CUDA-ish DSL that compiles
//! into `fourierd3_engine::ir::stmt` IR. Anything outside the grammar is a span-pointed
//! compile error rather than a silent passthrough.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

mod ast;
mod declarations;
mod emit;
mod expressions;
mod keywords;
mod statements;
mod type_names;
mod xla_handler;

use ast::CudaInput;
use emit::emit_pushes;

#[proc_macro]
pub fn cuda(input: TokenStream) -> TokenStream {
    let input: CudaInput = parse_macro_input!(input);
    let target = &input.target;
    let pushes = emit_pushes(&quote! { #target }, &input.stmts);
    quote! { { #pushes } }.into()
}

#[proc_macro_attribute]
pub fn handler(attr: TokenStream, item: TokenStream) -> TokenStream {
    xla_handler::expand(attr, item)
}
