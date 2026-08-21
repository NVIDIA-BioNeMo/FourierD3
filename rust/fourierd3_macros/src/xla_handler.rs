// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Attribute, FnArg, Ident, ItemFn, Pat, PatType, Signature, Type, parse_macro_input,
    spanned::Spanned,
};

pub(crate) fn expand(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let user_fn = parse_macro_input!(item as ItemFn);
    expand_handler(user_fn)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

fn expand_handler(user_fn: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let user_vis = &user_fn.vis;
    let user_sig = &user_fn.sig;
    let user_body = &user_fn.block;
    let user_name = &user_sig.ident;
    let user_name_orig = Ident::new(&format!("__xla_ffi_impl_{user_name}"), user_name.span());

    validate_signature(user_sig)?;
    let params = classify_params(user_sig)?;
    let (decode_stmts, call_args) = decode_parameters(&params);
    let stripped_inputs = strip_param_attributes(user_sig);

    let user_attrs = &user_fn.attrs;
    let user_output = &user_sig.output;

    Ok(quote! {
        #(#user_attrs)*
        #[doc(hidden)]
        #[inline(always)]
        #user_vis fn #user_name_orig(#(#stripped_inputs),*) #user_output {
            #user_body
        }

        #[unsafe(no_mangle)]
        #user_vis unsafe extern "C" fn #user_name(
            call_frame: *mut ::fourierd3_engine::xla_ffi::sys::XLA_FFI_CallFrame,
        ) -> *mut ::fourierd3_engine::xla_ffi::sys::XLA_FFI_Error {
            unsafe {
                ::fourierd3_engine::xla_ffi::dispatch(call_frame, |state| {
                    #(#decode_stmts)*
                    #user_name_orig(#(#call_args),*)
                })
            }
        }
    })
}

fn validate_signature(signature: &Signature) -> syn::Result<()> {
    if signature.asyncness.is_some() {
        return Err(syn::Error::new(
            signature.asyncness.span(),
            "xla_ffi::handler does not support async fns",
        ));
    }
    if signature.unsafety.is_some() {
        return Err(syn::Error::new(
            signature.unsafety.span(),
            "xla_ffi::handler fn must be safe (decoding errors are surfaced via Result)",
        ));
    }
    if !signature.generics.params.is_empty() {
        return Err(syn::Error::new(
            signature.generics.span(),
            "xla_ffi::handler fn cannot be generic",
        ));
    }
    Ok(())
}

fn classify_params(signature: &Signature) -> syn::Result<Vec<Param>> {
    let mut params = Vec::new();
    let mut unique = UniqueParams::default();
    for input in &signature.inputs {
        let FnArg::Typed(pat_type) = input else {
            return Err(syn::Error::new(
                input.span(),
                "xla_ffi::handler fn cannot take `self`",
            ));
        };
        let p = classify_param(pat_type)?;
        unique.record(&p.kind, pat_type)?;
        params.push(p);
    }
    Ok(params)
}

fn decode_parameters(
    params: &[Param],
) -> (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>) {
    let mut decode_stmts = Vec::with_capacity(params.len());
    let mut call_args = Vec::with_capacity(params.len());
    for p in params {
        let binding = &p.binding;
        call_args.push(quote!(#binding));
        let stmt = match &p.kind {
            ParamKind::Stream => quote!(let #binding = state.stream()?;),
            ParamKind::RemainingArgs => quote!(let #binding = state.remaining_args();),
            ParamKind::RemainingRets => quote!(let #binding = state.remaining_rets();),
            ParamKind::AttrScalar { name, ty } => {
                quote!(let #binding: #ty = state.attr_scalar::<#ty>(#name)?;)
            }
        };
        decode_stmts.push(stmt);
    }
    (decode_stmts, call_args)
}

fn strip_param_attributes(signature: &Signature) -> Vec<FnArg> {
    signature
        .inputs
        .iter()
        .cloned()
        .map(|mut input| {
            if let FnArg::Typed(pt) = &mut input {
                pt.attrs.retain(|a| !a.path().is_ident("attr"));
            }
            input
        })
        .collect()
}

#[derive(Default)]
struct UniqueParams {
    remaining_args: bool,
    remaining_rets: bool,
}

impl UniqueParams {
    fn record(&mut self, kind: &ParamKind, param: &PatType) -> syn::Result<()> {
        match kind {
            ParamKind::RemainingArgs if self.remaining_args => Err(syn::Error::new(
                param.span(),
                "duplicate RemainingArgs parameter",
            )),
            ParamKind::RemainingRets if self.remaining_rets => Err(syn::Error::new(
                param.span(),
                "duplicate RemainingRets parameter",
            )),
            ParamKind::RemainingArgs => {
                self.remaining_args = true;
                Ok(())
            }
            ParamKind::RemainingRets => {
                self.remaining_rets = true;
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

struct Param {
    binding: Ident,
    kind: ParamKind,
}

enum ParamKind {
    Stream,
    RemainingArgs,
    RemainingRets,
    AttrScalar { name: String, ty: Box<Type> },
}

fn classify_param(pat_type: &PatType) -> syn::Result<Param> {
    let binding = match pat_type.pat.as_ref() {
        Pat::Ident(pi) => pi.ident.clone(),
        other => {
            return Err(syn::Error::new(
                other.span(),
                "xla_ffi::handler params must be plain identifiers",
            ));
        }
    };

    let attr_override = parse_attr_attribute(&pat_type.attrs)?;
    if let Some(attr_name) = attr_override {
        let name = attr_name.unwrap_or_else(|| binding.to_string());
        let kind = classify_attr_type(&pat_type.ty, &name)?;
        return Ok(Param { binding, kind });
    }

    let kind = classify_positional_type(&pat_type.ty)?;
    Ok(Param { binding, kind })
}

fn parse_attr_attribute(attrs: &[Attribute]) -> syn::Result<Option<Option<String>>> {
    let mut found: Option<Option<String>> = None;
    for attr in attrs {
        if !attr.path().is_ident("attr") {
            return Err(syn::Error::new(
                attr.span(),
                "xla_ffi::handler params accept only `#[attr]` / `#[attr(\"name\")]`",
            ));
        }
        if found.is_some() {
            return Err(syn::Error::new(
                attr.span(),
                "duplicate `#[attr]` on a single parameter",
            ));
        }
        let name = match &attr.meta {
            syn::Meta::Path(_) => None,
            syn::Meta::List(list) => {
                let lit: syn::LitStr = syn::parse2(list.tokens.clone())?;
                Some(lit.value())
            }
            syn::Meta::NameValue(_) => {
                return Err(syn::Error::new(
                    attr.span(),
                    "`#[attr = \"…\"]` is not supported; use `#[attr(\"…\")]`",
                ));
            }
        };
        found = Some(name);
    }
    Ok(found)
}

fn classify_positional_type(ty: &Type) -> syn::Result<ParamKind> {
    let Type::Path(path) = ty else {
        return Err(syn::Error::new(ty.span(), "expected a named type"));
    };
    let last = path
        .path
        .segments
        .last()
        .ok_or_else(|| syn::Error::new(ty.span(), "empty type path"))?;
    let name = last.ident.to_string();
    match name.as_str() {
        "Stream" => Ok(ParamKind::Stream),
        "RemainingArgs" => Ok(ParamKind::RemainingArgs),
        "RemainingRets" => Ok(ParamKind::RemainingRets),
        other => Err(syn::Error::new(
            ty.span(),
            format!(
                "unsupported positional parameter type `{other}` — \
                 expected Stream / RemainingArgs / RemainingRets, or annotate with `#[attr]`"
            ),
        )),
    }
}

fn classify_attr_type(ty: &Type, name: &str) -> syn::Result<ParamKind> {
    match ty {
        Type::Path(_) => Ok(ParamKind::AttrScalar {
            name: name.to_string(),
            ty: Box::new(ty.clone()),
        }),
        _ => Err(syn::Error::new(
            ty.span(),
            "`#[attr]` parameter type must be a scalar",
        )),
    }
}
