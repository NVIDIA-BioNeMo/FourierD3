// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lowering the parsed syntax tree into the `quote!`d `fourierd3_engine` constructor
//! calls the macro expands to.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

use crate::ast::{CAssignOp, CExpr, CName, COp, CSize, CStmt, CType, CUnroll};

pub(crate) fn emit_pushes(target: &TokenStream2, stmts: &[CStmt]) -> TokenStream2 {
    let mut out = TokenStream2::new();
    for stmt in stmts {
        match stmt {
            CStmt::Splice(ts) => {
                out.extend(quote! { #target.extend(#ts); });
            }
            _ => {
                let push_expr = emit_stmt(stmt);
                out.extend(quote! { #target.push(#push_expr); });
            }
        }
    }
    out
}

pub(crate) fn emit_body_vec(body: &[CStmt]) -> TokenStream2 {
    let pushes = emit_pushes(&quote! { __body }, body);
    quote! {
        {
            let mut __body: ::std::vec::Vec<::fourierd3_engine::ir::stmt::Stmt> = ::std::vec::Vec::new();
            #pushes
            __body
        }
    }
}

pub(crate) fn emit_decl(ty: &CType, decls: &[(CName, Option<CExpr>)]) -> TokenStream2 {
    let ty = emit_ty(ty);
    let emit_init = |init: &Option<CExpr>| match init {
        Some(e) => {
            let e = emit_expr(e);
            quote! { ::std::option::Option::Some(#e) }
        }
        None => quote! { ::std::option::Option::None },
    };
    if let [(name, init)] = decls {
        let name = emit_name(name);
        let init = emit_init(init);
        quote! { ::fourierd3_engine::ir::stmt::Decl::scalar(#ty, #name, #init) }
    } else {
        let entries: Vec<_> = decls
            .iter()
            .map(|(name, init)| {
                let name = emit_name(name);
                let init = emit_init(init);
                quote! { (#name, #init) }
            })
            .collect();
        quote! { ::fourierd3_engine::ir::stmt::Decl::multi(#ty, ::std::vec![ #(#entries),* ]) }
    }
}

pub(crate) fn emit_for_init(init: &CStmt) -> TokenStream2 {
    match init {
        CStmt::Decl { ty, decls } => {
            let decl = emit_decl(ty, decls);
            quote! { ::fourierd3_engine::ir::stmt::ForInit::Decl(#decl) }
        }
        CStmt::Eval(e) => {
            let e = emit_expr(e);
            quote! { ::fourierd3_engine::ir::stmt::ForInit::Expr(#e) }
        }
        _ => quote! {
            compile_error!("`for` init must be a declaration or an expression statement")
        },
    }
}

pub(crate) fn emit_stmt(stmt: &CStmt) -> TokenStream2 {
    match stmt {
        CStmt::Decl { ty, decls } => {
            let decl = emit_decl(ty, decls);
            quote! { ::fourierd3_engine::ir::stmt::Stmt::Decl(#decl) }
        }
        CStmt::ArrayDecl { ty, name, dims } => {
            let ty = emit_ty(ty);
            let name = emit_name(name);
            let dims: Vec<_> = dims.iter().map(emit_size).collect();
            quote! {
                ::fourierd3_engine::ir::stmt::Stmt::array_decl(#ty, #name, ::std::vec![ #(#dims),* ])
            }
        }
        CStmt::ArrayInitDecl {
            ty,
            name,
            size,
            init,
        } => {
            let ty = emit_ty(ty);
            let name = emit_name(name);
            let size = match size {
                Some(s) => {
                    let s = emit_size(s);
                    quote! { ::std::option::Option::Some(#s) }
                }
                None => quote! { ::std::option::Option::None },
            };
            let init: Vec<_> = init.iter().map(emit_expr).collect();
            quote! {
                ::fourierd3_engine::ir::stmt::Stmt::array_init_decl(#ty, #name, #size, ::std::vec![ #(#init),* ])
            }
        }
        CStmt::ExternSharedDecl { ty, name } => {
            let ty = emit_ty(ty);
            let name = emit_name(name);
            quote! { ::fourierd3_engine::ir::stmt::Stmt::extern_shared_decl(#ty, #name) }
        }
        CStmt::ExternDeviceDecl { name, param_types } => {
            let name = emit_name(name);
            let types: Vec<_> = param_types.iter().map(emit_ty).collect();
            quote! {
                ::fourierd3_engine::ir::stmt::Stmt::ExternDeviceDecl {
                    name: #name,
                    param_types: vec![ #(#types),* ],
                }
            }
        }
        CStmt::Eval(expr) => {
            let expr = emit_expr(expr);
            quote! { ::fourierd3_engine::ir::stmt::Stmt::Eval(#expr) }
        }
        CStmt::If { cond, then_, else_ } => {
            let cond = emit_expr(cond);
            let then_vec = emit_body_vec(then_);
            let else_t = match else_ {
                Some(stmts) => {
                    let v = emit_body_vec(stmts);
                    quote! { ::std::option::Option::Some(#v) }
                }
                None => quote! { ::std::option::Option::None },
            };
            quote! {
                ::fourierd3_engine::ir::stmt::Stmt::If {
                    cond: #cond,
                    then_: #then_vec,
                    else_: #else_t,
                }
            }
        }
        CStmt::For {
            init,
            cond,
            step,
            body,
            unroll,
        } => {
            let init = emit_for_init(init);
            let cond = emit_expr(cond);
            let step = emit_expr(step);
            let unroll = emit_unroll(unroll);
            let body_vec = emit_body_vec(body);
            quote! {
                ::fourierd3_engine::ir::stmt::Stmt::For {
                    init: #init,
                    cond: #cond,
                    step: #step,
                    body: #body_vec,
                    unroll: #unroll,
                }
            }
        }
        CStmt::While { cond, body } => {
            let cond = emit_expr(cond);
            let body_vec = emit_body_vec(body);
            quote! {
                ::fourierd3_engine::ir::stmt::Stmt::While { cond: #cond, body: #body_vec }
            }
        }
        CStmt::Continue => quote! { ::fourierd3_engine::ir::stmt::Stmt::Continue },
        CStmt::Return(value) => {
            let inner = match value {
                Some(v) => {
                    let v = emit_expr(v);
                    quote! { Some(#v) }
                }
                None => quote! { None },
            };
            quote! { ::fourierd3_engine::ir::stmt::Stmt::Return(#inner) }
        }
        CStmt::Blank => quote! { ::fourierd3_engine::ir::stmt::Stmt::Blank },
        CStmt::Splice(_) => {
            unreachable!("CStmt::Splice should be handled by emit_pushes, not emit_stmt");
        }
    }
}

pub(crate) fn emit_unroll(unroll: &CUnroll) -> TokenStream2 {
    match unroll {
        CUnroll::None => quote! { ::fourierd3_engine::ir::stmt::Unroll::None },
        CUnroll::All => quote! { ::fourierd3_engine::ir::stmt::Unroll::All },
    }
}

pub(crate) fn emit_ty(ty: &CType) -> TokenStream2 {
    match ty {
        CType::Lit(s) => {
            let s = s.as_str();
            quote! { ::std::string::String::from(#s) }
        }
        CType::Interp(ts) => quote! { ::std::string::String::from((#ts).clone()) },
    }
}

pub(crate) fn emit_name(name: &CName) -> TokenStream2 {
    match name {
        CName::Lit(s) => {
            let s = s.as_str();
            quote! { ::std::string::String::from(#s) }
        }
        CName::Interp(ts) => quote! { ::std::string::String::from((#ts).clone()) },
    }
}

pub(crate) fn emit_call_args(args: &[CExpr]) -> TokenStream2 {
    let args: Vec<_> = args.iter().map(emit_expr).collect();
    quote! { ::std::vec![ #(#args),* ] }
}

pub(crate) fn emit_size(size: &CSize) -> TokenStream2 {
    match size {
        CSize::Lit(v) => {
            let v = *v;
            quote! { #v }
        }
        CSize::Interp(ts) => quote! { (#ts) as i64 },
    }
}

pub(crate) fn emit_assignop(op: CAssignOp) -> TokenStream2 {
    match op {
        CAssignOp::Set => quote! { ::fourierd3_engine::ir::expr::AssignOp::Set },
        CAssignOp::AddAssign => quote! { ::fourierd3_engine::ir::expr::AssignOp::AddAssign },
        CAssignOp::SubAssign => quote! { ::fourierd3_engine::ir::expr::AssignOp::SubAssign },
        CAssignOp::DivAssign => quote! { ::fourierd3_engine::ir::expr::AssignOp::DivAssign },
        CAssignOp::BitAndAssign => quote! { ::fourierd3_engine::ir::expr::AssignOp::BitAndAssign },
        CAssignOp::ShrAssign => quote! { ::fourierd3_engine::ir::expr::AssignOp::ShrAssign },
    }
}

pub(crate) fn emit_expr(e: &CExpr) -> TokenStream2 {
    match e {
        CExpr::Var(name) => {
            let name = name.as_str();
            quote! { ::fourierd3_engine::ir::expr::Expr::var(::std::string::String::from(#name)) }
        }
        CExpr::Lit(v) => {
            let v = *v;
            quote! { ::fourierd3_engine::ir::expr::Expr::lit(#v) }
        }
        CExpr::Interp(ts) => {
            quote! { ::fourierd3_engine::ir::expr::IntoExpr::into_expr((#ts).clone()) }
        }
        CExpr::BinOp(op, lhs, rhs) => {
            let lhs = emit_expr(lhs);
            let rhs = emit_expr(rhs);
            let ctor = match op {
                COp::Add => quote! { add },
                COp::Sub => quote! { sub },
                COp::Mul => quote! { mul },
                COp::Div => quote! { div },
                COp::Mod => quote! { rem },
                COp::Eq => quote! { eq },
                COp::Ne => quote! { ne },
                COp::Lt => quote! { lt },
                COp::Le => quote! { le },
                COp::Gt => quote! { gt },
                COp::Ge => quote! { ge },
                COp::Shl => quote! { shl },
                COp::Shr => quote! { shr },
                COp::BitAnd => quote! { band },
                COp::BitOr => quote! { bor },
                COp::LogicalAnd => quote! { land },
                COp::LogicalOr => quote! { lor },
            };
            quote! { ::fourierd3_engine::ir::expr::Expr::#ctor(#lhs, #rhs) }
        }
        CExpr::Ternary { cond, then_, else_ } => {
            let cond = emit_expr(cond);
            let then_ = emit_expr(then_);
            let else_ = emit_expr(else_);
            quote! {
                ::fourierd3_engine::ir::expr::Expr::ternary(#cond, #then_, #else_)
            }
        }
        CExpr::Assign { op, target, value } => {
            let op = emit_assignop(*op);
            let target = emit_expr(target);
            let value = emit_expr(value);
            quote! {
                ::fourierd3_engine::ir::expr::Expr::assign(#op, #target, #value)
            }
        }
        CExpr::Cast { ty, expr } => {
            let ty = emit_ty(ty);
            let expr = emit_expr(expr);
            quote! {
                ::fourierd3_engine::ir::expr::Expr::cast(#ty, #expr)
            }
        }
        CExpr::Index(arr, idx) => {
            let arr = emit_expr(arr);
            let idx = emit_expr(idx);
            quote! { ::fourierd3_engine::ir::expr::Expr::index(#arr, #idx) }
        }
        CExpr::Addr(inner) => {
            let inner = emit_expr(inner);
            quote! { ::fourierd3_engine::ir::expr::Expr::addr(#inner) }
        }
        CExpr::Neg(inner) => {
            let inner = emit_expr(inner);
            quote! { ::fourierd3_engine::ir::expr::Expr::neg(#inner) }
        }
        CExpr::PostInc(target) => {
            let target = emit_expr(target);
            quote! { ::fourierd3_engine::ir::expr::Expr::post_inc(#target) }
        }
        CExpr::Call(name, args) => {
            let callee = emit_name(name);
            let args_build = emit_call_args(args);
            quote! {
                ::fourierd3_engine::ir::expr::Expr::call(#callee, #args_build)
            }
        }
        CExpr::Member(base, name) => {
            let base = emit_expr(base);
            let s = name.as_str();
            quote! {
                ::fourierd3_engine::ir::expr::Expr::member(#base, ::std::string::String::from(#s))
            }
        }
    }
}
