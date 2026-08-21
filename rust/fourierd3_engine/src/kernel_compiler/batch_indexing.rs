// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use fourierd3_engine::cuda;
use fourierd3_engine::ir::expr::{Expr, IntoExpr};
use fourierd3_engine::ir::stmt::Stmt;

pub(crate) fn push_batch_decode(stmts: &mut Vec<Stmt>, batch_sizes: &[i64], flat: impl IntoExpr) {
    let ctype = String::from("int");
    let prefix = String::from("batch_idx_");
    push_batch_decode_typed(stmts, batch_sizes, flat, &ctype, &prefix);
}

pub(crate) fn push_batch_decode_typed(
    stmts: &mut Vec<Stmt>,
    batch_sizes: &[i64],
    flat: impl IntoExpr,
    ctype: &str,
    axis_prefix: &str,
) {
    let flat = flat.into_expr();
    match batch_sizes {
        [] => {}
        [_] => {
            let name = format!("{axis_prefix}0");
            cuda! { stmts => #ctype #name = #flat; }
        }
        _ => {
            let mut axis_decls: Vec<Stmt> = Vec::new();
            for i in (1..batch_sizes.len()).rev() {
                let name = format!("{axis_prefix}{i}");
                let sz = batch_sizes[i];
                cuda! { axis_decls =>
                    #ctype #name = _rem % #sz;
                    _rem /= #sz;
                }
            }
            let first = format!("{axis_prefix}0");
            cuda! { stmts =>
                #ctype _rem = #flat;
                splice!(axis_decls);
                #ctype #first = _rem;
            }
        }
    }
}

pub(crate) fn row_major_offset_expr(
    ic_entry: &[i32],
    buf_extents: &[i64],
    idx_offset_vars: &HashMap<usize, String>,
) -> Expr {
    let batch_prefix = String::from("batch_idx_");
    let idx_prefix = String::from("idx_");
    let term = |axis: usize| -> Expr {
        let r = ic_entry[axis];
        if r < 0 {
            if buf_extents[axis] == 1 {
                Expr::lit(0)
            } else {
                Expr::var(format!("{batch_prefix}{axis}"))
            }
        } else {
            let i = r as usize;
            let v = idx_offset_vars
                .get(&i)
                .expect("idx_offset_vars missing entry");
            Expr::index(format!("{idx_prefix}{i}"), Expr::var(v.clone()))
        }
    };
    if ic_entry.is_empty() {
        return Expr::lit(0);
    }
    // The (i64) cast keeps every `acc * extent + idx` step 64-bit so neither
    // the accumulation nor the `off * elem_size` pointer base can overflow past 2^31.
    let seed = Expr::cast(String::from("long long"), term(0));
    (1..ic_entry.len()).fold(seed, |e, axis| {
        Expr::add(Expr::mul(e, Expr::lit(buf_extents[axis])), term(axis))
    })
}

pub(crate) fn push_row_major_offset_decl(
    stmts: &mut Vec<Stmt>,
    name: &str,
    ic_entry: &[i32],
    buf_extents: &[i64],
    idx_offset_vars: &HashMap<usize, String>,
) {
    let init = row_major_offset_expr(ic_entry, buf_extents, idx_offset_vars);
    cuda! { stmts => i64 #name = #init; }
}
