// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rendering declarations: storage class, declarators, and initializers.

use crate::ir::code_builder::CodeBuilder;
use crate::ir::expr::Expr;
use crate::ir::stmt::{Decl, DeclInit, Declarator, Storage};
use crate::{cb, emit};

impl Decl {
    pub fn scalar(ty: impl Into<String>, name: impl Into<String>, init: Option<Expr>) -> Self {
        Decl {
            storage: Storage::Plain,
            ty: ty.into(),
            declarators: vec![Declarator {
                name: name.into(),
                dims: vec![],
                init: init.map(DeclInit::Scalar),
            }],
        }
    }

    pub fn multi(ty: impl Into<String>, decls: Vec<(String, Option<Expr>)>) -> Self {
        Decl {
            storage: Storage::Plain,
            ty: ty.into(),
            declarators: decls
                .into_iter()
                .map(|(name, init)| Declarator {
                    name,
                    dims: vec![],
                    init: init.map(DeclInit::Scalar),
                })
                .collect(),
        }
    }

    pub fn array(ty: impl Into<String>, name: impl Into<String>, dims: Vec<i64>) -> Self {
        Decl {
            storage: Storage::Plain,
            ty: ty.into(),
            declarators: vec![Declarator {
                name: name.into(),
                dims: dims.into_iter().map(Some).collect(),
                init: None,
            }],
        }
    }

    pub fn array_init(
        ty: impl Into<String>,
        name: impl Into<String>,
        size: Option<i64>,
        init: Vec<Expr>,
    ) -> Self {
        Decl {
            storage: Storage::Plain,
            ty: ty.into(),
            declarators: vec![Declarator {
                name: name.into(),
                dims: vec![size],
                init: Some(DeclInit::List(init)),
            }],
        }
    }

    pub fn shared(ty: impl Into<String>, name: impl Into<String>, size: Option<i64>) -> Self {
        Decl {
            storage: Storage::Shared,
            ty: ty.into(),
            declarators: vec![Declarator {
                name: name.into(),
                dims: size.map(|n| vec![Some(n)]).unwrap_or_default(),
                init: None,
            }],
        }
    }

    pub fn extern_shared(ty: impl Into<String>, name: impl Into<String>) -> Self {
        Decl {
            storage: Storage::ExternShared,
            ty: ty.into(),
            declarators: vec![Declarator {
                name: name.into(),
                dims: vec![None],
                init: None,
            }],
        }
    }

    pub fn constant_array(
        ty: impl Into<String>,
        name: impl Into<String>,
        size: Option<i64>,
        init: Vec<Expr>,
    ) -> Self {
        Decl {
            storage: Storage::Constant,
            ty: ty.into(),
            declarators: vec![Declarator {
                name: name.into(),
                dims: vec![size],
                init: Some(DeclInit::List(init)),
            }],
        }
    }

    pub(crate) fn emit(&self, cb: &mut CodeBuilder) {
        self.storage.emit_prefix(cb);
        cb.push_str(&self.ty);
        emit!(cb, " ");
        for (i, d) in self.declarators.iter().enumerate() {
            if i > 0 {
                emit!(cb, ", ");
            }
            d.emit(cb);
        }
    }
}

impl Declarator {
    fn emit(&self, cb: &mut CodeBuilder) {
        cb.push_str(&self.name);
        for dim in &self.dims {
            emit!(cb, "[");
            if let Some(n) = dim {
                cb!(cb, "{}", n);
            }
            emit!(cb, "]");
        }
        if let Some(init) = &self.init {
            init.emit(cb);
        }
    }
}

impl DeclInit {
    fn emit(&self, cb: &mut CodeBuilder) {
        emit!(cb, " = ");
        match self {
            DeclInit::Scalar(expr) => expr.emit(cb),
            DeclInit::List(exprs) => {
                emit!(cb, "{");
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        emit!(cb, ", ");
                    }
                    expr.emit(cb);
                }
                emit!(cb, "}");
            }
        }
    }
}
