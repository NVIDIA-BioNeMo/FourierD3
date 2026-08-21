// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden CUDA-source renders for every statement form.

use super::{Decl, ForInit, Param, Stmt, Unroll};
use crate::ir::code_builder::CodeBuilder;
use crate::ir::expr::{AssignOp, Expr};
use crate::ir::stmt::kernel_params::emit_param_list;

#[test]
fn kernel_params_grammar() {
    let ct = "double";
    let params = crate::kernel_params![
        i32 n,
        const f32 * restrict pos,
        f32 * out,
        const u32 * g_count,
        u32 * restrict idx,
        const #ct * restrict lat,
        #{ct} * scratch,
    ];
    let mut cb = CodeBuilder::new();
    emit_param_list(&mut cb, &params);
    assert_eq!(
        cb.finish_string(),
        "int n,\nconst float* __restrict__ pos,\nfloat* out,\n\
         const unsigned int* g_count,\nunsigned int* __restrict__ idx,\n\
         const double* __restrict__ lat,\ndouble* scratch"
    );
}

#[test]
#[allow(clippy::cognitive_complexity)]
fn dsl_type_mnemonics_match_dtype_ctype() {
    use crate::dtype::Dtype;
    macro_rules! check {
        ($($mn:ident => $dt:ident),* $(,)?) => {{
            $(
                let mut v: Vec<Stmt> = Vec::new();
                crate::cuda! { v => $mn x = 0; };
                let Stmt::Decl(Decl { ty, .. }) = &v[0] else {
                    panic!("expected Decl for {}", stringify!($mn));
                };
                assert_eq!(ty, Dtype::$dt.ctype(), "mnemonic {}", stringify!($mn));
            )*
        }};
    }
    check! {
        f32 => F32, f64 => F64, f16 => F16, bf16 => Bf16,
        bool => Bool,
        i8 => I8, i16 => I16, i32 => I32, i64 => I64,
        u8 => U8, u16 => U16, u32 => U32, u64 => U64,
        c64 => Complex64, c128 => Complex128,
    }
}

#[test]
fn decl_renders() {
    let mut cb = CodeBuilder::new();
    Stmt::decl("int", "x", Some(Expr::lit(42))).emit(&mut cb);
    assert_eq!(cb.finish_string(), "int x = 42;\n");
}

#[test]
fn decl_uninitialized_omits_init() {
    let mut cb = CodeBuilder::new();
    Stmt::decl("bf16x8", "r", None).emit(&mut cb);
    assert_eq!(cb.finish_string(), "bf16x8 r;\n");
}

#[test]
fn extern_device_decl_renders() {
    let mut cb = CodeBuilder::new();
    Stmt::ExternDeviceDecl {
        name: "cufftdx_execute_abc".into(),
        param_types: vec!["float2*".into(), "char*".into()],
    }
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "extern \"C\" __device__ void cufftdx_execute_abc(float2*, char*);\n"
    );
}

#[test]
fn array_decl_renders() {
    let mut cb = CodeBuilder::new();
    Stmt::array_decl("float", "scratch", vec![4]).emit(&mut cb);
    assert_eq!(cb.finish_string(), "float scratch[4];\n");
}

#[test]
fn array_decl_2d_renders() {
    let mut cb = CodeBuilder::new();
    Stmt::array_decl("float", "data", vec![4, 3]).emit(&mut cb);
    assert_eq!(cb.finish_string(), "float data[4][3];\n");
}

#[test]
fn array_init_decl_size_deduced() {
    let mut cb = CodeBuilder::new();
    Stmt::array_init_decl(
        "int",
        "_sup",
        None,
        vec![
            Expr::index("SUPPORT", Expr::mul(Expr::var("s"), Expr::lit(3))),
            Expr::index(
                "SUPPORT",
                Expr::add(Expr::mul(Expr::var("s"), Expr::lit(3)), Expr::lit(1)),
            ),
            Expr::index(
                "SUPPORT",
                Expr::add(Expr::mul(Expr::var("s"), Expr::lit(3)), Expr::lit(2)),
            ),
        ],
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "int _sup[] = {SUPPORT[s * 3], SUPPORT[s * 3 + 1], SUPPORT[s * 3 + 2]};\n"
    );
}

#[test]
fn array_init_decl_explicit_size() {
    let mut cb = CodeBuilder::new();
    Stmt::array_init_decl(
        "float",
        "w",
        Some(4),
        vec![Expr::lit(0), Expr::lit(0), Expr::lit(0), Expr::lit(0)],
    )
    .emit(&mut cb);
    assert_eq!(cb.finish_string(), "float w[4] = {0, 0, 0, 0};\n");
}

#[test]
fn shared_scalar_decl_renders() {
    let mut cb = CodeBuilder::new();
    Stmt::shared_decl("int", "anchor_x", None).emit(&mut cb);
    assert_eq!(cb.finish_string(), "__shared__ int anchor_x;\n");
}

#[test]
fn shared_array_decl_renders() {
    let mut cb = CodeBuilder::new();
    Stmt::shared_decl("float", "cache_val", Some(1024)).emit(&mut cb);
    assert_eq!(cb.finish_string(), "__shared__ float cache_val[1024];\n");
}

#[test]
fn assign_compound() {
    let mut cb = CodeBuilder::new();
    Stmt::assign(Expr::var("_rem"), AssignOp::DivAssign, Expr::lit(6)).emit(&mut cb);
    assert_eq!(cb.finish_string(), "_rem /= 6;\n");
}

#[test]
fn atomic_add_is_a_call_statement() {
    let mut cb = CodeBuilder::new();
    Stmt::Eval(Expr::call(
        "atomicAdd",
        vec![
            Expr::addr(Expr::index("out_0", Expr::var("off"))),
            Expr::index("_out_0", Expr::lit(0)),
        ],
    ))
    .emit(&mut cb);
    assert_eq!(cb.finish_string(), "atomicAdd(&out_0[off], _out_0[0]);\n");
}

#[test]
fn if_then_only() {
    let mut cb = CodeBuilder::new();
    Stmt::If {
        cond: Expr::eq(Expr::var("threadIdx.x"), Expr::lit(0)),
        then_: vec![Stmt::assign(
            Expr::var("anchor_x"),
            AssignOp::Set,
            Expr::lit(0),
        )],
        else_: None,
    }
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "if (threadIdx.x == 0) {\n    anchor_x = 0;\n}\n"
    );
}

#[test]
fn if_then_else() {
    let mut cb = CodeBuilder::new();
    Stmt::If {
        cond: Expr::lt(Expr::var("x"), Expr::lit(0)),
        then_: vec![Stmt::assign(Expr::var("y"), AssignOp::Set, Expr::lit(1))],
        else_: Some(vec![Stmt::assign(
            Expr::var("y"),
            AssignOp::Set,
            Expr::lit(2),
        )]),
    }
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "if (x < 0) {\n    y = 1;\n} else {\n    y = 2;\n}\n"
    );
}

#[test]
fn else_if_renders_inline() {
    let mut cb = CodeBuilder::new();
    Stmt::If {
        cond: Expr::eq(Expr::var("claimed"), Expr::var("grid_idx")),
        then_: vec![Stmt::assign(Expr::var("a"), AssignOp::Set, Expr::lit(1))],
        else_: Some(vec![Stmt::If {
            cond: Expr::eq(Expr::var("claimed"), Expr::lit(-1)),
            then_: vec![Stmt::assign(Expr::var("a"), AssignOp::Set, Expr::lit(2))],
            else_: Some(vec![Stmt::assign(
                Expr::var("a"),
                AssignOp::Set,
                Expr::lit(3),
            )]),
        }]),
    }
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "if (claimed == grid_idx) {\n    a = 1;\n} else if (claimed == -1) {\n    a = 2;\n} else {\n    a = 3;\n}\n"
    );
}

fn post(var: &str) -> Expr {
    Expr::post_inc(Expr::var(var))
}

fn step_assign(var: &str, op: AssignOp, rhs: i64) -> Expr {
    Expr::assign(op, Expr::var(var), Expr::lit(rhs))
}

fn counted_for(var: &str, start: Expr, cond: Expr, step: Expr, unroll: Unroll) -> Stmt {
    Stmt::For {
        init: ForInit::Decl(Decl::scalar("int", var, Some(start))),
        cond,
        step,
        body: vec![Stmt::assign(
            Expr::index("y", Expr::var(var)),
            AssignOp::Set,
            Expr::lit(0),
        )],
        unroll,
    }
}

#[test]
fn for_canonical_inc() {
    let mut cb = CodeBuilder::new();
    counted_for(
        "_a",
        Expr::lit(0),
        Expr::lt(Expr::var("_a"), Expr::lit(3)),
        post("_a"),
        Unroll::None,
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "for (int _a = 0; _a < 3; _a++) {\n    y[_a] = 0;\n}\n"
    );
}

#[test]
fn for_non_zero_start_renders() {
    let mut cb = CodeBuilder::new();
    counted_for(
        "s",
        Expr::var("_s0"),
        Expr::lt(Expr::var("s"), Expr::add(Expr::var("_s0"), Expr::lit(4))),
        post("s"),
        Unroll::None,
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "for (int s = _s0; s < _s0 + 4; s++) {\n    y[s] = 0;\n}\n"
    );
}

#[test]
fn for_add_assign_step() {
    let mut cb = CodeBuilder::new();
    counted_for(
        "_off",
        Expr::lit(0),
        Expr::lt(Expr::var("_off"), Expr::lit(32)),
        step_assign("_off", AssignOp::AddAssign, 2),
        Unroll::None,
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "for (int _off = 0; _off < 32; _off += 2) {\n    y[_off] = 0;\n}\n"
    );
}

#[test]
fn for_reverse_shr_assign() {
    let mut cb = CodeBuilder::new();
    counted_for(
        "_off",
        Expr::lit(16),
        Expr::ge(Expr::var("_off"), Expr::lit(1)),
        step_assign("_off", AssignOp::ShrAssign, 1),
        Unroll::None,
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "for (int _off = 16; _off >= 1; _off >>= 1) {\n    y[_off] = 0;\n}\n"
    );
}

#[test]
fn for_unroll_all_prefix() {
    let mut cb = CodeBuilder::new();
    counted_for(
        "i",
        Expr::lit(0),
        Expr::lt(Expr::var("i"), Expr::lit(4)),
        post("i"),
        Unroll::All,
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "#pragma unroll\nfor (int i = 0; i < 4; i++) {\n    y[i] = 0;\n}\n"
    );
}

#[test]
fn for_unroll_count_prefix() {
    let mut cb = CodeBuilder::new();
    counted_for(
        "i",
        Expr::lit(0),
        Expr::lt(Expr::var("i"), Expr::lit(8)),
        post("i"),
        Unroll::Count(4),
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "#pragma unroll 4\nfor (int i = 0; i < 8; i++) {\n    y[i] = 0;\n}\n"
    );
}

#[test]
fn constant_array_decl_renders() {
    let mut cb = CodeBuilder::new();
    Stmt::constant_array_decl(
        "unsigned int",
        "d_lut",
        Some(4),
        vec![Expr::lit(0), Expr::lit(1), Expr::lit(2), Expr::lit(3)],
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "__constant__ unsigned int d_lut[4] = {0, 1, 2, 3};\n"
    );
}

#[test]
fn constant_array_decl_size_deduced() {
    let mut cb = CodeBuilder::new();
    Stmt::constant_array_decl(
        "int",
        "d_offsets",
        None,
        vec![Expr::lit(-1), Expr::lit(0), Expr::lit(1)],
    )
    .emit(&mut cb);
    assert_eq!(
        cb.finish_string(),
        "__constant__ int d_offsets[] = {-1, 0, 1};\n"
    );
}

#[test]
fn param_pointer_emits_qualifiers() {
    let mut cb = CodeBuilder::new();
    Param::Pointer {
        const_: true,
        restrict: true,
        ctype: "float".into(),
        name: "x".into(),
    }
    .emit(&mut cb);
    assert_eq!(cb.finish_string(), "const float* __restrict__ x");
}

#[test]
fn emit_param_list_inserts_commas() {
    let mut cb = CodeBuilder::new();
    let params = vec![
        Param::Pointer {
            const_: true,
            restrict: true,
            ctype: "float".into(),
            name: "a".into(),
        },
        Param::Pointer {
            const_: false,
            restrict: true,
            ctype: "int".into(),
            name: "b".into(),
        },
    ];
    emit_param_list(&mut cb, &params);
    assert_eq!(
        cb.finish_string(),
        "const float* __restrict__ a,\nint* __restrict__ b"
    );
}
