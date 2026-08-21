// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden renders and folding behaviour for every expression form.

use super::{Expr, FloatBits, IntoExpr};

#[test]
fn float_lit_renders_with_suffix_for_f32() {
    assert_eq!(Expr::FloatLit(1.0, FloatBits::F32).render(), "1e0f");
    assert_eq!(Expr::FloatLit(1.0, FloatBits::F64).render(), "1e0");
    assert_eq!(Expr::FloatLit(0.5, FloatBits::F32).render(), "5e-1f");
}

#[test]
fn into_expr_floats() {
    assert_eq!(1.0_f32.into_expr().render(), "1e0f");
    assert_eq!(1.0_f64.into_expr().render(), "1e0");
}

#[test]
fn product_folds_left() {
    let e = Expr::product([Expr::var("a"), Expr::var("b"), Expr::var("c")]);
    assert_eq!(e.render(), "a * b * c");
    let e = Expr::product([Expr::lit(2), Expr::var("x"), Expr::lit(3)]);
    assert_eq!(e.render(), "2 * x * 3");
}

#[test]
fn folding_add_zero() {
    let e = Expr::add(Expr::lit(0), Expr::var("x"));
    assert_eq!(e.render(), "x");
    let e = Expr::add(Expr::var("x"), Expr::lit(0));
    assert_eq!(e.render(), "x");
}

#[test]
fn folding_mul_zero_one() {
    assert_eq!(Expr::mul(Expr::var("x"), Expr::lit(0)).render(), "0");
    assert_eq!(Expr::mul(Expr::lit(1), Expr::var("x")).render(), "x");
    assert_eq!(Expr::mul(Expr::var("x"), Expr::lit(1)).render(), "x");
}

#[test]
fn lit_lit_folds() {
    let e = Expr::add(Expr::mul(Expr::lit(2), Expr::lit(3)), Expr::lit(4));
    assert_eq!(e, Expr::Lit(10));
    assert_eq!(e.render(), "10");
}

#[test]
fn precedence_no_parens_needed() {
    let e = Expr::add(Expr::mul(Expr::var("a"), Expr::var("b")), Expr::var("c"));
    assert_eq!(e.render(), "a * b + c");
}

#[test]
fn precedence_parens_around_lhs_add() {
    let e = Expr::mul(Expr::add(Expr::var("a"), Expr::var("b")), Expr::var("c"));
    assert_eq!(e.render(), "(a + b) * c");
}

#[test]
fn precedence_left_assoc_sub() {
    let inner = Expr::sub(Expr::var("b"), Expr::var("c"));
    let e = Expr::sub(Expr::var("a"), inner);
    assert_eq!(e.render(), "a - (b - c)");
    let e = Expr::sub(Expr::sub(Expr::var("a"), Expr::var("b")), Expr::var("c"));
    assert_eq!(e.render(), "a - b - c");
}

#[test]
fn index_and_call_render() {
    let e = Expr::index("idx_0", Expr::var("off_idx_0"));
    assert_eq!(e.render(), "idx_0[off_idx_0]");
    let e = Expr::call(
        "atomicAdd",
        vec![Expr::index("a", Expr::lit(3)), Expr::var("v")],
    );
    assert_eq!(e.render(), "atomicAdd(a[3], v)");
}

#[test]
fn nested_index() {
    let inner = Expr::index("data", Expr::var("ix"));
    let outer = Expr::index(inner, Expr::lit(0));
    assert_eq!(outer.render(), "data[ix][0]");
}

#[test]
fn addr_of_index() {
    let e = Expr::addr(Expr::index("buf", Expr::var("i")));
    assert_eq!(e.render(), "&buf[i]");
}

#[test]
fn neg_render_and_fold() {
    assert_eq!(Expr::neg(Expr::var("x")).render(), "-x");
    assert_eq!(Expr::neg(Expr::lit(5)), Expr::Lit(-5));
    assert_eq!(
        Expr::neg(Expr::FloatLit(0.5, FloatBits::F32)),
        Expr::FloatLit(-0.5, FloatBits::F32)
    );
    assert_eq!(
        Expr::neg(Expr::add(Expr::var("a"), Expr::var("b"))).render(),
        "-(a + b)"
    );
}

#[test]
fn comparison_render() {
    assert_eq!(
        Expr::ge(Expr::var("idx"), Expr::lit(1024)).render(),
        "idx >= 1024"
    );
    assert_eq!(Expr::lt(Expr::var("x"), Expr::var("y")).render(), "x < y");
    assert_eq!(Expr::eq(Expr::var("a"), Expr::lit(0)).render(), "a == 0");
    assert_eq!(Expr::ne(Expr::var("a"), Expr::lit(0)).render(), "a != 0");
}

#[test]
fn into_expr_blanket() {
    fn take(x: impl IntoExpr) -> Expr {
        x.into_expr()
    }
    assert_eq!(take("x").render(), "x");
    assert_eq!(take(5_i64).render(), "5");
    assert_eq!(take(Expr::lit(7)).render(), "7");
}

#[test]
fn offset_row_simplifies() {
    let extents = [2_i64, 1, 4];
    let term0 = Expr::var("batch_idx_0");
    let term1 = Expr::lit(0);
    let term2 = Expr::index("idx_0", Expr::var("off_idx_0"));

    let mut e = term0;
    e = Expr::add(Expr::mul(e, Expr::lit(extents[1])), term1);
    assert_eq!(e.render(), "batch_idx_0");
    e = Expr::add(Expr::mul(e, Expr::lit(extents[2])), term2);
    assert_eq!(e.render(), "batch_idx_0 * 4 + idx_0[off_idx_0]");
}

#[test]
fn atomic_cas_render() {
    let e = Expr::call(
        "atomicCAS",
        vec![
            Expr::addr(Expr::index("cache_idx", Expr::var("_slot"))),
            Expr::lit(-1),
            Expr::var("_va"),
        ],
    );
    assert_eq!(e.render(), "atomicCAS(&cache_idx[_slot], -1, _va)");
}

#[test]
fn shfl_sync_render() {
    let e = Expr::call(
        "__shfl_sync",
        vec![
            Expr::lit(0xFFFFFFFF),
            Expr::var("src"),
            Expr::add(Expr::var("_base"), Expr::var("_src")),
        ],
    );
    assert_eq!(e.render(), "__shfl_sync(4294967295, src, _base + _src)");
}

#[test]
fn shfl_sync_with_width_renders_fourth_arg() {
    let e = Expr::call(
        "__shfl_sync",
        vec![
            Expr::lit(0xFFFFFFFF),
            Expr::var("ci"),
            Expr::var("src_16"),
            Expr::lit(16),
        ],
    );
    assert_eq!(e.render(), "__shfl_sync(4294967295, ci, src_16, 16)");
}

#[test]
fn shr_render_and_fold() {
    assert_eq!(
        Expr::shr(Expr::var("threadIdx.x"), Expr::lit(5)).render(),
        "threadIdx.x >> 5"
    );
    assert_eq!(Expr::shr(Expr::var("x"), Expr::lit(0)).render(), "x");
    assert_eq!(Expr::shr(Expr::lit(64), Expr::lit(2)), Expr::Lit(16));
}

#[test]
fn band_render_and_fold() {
    assert_eq!(
        Expr::band(Expr::var("threadIdx.x"), Expr::lit(31)).render(),
        "threadIdx.x & 31"
    );
    assert_eq!(Expr::band(Expr::var("x"), Expr::lit(0)), Expr::Lit(0));
    assert_eq!(
        Expr::band(Expr::lit(0xFF), Expr::lit(0x0F)),
        Expr::Lit(0x0F)
    );
}

#[test]
fn land_lor_render() {
    let e = Expr::lor(
        Expr::var("_inactive"),
        Expr::ge(Expr::var("sorted_idx"), Expr::var("TOTAL")),
    );
    assert_eq!(e.render(), "_inactive || sorted_idx >= TOTAL");
    let e = Expr::land(Expr::var("a"), Expr::var("b"));
    assert_eq!(e.render(), "a && b");
}

#[test]
fn precedence_logical_vs_comparison() {
    let e = Expr::land(
        Expr::ge(Expr::var("a"), Expr::var("b")),
        Expr::lt(Expr::var("c"), Expr::var("d")),
    );
    assert_eq!(e.render(), "a >= b && c < d");
}

#[test]
fn precedence_or_around_and() {
    let e = Expr::lor(Expr::var("a"), Expr::land(Expr::var("b"), Expr::var("c")));
    assert_eq!(e.render(), "a || b && c");
    let e = Expr::land(Expr::lor(Expr::var("a"), Expr::var("b")), Expr::var("c"));
    assert_eq!(e.render(), "(a || b) && c");
}

#[test]
fn bor_render_and_fold() {
    assert_eq!(Expr::bor(Expr::var("a"), Expr::var("b")).render(), "a | b");
    assert_eq!(Expr::bor(Expr::var("x"), Expr::lit(0)), Expr::var("x"));
    assert_eq!(Expr::bor(Expr::lit(0x0F), Expr::lit(0xF0)), Expr::Lit(0xFF));
}

#[test]
fn shl_render_and_fold() {
    assert_eq!(Expr::shl(Expr::var("a"), Expr::lit(8)).render(), "a << 8");
    assert_eq!(Expr::shl(Expr::var("x"), Expr::lit(0)), Expr::var("x"));
    assert_eq!(Expr::shl(Expr::lit(1), Expr::lit(4)), Expr::Lit(16));
}

#[test]
fn ternary_renders() {
    let e = Expr::ternary(Expr::var("c"), Expr::var("a"), Expr::var("b"));
    assert_eq!(e.render(), "c ? a : b");
}

#[test]
fn ternary_inside_binop_wraps() {
    let t = Expr::ternary(Expr::var("c"), Expr::lit(1), Expr::lit(2));
    let e = Expr::add(t, Expr::lit(5));
    assert_eq!(e.render(), "(c ? 1 : 2) + 5");
}

#[test]
fn ternary_right_associative() {
    // else-branch at same precedence renders without inner parens
    let inner = Expr::ternary(Expr::var("c"), Expr::var("d"), Expr::var("e"));
    let outer = Expr::ternary(Expr::var("a"), Expr::var("b"), inner);
    assert_eq!(outer.render(), "a ? b : c ? d : e");
}

#[test]
fn cast_renders() {
    let e = Expr::cast(
        "int",
        Expr::mul(Expr::var("blockIdx.x"), Expr::var("blockDim.x")),
    );
    assert_eq!(e.render(), "(int)(blockIdx.x * blockDim.x)");
}

#[test]
fn ldg_render() {
    let e = Expr::call(
        "__ldg",
        vec![Expr::addr(Expr::index("grid_in_0", Expr::var("cell")))],
    );
    assert_eq!(e.render(), "__ldg(&grid_in_0[cell])");
}
