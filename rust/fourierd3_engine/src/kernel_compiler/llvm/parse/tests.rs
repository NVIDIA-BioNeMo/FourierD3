// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing of the LLVM text jax lowers to, form by form.

use super::{BinOpKind, Instr, LlvmType, Operand, parse_module};

const TRIVIAL: &str = r#"; ModuleID = "V"
target triple = "nvptx64-nvidia-cuda"
target datalayout = ""

define void @"V"(i32* %"sidx", i32* %"sup", float* %"x", float* %"out0")
{
entry:
  %"v0" = load float, float* %"x"
  %"v1" = fmul float %"v0", %"v0"
  %"v2" = fadd float %"v1", 0x3ff0000000000000
  store float %"v2", float* %"out0"
  ret void
}
"#;

#[test]
fn parses_frem_as_binop_kind() {
    const IR: &str = r#"; ModuleID = "V"
target triple = "nvptx64-nvidia-cuda"

define void @"V"(float* %"x", float* %"y", float* %"out0")
{
entry:
  %"a" = load float, float* %"x"
  %"b" = load float, float* %"y"
  %"r" = frem float %"a", %"b"
  store float %"r", float* %"out0"
  ret void
}
"#;
    let m = parse_module(IR).unwrap();
    match &m.function.instrs[2] {
        Instr::BinOp { op, .. } => assert_eq!(*op, BinOpKind::FRem),
        other => panic!("expected frem binop, got {other:?}"),
    }
    // emit lowers FRem to a libdevice call, no CUDA infix.
    let cuda = super::super::emit_cuda(&m, m.function.params.len() - 1);
    assert!(
        cuda.contains("fmodf("),
        "expected fmodf call in emitted CUDA, got:\n{cuda}"
    );
}

#[test]
fn parses_trivial_v() {
    let m = parse_module(TRIVIAL).unwrap();
    let f = &m.function;
    assert_eq!(f.name, "V");
    assert_eq!(f.params.len(), 4);
    assert_eq!(f.params[2].name, "x");
    assert_eq!(f.params[2].elem_ty, LlvmType::F32);
    assert_eq!(f.instrs.len(), 5);
    match &f.instrs[1] {
        Instr::BinOp { op, .. } => assert_eq!(*op, BinOpKind::FMul),
        other => panic!("expected binop, got {other:?}"),
    }
    match &f.instrs[2] {
        Instr::BinOp {
            rhs: Operand::FloatHex(bits),
            ..
        } => {
            assert_eq!(f64::from_bits(*bits), 1.0);
        }
        other => panic!("expected fadd with float constant, got {other:?}"),
    }
    match &f.instrs[4] {
        Instr::RetVoid => {}
        other => panic!("expected ret void, got {other:?}"),
    }
}

const REAL_V: &str = r#"; ModuleID = "V"
target triple = "nvptx64-nvidia-cuda"

define void @"V"(i32* %"sidx", i32* %"sup", float* %"q", float* %"r", float* %"out0")
{
entry:
  %"v0" = load float, float* %"r"
  %"v1" = call float @"__nv_fmaxf"(float %"v0", float 0x3eb0c6f7a0000000)
  %"v2" = fdiv float 0x3ff0000000000000, %"v1"
  %"v3" = load float, float* %"q"
  %"v4" = fmul float %"v3", %"v2"
  %"v5" = fneg float %"v0"
  %"v6" = call float @"__nv_expf"(float %"v5")
  %"v7" = fmul float %"v4", %"v6"
  store float %"v7", float* %"out0"
  ret void
}

declare float @"__nv_fmaxf"(float %".1", float %".2")
declare float @"__nv_expf"(float %".1")
"#;

#[test]
fn parses_real_v() {
    let m = parse_module(REAL_V).unwrap();
    let f = &m.function;
    assert_eq!(f.name, "V");
    assert_eq!(f.params.len(), 5);
    let calls: Vec<&Instr> = f
        .instrs
        .iter()
        .filter(|i| matches!(i, Instr::Call { .. }))
        .collect();
    assert_eq!(calls.len(), 2);
    match calls[0] {
        Instr::Call { callee, args, .. } => {
            assert_eq!(callee, "__nv_fmaxf");
            assert_eq!(args.len(), 2);
        }
        _ => unreachable!(),
    }
    match calls[1] {
        Instr::Call { callee, args, .. } => {
            assert_eq!(callee, "__nv_expf");
            assert_eq!(args.len(), 1);
        }
        _ => unreachable!(),
    }
    assert!(matches!(&f.instrs[5], Instr::FNeg { .. }));
    match &f.instrs[2] {
        Instr::BinOp { op, .. } => assert_eq!(*op, BinOpKind::FDiv),
        other => panic!("expected fdiv, got {other:?}"),
    }
}
