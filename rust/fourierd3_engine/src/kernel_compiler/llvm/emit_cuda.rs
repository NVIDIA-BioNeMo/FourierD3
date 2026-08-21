// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::parse::{BinOpKind, Instr, LlvmFunction, LlvmGlobal, LlvmModule, LlvmType, Operand};
use std::collections::HashMap;

pub(crate) fn emit_cuda(module: &LlvmModule, n_in: usize) -> String {
    let mut out = String::new();
    for g in &module.globals {
        emit_constant_global(&mut out, g);
    }
    if !module.globals.is_empty() {
        out.push('\n');
    }
    emit_function(&mut out, &module.function, n_in);
    out
}

fn emit_constant_global(out: &mut String, g: &LlvmGlobal) {
    let qualifier = match g.addrspace {
        4 => String::from("__constant__"),
        _ => String::from("static __device__ const"),
    };
    let ctype = g.elem_ty.cuda_ctype();
    let len = g.values.len();
    let elems: Vec<String> = g
        .values
        .iter()
        .map(|op| fmt_operand(op, &g.elem_ty))
        .collect();
    out.push_str(&format!(
        "{qualifier} {ctype} {}[{len}] = {{{}}};\n",
        g.name,
        elems.join(", "),
    ));
}

fn emit_function(out: &mut String, func: &LlvmFunction, n_in: usize) {
    let mut params = Vec::with_capacity(func.params.len());
    for (i, p) in func.params.iter().enumerate() {
        let qual = if i < n_in { "const " } else { "" };
        params.push(format!("{qual}{}* {}", p.elem_ty.cuda_ctype(), p.name));
    }
    let sig = format!(
        "{} {}({})",
        String::from("__device__ __forceinline__ void"),
        func.name,
        params.join(", ")
    );

    // GEP-then-load/store collapse: `%p = gep %base, i32 N` followed by
    // `load %p` becomes `T v = base[N];`. We don't emit a CUDA line for
    // the GEP itself. Offset may be either a literal (static index) or
    // a runtime SSA value (dynamic_slice on a `__constant__` table).
    let mut gep_table: HashMap<&str, (&str, &Operand)> = HashMap::new();
    let mut lines: Vec<String> = Vec::new();
    for instr in &func.instrs {
        if let Some(line) = emit_instruction(instr, &mut gep_table) {
            lines.push(line);
        }
    }

    out.push_str(&sig);
    out.push_str(" {\n");
    for l in &lines {
        out.push_str("    ");
        out.push_str(l);
        out.push('\n');
    }
    out.push_str("}\n");
}

fn emit_instruction<'a>(
    instr: &'a Instr,
    gep_table: &mut HashMap<&'a str, (&'a str, &'a Operand)>,
) -> Option<String> {
    match instr {
        Instr::Gep {
            dst, base, offset, ..
        } => {
            gep_table.insert(dst, (base, offset));
            None
        }
        Instr::Load { dst, ty, src } => {
            let (base, offset) = gep_address(src, gep_table);
            Some(format!(
                "{} {} = {}[{}];",
                ty.cuda_ctype(),
                dst,
                base,
                offset
            ))
        }
        Instr::BinOp {
            dst,
            ty,
            op,
            lhs,
            rhs,
        } => Some(emit_binary(dst, ty, op, lhs, rhs)),
        Instr::FNeg { dst, ty, operand } => Some(format!(
            "{} {} = -{};",
            ty.cuda_ctype(),
            dst,
            fmt_operand(operand, ty),
        )),
        Instr::Call {
            dst,
            ret_ty,
            callee,
            args,
        } => Some(emit_call(dst, ret_ty, callee, args)),
        Instr::Cmp {
            dst,
            pred,
            operand_ty,
            lhs,
            rhs,
            ..
        } => Some(format!(
            "bool {} = {} {} {};",
            dst,
            fmt_operand(lhs, operand_ty),
            pred.cuda_op(),
            fmt_operand(rhs, operand_ty),
        )),
        Instr::Select {
            dst,
            cond,
            ty,
            true_val,
            false_val,
        } => Some(format!(
            "{} {} = {} ? {} : {};",
            ty.cuda_ctype(),
            dst,
            fmt_operand(cond, &LlvmType::I1),
            fmt_operand(true_val, ty),
            fmt_operand(false_val, ty),
        )),
        Instr::Cast {
            dst,
            op,
            src_ty,
            dst_ty,
            operand,
        } => Some(format!(
            "{} {} = {};",
            dst_ty.cuda_ctype(),
            dst,
            emit_cast(op, src_ty, dst_ty, operand)
        )),
        Instr::Store { ty, val, dst } => {
            let (base, offset) = gep_address(dst, gep_table);
            Some(format!("{}[{}] = {};", base, offset, fmt_operand(val, ty)))
        }
        Instr::RetVoid => None,
    }
}

fn gep_address<'a>(
    source: &'a str,
    gep_table: &HashMap<&'a str, (&'a str, &'a Operand)>,
) -> (&'a str, String) {
    match gep_table.get(source).copied() {
        Some((base, offset)) => (base, fmt_operand(offset, &LlvmType::I32)),
        None => (source, "0".to_string()),
    }
}

fn emit_binary(dst: &str, ty: &LlvmType, op: &BinOpKind, lhs: &Operand, rhs: &Operand) -> String {
    let lhs = fmt_operand(lhs, ty);
    let rhs = fmt_operand(rhs, ty);
    let expr = binary_expression(op, ty, &lhs, &rhs);
    format!("{} {} = {};", ty.cuda_ctype(), dst, expr)
}

fn binary_expression(op: &BinOpKind, ty: &LlvmType, lhs: &str, rhs: &str) -> String {
    if let Some(infix) = op.cuda_op() {
        return format!("{lhs} {infix} {rhs}");
    }
    match op {
        BinOpKind::FRem => format!("{}({lhs}, {rhs})", float_remainder_name(ty)),
        BinOpKind::URem | BinOpKind::LShr => {
            let unsigned = unsigned_type(ty);
            let infix = if matches!(op, BinOpKind::URem) {
                "%"
            } else {
                ">>"
            };
            format!("(({unsigned}){lhs} {infix} ({unsigned}){rhs})")
        }
        other => unreachable!("cuda_op returned None for {other:?}"),
    }
}

fn float_remainder_name(ty: &LlvmType) -> &'static str {
    match ty {
        LlvmType::F32 => "fmodf",
        LlvmType::F64 => "fmod",
        other => panic!("frem requires a float operand type, got {other:?}"),
    }
}

fn unsigned_type(ty: &LlvmType) -> &'static str {
    match ty {
        LlvmType::I32 => "unsigned int",
        LlvmType::I64 => "unsigned long long",
        other => panic!("urem/lshr requires an integer operand type, got {other:?}"),
    }
}

fn emit_call(dst: &str, ret_ty: &LlvmType, callee: &str, args: &[(LlvmType, Operand)]) -> String {
    let callee = map_libdevice_name(callee).unwrap_or_else(|| callee.to_string());
    let args = args
        .iter()
        .map(|(ty, operand)| fmt_operand(operand, ty))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} {} = {}({});", ret_ty.cuda_ctype(), dst, callee, args)
}

fn emit_cast(op: &str, src_ty: &LlvmType, dst_ty: &LlvmType, operand: &Operand) -> String {
    let operand = fmt_operand(operand, src_ty);
    if op != "bitcast" {
        return format!("({}){}", dst_ty.cuda_ctype(), operand);
    }
    match (src_ty, dst_ty) {
        (LlvmType::F32, LlvmType::I32) => format!("__float_as_int({operand})"),
        (LlvmType::I32, LlvmType::F32) => format!("__int_as_float({operand})"),
        (LlvmType::F64, LlvmType::I64) => format!("__double_as_longlong({operand})"),
        (LlvmType::I64, LlvmType::F64) => format!("__longlong_as_double({operand})"),
        (a, b) => panic!("bitcast between {a:?} and {b:?} not supported"),
    }
}

fn fmt_operand(op: &Operand, ty: &LlvmType) -> String {
    match op {
        Operand::Ssa(name) => name.clone(),
        Operand::IntLit(v) => v.to_string(),
        Operand::FloatHex(bits) => fmt_float_literal(f64::from_bits(*bits), ty),
    }
}

fn fmt_float_literal(v: f64, ty: &LlvmType) -> String {
    if let Some(s) = nonfinite_literal(v, ty) {
        return s;
    }
    match ty {
        LlvmType::F32 => {
            let s = format!("{}", v as f32);
            if s.contains('.') || s.contains('e') || s.contains('E') {
                format!("{s}f")
            } else {
                format!("{s}.0f")
            }
        }
        LlvmType::F64 => {
            let s = format!("{v}");
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{s}.0")
            }
        }
        LlvmType::F16 | LlvmType::BF16 => {
            // `__half` / `__nv_bfloat16` have no literal suffix; the
            // `<cuda_fp16.h>` / `<cuda_bf16.h>` headers provide
            // `__float2half` / `__float2bfloat16_rn` constructors.
            let f = v as f32;
            let s = if f.fract() == 0.0 {
                format!("{f}.0f")
            } else {
                format!("{f}f")
            };
            narrow_float_call(ty, &s)
        }
        _ => unreachable!("float operand on non-float type"),
    }
}

fn narrow_float_call(ty: &LlvmType, f32_expr: &str) -> String {
    let ctor = match ty {
        LlvmType::F16 => "__float2half",
        LlvmType::BF16 => "__float2bfloat16_rn",
        other => panic!("narrow_float_call on non-narrow-float {other:?}"),
    };
    format!("{ctor}({f32_expr})")
}

fn nonfinite_literal(v: f64, ty: &LlvmType) -> Option<String> {
    if v.is_finite() {
        return None;
    }
    Some(match ty {
        LlvmType::F16 | LlvmType::BF16 => {
            let f32_lit = nonfinite_literal(v, &LlvmType::F32)?;
            narrow_float_call(ty, &f32_lit)
        }
        LlvmType::F32 => {
            let bits = (v as f32).to_bits();
            format!("__int_as_float(0x{bits:08x})")
        }
        LlvmType::F64 => {
            let bits = v.to_bits();
            format!("__longlong_as_double(0x{bits:016x}ULL)")
        }
        _ => unreachable!("float operand on non-float type"),
    })
}

fn map_libdevice_name(callee: &str) -> Option<String> {
    callee.strip_prefix("__nv_").map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse_module;
    use super::*;

    const TRIVIAL_IR: &str = r#"; ModuleID = "V"
target triple = "nvptx64-nvidia-cuda"

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
    fn lowers_trivial_v_to_cuda() {
        let m = parse_module(TRIVIAL_IR).unwrap();
        let cuda = emit_cuda(&m, 3);
        let expected = "\
__device__ __forceinline__ void V(const int* sidx, const int* sup, const float* x, float* out0) {
    float v0 = x[0];
    float v1 = v0 * v0;
    float v2 = v1 + 1.0f;
    out0[0] = v2;
}
";
        assert_eq!(cuda, expected);
    }

    const REAL_V_IR: &str = r#"; ModuleID = "V"
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
    fn lowers_real_v_to_cuda() {
        let m = parse_module(REAL_V_IR).unwrap();
        let cuda = emit_cuda(&m, 4);
        assert!(cuda.contains("fmaxf("));
        assert!(cuda.contains("expf("));
        assert!(cuda.contains("v5 = -v0"));
        assert!(cuda.contains("v2 = 1.0f / v1"));
        assert!(cuda.contains("__device__ __forceinline__ void V("));
        assert!(cuda.contains("const float* q"));
        assert!(cuda.contains("float* out0"));
    }
}
