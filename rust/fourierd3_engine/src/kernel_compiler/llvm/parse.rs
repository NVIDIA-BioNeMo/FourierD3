// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The textual LLVM IR jax lowers a device function to, as a typed module.
//!
//! Only the straight-line subset the lowering emits is accepted: one function,
//! one basic block, no phi nodes. Anything else is an error, not a silent skip.

use fourierd3_engine::dtype::Dtype;

mod instruction;
mod lexer;
mod module;
#[cfg(test)]
mod tests;

pub(crate) use module::parse_module;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum LlvmType {
    I1,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

struct TypeInfo {
    variant: LlvmType,
    ir_text: &'static str,
    cuda_ctype: &'static str,
    is_float: bool,
}

const TYPES: &[TypeInfo] = &[
    TypeInfo {
        variant: LlvmType::I1,
        ir_text: "i1",
        cuda_ctype: "bool",
        is_float: false,
    },
    TypeInfo {
        variant: LlvmType::I8,
        ir_text: "i8",
        cuda_ctype: "signed char",
        is_float: false,
    },
    TypeInfo {
        variant: LlvmType::I16,
        ir_text: "i16",
        cuda_ctype: "short",
        is_float: false,
    },
    TypeInfo {
        variant: LlvmType::I32,
        ir_text: "i32",
        cuda_ctype: "int",
        is_float: false,
    },
    TypeInfo {
        variant: LlvmType::I64,
        ir_text: "i64",
        cuda_ctype: "long long",
        is_float: false,
    },
    TypeInfo {
        variant: LlvmType::F16,
        ir_text: "half",
        cuda_ctype: "__half",
        is_float: true,
    },
    TypeInfo {
        variant: LlvmType::BF16,
        ir_text: "bfloat",
        cuda_ctype: "__nv_bfloat16",
        is_float: true,
    },
    TypeInfo {
        variant: LlvmType::F32,
        ir_text: "float",
        cuda_ctype: "float",
        is_float: true,
    },
    TypeInfo {
        variant: LlvmType::F64,
        ir_text: "double",
        cuda_ctype: "double",
        is_float: true,
    },
];

impl LlvmType {
    fn info(&self) -> &'static TypeInfo {
        TYPES
            .iter()
            .find(|t| t.variant == *self)
            .expect("every LlvmType variant must have a TYPES entry")
    }

    pub(crate) fn cuda_ctype(&self) -> &'static str {
        self.info().cuda_ctype
    }

    pub(crate) fn dtype(&self) -> Dtype {
        match self {
            LlvmType::I1 => Dtype::Bool,
            LlvmType::I8 => Dtype::I8,
            LlvmType::I16 => Dtype::I16,
            LlvmType::I32 => Dtype::I32,
            LlvmType::I64 => Dtype::I64,
            LlvmType::F16 => Dtype::F16,
            LlvmType::BF16 => Dtype::Bf16,
            LlvmType::F32 => Dtype::F32,
            LlvmType::F64 => Dtype::F64,
        }
    }

    pub(crate) fn is_float(&self) -> bool {
        self.info().is_float
    }

    fn parse(s: &str) -> Result<Self, String> {
        TYPES
            .iter()
            .find(|t| t.ir_text == s)
            .map(|t| t.variant.clone())
            .ok_or_else(|| format!("unsupported llvm type {s:?}"))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LlvmParam {
    pub name: String,
    pub elem_ty: LlvmType,
}

#[derive(Debug, Clone)]
pub(crate) enum Operand {
    Ssa(String),
    FloatHex(u64),
    IntLit(i64),
}

#[derive(Debug, Clone)]
pub(crate) enum Instr {
    Load {
        dst: String,
        ty: LlvmType,
        src: String,
    },
    BinOp {
        dst: String,
        ty: LlvmType,
        op: BinOpKind,
        lhs: Operand,
        rhs: Operand,
    },
    FNeg {
        dst: String,
        ty: LlvmType,
        operand: Operand,
    },
    Call {
        dst: String,
        ret_ty: LlvmType,
        callee: String,
        args: Vec<(LlvmType, Operand)>,
    },
    Store {
        ty: LlvmType,
        val: Operand,
        dst: String,
    },
    Cmp {
        dst: String,
        pred: CmpPred,
        operand_ty: LlvmType,
        lhs: Operand,
        rhs: Operand,
    },
    Select {
        dst: String,
        cond: Operand,
        ty: LlvmType,
        true_val: Operand,
        false_val: Operand,
    },
    Gep {
        dst: String,
        ty: LlvmType,
        base: String,
        base_is_global: bool,
        offset: Operand,
    },
    Cast {
        dst: String,
        op: &'static str,
        src_ty: LlvmType,
        dst_ty: LlvmType,
        operand: Operand,
    },
    RetVoid,
}

pub(super) const CAST_OPS: &[&str] = &[
    "sitofp", "uitofp", "fptosi", "fptoui", "sext", "zext", "trunc", "fpext", "fptrunc", "bitcast",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CmpPred {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

impl CmpPred {
    pub(crate) fn cuda_op(self) -> &'static str {
        match self {
            CmpPred::Eq => "==",
            CmpPred::Ne => "!=",
            CmpPred::Gt => ">",
            CmpPred::Ge => ">=",
            CmpPred::Lt => "<",
            CmpPred::Le => "<=",
        }
    }

    fn parse_fcmp(s: &str) -> Option<Self> {
        // Ordered (NaN → false) variants are what jaxpr emits.
        Some(match s {
            "oeq" | "ueq" => CmpPred::Eq,
            "one" | "une" => CmpPred::Ne,
            "ogt" | "ugt" => CmpPred::Gt,
            "oge" | "uge" => CmpPred::Ge,
            "olt" | "ult" => CmpPred::Lt,
            "ole" | "ule" => CmpPred::Le,
            _ => return None,
        })
    }

    fn parse_icmp(s: &str) -> Option<Self> {
        Some(match s {
            "eq" => CmpPred::Eq,
            "ne" => CmpPred::Ne,
            "sgt" | "ugt" => CmpPred::Gt,
            "sge" | "uge" => CmpPred::Ge,
            "slt" | "ult" => CmpPred::Lt,
            "sle" | "ule" => CmpPred::Le,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinOpKind {
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    Add,
    Sub,
    Mul,
    SRem,
    URem,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

impl BinOpKind {
    pub(crate) fn cuda_op(self) -> Option<&'static str> {
        Some(match self {
            BinOpKind::FAdd | BinOpKind::Add => "+",
            BinOpKind::FSub | BinOpKind::Sub => "-",
            BinOpKind::FMul | BinOpKind::Mul => "*",
            BinOpKind::FDiv => "/",
            BinOpKind::SRem => "%",
            // Bitwise on the LLVM side; for `i1` operands bitwise is
            // equivalent to logical, and CUDA accepts both.
            BinOpKind::And => "&",
            BinOpKind::Or => "|",
            BinOpKind::Xor => "^",
            BinOpKind::Shl => "<<",
            BinOpKind::AShr => ">>",
            BinOpKind::FRem | BinOpKind::URem | BinOpKind::LShr => return None,
        })
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "fadd" => BinOpKind::FAdd,
            "fsub" => BinOpKind::FSub,
            "fmul" => BinOpKind::FMul,
            "fdiv" => BinOpKind::FDiv,
            "frem" => BinOpKind::FRem,
            "add" => BinOpKind::Add,
            "sub" => BinOpKind::Sub,
            "mul" => BinOpKind::Mul,
            "srem" => BinOpKind::SRem,
            "urem" => BinOpKind::URem,
            "and" => BinOpKind::And,
            "or" => BinOpKind::Or,
            "xor" => BinOpKind::Xor,
            "shl" => BinOpKind::Shl,
            "lshr" => BinOpKind::LShr,
            "ashr" => BinOpKind::AShr,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LlvmFunction {
    pub name: String,
    pub params: Vec<LlvmParam>,
    pub instrs: Vec<Instr>,
}

#[derive(Debug, Clone)]
pub(crate) struct LlvmGlobal {
    pub name: String,
    pub addrspace: u32,
    pub elem_ty: LlvmType,
    pub values: Vec<Operand>,
}

#[derive(Debug, Clone)]
pub(crate) struct LlvmModule {
    pub globals: Vec<LlvmGlobal>,
    pub function: LlvmFunction,
}
