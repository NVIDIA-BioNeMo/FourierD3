// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing one instruction: the opcode dispatch and the per-opcode
//! operand grammars.

use super::lexer::{parse_at_ident, parse_operand, parse_percent_ident, split_comma, split_token};
use super::{BinOpKind, CAST_OPS, CmpPred, Instr, LlvmType, Operand};

pub(super) fn parse_instr(line: &str) -> Result<Instr, String> {
    if line == "ret void" {
        return Ok(Instr::RetVoid);
    }
    if let Some(rest) = line.strip_prefix("store ") {
        return parse_store(rest);
    }
    let (dst, rest) = parse_percent_ident(line)?;
    let rest = rest.trim_start();
    let rest = rest
        .strip_prefix('=')
        .ok_or_else(|| format!("expected `=` after dst in: {line:?}"))?
        .trim_start();
    let (op, body) = split_token(rest);
    let body = body.trim_start();
    match op {
        "load" => parse_load(dst, body),
        "fneg" => parse_fneg(dst, body),
        "call" => parse_call(dst, body),
        "fcmp" => parse_cmp(dst, body, true),
        "icmp" => parse_cmp(dst, body, false),
        "select" => parse_select(dst, body),
        "getelementptr" => parse_gep(dst, body),
        _ if BinOpKind::parse(op).is_some() => {
            parse_binop(dst, BinOpKind::parse(op).unwrap(), body)
        }
        _ if CAST_OPS.contains(&op) => parse_cast(dst, op, body),
        _ => Err(format!("unsupported opcode {op:?} in: {line:?}")),
    }
}

fn parse_cast(dst: String, op: &str, body: &str) -> Result<Instr, String> {
    let op = CAST_OPS
        .iter()
        .copied()
        .find(|name| *name == op)
        .expect("caller already checked CAST_OPS");
    let (src_ty_str, rest) = split_token(body);
    let src_ty = LlvmType::parse(src_ty_str)?;
    let rest = rest.trim_start();
    let (operand_str, rest) = split_token(rest);
    let operand = parse_operand(operand_str.trim_end_matches(','), &src_ty)?;
    let rest = rest.trim_start();
    let rest = rest
        .strip_prefix("to ")
        .ok_or_else(|| format!("expected `to <type>` in cast, got {rest:?}"))?;
    let dst_ty = LlvmType::parse(rest.trim())?;
    Ok(Instr::Cast {
        dst,
        op,
        src_ty,
        dst_ty,
        operand,
    })
}

fn parse_gep(dst: String, body: &str) -> Result<Instr, String> {
    let body = body.strip_prefix("inbounds ").unwrap_or(body);
    let body = body.trim_start();
    if body.starts_with('[') {
        return parse_gep_array(dst, body);
    }
    let (ty_str, rest) = split_token(body);
    let ty_str = ty_str.trim_end_matches(',');
    let ty = LlvmType::parse(ty_str)?;
    let rest = rest.trim_start();
    let (ty2_str, rest) = split_token(rest);
    let _ = ty2_str
        .strip_suffix('*')
        .ok_or_else(|| format!("expected pointer type in gep, got {ty2_str:?}"))?;
    let rest = rest.trim_start_matches([' ', ',']);
    let (base, rest) = parse_percent_ident(rest)?;
    let rest = rest.trim_start_matches([' ', ',']);
    let (idx_ty_str, rest) = split_token(rest);
    let idx_ty = LlvmType::parse(idx_ty_str)?;
    if !matches!(idx_ty, LlvmType::I32 | LlvmType::I64) {
        return Err(format!("gep index type {idx_ty_str:?} (only i32/i64)"));
    }
    let offset = parse_operand(rest.trim(), &idx_ty)?;
    Ok(Instr::Gep {
        dst,
        ty,
        base,
        base_is_global: false,
        offset,
    })
}

fn parse_gep_array(dst: String, body: &str) -> Result<Instr, String> {
    let end = body
        .find(']')
        .ok_or_else(|| format!("unterminated array type in gep: {body:?}"))?;
    let inside = &body[1..end];
    let (_count, ty_str) = inside
        .split_once(" x ")
        .ok_or_else(|| format!("malformed array type {inside:?}"))?;
    let ty = LlvmType::parse(ty_str.trim())?;
    let rest = body[end + 1..].trim_start_matches([' ', ',']);
    let end2 = rest
        .find(']')
        .ok_or_else(|| format!("missing second array type in gep: {rest:?}"))?;
    let rest = rest[end2 + 1..].trim_start();
    let rest = if let Some(after) = rest.strip_prefix("addrspace(") {
        let close = after
            .find(')')
            .ok_or_else(|| format!("malformed addrspace in gep: {after:?}"))?;
        after[close + 1..].trim_start()
    } else {
        rest
    };
    let rest = rest
        .strip_prefix('*')
        .ok_or_else(|| format!("expected `*` after array type in gep: {rest:?}"))?;
    let rest = rest.trim_start();
    let (base, rest) = parse_at_ident(rest)?;
    let rest = rest.trim_start_matches([' ', ',']);
    let (idx1_ty_str, rest) = split_token(rest);
    let _ = LlvmType::parse(idx1_ty_str)?;
    let rest = rest.trim_start();
    let comma = rest
        .find(',')
        .ok_or_else(|| format!("expected two indices in array gep: {rest:?}"))?;
    let first_idx: i64 = rest[..comma]
        .trim()
        .parse()
        .map_err(|e| format!("bad first gep index: {e}"))?;
    if first_idx != 0 {
        return Err(format!("first array-gep index must be 0, got {first_idx}"));
    }
    let rest = rest[comma + 1..].trim_start();
    let (idx2_ty_str, rest) = split_token(rest);
    let idx2_ty = LlvmType::parse(idx2_ty_str)?;
    let offset = parse_operand(rest.trim(), &idx2_ty)?;
    Ok(Instr::Gep {
        dst,
        ty,
        base,
        base_is_global: true,
        offset,
    })
}

fn parse_cmp(dst: String, body: &str, is_float: bool) -> Result<Instr, String> {
    let (pred_str, rest) = split_token(body);
    let pred = if is_float {
        CmpPred::parse_fcmp(pred_str)
    } else {
        CmpPred::parse_icmp(pred_str)
    }
    .ok_or_else(|| format!("unsupported cmp predicate {pred_str:?}"))?;
    let rest = rest.trim_start();
    let (ty_str, rest) = split_token(rest);
    let operand_ty = LlvmType::parse(ty_str)?;
    let rest = rest.trim_start();
    let (lhs_str, rhs_str) = split_comma(rest)?;
    let lhs = parse_operand(lhs_str.trim(), &operand_ty)?;
    let rhs = parse_operand(rhs_str.trim(), &operand_ty)?;
    Ok(Instr::Cmp {
        dst,
        pred,
        operand_ty,
        lhs,
        rhs,
    })
}

fn parse_select(dst: String, body: &str) -> Result<Instr, String> {
    let (cond_ty_str, rest) = split_token(body);
    if LlvmType::parse(cond_ty_str)? != LlvmType::I1 {
        return Err(format!(
            "select condition type {cond_ty_str:?} (only i1 supported)"
        ));
    }
    let rest = rest.trim_start();
    let cond_end = rest
        .find(',')
        .ok_or_else(|| format!("select missing first comma: {rest:?}"))?;
    let cond = parse_operand(rest[..cond_end].trim(), &LlvmType::I1)?;
    let rest = rest[cond_end + 1..].trim_start();
    let (ty_str, rest) = split_token(rest);
    let ty = LlvmType::parse(ty_str)?;
    let rest = rest.trim_start();
    let (t_str, rest) = split_comma(rest)?;
    let true_val = parse_operand(t_str.trim(), &ty)?;
    let rest = rest.trim_start();
    let (ty2_str, rest) = split_token(rest);
    if LlvmType::parse(ty2_str)? != ty {
        return Err(format!(
            "select branch type mismatch: {ty_str:?} vs {ty2_str:?}"
        ));
    }
    let false_val = parse_operand(rest.trim(), &ty)?;
    Ok(Instr::Select {
        dst,
        cond,
        ty,
        true_val,
        false_val,
    })
}

fn parse_load(dst: String, body: &str) -> Result<Instr, String> {
    let (ty1_str, rest) = split_token(body);
    let ty1_str = ty1_str.trim_end_matches(',');
    let ty = LlvmType::parse(ty1_str)?;
    let rest = rest.trim_start();
    let rest = skip_pointer_type(rest)?;
    let (src, _) = parse_percent_ident(rest.trim_start_matches([' ', ',']))?;
    Ok(Instr::Load { dst, ty, src })
}

fn skip_pointer_type(s: &str) -> Result<&str, String> {
    let (tok, rest) = split_token(s);
    if let Some(_stripped) = tok.strip_suffix('*') {
        return Ok(rest.trim_start());
    }
    let _ = LlvmType::parse(tok)?;
    let rest = rest.trim_start();
    let rest = if let Some(after) = rest.strip_prefix("addrspace(") {
        let close = after
            .find(')')
            .ok_or_else(|| format!("malformed addrspace in: {after:?}"))?;
        after[close + 1..].trim_start()
    } else {
        rest
    };
    rest.strip_prefix('*')
        .map(str::trim_start)
        .ok_or_else(|| format!("expected `*` after pointee type, got: {rest:?}"))
}

fn parse_binop(dst: String, op: BinOpKind, body: &str) -> Result<Instr, String> {
    let (ty_str, rest) = split_token(body);
    let ty = LlvmType::parse(ty_str)?;
    let rest = rest.trim_start();
    let (lhs_str, rhs_str) = split_comma(rest)?;
    let lhs = parse_operand(lhs_str.trim(), &ty)?;
    let rhs = parse_operand(rhs_str.trim(), &ty)?;
    Ok(Instr::BinOp {
        dst,
        ty,
        op,
        lhs,
        rhs,
    })
}

fn parse_fneg(dst: String, body: &str) -> Result<Instr, String> {
    let (ty_str, rest) = split_token(body);
    let ty = LlvmType::parse(ty_str)?;
    let operand = parse_operand(rest.trim(), &ty)?;
    Ok(Instr::FNeg { dst, ty, operand })
}

fn parse_call(dst: String, body: &str) -> Result<Instr, String> {
    let (ret_ty_str, rest) = split_token(body);
    let ret_ty = LlvmType::parse(ret_ty_str)?;
    let rest = rest.trim_start();
    let (callee, rest) = parse_at_ident(rest)?;
    let rest = rest.trim_start();
    let inner = rest
        .strip_prefix('(')
        .and_then(|r| r.rfind(')').map(|i| &r[..i]))
        .ok_or_else(|| format!("missing parenthesised call arg list: {rest:?}"))?;

    let mut args = Vec::new();
    if !inner.trim().is_empty() {
        for chunk in inner.split(',') {
            args.push(parse_call_arg(chunk.trim())?);
        }
    }
    Ok(Instr::Call {
        dst,
        ret_ty,
        callee,
        args,
    })
}

fn parse_call_arg(s: &str) -> Result<(LlvmType, Operand), String> {
    let (ty_str, rest) = split_token(s);
    let ty = LlvmType::parse(ty_str)?;
    let operand = parse_operand(rest.trim(), &ty)?;
    Ok((ty, operand))
}

fn parse_store(body: &str) -> Result<Instr, String> {
    let (ty_str, rest) = split_token(body);
    let ty = LlvmType::parse(ty_str)?;
    let rest = rest.trim_start();
    let (val_str, rhs) = split_comma(rest)?;
    let val = parse_operand(val_str.trim(), &ty)?;
    let rhs = rhs.trim_start();
    let (ty2_str, rest) = split_token(rhs);
    let _ = ty2_str
        .strip_suffix('*')
        .ok_or_else(|| format!("expected pointer type in store, got {ty2_str:?}"))?;
    let (dst, _) = parse_percent_ident(rest.trim_start())?;
    Ok(Instr::Store { ty, val, dst })
}
