// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::parse::{Instr, LlvmFunction, LlvmParam, LlvmType, Operand};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct LicmSplit {
    pub pre: LlvmFunction,
    pub step: LlvmFunction,
    pub n_state: usize,
    pub pre_indices: Vec<usize>,
    pub direct_indices: Vec<usize>,
}

pub(crate) fn split_loop_invariant(
    func: &LlvmFunction,
    loop_varying_param_indices: &[usize],
    n_outputs: usize,
) -> Result<LicmSplit, String> {
    let n_params = func.params.len();
    let n_inputs = validate_parameter_ranges(n_params, n_outputs, loop_varying_param_indices)?;
    let varying_param_set: HashSet<usize> = loop_varying_param_indices.iter().copied().collect();
    let varying_param_names: HashSet<String> = loop_varying_param_indices
        .iter()
        .map(|&i| func.params[i].name.clone())
        .collect();

    let gep_base = collect_gep_bases(func);
    let mut partition = partition_instructions(func, &varying_param_names, &gep_base);
    duplicate_step_geps(&partition.pre, &mut partition.step, &partition.stores);
    let (state_names, state_types) = collect_state_values(
        &partition.pre,
        &partition.step,
        &partition.stores,
        &partition.varying_ssa,
        &varying_param_names,
    );
    let n_state = state_names.len();
    let (pre_uses, step_uses) = collect_parameter_uses(
        &partition.pre,
        &partition.step,
        &partition.stores,
        &gep_base,
        &varying_param_names,
    );
    let (pre_indices, direct_indices) =
        classify_parameter_indices(func, n_inputs, &varying_param_set, &pre_uses, &step_uses);
    let pre = build_pre_function(
        func,
        &partition.pre,
        &pre_indices,
        &state_names,
        &state_types,
    );
    let step = build_step_function(
        func,
        n_inputs,
        loop_varying_param_indices,
        &direct_indices,
        &state_names,
        &state_types,
        partition.step,
        partition.stores,
    );

    Ok(LicmSplit {
        pre,
        step,
        n_state,
        pre_indices,
        direct_indices,
    })
}

fn validate_parameter_ranges(
    n_params: usize,
    n_outputs: usize,
    varying: &[usize],
) -> Result<usize, String> {
    if n_outputs > n_params {
        return Err(format!(
            "n_outputs={n_outputs} exceeds param count {n_params}"
        ));
    }
    let n_inputs = n_params - n_outputs;
    if let Some(index) = varying.iter().find(|&&index| index >= n_inputs) {
        return Err(format!(
            "loop_varying index {index} is in output range (n_inputs={n_inputs})"
        ));
    }
    Ok(n_inputs)
}

fn collect_gep_bases(func: &LlvmFunction) -> HashMap<String, String> {
    func.instrs
        .iter()
        .filter_map(|instr| match instr {
            Instr::Gep {
                dst,
                base,
                base_is_global,
                ..
            } if !base_is_global => Some((dst.clone(), base.clone())),
            _ => None,
        })
        .collect()
}

fn resolve_load_source(src: &str, gep_bases: &HashMap<String, String>) -> String {
    gep_bases
        .get(src)
        .cloned()
        .unwrap_or_else(|| src.to_string())
}

struct InstructionPartition {
    pre: Vec<Instr>,
    step: Vec<Instr>,
    stores: Vec<Instr>,
    varying_ssa: HashSet<String>,
}

fn partition_instructions(
    func: &LlvmFunction,
    varying_params: &HashSet<String>,
    gep_bases: &HashMap<String, String>,
) -> InstructionPartition {
    let mut result = InstructionPartition {
        pre: Vec::new(),
        step: Vec::new(),
        stores: Vec::new(),
        varying_ssa: HashSet::new(),
    };
    for instr in &func.instrs {
        if matches!(instr, Instr::Store { .. }) {
            result.stores.push(instr.clone());
        } else if !matches!(instr, Instr::RetVoid) {
            classify_instruction(instr, varying_params, gep_bases, &mut result);
        }
    }
    result
}

fn classify_instruction(
    instr: &Instr,
    varying_params: &HashSet<String>,
    gep_bases: &HashMap<String, String>,
    result: &mut InstructionPartition,
) {
    let dst = instr_dst(instr).expect("instruction should have a destination");
    let depends_on_varying = instr_operands(instr).any(|op| {
        matches!(op, Operand::Ssa(name) if result.varying_ssa.contains(name) || varying_params.contains(name))
    }) || instr_loads_param(instr, varying_params, gep_bases)
        || instr_geps_varying_base(instr, varying_params, &result.varying_ssa);
    let cheap_output = matches!(
        instr_dst_type(instr),
        Some(LlvmType::I1 | LlvmType::I32 | LlvmType::I64)
    );
    if depends_on_varying || cheap_output {
        result.varying_ssa.insert(dst.to_string());
        result.step.push(instr.clone());
    } else {
        result.pre.push(instr.clone());
    }
}

fn duplicate_step_geps(pre: &[Instr], step: &mut Vec<Instr>, stores: &[Instr]) {
    let indices: HashMap<String, usize> = pre
        .iter()
        .enumerate()
        .filter_map(|(index, instr)| match instr {
            Instr::Gep { dst, .. } => Some((dst.clone(), index)),
            _ => None,
        })
        .collect();
    let mut referenced = HashSet::new();
    for instr in step.iter().chain(stores) {
        referenced.extend(referenced_geps(instr, &indices));
    }
    let copies = referenced
        .iter()
        .map(|name| pre[indices[name]].clone())
        .collect::<Vec<_>>();
    *step = copies.into_iter().chain(std::mem::take(step)).collect();
}

fn referenced_geps(instr: &Instr, indices: &HashMap<String, usize>) -> Vec<String> {
    if let Instr::Store { dst, .. } = instr {
        return indices
            .contains_key(dst)
            .then(|| dst.clone())
            .into_iter()
            .collect();
    }
    instr_operands(instr)
        .filter_map(|op| match op {
            Operand::Ssa(name) if indices.contains_key(name) => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn collect_state_values(
    pre: &[Instr],
    step: &[Instr],
    stores: &[Instr],
    varying_ssa: &HashSet<String>,
    varying_params: &HashSet<String>,
) -> (Vec<String>, Vec<LlvmType>) {
    let pre_types: HashMap<String, LlvmType> = pre
        .iter()
        .filter_map(|instr| Some((instr_dst(instr)?.to_string(), instr_dst_type(instr)?)))
        .collect();
    let mut names = Vec::new();
    let mut seen = HashSet::new();
    for operand in step.iter().chain(stores).flat_map(instr_operands) {
        if let Operand::Ssa(name) = operand
            && !varying_ssa.contains(name)
            && !varying_params.contains(name)
            && pre_types.contains_key(name)
            && seen.insert(name.clone())
        {
            names.push(name.clone());
        }
    }
    let types = names.iter().map(|name| pre_types[name].clone()).collect();
    (names, types)
}

fn collect_parameter_uses(
    pre: &[Instr],
    step: &[Instr],
    stores: &[Instr],
    gep_bases: &HashMap<String, String>,
    varying_params: &HashSet<String>,
) -> (HashSet<String>, HashSet<String>) {
    let pre_uses = load_sources(pre.iter(), gep_bases).collect();
    let step_uses = load_sources(step.iter().chain(stores), gep_bases)
        .filter(|source| !varying_params.contains(source))
        .collect();
    (pre_uses, step_uses)
}

fn load_sources<'a>(
    instrs: impl Iterator<Item = &'a Instr> + 'a,
    gep_bases: &'a HashMap<String, String>,
) -> impl Iterator<Item = String> + 'a {
    instrs.filter_map(move |instr| match instr {
        Instr::Load { src, .. } => Some(resolve_load_source(src, gep_bases)),
        _ => None,
    })
}

fn classify_parameter_indices(
    func: &LlvmFunction,
    n_inputs: usize,
    varying: &HashSet<usize>,
    pre_uses: &HashSet<String>,
    step_uses: &HashSet<String>,
) -> (Vec<usize>, Vec<usize>) {
    let invariant = (0..n_inputs).filter(|index| !varying.contains(index));
    let pre = invariant
        .clone()
        .filter(|&index| pre_uses.contains(&func.params[index].name))
        .collect();
    let direct = invariant
        .filter(|&index| step_uses.contains(&func.params[index].name))
        .collect();
    (pre, direct)
}

fn state_params(names: &[String], types: &[LlvmType]) -> Vec<LlvmParam> {
    names
        .iter()
        .zip(types)
        .map(|(name, ty)| LlvmParam {
            name: format!("_state_{name}"),
            elem_ty: ty.clone(),
        })
        .collect()
}

fn build_pre_function(
    func: &LlvmFunction,
    instructions: &[Instr],
    pre_indices: &[usize],
    state_names: &[String],
    state_types: &[LlvmType],
) -> LlvmFunction {
    let states = state_params(state_names, state_types);
    let params = pre_indices
        .iter()
        .map(|&index| func.params[index].clone())
        .chain(states.iter().cloned())
        .collect();
    let mut instrs = instructions.to_vec();
    instrs.extend(
        state_names
            .iter()
            .zip(&states)
            .map(|(name, param)| Instr::Store {
                ty: param.elem_ty.clone(),
                val: Operand::Ssa(name.clone()),
                dst: param.name.clone(),
            }),
    );
    instrs.push(Instr::RetVoid);
    LlvmFunction {
        name: format!("{}_pre", func.name),
        params,
        instrs,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_step_function(
    func: &LlvmFunction,
    n_inputs: usize,
    varying_indices: &[usize],
    direct_indices: &[usize],
    state_names: &[String],
    state_types: &[LlvmType],
    step: Vec<Instr>,
    stores: Vec<Instr>,
) -> LlvmFunction {
    let states = state_params(state_names, state_types);
    let varying_params = varying_indices
        .iter()
        .map(|&index| func.params[index].clone());
    let direct_params = direct_indices
        .iter()
        .map(|&index| func.params[index].clone());
    let params = varying_params
        .chain(states.iter().cloned())
        .chain(direct_params)
        .chain(func.params[n_inputs..].iter().cloned())
        .collect();
    let mut instrs = state_names
        .iter()
        .zip(state_types)
        .map(|(name, ty)| Instr::Load {
            dst: name.clone(),
            ty: ty.clone(),
            src: format!("_state_{name}"),
        })
        .collect::<Vec<_>>();
    instrs.extend(step);
    instrs.extend(stores);
    instrs.push(Instr::RetVoid);
    LlvmFunction {
        name: format!("{}_step", func.name),
        params,
        instrs,
    }
}

fn instr_dst(instr: &Instr) -> Option<&str> {
    match instr {
        Instr::Load { dst, .. }
        | Instr::BinOp { dst, .. }
        | Instr::FNeg { dst, .. }
        | Instr::Call { dst, .. }
        | Instr::Cmp { dst, .. }
        | Instr::Select { dst, .. }
        | Instr::Gep { dst, .. }
        | Instr::Cast { dst, .. } => Some(dst),
        Instr::Store { .. } | Instr::RetVoid => None,
    }
}

fn instr_dst_type(instr: &Instr) -> Option<LlvmType> {
    match instr {
        Instr::Load { ty, .. } => Some(ty.clone()),
        Instr::BinOp { ty, .. } => Some(ty.clone()),
        Instr::FNeg { ty, .. } => Some(ty.clone()),
        Instr::Call { ret_ty, .. } => Some(ret_ty.clone()),
        Instr::Cmp { .. } => Some(LlvmType::I1),
        Instr::Select { ty, .. } => Some(ty.clone()),
        Instr::Gep { ty, .. } => Some(ty.clone()),
        Instr::Cast { dst_ty, .. } => Some(dst_ty.clone()),
        Instr::Store { .. } | Instr::RetVoid => None,
    }
}

fn instr_operands(instr: &Instr) -> Box<dyn Iterator<Item = &Operand> + '_> {
    match instr {
        Instr::Load { .. } => Box::new(std::iter::empty()),
        Instr::Gep { offset, .. } => Box::new(std::iter::once(offset)),
        Instr::BinOp { lhs, rhs, .. } => Box::new([lhs, rhs].into_iter()),
        Instr::FNeg { operand, .. } => Box::new(std::iter::once(operand)),
        Instr::Cast { operand, .. } => Box::new(std::iter::once(operand)),
        Instr::Call { args, .. } => Box::new(args.iter().map(|(_, op)| op)),
        Instr::Cmp { lhs, rhs, .. } => Box::new([lhs, rhs].into_iter()),
        Instr::Select {
            cond,
            true_val,
            false_val,
            ..
        } => Box::new([cond, true_val, false_val].into_iter()),
        Instr::Store { val, .. } => Box::new(std::iter::once(val)),
        Instr::RetVoid => Box::new(std::iter::empty()),
    }
}

fn instr_loads_param(
    instr: &Instr,
    varying_param_names: &HashSet<String>,
    gep_bases: &HashMap<String, String>,
) -> bool {
    if let Instr::Load { src, .. } = instr {
        varying_param_names.contains(&resolve_load_source(src, gep_bases))
    } else {
        false
    }
}

fn instr_geps_varying_base(
    instr: &Instr,
    varying_param_names: &HashSet<String>,
    varying_ssa: &HashSet<String>,
) -> bool {
    if let Instr::Gep {
        base,
        base_is_global,
        ..
    } = instr
        && !*base_is_global
    {
        return varying_param_names.contains(base) || varying_ssa.contains(base);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::parse::parse_module;
    use super::*;

    const SPLIT_V: &str = r#"; ModuleID = "V"
target triple = "nvptx64-nvidia-cuda"

define void @"V"(float* %"k", float* %"x", float* %"out0")
{
entry:
  %"v0" = load float, float* %"k"
  %"v1" = call float @"__nv_expf"(float %"v0")
  %"v2" = load float, float* %"x"
  %"v3" = fmul float %"v1", %"v2"
  store float %"v3", float* %"out0"
  ret void
}

declare float @"__nv_expf"(float %".1")
"#;

    #[test]
    fn licm_splits_simple_v() {
        let module = parse_module(SPLIT_V).unwrap();
        let split = split_loop_invariant(&module.function, &[1], 1).unwrap();
        assert_eq!(split.n_state, 1);
        assert_eq!(split.pre_indices, vec![0]);
        assert_eq!(split.direct_indices, Vec::<usize>::new());

        let pre_callees: Vec<&str> = split
            .pre
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::Call { callee, .. } => Some(callee.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(pre_callees, vec!["__nv_expf"]);

        assert!(
            split
                .step
                .instrs
                .iter()
                .any(|i| matches!(i, Instr::BinOp { .. }))
        );
        assert!(
            split
                .step
                .instrs
                .iter()
                .any(|i| matches!(i, Instr::Store { .. }))
        );
    }

    #[test]
    fn gep_into_varying_buf_lands_in_step() {
        const IR: &str = r#"; ModuleID = "V"
target triple = "nvptx64-nvidia-cuda"

define void @"V"(float* %"k", float* %"x", float* %"out0")
{
entry:
  %".7" = getelementptr inbounds float, float* %"k", i32 0
  %"v0" = load float, float* %".7"
  %".8" = getelementptr inbounds float, float* %"x", i32 0
  %"v1" = load float, float* %".8"
  %"v2" = fmul float %"v0", %"v1"
  store float %"v2", float* %"out0"
  ret void
}
"#;
        let module = parse_module(IR).unwrap();
        let split = split_loop_invariant(&module.function, &[1], 1).unwrap();
        let has_gep_into = |body: &[Instr], buf: &str| {
            body.iter().any(|i| match i {
                Instr::Gep { base, .. } => base == buf,
                _ => false,
            })
        };
        assert!(
            has_gep_into(&split.step.instrs, "x"),
            "GEP into varying buf `x` must appear in step"
        );
        assert!(
            !has_gep_into(&split.pre.instrs, "x"),
            "GEP into varying buf `x` must NOT remain in pre"
        );
    }
}
