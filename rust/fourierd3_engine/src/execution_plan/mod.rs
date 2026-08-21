// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod blob;
mod builder;
mod ir;
pub(crate) mod layout;
mod splice;
pub(crate) mod wire;

pub(crate) use blob::Blob;
pub(crate) use builder::*;
pub(crate) use ir::*;
#[cfg(test)]
pub(crate) use wire::deserialize;
pub(crate) use wire::{WireError, deserialize_shared, serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FlatPlan(ExecutionPlan);

impl FlatPlan {
    pub(crate) fn assume_flat(plan: ExecutionPlan) -> Self {
        Self(plan)
    }

    pub(crate) fn into_plan(self) -> ExecutionPlan {
        self.0
    }
}

impl std::ops::Deref for FlatPlan {
    type Target = ExecutionPlan;
    fn deref(&self) -> &ExecutionPlan {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlanError {
    DepNotEarlier { node: usize, dep: usize },
    ModuleOutOfRange { node: usize, module: usize },
    WorkspaceOutOfRange { node: usize, index: usize },
    InputOutOfRange { node: usize, index: usize },
    OutputOutOfRange { node: usize, index: usize },
    EmptyChoice { node: usize },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::DepNotEarlier { node, dep } => {
                write!(
                    f,
                    "node {node} depends on {dep}, not a strictly-earlier node"
                )
            }
            PlanError::ModuleOutOfRange { node, module } => {
                write!(f, "node {node} references module {module}, out of range")
            }
            PlanError::WorkspaceOutOfRange { node, index } => {
                write!(f, "node {node} references workspace {index}, out of range")
            }
            PlanError::InputOutOfRange { node, index } => {
                write!(
                    f,
                    "node {node} references input binding {index}, out of range"
                )
            }
            PlanError::OutputOutOfRange { node, index } => {
                write!(
                    f,
                    "node {node} references output binding {index}, out of range"
                )
            }
            PlanError::EmptyChoice { node } => {
                write!(f, "node {node} is a choice with no candidates")
            }
        }
    }
}

impl std::error::Error for PlanError {}

impl ExecutionPlan {
    pub(crate) fn validate(&self, n_inputs: usize, n_outputs: usize) -> Result<(), PlanError> {
        for (node, n) in self.nodes.iter().enumerate() {
            Self::check_dependencies(node, &n.deps)?;
            self.check_operation(node, &n.op, n_inputs, n_outputs)?;
        }
        Ok(())
    }

    fn check_dependencies(node: usize, dependencies: &[usize]) -> Result<(), PlanError> {
        for &dep in dependencies {
            if dep >= node {
                return Err(PlanError::DepNotEarlier { node, dep });
            }
        }
        Ok(())
    }

    fn check_operation(
        &self,
        node: usize,
        op: &Op,
        n_inputs: usize,
        n_outputs: usize,
    ) -> Result<(), PlanError> {
        match op {
            Op::KernelLaunch { module, args, .. } => {
                self.check_kernel(node, *module, args, n_inputs, n_outputs)
            }
            Op::Memset { target, .. } => self.check_buf(node, target, n_outputs),
            Op::Choice {
                candidates,
                input_binding,
                output_binding,
            } => self.check_choice(
                node,
                candidates,
                input_binding,
                output_binding,
                n_inputs,
                n_outputs,
            ),
        }
    }

    fn check_kernel(
        &self,
        node: usize,
        module: usize,
        args: &[Arg],
        n_inputs: usize,
        n_outputs: usize,
    ) -> Result<(), PlanError> {
        if module >= self.modules.len() {
            return Err(PlanError::ModuleOutOfRange { node, module });
        }
        for arg in args {
            self.check_bufref(node, &arg.buf, n_inputs, n_outputs)?;
        }
        Ok(())
    }

    fn check_choice(
        &self,
        node: usize,
        candidates: &[ExecutionPlan],
        input_binding: &[BufRef],
        output_binding: &[BufRef],
        n_inputs: usize,
        n_outputs: usize,
    ) -> Result<(), PlanError> {
        if candidates.is_empty() {
            return Err(PlanError::EmptyChoice { node });
        }
        for binding in input_binding.iter().chain(output_binding) {
            self.check_bufref(node, binding, n_inputs, n_outputs)?;
        }
        for candidate in candidates {
            candidate.validate(input_binding.len(), output_binding.len())?;
        }
        Ok(())
    }

    fn check_bufref(
        &self,
        node: usize,
        buf: &BufRef,
        n_inputs: usize,
        n_outputs: usize,
    ) -> Result<(), PlanError> {
        match buf {
            BufRef::Input(index) => {
                if *index >= n_inputs {
                    return Err(PlanError::InputOutOfRange {
                        node,
                        index: *index,
                    });
                }
            }
            BufRef::Output(index) => {
                if *index >= n_outputs {
                    return Err(PlanError::OutputOutOfRange {
                        node,
                        index: *index,
                    });
                }
            }
            BufRef::Workspace(index) => {
                if *index >= self.workspace.len() {
                    return Err(PlanError::WorkspaceOutOfRange {
                        node,
                        index: *index,
                    });
                }
            }
        }
        Ok(())
    }

    fn check_buf(&self, node: usize, buf: &WritableBuf, n_outputs: usize) -> Result<(), PlanError> {
        match buf {
            WritableBuf::Output(index) => {
                if *index >= n_outputs {
                    return Err(PlanError::OutputOutOfRange {
                        node,
                        index: *index,
                    });
                }
            }
            WritableBuf::Workspace(index) => {
                if *index >= self.workspace.len() {
                    return Err(PlanError::WorkspaceOutOfRange {
                        node,
                        index: *index,
                    });
                }
            }
        }
        Ok(())
    }
}
