// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod bind;
mod device;
mod exec;
mod layout;
mod resident;
mod selection;
mod tune;

use crate::cuda_driver::CUdeviceptr;

pub(crate) use exec::execute;
pub(crate) use resident::ResidentPlan;
pub(crate) use tune::{ChoiceReport, TuneReport, kernel_entries, tune, tune_reported};

pub(crate) struct Bindings<'a> {
    pub inputs: &'a [CUdeviceptr],
    pub outputs: &'a [CUdeviceptr],
    pub workspace: CUdeviceptr,
}

#[derive(Clone, Debug)]
pub(crate) enum Error {
    Driver(crate::cuda_driver::Error),
    MissingSymbol(String),
    InputUnbound(usize),
    OutputUnbound(usize),
    WorkspaceUnbound(usize),
    UnresolvedChoice { node: usize },
    ContextMismatch,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(error) => write!(f, "CUDA driver error: {error}"),
            Self::MissingSymbol(symbol) => write!(f, "missing kernel symbol `{symbol}`"),
            Self::InputUnbound(index) => write!(f, "input {index} is unbound"),
            Self::OutputUnbound(index) => write!(f, "output {index} is unbound"),
            Self::WorkspaceUnbound(index) => write!(f, "workspace buffer {index} is unbound"),
            Self::UnresolvedChoice { node } => write!(f, "choice node {node} is unresolved"),
            Self::ContextMismatch => f.write_str("CUDA context mismatch"),
        }
    }
}
