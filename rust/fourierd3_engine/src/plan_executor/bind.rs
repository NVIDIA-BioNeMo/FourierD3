// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;

use crate::cuda_driver::{CUdeviceptr, CUstream, StreamRef};

use crate::execution_plan::{Arg, BufRef, ExecutionPlan, WritableBuf};

use crate::plan_executor::Error;

pub(crate) struct Ports<'a> {
    pub(crate) inputs: &'a [CUdeviceptr],
    pub(crate) outputs: &'a [CUdeviceptr],
    pub(crate) workspace: &'a [CUdeviceptr],
}

pub(crate) unsafe fn seed_workspace(
    plan: &ExecutionPlan,
    ports: &Ports,
    stream: CUstream,
) -> Result<(), Error> {
    let stream = StreamRef::from_raw(stream);
    for (i, buf) in plan.workspace.iter().enumerate() {
        if let Some(init) = &buf.init {
            let dst = workspace_ptr(ports, i)?;
            unsafe { stream.memcpy_h2d(dst, init.as_ptr() as *const c_void, init.len()) }
                .map_err(Error::Driver)?;
        }
    }
    Ok(())
}

pub(crate) struct KernelParams {
    #[allow(dead_code)]
    values: Vec<CUdeviceptr>,
    params: Vec<*mut c_void>,
}

impl KernelParams {
    pub(crate) fn as_slice(&self) -> &[*mut c_void] {
        &self.params
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [*mut c_void] {
        &mut self.params
    }
}

pub(crate) fn kernel_params(args: &[Arg], ports: &Ports) -> Result<KernelParams, Error> {
    let mut values = args
        .iter()
        .map(|arg| Ok(bufref_ptr(ports, &arg.buf)? + arg.offset as CUdeviceptr))
        .collect::<Result<Vec<_>, Error>>()?;
    let params = values
        .iter_mut()
        .map(|value| (value as *mut CUdeviceptr).cast())
        .collect();

    Ok(KernelParams { values, params })
}

pub(crate) fn bufref_ptr(ports: &Ports, buf: &BufRef) -> Result<CUdeviceptr, Error> {
    match buf {
        BufRef::Input(i) => input_ptr(ports, *i),
        BufRef::Output(i) => output_ptr(ports, *i),
        BufRef::Workspace(i) => workspace_ptr(ports, *i),
    }
}

pub(crate) fn input_ptr(ports: &Ports, index: usize) -> Result<CUdeviceptr, Error> {
    ports
        .inputs
        .get(index)
        .copied()
        .ok_or(Error::InputUnbound(index))
}

pub(crate) fn output_ptr(ports: &Ports, index: usize) -> Result<CUdeviceptr, Error> {
    ports
        .outputs
        .get(index)
        .copied()
        .ok_or(Error::OutputUnbound(index))
}

pub(crate) fn workspace_ptr(ports: &Ports, index: usize) -> Result<CUdeviceptr, Error> {
    ports
        .workspace
        .get(index)
        .copied()
        .ok_or(Error::WorkspaceUnbound(index))
}

pub(crate) fn buf_ptr(ports: &Ports, buf: &WritableBuf) -> Result<CUdeviceptr, Error> {
    match buf {
        WritableBuf::Output(i) => output_ptr(ports, *i),
        WritableBuf::Workspace(i) => workspace_ptr(ports, *i),
    }
}
