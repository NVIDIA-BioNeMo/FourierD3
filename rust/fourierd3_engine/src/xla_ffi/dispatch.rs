// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::panic;

use crate::xla_ffi::sys::{
    XLA_FFI_API_MAJOR, XLA_FFI_API_MINOR, XLA_FFI_Api_Version, XLA_FFI_Api_Version_STRUCT_SIZE,
    XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_ExecutionStage_EXECUTE, XLA_FFI_Extension_Metadata,
    XLA_FFI_Metadata_Extension, XLA_FFI_TypeId,
};

use crate::xla_ffi::error::Error;
use crate::xla_ffi::state::DecodeState;

unsafe fn populate_metadata(cf: &XLA_FFI_CallFrame) {
    let ext = cf.extension_start as *mut XLA_FFI_Metadata_Extension;
    let md = unsafe { (*ext).metadata };
    if md.is_null() {
        return;
    }
    // NGC JAX 26.03 ships XLA FFI API 0.2. The vendored header can be
    // newer than the runtime, and XLA silently rejects handlers that
    // announce a higher minor version at registration time. The fields
    // used by this wrapper are covered by 0.2, so report that compatible
    // floor while continuing to build against the vendored definitions.
    let minor_version = (XLA_FFI_API_MINOR as i32).min(2);
    unsafe {
        (*md).api_version = XLA_FFI_Api_Version {
            struct_size: XLA_FFI_Api_Version_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            major_version: XLA_FFI_API_MAJOR as i32,
            minor_version,
        };
        (*md).traits = 0;
        (*md).state_type_id = XLA_FFI_TypeId { type_id: 0 };
    }
}

pub(crate) unsafe fn dispatch<F>(call_frame: *mut XLA_FFI_CallFrame, body: F) -> *mut XLA_FFI_Error
where
    F: FnOnce(&mut DecodeState<'_>) -> crate::xla_ffi::Result<()> + panic::UnwindSafe,
{
    if call_frame.is_null() {
        return std::ptr::null_mut();
    }
    let cf = unsafe { &mut *call_frame };
    let api = cf.api;

    if !cf.extension_start.is_null()
        && unsafe { (*cf.extension_start).type_ } == XLA_FFI_Extension_Metadata
    {
        unsafe { populate_metadata(cf) };
        return std::ptr::null_mut();
    }

    // The other stages (instantiate / prepare / initialize) leave args
    // and rets unpopulated, so any handler that decodes either would
    // miscompile. Succeed silently for those — XLA accepts a null
    // pointer as OK — and only run user code in EXECUTE.
    if cf.stage != XLA_FFI_ExecutionStage_EXECUTE {
        return std::ptr::null_mut();
    }

    let r = panic::catch_unwind(|| {
        let mut state = DecodeState::new(cf);
        body(&mut state)
    });
    match r {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(e)) => unsafe { e.into_xla(api) },
        Err(_) => unsafe { Error::internal("panic in FFI handler").into_xla(api) },
    }
}
