// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::load_sym;
use libloading::Library;
use std::ffi::{CStr, c_char, c_void};
use std::os::raw::c_int;
use std::path::PathBuf;
use std::ptr;

pub(super) type NvrtcProgram = *mut c_void;
pub(super) type NvrtcResult = c_int;

pub(super) const NVRTC_SUCCESS: NvrtcResult = 0;

fn load_named(stem: &str, majors: &[&str], lib_dirs: &[PathBuf]) -> Option<Library> {
    let owned = crate::dynamic_library::sonames(stem, majors);
    let names: Vec<&str> = owned.iter().map(String::as_str).collect();
    let attempts: Vec<PathBuf> = [
        cross_product(lib_dirs, &names),
        names.iter().map(PathBuf::from).collect(),
    ]
    .concat();
    crate::dynamic_library::open_first(&attempts)
}

fn cross_product(dirs: &[PathBuf], names: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::with_capacity(dirs.len() * names.len());
    for d in dirs {
        for n in names {
            out.push(d.join(n));
        }
    }
    out
}

#[allow(non_snake_case)]
fn load_nvrtc_builtins(
    nvrtcVersion: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult,
    lib_dirs: &[PathBuf],
) -> Option<Library> {
    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    // SAFETY: nvrtcVersion is a function pointer freshly resolved from
    // the loaded libnvrtc; the two out-pointers are stack locals.
    if unsafe { nvrtcVersion(&mut major, &mut minor) } != NVRTC_SUCCESS {
        return None;
    }
    let soname = format!("libnvrtc-builtins.so.{major}.{minor}");
    let names = [soname.as_str()];
    // SAFETY: nvrtcVersion is a function pointer mapped via dlopen.
    let companion: Vec<_> =
        unsafe { crate::dynamic_library::loaded_lib_dir(nvrtcVersion as *const c_void) }
            .map(|d| vec![d.join(&soname)])
            .unwrap_or_default();
    let attempts: Vec<_> = [
        companion,
        cross_product(lib_dirs, &names),
        vec![PathBuf::from(&soname)],
    ]
    .concat();
    crate::dynamic_library::open_first(&attempts)
}

#[allow(non_snake_case)]
pub(super) struct NvrtcCompiler {
    _lib: Library,
    _builtins: Option<Library>,
    nvrtcCreateProgram: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    nvrtcDestroyProgram: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
    nvrtcCompileProgram:
        unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    nvrtcGetProgramLogSize: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtcGetProgramLog: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtcGetCUBINSize: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtcGetCUBIN: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtcGetLTOIRSize: Option<unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult>,
    nvrtcGetLTOIR: Option<unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult>,
    pub(crate) nvrtcVersion: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult,
}

impl NvrtcCompiler {
    #[allow(non_snake_case)]
    pub(crate) fn load(lib_dirs: &[PathBuf]) -> Option<Self> {
        let lib = load_named("libnvrtc", &["13", "12", "11"], lib_dirs)?;
        let nvrtcCreateProgram = load_sym!(
            &lib,
            "nvrtcCreateProgram",
            unsafe extern "C" fn(
                *mut NvrtcProgram,
                *const c_char,
                *const c_char,
                c_int,
                *const *const c_char,
                *const *const c_char,
            ) -> NvrtcResult
        );
        let nvrtcDestroyProgram = load_sym!(
            &lib,
            "nvrtcDestroyProgram",
            unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult
        );
        let nvrtcCompileProgram = load_sym!(
            &lib,
            "nvrtcCompileProgram",
            unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult
        );
        let nvrtcGetProgramLogSize = load_sym!(
            &lib,
            "nvrtcGetProgramLogSize",
            unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult
        );
        let nvrtcGetProgramLog = load_sym!(
            &lib,
            "nvrtcGetProgramLog",
            unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult
        );
        let nvrtcGetCUBINSize = load_sym!(
            &lib,
            "nvrtcGetCUBINSize",
            unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult
        );
        let nvrtcGetCUBIN = load_sym!(
            &lib,
            "nvrtcGetCUBIN",
            unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult
        );
        let nvrtcVersion = load_sym!(
            &lib,
            "nvrtcVersion",
            unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult
        );

        let _builtins = load_nvrtc_builtins(nvrtcVersion, lib_dirs);

        type LtoSizeFn = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
        type LtoFn = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
        let nvrtcGetLTOIRSize: Option<LtoSizeFn> = unsafe {
            lib.get::<LtoSizeFn>(b"nvrtcGetLTOIRSize\0")
                .ok()
                .map(|s| *s)
        };
        let nvrtcGetLTOIR: Option<LtoFn> =
            unsafe { lib.get::<LtoFn>(b"nvrtcGetLTOIR\0").ok().map(|s| *s) };

        Some(Self {
            _lib: lib,
            _builtins,
            nvrtcCreateProgram,
            nvrtcDestroyProgram,
            nvrtcCompileProgram,
            nvrtcGetProgramLogSize,
            nvrtcGetProgramLog,
            nvrtcGetCUBINSize,
            nvrtcGetCUBIN,
            nvrtcGetLTOIRSize,
            nvrtcGetLTOIR,
            nvrtcVersion,
        })
    }

    pub(crate) fn version(&self) -> i32 {
        let mut major = 0;
        let mut minor = 0;
        unsafe {
            if (self.nvrtcVersion)(&mut major, &mut minor) == NVRTC_SUCCESS {
                major * 1000 + minor * 10
            } else {
                0
            }
        }
    }

    pub(crate) fn cubin_from_plaintext(
        &self,
        code: &str,
        filename: &str,
        sm: i32,
        extra_opts: &[String],
        include_dirs: &[PathBuf],
    ) -> Result<Vec<u8>, String> {
        let code_c = std::ffi::CString::new(code).map_err(|e| format!("invalid code: {e}"))?;
        let file_name =
            std::ffi::CString::new(format!("{filename}.cu")).map_err(|e| format!("{e}"))?;

        let mut prog: NvrtcProgram = ptr::null_mut();
        unsafe {
            let r = (self.nvrtcCreateProgram)(
                &mut prog,
                code_c.as_ptr(),
                file_name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            );
            if r != NVRTC_SUCCESS {
                return Err(format!("nvrtcCreateProgram failed: {r}"));
            }
        }

        let (opts, _hold) = self.build_opts(sm, false, extra_opts, include_dirs);
        let compile_res =
            unsafe { (self.nvrtcCompileProgram)(prog, opts.len() as c_int, opts.as_ptr()) };
        if compile_res != NVRTC_SUCCESS {
            return Err(format!(
                "NVRTC compilation failed:\n{}",
                self.take_log(&mut prog)
            ));
        }
        Ok(self.take_cubin(&mut prog))
    }

    pub(crate) fn ltoir_from_plaintext(
        &self,
        code: &str,
        filename: &str,
        sm: i32,
        extra_opts: &[String],
        include_dirs: &[PathBuf],
    ) -> Result<Vec<u8>, String> {
        let (get_lto_size, get_lto) = self.lto_getters()?;

        let code_c = std::ffi::CString::new(code).map_err(|e| format!("invalid code: {e}"))?;
        let file_name =
            std::ffi::CString::new(format!("{filename}.cu")).map_err(|e| format!("{e}"))?;

        let mut prog: NvrtcProgram = ptr::null_mut();
        unsafe {
            let r = (self.nvrtcCreateProgram)(
                &mut prog,
                code_c.as_ptr(),
                file_name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            );
            if r != NVRTC_SUCCESS {
                return Err(format!("nvrtcCreateProgram failed: {r}"));
            }
        }

        let (opts, _hold) = self.build_opts(sm, true, extra_opts, include_dirs);
        let compile_res =
            unsafe { (self.nvrtcCompileProgram)(prog, opts.len() as c_int, opts.as_ptr()) };
        if compile_res != NVRTC_SUCCESS {
            return Err(format!(
                "NVRTC -dlto compilation failed:\n{}",
                self.take_log(&mut prog)
            ));
        }
        Ok(self.take_ltoir(&mut prog, get_lto_size, get_lto))
    }

    #[allow(clippy::type_complexity)]
    fn lto_getters(
        &self,
    ) -> Result<
        (
            unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
            unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
        ),
        String,
    > {
        let s = self.nvrtcGetLTOIRSize.ok_or_else(|| {
            "nvrtcGetLTOIRSize not available (need CUDA 12+ libnvrtc)".to_string()
        })?;
        let g = self
            .nvrtcGetLTOIR
            .ok_or_else(|| "nvrtcGetLTOIR not available (need CUDA 12+ libnvrtc)".to_string())?;
        Ok((s, g))
    }

    fn build_opts(
        &self,
        sm: i32,
        dlto: bool,
        extra_opts: &[String],
        include_dirs: &[PathBuf],
    ) -> (Vec<*const c_char>, Vec<std::ffi::CString>) {
        let mut owned: Vec<std::ffi::CString> = Vec::new();
        owned.push(std::ffi::CString::new(format!("--gpu-architecture=sm_{sm}")).unwrap());
        owned.push(std::ffi::CString::new("--std=c++17").unwrap());
        if dlto {
            owned.push(std::ffi::CString::new("-dlto").unwrap());
        }
        for p in resolve_include_dirs(include_dirs) {
            owned.push(std::ffi::CString::new(format!("-I{p}")).unwrap());
        }
        for o in extra_opts {
            owned.push(std::ffi::CString::new(o.as_str()).expect("opt has nul byte"));
        }
        let ptrs: Vec<*const c_char> = owned.iter().map(|c| c.as_ptr()).collect();
        (ptrs, owned)
    }

    fn take_log(&self, prog: &mut NvrtcProgram) -> String {
        let mut log_size: usize = 0;
        unsafe {
            (self.nvrtcGetProgramLogSize)(*prog, &mut log_size);
        }
        let mut log = vec![0u8; log_size];
        unsafe {
            (self.nvrtcGetProgramLog)(*prog, log.as_mut_ptr() as *mut c_char);
            (self.nvrtcDestroyProgram)(prog);
        }
        CStr::from_bytes_until_nul(&log)
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from_utf8_lossy(&log).into_owned())
    }

    fn take_cubin(&self, prog: &mut NvrtcProgram) -> Vec<u8> {
        let mut cubin_size: usize = 0;
        unsafe {
            (self.nvrtcGetCUBINSize)(*prog, &mut cubin_size);
        }
        let mut cubin = vec![0u8; cubin_size];
        unsafe {
            (self.nvrtcGetCUBIN)(*prog, cubin.as_mut_ptr() as *mut c_char);
            (self.nvrtcDestroyProgram)(prog);
        }
        cubin
    }

    fn take_ltoir(
        &self,
        prog: &mut NvrtcProgram,
        get_lto_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
        get_lto: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    ) -> Vec<u8> {
        let mut lto_size: usize = 0;
        unsafe {
            (get_lto_size)(*prog, &mut lto_size);
        }
        let mut lto = vec![0u8; lto_size];
        unsafe {
            (get_lto)(*prog, lto.as_mut_ptr() as *mut c_char);
            (self.nvrtcDestroyProgram)(prog);
        }
        lto
    }
}

fn resolve_include_dirs(extra: &[PathBuf]) -> Vec<String> {
    extra
        .iter()
        .filter(|p| p.is_dir())
        .cloned()
        .flat_map(|p| {
            let mut dirs = vec![p.to_string_lossy().into_owned()];
            let cccl = p.join("cccl");
            if cccl.is_dir() {
                dirs.push(cccl.to_string_lossy().into_owned());
            }
            dirs
        })
        .collect()
}
