// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::load_sym;
use libloading::Library;
use std::ffi::{c_char, c_int, c_void};
use std::path::PathBuf;

use crate::cuda_compiler::nvrtc::NvrtcCompiler;

pub(super) type NvJitLinkHandle = *mut c_void;
pub(super) type NvJitLinkResult = c_int;

pub(super) const NVJITLINK_SUCCESS: NvJitLinkResult = 0;

const NVJITLINK_INPUT_LTOIR: c_int = 3;

#[allow(non_snake_case)]
pub(super) struct NvJitLink {
    _lib: Library,
    loaded_path: Option<PathBuf>,
    version: Option<(u32, u32, u32)>,
    lib_dirs: Vec<PathBuf>,
    nvJitLinkCreate:
        unsafe extern "C" fn(*mut NvJitLinkHandle, u32, *const *const c_char) -> NvJitLinkResult,
    nvJitLinkDestroy: unsafe extern "C" fn(*mut NvJitLinkHandle) -> NvJitLinkResult,
    nvJitLinkAddData: unsafe extern "C" fn(
        NvJitLinkHandle,
        c_int,
        *const c_void,
        usize,
        *const c_char,
    ) -> NvJitLinkResult,
    nvJitLinkComplete: unsafe extern "C" fn(NvJitLinkHandle) -> NvJitLinkResult,
    nvJitLinkGetLinkedCubinSize:
        unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    nvJitLinkGetLinkedCubin: unsafe extern "C" fn(NvJitLinkHandle, *mut c_void) -> NvJitLinkResult,
    nvJitLinkGetErrorLogSize: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    nvJitLinkGetErrorLog: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
}

// SAFETY: the loaded function pointers refer to a library kept alive for the
// lifetime of this struct.
unsafe impl Send for NvJitLink {}
unsafe impl Sync for NvJitLink {}

/// Both CUDA majors this engine builds against, newest first, then the
/// unversioned link. The pip wheels ship only the versioned soname
/// (`nvidia-nvjitlink-cu12` installs `libnvJitLink.so.12` with no
/// `libnvJitLink.so`), so a major missing from this list is a CUDA release the
/// engine cannot load at all — the list must track `NvrtcCompiler`'s.
const SONAMES: [&str; 3] = [
    "libnvJitLink.so.13",
    "libnvJitLink.so.12",
    "libnvJitLink.so",
];

fn cross_product(dirs: &[PathBuf], names: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::with_capacity(dirs.len() * names.len());
    for d in dirs {
        for n in names {
            out.push(d.join(n));
        }
    }
    out
}

impl NvJitLink {
    pub(crate) fn load(lib_dirs: &[PathBuf], nvrtc: &NvrtcCompiler) -> Option<Self> {
        // SAFETY: `nvrtcVersion` is a live function pointer in a
        // dlopen-mapped library — exactly what dladdr expects.
        let companion: Vec<PathBuf> =
            unsafe { crate::dynamic_library::loaded_lib_dir(nvrtc.nvrtcVersion as *const c_void) }
                .map(|d| SONAMES.iter().map(|n| d.join(n)).collect())
                .unwrap_or_default();
        // Reuse a resident libnvJitLink before opening an absolute path: two
        // separately mapped copies can corrupt their shared JIT-LTO state.
        let fallback: Vec<_> = [companion, SONAMES.iter().map(PathBuf::from).collect()].concat();
        let configured = cross_product(lib_dirs, &SONAMES);
        let lib = crate::dynamic_library::open_resident(&SONAMES)
            .or_else(|| crate::dynamic_library::open_first(&configured))
            .or_else(|| crate::dynamic_library::open_first(&fallback))?;
        type VerFn = unsafe extern "C" fn(*mut u32, *mut u32) -> NvJitLinkResult;
        let version_fn: Option<VerFn> = unsafe { lib.get::<VerFn>(b"nvJitLinkVersion\0") }
            .ok()
            .map(|s| *s);
        let version = version_fn.and_then(|f| {
            let (mut major, mut minor) = (0u32, 0u32);
            (unsafe { f(&mut major, &mut minor) } == NVJITLINK_SUCCESS).then_some((major, minor, 0))
        });
        // SAFETY: `nvJitLinkCreate` is a freshly resolved symbol in the
        // loaded library. We don't call it — just hand its address to
        // dladdr.
        let create_sym: libloading::Symbol<unsafe extern "C" fn()> =
            unsafe { lib.get(b"nvJitLinkCreate\0") }.ok()?;
        let loaded_path =
            unsafe { crate::dynamic_library::loaded_lib_path(*create_sym as *const c_void) };
        Some(Self {
            loaded_path,
            version,
            lib_dirs: lib_dirs.to_vec(),
            nvJitLinkCreate: load_sym!(
                &lib,
                "nvJitLinkCreate",
                unsafe extern "C" fn(
                    *mut NvJitLinkHandle,
                    u32,
                    *const *const c_char,
                ) -> NvJitLinkResult
            ),
            nvJitLinkDestroy: load_sym!(
                &lib,
                "nvJitLinkDestroy",
                unsafe extern "C" fn(*mut NvJitLinkHandle) -> NvJitLinkResult
            ),
            nvJitLinkAddData: load_sym!(
                &lib,
                "nvJitLinkAddData",
                unsafe extern "C" fn(
                    NvJitLinkHandle,
                    c_int,
                    *const c_void,
                    usize,
                    *const c_char,
                ) -> NvJitLinkResult
            ),
            nvJitLinkComplete: load_sym!(
                &lib,
                "nvJitLinkComplete",
                unsafe extern "C" fn(NvJitLinkHandle) -> NvJitLinkResult
            ),
            nvJitLinkGetLinkedCubinSize: load_sym!(
                &lib,
                "nvJitLinkGetLinkedCubinSize",
                unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult
            ),
            nvJitLinkGetLinkedCubin: load_sym!(
                &lib,
                "nvJitLinkGetLinkedCubin",
                unsafe extern "C" fn(NvJitLinkHandle, *mut c_void) -> NvJitLinkResult
            ),
            nvJitLinkGetErrorLogSize: load_sym!(
                &lib,
                "nvJitLinkGetErrorLogSize",
                unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult
            ),
            nvJitLinkGetErrorLog: load_sym!(
                &lib,
                "nvJitLinkGetErrorLog",
                unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult
            ),
            _lib: lib,
        })
    }

    fn provenance(&self) -> String {
        let path = self
            .loaded_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        let version = self
            .version
            .map(|(a, b, c)| format!("{a}.{b}.{c}"))
            .unwrap_or_else(|| "<unknown>".to_string());
        let mut out = format!("loaded libnvJitLink: {path} (version {version})");
        let mut alternates: Vec<PathBuf> = cross_product(&self.lib_dirs, &SONAMES)
            .into_iter()
            .filter(|p| p.exists())
            .filter(|p| {
                self.loaded_path.as_deref().is_none_or(|loaded| {
                    std::fs::canonicalize(p).ok() != std::fs::canonicalize(loaded).ok()
                })
            })
            .collect();
        alternates.dedup();
        if !alternates.is_empty() {
            out.push_str("\nother libnvJitLink copies known to the engine:");
            for p in &alternates {
                out.push_str("\n  ");
                out.push_str(&p.display().to_string());
            }
            out.push_str(
                "\nIf the loaded copy is older than required: the engine reused \
                 the libnvJitLink already mapped into this process, to avoid a \
                 second, conflicting instance. Make the process load a newer one \
                 first — prepend it to $LD_LIBRARY_PATH or upgrade the system \
                 CUDA toolkit.",
            );
        }
        out
    }

    fn error_log(&self, handle: NvJitLinkHandle) -> String {
        let mut n: usize = 0;
        if unsafe { (self.nvJitLinkGetErrorLogSize)(handle, &mut n) } != NVJITLINK_SUCCESS || n == 0
        {
            return String::new();
        }
        let mut buf = vec![0u8; n];
        if unsafe { (self.nvJitLinkGetErrorLog)(handle, buf.as_mut_ptr() as *mut c_char) }
            != NVJITLINK_SUCCESS
        {
            return String::new();
        }
        if let Some(p) = buf.iter().position(|&b| b == 0) {
            buf.truncate(p);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    pub(crate) fn link_cubin(&self, sm: i32, inputs: &[LtoInput]) -> Result<Vec<u8>, String> {
        let arch = std::ffi::CString::new(format!("-arch=sm_{sm}")).unwrap();
        let lto_opt = std::ffi::CString::new("-lto").unwrap();
        let opt_ptrs = [arch.as_ptr(), lto_opt.as_ptr()];

        let mut handle: NvJitLinkHandle = std::ptr::null_mut();
        let r = unsafe {
            (self.nvJitLinkCreate)(&mut handle, opt_ptrs.len() as u32, opt_ptrs.as_ptr())
        };
        if r != NVJITLINK_SUCCESS {
            return Err(format!("nvJitLinkCreate failed: {r}"));
        }
        struct LinkGuard<'a> {
            h: NvJitLinkHandle,
            lib: &'a NvJitLink,
        }
        impl Drop for LinkGuard<'_> {
            fn drop(&mut self) {
                if !self.h.is_null() {
                    unsafe { (self.lib.nvJitLinkDestroy)(&mut self.h) };
                }
            }
        }
        let _g = LinkGuard {
            h: handle,
            lib: self,
        };

        for inp in inputs {
            let name = std::ffi::CString::new(inp.name).map_err(|e| e.to_string())?;
            let r = unsafe {
                (self.nvJitLinkAddData)(
                    handle,
                    NVJITLINK_INPUT_LTOIR,
                    inp.data.as_ptr() as *const c_void,
                    inp.data.len(),
                    name.as_ptr(),
                )
            };
            if r != NVJITLINK_SUCCESS {
                return Err(format!(
                    "nvJitLinkAddData({}) failed: {r}: {}\n{}",
                    inp.name,
                    self.error_log(handle),
                    self.provenance(),
                ));
            }
        }

        let r = unsafe { (self.nvJitLinkComplete)(handle) };
        if r != NVJITLINK_SUCCESS {
            return Err(format!(
                "nvJitLinkComplete failed: {r}: {}\n{}",
                self.error_log(handle),
                self.provenance(),
            ));
        }

        let mut n: usize = 0;
        let r = unsafe { (self.nvJitLinkGetLinkedCubinSize)(handle, &mut n) };
        if r != NVJITLINK_SUCCESS {
            return Err(format!("nvJitLinkGetLinkedCubinSize failed: {r}"));
        }
        let mut cubin = vec![0u8; n];
        let r =
            unsafe { (self.nvJitLinkGetLinkedCubin)(handle, cubin.as_mut_ptr() as *mut c_void) };
        if r != NVJITLINK_SUCCESS {
            return Err(format!("nvJitLinkGetLinkedCubin failed: {r}"));
        }
        Ok(cubin)
    }
}

pub(super) struct LtoInput<'a> {
    pub data: &'a [u8],
    pub name: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_compiler::CudaCompilerConfig;

    fn loaded() -> Option<crate::cuda_compiler::CudaCompiler> {
        crate::cuda_compiler::CudaCompiler::load(CudaCompilerConfig::default()).ok()
    }

    #[test]
    fn library_loads() {
        if loaded().is_none() {
            eprintln!("skip: CUDA JIT toolchain not loadable in this environment");
        }
    }

    #[test]
    fn link_with_bogus_input_returns_error() {
        let Some(jit) = loaded() else {
            eprintln!("skip: CUDA JIT toolchain not loadable in this environment");
            return;
        };
        let err = jit
            .nvjitlink
            .link_cubin(
                120,
                &[LtoInput {
                    data: b"not actually LTOIR",
                    name: "bogus",
                }],
            )
            .expect_err("bogus LTOIR must be rejected");
        assert!(!err.is_empty());
    }
}
