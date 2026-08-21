// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Locating a shared library by soname or path, opening it, and binding its
//! symbols to caller-declared signatures.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

pub(crate) use libloading::Library;

pub(crate) fn load_symbol<T: Copy>(lib: &Library, name: &[u8]) -> T {
    let symbol = unsafe { lib.get::<T>(name) }
        .unwrap_or_else(|_| panic!("Failed to load symbol: {}", String::from_utf8_lossy(name)));
    *symbol
}

#[macro_export]
macro_rules! load_sym {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let name = concat!($name, "\0").as_bytes();
        $crate::dynamic_library::load_symbol::<$ty>($lib, name)
    }};
}

pub(crate) fn sonames(stem: &str, majors: &[&str]) -> Vec<String> {
    std::iter::once(format!("{stem}.so"))
        .chain(majors.iter().map(|m| format!("{stem}.so.{m}")))
        .collect()
}

pub(crate) fn open_first(candidates: &[PathBuf]) -> Option<Library> {
    candidates
        .iter()
        // SAFETY: dlopen of a shared library by absolute path or soname.
        .find_map(|p| unsafe { Library::new(p) }.ok())
}

pub(crate) fn open_named(stem: &str, majors: &[&str]) -> Option<Library> {
    open_first(
        &sonames(stem, majors)
            .iter()
            .map(PathBuf::from)
            .collect::<Vec<_>>(),
    )
}

const RTLD_NOLOAD: c_int = 0x4;

pub(crate) fn open_resident(sonames: &[&str]) -> Option<Library> {
    sonames.iter().find_map(|n| {
        // SAFETY: with RTLD_NOLOAD `dlopen` performs no load; it returns a
        // handle only when a library of this soname is already mapped, else an
        // error we map to `None`.
        unsafe {
            libloading::os::unix::Library::open(
                Some(*n),
                libloading::os::unix::RTLD_NOW | RTLD_NOLOAD,
            )
        }
        .ok()
        .map(Library::from)
    })
}

/// # Safety
/// `symbol` must point into a `dlopen`-mapped library — typically a function
/// pointer obtained via [`libloading::Symbol`]. Passing heap, stack, or
/// unmapped addresses is UB.
pub(crate) unsafe fn loaded_lib_path(symbol: *const c_void) -> Option<PathBuf> {
    #[repr(C)]
    struct DlInfo {
        dli_fname: *const c_char,
        dli_fbase: *mut c_void,
        dli_sname: *const c_char,
        dli_saddr: *mut c_void,
    }
    unsafe extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    }
    let mut info = DlInfo {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    // SAFETY: caller guarantees `symbol` is a valid dlopen-mapped address.
    if unsafe { dladdr(symbol, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: dladdr guarantees `dli_fname` points to a NUL-terminated
    // C-string owned by the dynamic linker for the lifetime of the loaded
    // library.
    let p = unsafe { CStr::from_ptr(info.dli_fname) }.to_str().ok()?;
    Some(PathBuf::from(p))
}

/// # Safety
/// Same as [`loaded_lib_path`].
pub(crate) unsafe fn loaded_lib_dir(symbol: *const c_void) -> Option<PathBuf> {
    // SAFETY: forwarded to `loaded_lib_path`.
    unsafe { loaded_lib_path(symbol) }
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
}

pub(crate) fn preload_dir_siblings(dir: &Path, prefix: &str) -> Vec<Library> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.starts_with(prefix) {
            continue;
        }
        // SAFETY: dlopen of a shared library located inside a peer library's
        // own directory. The OS loader handles soname dedup.
        if let Ok(lib) = unsafe { Library::new(&path) } {
            out.push(lib);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_resident_binds_only_to_mapped_libraries() {
        assert!(open_resident(&["libc.so.6"]).is_some());
        assert!(open_resident(&["dynamic-library-no-such-lib.so.999"]).is_none());
        assert!(open_resident(&["dynamic-library-no-such-lib.so.999", "libc.so.6"]).is_some());
    }
}
