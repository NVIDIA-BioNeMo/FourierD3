// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

fn main() {
    pyo3_build_config::add_extension_module_link_args();

    let header = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include/xla_ffi_c_api.h");
    println!("cargo:rerun-if-changed={}", header.display());

    let mut builder = bindgen::Builder::default().header(header.to_str().unwrap());
    if let Some(include) = find_gcc_include() {
        builder = builder.clang_arg(format!("-I{}", include.display()));
    }
    let bindings = builder
        .allowlist_type("XLA_FFI_.*")
        .allowlist_function("XLA_FFI_.*")
        .allowlist_var("XLA_FFI_.*")
        .derive_default(true)
        .prepend_enum_name(false)
        .layout_tests(false)
        .generate()
        .expect("bindgen failed");

    let output = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings.write_to_file(output).expect("write bindings");
}

fn find_gcc_include() -> Option<PathBuf> {
    for triple in ["x86_64-linux-gnu", "aarch64-linux-gnu"] {
        let base = PathBuf::from(format!("/usr/lib/gcc/{triple}"));
        let Ok(entries) = std::fs::read_dir(base) else {
            continue;
        };
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.join("include/stddef.h").exists())
            .collect();
        versions.sort();
        if let Some(latest) = versions.pop() {
            return Some(latest.join("include"));
        }
    }
    None
}
