// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::sync::OnceLock;

use crate::cuda_compiler::{CudaCompiler, CudaCompilerConfig};
use parking_lot::Mutex;

struct State {
    config: Option<CudaCompilerConfig>,
    frozen: bool,
}
static STATE: Mutex<State> = Mutex::new(State {
    config: None,
    frozen: false,
});

static JIT: OnceLock<CudaCompiler> = OnceLock::new();

static CACHE: OnceLock<crate::artifact_cache::Cache> = OnceLock::new();

fn effective_config(mut cfg: CudaCompilerConfig) -> CudaCompilerConfig {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = std::env::var(var) {
            push_unique(&mut cfg.include_dirs, PathBuf::from(root).join("include"));
        }
    }
    push_unique(
        &mut cfg.include_dirs,
        PathBuf::from("/usr/local/cuda/include"),
    );
    cfg
}

pub(crate) fn jit() -> &'static CudaCompiler {
    JIT.get_or_init(|| {
        let cfg = {
            let mut st = STATE.lock();
            st.frozen = true;
            effective_config(st.config.get_or_insert_default().clone())
        };
        CudaCompiler::load(cfg).expect("failed to load the CUDA JIT toolchain")
    })
}

pub(crate) fn cache() -> &'static crate::artifact_cache::Cache {
    CACHE.get_or_init(crate::artifact_cache::Cache::new)
}

pub(crate) fn add_include_dir(dir: PathBuf) -> Result<(), String> {
    with_config("include", |cfg| push_unique(&mut cfg.include_dirs, dir))
}

pub(crate) fn add_lib_dir(dir: PathBuf) -> Result<(), String> {
    with_config("library", |cfg| push_unique(&mut cfg.lib_dirs, dir))
}

pub(crate) fn lib_dirs() -> Vec<PathBuf> {
    config_snapshot().lib_dirs
}

fn with_config(kind: &str, f: impl FnOnce(&mut CudaCompilerConfig)) -> Result<(), String> {
    let mut st = STATE.lock();
    if st.frozen {
        return Err(format!(
            "cannot add a CUDA {kind} dir: the JIT toolchain is already built. \
             Set CUDA include/library paths before the first kernel compile."
        ));
    }
    f(st.config.get_or_insert_default());
    Ok(())
}

fn config_snapshot() -> CudaCompilerConfig {
    let mut state = STATE.lock();
    effective_config(state.config.get_or_insert_default().clone())
}

fn push_unique(list: &mut Vec<PathBuf>, dir: PathBuf) {
    if !list.iter().any(|d| d == &dir) {
        list.push(dir);
    }
}

pub(crate) fn compile_cubin(
    src: &[u8],
    filename: Option<&str>,
    opts: &[String],
    ltoir_blobs: &[&[u8]],
) -> Result<Vec<u8>, String> {
    let jit = jit();
    let sm = crate::cuda_driver::Device::current().sm_arch().unwrap_or(0);
    let key = jit.cubin_key(src, opts, ltoir_blobs, sm);
    cache().get_or_insert(&key, || jit.to_cubin(src, filename, opts, ltoir_blobs, sm))
}

#[cfg(test)]
pub(crate) fn populate_from_python_for_tests() {
    use std::fs;
    let Ok(output) = std::process::Command::new("python")
        .args(["-c", "import sys; print(sys.prefix)"])
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    let Ok(prefix) = std::str::from_utf8(&output.stdout) else {
        return;
    };
    let lib_dir = PathBuf::from(prefix.trim()).join("lib");
    let Ok(entries) = fs::read_dir(&lib_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let site = entry.path().join("site-packages");
        if !site.is_dir() {
            continue;
        }
        for sub in [
            "nvidia/cu13/include",
            "nvidia/cuda_runtime/include",
            "nvidia/cuda_cccl/include",
        ] {
            let p = site.join(sub);
            if p.is_dir() {
                let _ = add_include_dir(p);
            }
        }
        for sub in [
            "nvidia/cu13/lib",
            "nvidia/cu12/lib",
            "nvidia/cuda_nvrtc/lib",
            "nvidia/nvjitlink/lib",
            "nvidia/cuda_runtime/lib",
        ] {
            let p = site.join(sub);
            if p.is_dir() {
                let _ = add_lib_dir(p);
            }
        }
    }
}
