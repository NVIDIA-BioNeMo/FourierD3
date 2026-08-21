// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod nvjitlink;
mod nvrtc;

use std::path::PathBuf;

use nvrtc::NvrtcCompiler;

#[derive(Clone, Debug, Default)]
pub(crate) struct CudaCompilerConfig {
    pub lib_dirs: Vec<PathBuf>,
    pub include_dirs: Vec<PathBuf>,
}

pub(crate) struct CudaCompiler {
    nvrtc: NvrtcCompiler,
    nvjitlink: nvjitlink::NvJitLink,
    include_dirs: Vec<PathBuf>,
}

impl CudaCompiler {
    pub(crate) fn load(config: CudaCompilerConfig) -> Result<CudaCompiler, String> {
        let nvrtc = NvrtcCompiler::load(&config.lib_dirs).ok_or_else(|| {
            "libnvrtc not found. Tried libnvrtc.so, libnvrtc.so.13, \
             libnvrtc.so.12, libnvrtc.so.11 in the configured lib dirs."
                .to_string()
        })?;
        // nvJitLink is companion-loaded from the directory of the libnvrtc
        // we just mapped (via dladdr) so the two stay version-aligned, with
        // the configured lib dirs as the explicit fallback.
        let nvjitlink = nvjitlink::NvJitLink::load(&config.lib_dirs, &nvrtc).ok_or_else(|| {
            "libnvJitLink not found. Tried libnvJitLink.so.13, \
             libnvJitLink.so.12, libnvJitLink.so beside the loaded libnvrtc \
             and in the configured lib dirs. Install the matching CUDA \
             toolkit or `uv pip install nvidia-nvjitlink-cu13` (or \
             `nvidia-nvjitlink-cu12`), or add its directory to the toolchain \
             lib dirs."
                .to_string()
        })?;
        Ok(CudaCompiler {
            nvrtc,
            nvjitlink,
            include_dirs: config.include_dirs,
        })
    }

    pub(crate) fn nvrtc_version(&self) -> i32 {
        self.nvrtc.version()
    }

    pub(crate) fn to_cubin(
        &self,
        src: &[u8],
        filename: Option<&str>,
        opts: &[String],
        ltoir_blobs: &[&[u8]],
        sm: i32,
    ) -> Result<Vec<u8>, String> {
        let filename = filename.unwrap_or("module");
        if ltoir_blobs.is_empty() {
            let src = std::str::from_utf8(src).map_err(|e| e.to_string())?;
            self.nvrtc
                .cubin_from_plaintext(src, filename, sm, opts, &self.include_dirs)
        } else {
            let nvrtc_lto = self.to_ltoir(src, Some(filename), opts, sm)?;
            self.link_with_ltoir(&nvrtc_lto, filename, sm, ltoir_blobs)
        }
    }

    pub(crate) fn to_ltoir(
        &self,
        src: &[u8],
        filename: Option<&str>,
        opts: &[String],
        sm: i32,
    ) -> Result<Vec<u8>, String> {
        let filename = filename.unwrap_or("module");
        let src = std::str::from_utf8(src).map_err(|e| e.to_string())?;
        self.nvrtc
            .ltoir_from_plaintext(src, filename, sm, opts, &self.include_dirs)
    }

    pub(crate) fn cubin_key(
        &self,
        src: &[u8],
        opts: &[String],
        ltoir_blobs: &[&[u8]],
        sm: i32,
    ) -> Vec<u8> {
        let mut k: Vec<u8> = Vec::with_capacity(src.len() + 64);
        k.extend_from_slice(src);
        if ltoir_blobs.is_empty() {
            for o in opts {
                k.push(0);
                k.extend_from_slice(o.as_bytes());
            }
        } else {
            k.extend_from_slice(b"\0__ltoir\0");
            for blob in ltoir_blobs {
                k.extend_from_slice(&(blob.len() as u64).to_le_bytes());
                k.extend_from_slice(blob);
            }
            for o in opts {
                k.push(0);
                k.extend_from_slice(o.as_bytes());
            }
        }
        k.extend_from_slice(&sm.to_le_bytes());
        k.extend_from_slice(&self.nvrtc_version().to_le_bytes());
        k
    }

    fn link_with_ltoir(
        &self,
        nvrtc_lto: &[u8],
        filename: &str,
        sm: i32,
        ltoir_blobs: &[&[u8]],
    ) -> Result<Vec<u8>, String> {
        let mut inputs: Vec<nvjitlink::LtoInput> = Vec::with_capacity(1 + ltoir_blobs.len());
        inputs.push(nvjitlink::LtoInput {
            data: nvrtc_lto,
            name: filename,
        });
        for blob in ltoir_blobs.iter() {
            inputs.push(nvjitlink::LtoInput {
                data: blob,
                name: "lib_lto",
            });
        }
        // Each link runs on its own freshly-created nvJitLink handle with no
        // shared mutable state, so candidate links proceed concurrently —
        // together with the NVRTC compiles feeding them — and the FFT
        // pipeline's per-candidate `into_par_iter` actually fans out across
        // cores instead of queuing on a lock.
        self.nvjitlink.link_cubin(sm, &inputs)
    }
}
