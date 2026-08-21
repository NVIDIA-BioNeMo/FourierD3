// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::load_sym;
use libloading::Library;
use parking_lot::Mutex;
use std::ffi::{c_char, c_int, c_void};
use std::io::Read;
use std::sync::OnceLock;

use fourierd3_engine::dtype::Dtype;

type Status = c_int;
type Descriptor = i64;
type Code = i64;

const STATUS_SUCCESS: Status = 0;

const OP_SIZE: c_int = 0;
const OP_DIRECTION: c_int = 1;
const OP_TYPE: c_int = 2;
const OP_PRECISION: c_int = 3;
const OP_SM: c_int = 4;
const OP_EXECUTION: c_int = 5;
const OP_FFTS_PER_BLOCK: c_int = 6;
const OP_ELEMENTS_PER_THREAD: c_int = 7;
const OP_REAL_FFT_OPTIONS: c_int = 9;
const OP_API: c_int = 10;

const TRAIT_STORAGE_SIZE: c_int = 5;
const TRAIT_STRIDE: c_int = 6;
const TRAIT_BLOCK_DIM: c_int = 7;
const TRAIT_SHARED_MEMORY_SIZE: c_int = 8;
const TRAIT_FFTS_PER_BLOCK: c_int = 9;
const TRAIT_SYMBOL_NAME: c_int = 10;
const TRAIT_INPUT_LENGTH: c_int = 11;
const TRAIT_OUTPUT_LENGTH: c_int = 12;
const TRAIT_INPUT_EPT: c_int = 13;
const TRAIT_OUTPUT_EPT: c_int = 14;

const OPTION_TARGET_SM: c_int = 1;
const OPTION_EXTRA_NVRTC_ARGS: c_int = 4;

const PRECISION_F32: i64 = 5;
const PRECISION_F64: i64 = 6;

const EXECUTION_BLOCK: i64 = 1;

const TYPE_C2C: i64 = 0;
const TYPE_R2C: i64 = 1;
const TYPE_C2R: i64 = 2;

const DIR_FWD: i64 = 0;
const DIR_INV: i64 = 1;

const API_LMEM: i64 = 0;

const LAYOUT_NATURAL: i64 = 0;
const REAL_NORMAL: i64 = 0;

#[allow(non_snake_case)]
struct LibMathDx {
    _lib: Library,
    _builtins: Vec<Library>,
    cufftdxCreateDescriptor: unsafe extern "C" fn(*mut Descriptor) -> Status,
    cufftdxDestroyDescriptor: unsafe extern "C" fn(Descriptor) -> Status,
    cufftdxSetOperatorInt64: unsafe extern "C" fn(Descriptor, c_int, i64) -> Status,
    cufftdxSetOperatorInt64s: unsafe extern "C" fn(Descriptor, c_int, usize, *const i64) -> Status,
    cufftdxIsSupported: unsafe extern "C" fn(Descriptor, *mut c_int) -> Status,
    cufftdxGetTraitInt64: unsafe extern "C" fn(Descriptor, c_int, *mut i64) -> Status,
    cufftdxGetTraitInt64s: unsafe extern "C" fn(Descriptor, c_int, usize, *mut i64) -> Status,
    cufftdxGetTraitStrSize: unsafe extern "C" fn(Descriptor, c_int, *mut usize) -> Status,
    cufftdxGetTraitStr: unsafe extern "C" fn(Descriptor, c_int, usize, *mut c_char) -> Status,
    cufftdxFinalizeCode: unsafe extern "C" fn(Code, Descriptor) -> Status,
    cufftdxGetLTOIRSize: unsafe extern "C" fn(Descriptor, *mut usize) -> Status,
    cufftdxGetLTOIR: unsafe extern "C" fn(Descriptor, usize, *mut u8) -> Status,

    commondxCreateCode: unsafe extern "C" fn(*mut Code) -> Status,
    commondxDestroyCode: unsafe extern "C" fn(Code) -> Status,
    commondxSetCodeOptionInt64s: unsafe extern "C" fn(Code, c_int, usize, *const i64) -> Status,
    commondxSetCodeOptionStr: unsafe extern "C" fn(Code, c_int, *const c_char) -> Status,
    commondxGetLastErrorStrSize: unsafe extern "C" fn(*mut usize) -> Status,
    commondxGetLastErrorStr: unsafe extern "C" fn(*mut c_int, usize, *mut c_char) -> Status,
}

unsafe impl Send for LibMathDx {}
unsafe impl Sync for LibMathDx {}

fn preload_sibling_nvrtc_builtins(lib: &Library) -> Vec<Library> {
    let probe: libloading::Symbol<unsafe extern "C" fn(*mut Descriptor) -> Status> =
        match unsafe { lib.get(b"cufftdxCreateDescriptor\0") } {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
    let dir = match unsafe { crate::dynamic_library::loaded_lib_dir(*probe as *const c_void) } {
        Some(d) => d,
        None => return Vec::new(),
    };
    crate::dynamic_library::preload_dir_siblings(&dir, "libnvrtc-builtins.so.")
}

impl LibMathDx {
    fn load() -> Option<Self> {
        let _ = crate::kernel_compiler::cuda_toolchain::jit();
        let names = ["libmathdx.so.0", "libmathdx.so"];
        let lib_dirs = crate::kernel_compiler::cuda_toolchain::lib_dirs();
        let mut from_dirs: Vec<std::path::PathBuf> = Vec::new();
        for d in &lib_dirs {
            for n in &names {
                from_dirs.push(d.join(n));
            }
        }
        let attempts: Vec<_> = [
            from_dirs,
            names.iter().map(std::path::PathBuf::from).collect(),
        ]
        .concat();
        let lib = crate::dynamic_library::open_first(&attempts)?;

        Some(Self {
            cufftdxCreateDescriptor: load_sym!(
                &lib,
                "cufftdxCreateDescriptor",
                unsafe extern "C" fn(*mut Descriptor) -> Status
            ),
            cufftdxDestroyDescriptor: load_sym!(
                &lib,
                "cufftdxDestroyDescriptor",
                unsafe extern "C" fn(Descriptor) -> Status
            ),
            cufftdxSetOperatorInt64: load_sym!(
                &lib,
                "cufftdxSetOperatorInt64",
                unsafe extern "C" fn(Descriptor, c_int, i64) -> Status
            ),
            cufftdxSetOperatorInt64s: load_sym!(
                &lib,
                "cufftdxSetOperatorInt64s",
                unsafe extern "C" fn(Descriptor, c_int, usize, *const i64) -> Status
            ),
            cufftdxIsSupported: load_sym!(
                &lib,
                "cufftdxIsSupported",
                unsafe extern "C" fn(Descriptor, *mut c_int) -> Status
            ),
            cufftdxGetTraitInt64: load_sym!(
                &lib,
                "cufftdxGetTraitInt64",
                unsafe extern "C" fn(Descriptor, c_int, *mut i64) -> Status
            ),
            cufftdxGetTraitInt64s: load_sym!(
                &lib,
                "cufftdxGetTraitInt64s",
                unsafe extern "C" fn(Descriptor, c_int, usize, *mut i64) -> Status
            ),
            cufftdxGetTraitStrSize: load_sym!(
                &lib,
                "cufftdxGetTraitStrSize",
                unsafe extern "C" fn(Descriptor, c_int, *mut usize) -> Status
            ),
            cufftdxGetTraitStr: load_sym!(
                &lib,
                "cufftdxGetTraitStr",
                unsafe extern "C" fn(Descriptor, c_int, usize, *mut c_char) -> Status
            ),
            cufftdxFinalizeCode: load_sym!(
                &lib,
                "cufftdxFinalizeCode",
                unsafe extern "C" fn(Code, Descriptor) -> Status
            ),
            cufftdxGetLTOIRSize: load_sym!(
                &lib,
                "cufftdxGetLTOIRSize",
                unsafe extern "C" fn(Descriptor, *mut usize) -> Status
            ),
            cufftdxGetLTOIR: load_sym!(
                &lib,
                "cufftdxGetLTOIR",
                unsafe extern "C" fn(Descriptor, usize, *mut u8) -> Status
            ),
            commondxCreateCode: load_sym!(
                &lib,
                "commondxCreateCode",
                unsafe extern "C" fn(*mut Code) -> Status
            ),
            commondxDestroyCode: load_sym!(
                &lib,
                "commondxDestroyCode",
                unsafe extern "C" fn(Code) -> Status
            ),
            commondxSetCodeOptionInt64s: load_sym!(
                &lib,
                "commondxSetCodeOptionInt64s",
                unsafe extern "C" fn(Code, c_int, usize, *const i64) -> Status
            ),
            commondxSetCodeOptionStr: load_sym!(
                &lib,
                "commondxSetCodeOptionStr",
                unsafe extern "C" fn(Code, c_int, *const c_char) -> Status
            ),
            commondxGetLastErrorStrSize: load_sym!(
                &lib,
                "commondxGetLastErrorStrSize",
                unsafe extern "C" fn(*mut usize) -> Status
            ),
            commondxGetLastErrorStr: load_sym!(
                &lib,
                "commondxGetLastErrorStr",
                unsafe extern "C" fn(*mut c_int, usize, *mut c_char) -> Status
            ),
            _builtins: preload_sibling_nvrtc_builtins(&lib),
            _lib: lib,
        })
    }

    fn get() -> &'static Self {
        static INST: OnceLock<LibMathDx> = OnceLock::new();
        INST.get_or_init(|| {
            const INSTALL_HELP: &str = "libmathdx.so.0 not found. Run `pip install \
                 nvidia-libmathdx-cu13` (this is a transitive dep of the \
                 FourierD3 wheel). Otherwise register its directory through \
                 the toolchain API before the first compile.";
            Self::load().expect(INSTALL_HELP)
        })
    }

    fn last_error(&self) -> String {
        let mut n: usize = 0;
        let s = unsafe { (self.commondxGetLastErrorStrSize)(&mut n) };
        if s != STATUS_SUCCESS || n == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; n];
        let mut code: c_int = 0;
        let s = unsafe {
            (self.commondxGetLastErrorStr)(&mut code, n, buf.as_mut_ptr() as *mut c_char)
        };
        if s != STATUS_SUCCESS {
            return String::new();
        }
        if let Some(p) = buf.iter().position(|&b| b == 0) {
            buf.truncate(p);
        }
        String::from_utf8_lossy(&buf).into_owned()
    }
}

// libmathdx is not reentrant for descriptor mutation across threads; serialise.
static LIBMATHDX_LOCK: Mutex<()> = Mutex::new(());

fn precision_to_mathdx(precision: Dtype) -> i64 {
    match precision {
        Dtype::F32 => PRECISION_F32,
        Dtype::F64 => PRECISION_F64,
        other => panic!("libmathdx precision must be f32 or f64, got {other:?}"),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FftDirection {
    Forward,
    Inverse,
}

impl FftDirection {
    fn to_mathdx(self) -> i64 {
        match self {
            FftDirection::Forward => DIR_FWD,
            FftDirection::Inverse => DIR_INV,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FftType {
    C2C,
    R2C,
    C2R,
}

impl FftType {
    fn to_mathdx(self) -> i64 {
        match self {
            FftType::C2C => TYPE_C2C,
            FftType::R2C => TYPE_R2C,
            FftType::C2R => TYPE_C2R,
        }
    }
    fn is_real(self) -> bool {
        matches!(self, FftType::R2C | FftType::C2R)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FftSpec {
    pub size: u32,
    pub ty: FftType,
    pub direction: FftDirection,
    pub precision: Dtype,
    // SM encoded as `10*major + minor` (e.g. 120 for sm_120); rescaled to
    // `100*major + 10*minor + patch` at the libmathdx call site.
    pub sm: u32,
    pub ept: Option<u32>,
    pub fpb: Option<u32>,
}

impl FftSpec {
    #[cfg(test)]
    pub(crate) fn r2c_f32(size: u32, sm: u32) -> Self {
        Self {
            size,
            ty: FftType::R2C,
            direction: FftDirection::Forward,
            precision: Dtype::F32,
            sm,
            ept: None,
            fpb: None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct CufftdxFft {
    ltoir: Vec<u8>,
    symbol: String,
    shared_mem_bytes: u32,
    block_dim: [u32; 3],
    storage_size: u32,
    stride: u32,
    input_length: u32,
    output_length: u32,
    input_ept: u32,
    output_ept: u32,
    ffts_per_block: u32,
}

impl CufftdxFft {
    pub(crate) fn build(spec: &FftSpec) -> Result<Self, String> {
        let key = ltoir_cache_key(spec);
        let bytes = crate::kernel_compiler::cuda_toolchain::cache().get_or_insert(&key, || {
            Ok::<_, String>(Self::finalize(spec)?.to_cache_bytes())
        })?;
        Self::from_cache_bytes(&bytes)
            .ok_or_else(|| "corrupt cuFFTDx LTOIR cache entry".to_string())
    }

    fn finalize(spec: &FftSpec) -> Result<Self, String> {
        let lib = LibMathDx::get();
        let _g = LIBMATHDX_LOCK.lock();

        let chk = |status: Status, where_: &str| -> Result<(), String> {
            if status == STATUS_SUCCESS {
                return Ok(());
            }
            let mut msg = format!("libmathdx {where_} failed (status={status})");
            let detail = lib.last_error();
            if !detail.is_empty() {
                msg.push_str(": ");
                msg.push_str(&detail);
            }
            Err(msg)
        };

        let mut desc: Descriptor = 0;
        chk(
            unsafe { (lib.cufftdxCreateDescriptor)(&mut desc) },
            "create",
        )?;
        struct DescGuard<'a> {
            d: Descriptor,
            lib: &'a LibMathDx,
        }
        impl Drop for DescGuard<'_> {
            fn drop(&mut self) {
                unsafe { (self.lib.cufftdxDestroyDescriptor)(self.d) };
            }
        }
        let _desc_guard = DescGuard { d: desc, lib };

        let setop = |op: c_int, v: i64, name: &str| {
            chk(unsafe { (lib.cufftdxSetOperatorInt64)(desc, op, v) }, name)
        };
        // Engine encodes SM as `10*major + minor`; libmathdx wants `100*major + 10*minor + patch`.
        let mathdx_sm = (spec.sm as i64) * 10;
        setop(OP_API, API_LMEM, "setop(API)")?;
        setop(OP_SM, mathdx_sm, "setop(SM)")?;
        setop(OP_SIZE, spec.size as i64, "setop(SIZE)")?;
        setop(
            OP_PRECISION,
            precision_to_mathdx(spec.precision),
            "setop(PRECISION)",
        )?;
        setop(OP_TYPE, spec.ty.to_mathdx(), "setop(TYPE)")?;
        setop(OP_DIRECTION, spec.direction.to_mathdx(), "setop(DIRECTION)")?;

        if spec.ty.is_real() {
            let rf: [i64; 2] = [LAYOUT_NATURAL, REAL_NORMAL];
            chk(
                unsafe {
                    (lib.cufftdxSetOperatorInt64s)(desc, OP_REAL_FFT_OPTIONS, 2, rf.as_ptr())
                },
                "setop(REAL_FFT_OPTIONS)",
            )?;
        }

        setop(OP_EXECUTION, EXECUTION_BLOCK, "setop(EXECUTION)")?;
        if let Some(fpb) = spec.fpb {
            setop(OP_FFTS_PER_BLOCK, fpb as i64, "setop(FPB)")?;
        }
        if let Some(ept) = spec.ept {
            setop(OP_ELEMENTS_PER_THREAD, ept as i64, "setop(EPT)")?;
        }

        let mut supported: c_int = 0;
        chk(
            unsafe { (lib.cufftdxIsSupported)(desc, &mut supported) },
            "is_supported",
        )?;
        if supported == 0 {
            return Err(crate::kernel_compiler::infeasibility::infeasible(format!(
                "unsupported FftSpec {spec:?}"
            )));
        }

        let mut code: Code = 0;
        chk(
            unsafe { (lib.commondxCreateCode)(&mut code) },
            "code create",
        )?;
        struct CodeGuard<'a> {
            c: Code,
            lib: &'a LibMathDx,
        }
        impl Drop for CodeGuard<'_> {
            fn drop(&mut self) {
                unsafe { (self.lib.commondxDestroyCode)(self.c) };
            }
        }
        let _code_guard = CodeGuard { c: code, lib };

        let sm_arr: [i64; 1] = [mathdx_sm];
        chk(
            unsafe {
                (lib.commondxSetCodeOptionInt64s)(code, OPTION_TARGET_SM, 1, sm_arr.as_ptr())
            },
            "set TARGET_SM",
        )?;

        const NVRTC_ARGS: &[u8] = b"-gen-opt-lto\0";
        chk(
            unsafe {
                (lib.commondxSetCodeOptionStr)(
                    code,
                    OPTION_EXTRA_NVRTC_ARGS,
                    NVRTC_ARGS.as_ptr() as *const c_char,
                )
            },
            "set EXTRA_NVRTC_ARGS",
        )?;
        chk(unsafe { (lib.cufftdxFinalizeCode)(code, desc) }, "finalize")?;

        let mut lto_size: usize = 0;
        chk(
            unsafe { (lib.cufftdxGetLTOIRSize)(desc, &mut lto_size) },
            "lto size",
        )?;
        let mut ltoir = vec![0u8; lto_size];
        chk(
            unsafe { (lib.cufftdxGetLTOIR)(desc, lto_size, ltoir.as_mut_ptr()) },
            "lto fetch",
        )?;

        let trait_int = |t: c_int, name: &str| -> Result<i64, String> {
            let mut v: i64 = 0;
            chk(unsafe { (lib.cufftdxGetTraitInt64)(desc, t, &mut v) }, name)?;
            Ok(v)
        };
        let trait_str = |t: c_int, name: &str| -> Result<String, String> {
            let mut n: usize = 0;
            chk(
                unsafe { (lib.cufftdxGetTraitStrSize)(desc, t, &mut n) },
                name,
            )?;
            let mut buf = vec![0u8; n];
            chk(
                unsafe { (lib.cufftdxGetTraitStr)(desc, t, n, buf.as_mut_ptr() as *mut c_char) },
                name,
            )?;
            if let Some(p) = buf.iter().position(|&b| b == 0) {
                buf.truncate(p);
            }
            Ok(String::from_utf8_lossy(&buf).into_owned())
        };

        let mut bd: [i64; 3] = [0; 3];
        chk(
            unsafe { (lib.cufftdxGetTraitInt64s)(desc, TRAIT_BLOCK_DIM, 3, bd.as_mut_ptr()) },
            "trait BLOCK_DIM",
        )?;

        let out = CufftdxFft {
            ltoir,
            symbol: trait_str(TRAIT_SYMBOL_NAME, "trait SYMBOL_NAME")?,
            shared_mem_bytes: trait_int(TRAIT_SHARED_MEMORY_SIZE, "trait SHARED_MEMORY_SIZE")?
                as u32,
            block_dim: [bd[0] as u32, bd[1] as u32, bd[2] as u32],
            storage_size: trait_int(TRAIT_STORAGE_SIZE, "trait STORAGE_SIZE")? as u32,
            stride: trait_int(TRAIT_STRIDE, "trait STRIDE")? as u32,
            input_length: trait_int(TRAIT_INPUT_LENGTH, "trait INPUT_LENGTH")? as u32,
            output_length: trait_int(TRAIT_OUTPUT_LENGTH, "trait OUTPUT_LENGTH")? as u32,
            input_ept: trait_int(TRAIT_INPUT_EPT, "trait INPUT_EPT")? as u32,
            output_ept: trait_int(TRAIT_OUTPUT_EPT, "trait OUTPUT_EPT")? as u32,
            ffts_per_block: trait_int(TRAIT_FFTS_PER_BLOCK, "trait FFTS_PER_BLOCK")? as u32,
        };

        Ok(out)
    }

    pub(crate) fn build_candidates(
        base: &FftSpec,
        smem_budget: u32,
        fpb_cap: u32,
        order: &[(u32, u32)],
        cap: usize,
    ) -> Result<Vec<Self>, String> {
        let size = base.size;
        let epts: Vec<u32> = [1, 2, 4, 8, 16, 32, 64, 128]
            .into_iter()
            .filter(|&e| e <= size && size.is_multiple_of(e))
            .collect();
        let fpbs: Vec<u32> = [1, 2, 4, 8, 16, 32, 64, 128]
            .into_iter()
            .filter(|&f| f <= fpb_cap.max(1))
            .collect();

        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for &ept in &epts {
            for &fpb in &fpbs {
                pairs.push((ept, fpb));
            }
        }
        pairs.sort_by_key(|&(ept, fpb)| {
            let pos = order
                .iter()
                .position(|&o| o == (ept, fpb))
                .unwrap_or(usize::MAX);
            (pos, ept, fpb)
        });

        let mut out: Vec<CufftdxFft> = Vec::new();
        let mut seen: Vec<(u32, u32)> = Vec::new();
        for (ept, fpb) in pairs {
            if out.len() >= cap {
                break;
            }
            let spec = FftSpec {
                ept: Some(ept),
                fpb: Some(fpb),
                ..*base
            };
            let Ok(f) = Self::build(&spec) else { continue };
            if f.shared_mem_bytes > smem_budget {
                continue;
            }
            let key = (f.input_ept, f.ffts_per_block);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            out.push(f);
        }
        if out.is_empty() {
            out.push(Self::build(&FftSpec {
                ept: None,
                fpb: None,
                ..*base
            })?);
        }
        Ok(out)
    }

    pub(crate) fn ltoir(&self) -> &[u8] {
        &self.ltoir
    }
    pub(crate) fn symbol_name(&self) -> &str {
        &self.symbol
    }
    pub(crate) fn shared_mem_bytes(&self) -> u32 {
        self.shared_mem_bytes
    }
    pub(crate) fn block_dim(&self) -> [u32; 3] {
        self.block_dim
    }
    pub(crate) fn stride(&self) -> u32 {
        self.stride
    }
    #[cfg(test)]
    pub(crate) fn input_length(&self) -> u32 {
        self.input_length
    }
    #[cfg(test)]
    pub(crate) fn output_length(&self) -> u32 {
        self.output_length
    }
    pub(crate) fn input_ept(&self) -> u32 {
        self.input_ept
    }
    pub(crate) fn ffts_per_block(&self) -> u32 {
        self.ffts_per_block
    }

    fn to_cache_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.ltoir.len() + self.symbol.len() + 64);
        b.extend_from_slice(LTOIR_CACHE_MAGIC);
        for v in [
            self.shared_mem_bytes,
            self.block_dim[0],
            self.block_dim[1],
            self.block_dim[2],
            self.storage_size,
            self.stride,
            self.input_length,
            self.output_length,
            self.input_ept,
            self.output_ept,
            self.ffts_per_block,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&(self.symbol.len() as u64).to_le_bytes());
        b.extend_from_slice(self.symbol.as_bytes());
        b.extend_from_slice(&(self.ltoir.len() as u64).to_le_bytes());
        b.extend_from_slice(&self.ltoir);
        b
    }

    fn from_cache_bytes(bytes: &[u8]) -> Option<Self> {
        let mut c = std::io::Cursor::new(bytes);
        let mut magic = [0u8; LTOIR_CACHE_MAGIC.len()];
        c.read_exact(&mut magic).ok()?;
        if magic != *LTOIR_CACHE_MAGIC {
            return None;
        }
        let mut u32_le = || -> Option<u32> {
            let mut buf = [0u8; 4];
            c.read_exact(&mut buf).ok()?;
            Some(u32::from_le_bytes(buf))
        };
        let shared_mem_bytes = u32_le()?;
        let block_dim = [u32_le()?, u32_le()?, u32_le()?];
        let storage_size = u32_le()?;
        let stride = u32_le()?;
        let input_length = u32_le()?;
        let output_length = u32_le()?;
        let input_ept = u32_le()?;
        let output_ept = u32_le()?;
        let ffts_per_block = u32_le()?;
        let mut read_blob = || -> Option<Vec<u8>> {
            let mut len = [0u8; 8];
            c.read_exact(&mut len).ok()?;
            let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
            c.read_exact(&mut buf).ok()?;
            Some(buf)
        };
        let symbol = String::from_utf8(read_blob()?).ok()?;
        let ltoir = read_blob()?;
        Some(CufftdxFft {
            ltoir,
            symbol,
            shared_mem_bytes,
            block_dim,
            storage_size,
            stride,
            input_length,
            output_length,
            input_ept,
            output_ept,
            ffts_per_block,
        })
    }
}

const LTOIR_CACHE_MAGIC: &[u8; 8] = b"LTOIRc01";

fn ltoir_cache_key(spec: &FftSpec) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.extend_from_slice(b"fft");
    buf.extend_from_slice(&spec.size.to_le_bytes());
    buf.extend_from_slice(&spec.ty.to_mathdx().to_le_bytes());
    buf.extend_from_slice(&spec.direction.to_mathdx().to_le_bytes());
    buf.extend_from_slice(&precision_to_mathdx(spec.precision).to_le_bytes());
    buf.extend_from_slice(&spec.sm.to_le_bytes());
    buf.push(spec.ept.is_some() as u8);
    buf.extend_from_slice(&spec.ept.unwrap_or(0).to_le_bytes());
    buf.push(spec.fpb.is_some() as u8);
    buf.extend_from_slice(&spec.fpb.unwrap_or(0).to_le_bytes());
    buf.extend_from_slice(
        &crate::kernel_compiler::cuda_toolchain::jit()
            .nvrtc_version()
            .to_le_bytes(),
    );
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn libmathdx_available() -> bool {
        crate::kernel_compiler::cuda_toolchain::populate_from_python_for_tests();
        LibMathDx::load().is_some()
    }

    #[test]
    fn r2c_size_128_f32_supported_sms() {
        if !libmathdx_available() {
            eprintln!("skip: libmathdx not loadable in this environment");
            return;
        }
        for sm in [90, 120] {
            let spec = FftSpec::r2c_f32(128, sm);
            let fft = CufftdxFft::build(&spec).expect("build");
            assert!(!fft.ltoir().is_empty());
            assert!(fft.symbol_name().starts_with("cufftdx_execute_"));
            assert_eq!(fft.input_length(), 128);
            assert_eq!(fft.output_length(), 65);
            assert_eq!(fft.block_dim()[0], fft.stride());
        }
    }

    #[test]
    fn large_prime_unsupported() {
        if !libmathdx_available() {
            eprintln!("skip: libmathdx not loadable in this environment");
            return;
        }
        let mut spec = FftSpec::r2c_f32(127, 120);
        spec.ty = FftType::C2C;
        spec.direction = FftDirection::Forward;
        let err = match CufftdxFft::build(&spec) {
            Ok(_) => panic!("size 127 must be rejected"),
            Err(e) => e,
        };
        assert!(err.contains("unsupported"), "got: {err}");
    }

    #[test]
    fn c2c_inverse_size_64_f32() {
        if !libmathdx_available() {
            eprintln!("skip: libmathdx not loadable in this environment");
            return;
        }
        let spec = FftSpec {
            size: 64,
            ty: FftType::C2C,
            direction: FftDirection::Inverse,
            precision: Dtype::F32,
            sm: 120,
            ept: None,
            fpb: None,
        };
        let fft = CufftdxFft::build(&spec).expect("build");
        assert_eq!(fft.input_length(), 64);
        assert_eq!(fft.output_length(), 64);
    }

    #[test]
    fn smooth_nonpow2_size_60_works() {
        if !libmathdx_available() {
            eprintln!("skip: libmathdx not loadable in this environment");
            return;
        }
        let fft = CufftdxFft::build(&FftSpec::r2c_f32(60, 120)).expect("build");
        assert_eq!(fft.input_length(), 60);
        assert_eq!(fft.output_length(), 31);
    }
}
