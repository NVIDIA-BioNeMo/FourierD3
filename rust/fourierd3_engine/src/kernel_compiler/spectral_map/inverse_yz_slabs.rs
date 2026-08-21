// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kernel_compiler::libmathdx::{CufftdxFft, FftDirection, FftSpec, FftType};
use crate::kernel_compiler::spectral_map::specification::{complex_bytes, complex_ctype};
use fourierd3_engine::cuda;
use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::{Param, Stmt};

pub(crate) struct YzInvSlabEmit {
    pub source: Vec<Stmt>,
    pub ltoir: Vec<Vec<u8>>,
    pub kernel_name: String,
    pub block_dim: [u32; 3],
    pub shared_bytes: u32,
    pub grid_size: u32,
}

pub(crate) struct YzInvSlabSpec {
    pub total_slabs: u32,
    pub inner_size: u32,
    pub fft_lengths: [u32; 3],
    pub precision: Dtype,
    pub sm: u32,
    pub max_smem_bytes: u32,
    pub kernel_name: String,
    pub max_candidates: usize,
}

pub(crate) fn emit_variants(spec: &YzInvSlabSpec) -> Result<Vec<YzInvSlabEmit>, String> {
    let [_nx, ny, nz] = spec.fft_lengths;
    if ny != nz {
        return Err(format!(
            "yz_inv_slab requires ny == nz, got ny={ny} nz={nz}"
        ));
    }
    let nz_half = nz / 2 + 1;

    let per_chunk_bytes = ny * nz_half * complex_bytes(spec.precision);
    let fft_smem_budget = spec.max_smem_bytes.saturating_sub(per_chunk_bytes);
    let c2c_ffts = CufftdxFft::build_candidates(
        &FftSpec {
            size: ny,
            ty: FftType::C2C,
            direction: FftDirection::Inverse,
            precision: spec.precision,
            sm: spec.sm,
            ept: None,
            fpb: None,
        },
        fft_smem_budget,
        64,
        crate::kernel_compiler::spectral_map::candidate_budget::INV_YZ_ORDER,
        spec.max_candidates,
    )?;
    c2c_ffts
        .iter()
        .map(|c2c| {
            let c2r = CufftdxFft::build(&FftSpec {
                size: ny,
                ty: FftType::C2R,
                direction: FftDirection::Inverse,
                precision: spec.precision,
                sm: spec.sm,
                ept: Some(c2c.input_ept()),
                fpb: Some(c2c.ffts_per_block()),
            })?;
            emit_config(spec, c2c, &c2r)
        })
        .collect()
}

fn emit_config(
    spec: &YzInvSlabSpec,
    c2c: &CufftdxFft,
    c2r: &CufftdxFft,
) -> Result<YzInvSlabEmit, String> {
    let [_nx, ny, nz] = spec.fft_lengths;
    let nz_half = nz / 2 + 1;
    let per_chunk_bytes = ny * nz_half * complex_bytes(spec.precision);

    if c2r.block_dim() != c2c.block_dim() || c2r.stride() != c2c.stride() {
        return Err(crate::kernel_compiler::infeasibility::infeasible(format!(
            "yz_inv_slab: C2C/C2R geometry mismatch (c2r bd={:?} stride={}, c2c bd={:?} stride={})",
            c2r.block_dim(),
            c2r.stride(),
            c2c.block_dim(),
            c2c.stride(),
        )));
    }

    let ept = c2c.input_ept() as i64;
    let stride = c2c.stride() as i64;
    let fpb = c2c.ffts_per_block() as i64;
    let block_dim = c2c.block_dim();
    let y_symbol = c2c.symbol_name().to_string();
    let z_symbol = c2r.symbol_name().to_string();
    let fft_scratch_bytes = c2c.shared_mem_bytes();

    let real_t = spec.precision.ctype();
    let complex_t = complex_ctype(spec.precision);
    let mk = format!("make_{complex_t}");
    let ct_ptr = format!("{complex_t}*");
    let rt_ptr = format!("{real_t}*");
    let char_t = "char";
    let char_ptr = "char*";

    let smem_budget = spec
        .max_smem_bytes
        .checked_sub(fft_scratch_bytes)
        .ok_or_else(|| {
            crate::kernel_compiler::infeasibility::infeasible(format!(
                "yz_inv_slab: FFT scratch ({fft_scratch_bytes}B) exceeds max_smem_bytes ({})",
                spec.max_smem_bytes
            ))
        })?;
    let max_chunk = (smem_budget / per_chunk_bytes).max(1);
    let chunk = spec.inner_size.min(max_chunk);
    let staging_bytes = per_chunk_bytes * chunk;
    let shared_bytes = staging_bytes + fft_scratch_bytes;
    if shared_bytes > spec.max_smem_bytes {
        return Err(crate::kernel_compiler::infeasibility::infeasible(format!(
            "yz_inv_slab: needs {shared_bytes}B smem for chunk=1, budget {}",
            spec.max_smem_bytes
        )));
    }

    let staging_bytes_i = staging_bytes as i64;
    let total_slabs_i = spec.total_slabs as i64;
    let in_slab_elems = (ny * nz_half * spec.inner_size) as i64;
    let out_slab_elems = (ny * nz * spec.inner_size) as i64;
    let ny_i = ny as i64;
    let nz_i = nz as i64;
    let nz_half_i = nz_half as i64;
    let inner_size_i = spec.inner_size as i64;
    let zero = Expr::call(
        mk.clone(),
        vec![
            Expr::call(real_t, vec![Expr::lit(0)]),
            Expr::call(real_t, vec![Expr::lit(0)]),
        ],
    );

    let params = vec![
        Param::Pointer {
            const_: true,
            restrict: true,
            ctype: complex_t.into(),
            name: String::from("input"),
        },
        Param::Pointer {
            const_: false,
            restrict: true,
            ctype: real_t.into(),
            name: String::from("output"),
        },
    ];

    let mut prelude: Vec<Stmt> = Vec::new();
    cuda! { prelude =>
        extern "C" __device__ void #y_symbol(#ct_ptr, #char_ptr);
        extern "C" __device__ void #z_symbol(#ct_ptr, #char_ptr);
    }

    let mut body: Vec<Stmt> = Vec::new();
    cuda! { body =>
        extern __shared__ #char_t _shmem[];
        #ct_ptr staging = (#ct_ptr)_shmem;
        #char_ptr fft_scratch = _shmem + #staging_bytes_i;
        ;
        i32 slab_id = blockIdx.x;
        if (slab_id >= #total_slabs_i) return;
        ;
        i64 slab_in = (i64)slab_id * #in_slab_elems;
        i64 slab_out = (i64)slab_id * #out_slab_elems;
        #complex_t thread_data[#ept];
        #rt_ptr z_real = (#rt_ptr)thread_data;
        #real_t inv_ny = (#real_t)1 / (#real_t)#ny_i;
        #real_t inv_nz = (#real_t)1 / (#real_t)#nz_i;
    }

    let mut ic_off: u32 = 0;
    while ic_off < spec.inner_size {
        let cisz = chunk.min(spec.inner_size - ic_off) as i64;
        let ic_off_i = ic_off as i64;
        let n_y_ffts = nz_half_i * cisz;
        let n_z_ffts = ny_i * cisz;
        let nz_half_cisz = nz_half_i * cisz;
        let ny_inner = ny_i * inner_size_i;

        cuda! { body =>
            ;
            for (i32 g = 0; g < #n_y_ffts; g += #fpb) {
                i32 fft_idx = g + threadIdx.y;
                i32 kz = fft_idx / #cisz;
                i32 m = fft_idx % #cisz;
                unroll for (i32 i = 0; i < #ept; i++) {
                    i32 pos = threadIdx.x + i * #stride;
                    if (fft_idx < #n_y_ffts && pos < #ny_i) {
                        thread_data[i] = input[slab_in + kz * #ny_inner + (#ic_off_i + m) * #ny_i + pos];
                    } else {
                        thread_data[i] = #zero;
                    }
                }
                #y_symbol(thread_data, fft_scratch);
                if (fft_idx < #n_y_ffts) {
                    unroll for (i32 i = 0; i < #ept; i++) {
                        i32 pos = threadIdx.x + i * #stride;
                        if (pos < #ny_i) {
                            staging[pos * #nz_half_cisz + kz * #cisz + m] =
                                #mk(thread_data[i].x * inv_ny, thread_data[i].y * inv_ny);
                        }
                    }
                }
            }
            __syncthreads();
            ;
            for (i32 g = 0; g < #n_z_ffts; g += #fpb) {
                i32 fft_idx = g + threadIdx.y;
                i32 y = fft_idx / #cisz;
                i32 m = fft_idx % #cisz;
                unroll for (i32 i = 0; i < #ept; i++) {
                    i32 kz = threadIdx.x + i * #stride;
                    if (fft_idx < #n_z_ffts && kz < #nz_half_i) {
                        thread_data[i] = staging[y * #nz_half_cisz + kz * #cisz + m];
                    } else {
                        thread_data[i] = #zero;
                    }
                }
                #z_symbol(thread_data, fft_scratch);
                if (fft_idx < #n_z_ffts) {
                    unroll for (i32 i = 0; i < #ept; i++) {
                        i32 pos = threadIdx.x + i * #stride;
                        if (pos < #nz_i) {
                            output[slab_out + y * #nz_i * #inner_size_i + pos * #inner_size_i + #ic_off_i + m]
                                = z_real[i] * inv_nz;
                        }
                    }
                }
            }
            __syncthreads();
        }
        ic_off += cisz as u32;
    }

    let kernel_name = format!("{}_e{ept}_f{fpb}", spec.kernel_name);
    let mut module = prelude;
    module.push(Stmt::Kernel {
        name: kernel_name.clone(),
        launch_bounds: String::new(),
        params,
        body,
    });

    Ok(YzInvSlabEmit {
        source: module,
        ltoir: vec![c2c.ltoir().to_vec(), c2r.ltoir().to_vec()],
        kernel_name,
        block_dim,
        shared_bytes,
        grid_size: spec.total_slabs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn libmathdx_available() -> bool {
        crate::kernel_compiler::cuda_toolchain::populate_from_python_for_tests();
        crate::kernel_compiler::libmathdx::CufftdxFft::build(
            &crate::kernel_compiler::libmathdx::FftSpec::r2c_f32(8, 120),
        )
        .is_ok()
    }

    #[test]
    fn renders() {
        if !libmathdx_available() {
            return;
        }
        let variants = emit_variants(&YzInvSlabSpec {
            total_slabs: 16,
            inner_size: 1,
            max_smem_bytes: 65536,
            fft_lengths: [16, 16, 16],
            precision: Dtype::F32,
            sm: 120,
            kernel_name: "yz_inv_test".into(),
            max_candidates:
                crate::kernel_compiler::spectral_map::candidate_budget::DEFAULT_MAX_CANDIDATES,
        })
        .expect("emit_variants");
        let out = variants.into_iter().next().expect("≥1 variant");
        let src = fourierd3_engine::ir::stmt::render_module_string(&out.source);
        assert!(src.contains("yz_inv_test"));
        assert!(src.contains("extern \"C\" __device__ void cufftdx_execute_"));
        assert!(src.contains("inv_ny"));
        assert!(src.contains("inv_nz"));
        let calls = src.matches("(thread_data, fft_scratch);").count();
        assert_eq!(calls, 2, "expected 2 FFT execute calls\n{src}");
    }

    #[test]
    fn compiles_via_nvrtc_and_links_ltoir() {
        if !libmathdx_available() {
            return;
        }
        if crate::cuda_driver::ensure_context().is_err() {
            return;
        }
        let sm = crate::cuda_driver::Device::current().sm_arch().unwrap_or(0);
        if sm == 0 {
            return;
        }
        let variants = emit_variants(&YzInvSlabSpec {
            total_slabs: 16,
            inner_size: 1,
            max_smem_bytes: 65536,
            fft_lengths: [16, 16, 16],
            precision: Dtype::F32,
            sm: sm as u32,
            kernel_name: "yz_inv_smoke".into(),
            max_candidates:
                crate::kernel_compiler::spectral_map::candidate_budget::DEFAULT_MAX_CANDIDATES,
        })
        .expect("emit_variants");
        let out = variants.into_iter().next().expect("≥1 variant");
        let src = fourierd3_engine::ir::stmt::render_module_string(&out.source);
        let ltoir: Vec<&[u8]> = out.ltoir.iter().map(|b| b.as_slice()).collect();
        crate::kernel_compiler::cuda_toolchain::compile_cubin(
            src.as_bytes(),
            Some(out.kernel_name.as_str()),
            &[],
            &ltoir,
        )
        .expect("compile");
    }
}
