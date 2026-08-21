// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kernel_compiler::buffer::Buffer;
use crate::kernel_compiler::libmathdx::{CufftdxFft, FftDirection, FftSpec, FftType};
use crate::kernel_compiler::spectral_map::specification::{complex_bytes, complex_ctype};
use crate::kernel_compiler::spectral_map::{batch_offset_expr, push_batch_decompose, zero_complex};
use fourierd3_engine::cuda;
use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::stmt::{Param, Stmt};

pub(crate) struct ZyFwdSlabEmit {
    pub source: Vec<Stmt>,
    pub ltoir: Vec<Vec<u8>>,
    pub kernel_name: String,
    pub block_dim: [u32; 3],
    pub shared_bytes: u32,
    pub grid_size: u32,
}

pub(crate) struct ZyFwdSlabSpec<'a> {
    pub batch_shape: &'a [i64],
    pub grid_in_bufs: &'a [&'a Buffer],
    pub inner_size: u32,
    pub fft_lengths: [u32; 3],
    pub precision: Dtype,
    pub sm: u32,
    pub max_smem_bytes: u32,
    pub kernel_name: String,
    pub max_candidates: usize,
}

pub(crate) fn emit_variants(spec: &ZyFwdSlabSpec) -> Result<Vec<ZyFwdSlabEmit>, String> {
    let [_nx, ny, nz] = spec.fft_lengths;
    if ny != nz {
        return Err(format!(
            "zy_fwd_slab requires ny == nz, got ny={ny} nz={nz}"
        ));
    }
    let n = ny;
    let nz_half = nz / 2 + 1;
    let n_grids = spec.grid_in_bufs.len() as u32;

    let per_chunk_bytes = n_grids * ny * nz_half * complex_bytes(spec.precision);
    let fft_smem_budget = spec.max_smem_bytes.saturating_sub(per_chunk_bytes);
    let c2c_ffts = CufftdxFft::build_candidates(
        &FftSpec {
            size: n,
            ty: FftType::C2C,
            direction: FftDirection::Forward,
            precision: spec.precision,
            sm: spec.sm,
            ept: None,
            fpb: None,
        },
        fft_smem_budget,
        64,
        crate::kernel_compiler::spectral_map::candidate_budget::FWD_ZY_ORDER,
        spec.max_candidates,
    )?;
    c2c_ffts
        .iter()
        .map(|c2c| {
            let r2c = CufftdxFft::build(&FftSpec {
                size: n,
                ty: FftType::R2C,
                direction: FftDirection::Forward,
                precision: spec.precision,
                sm: spec.sm,
                ept: Some(c2c.input_ept()),
                fpb: Some(c2c.ffts_per_block()),
            })?;
            emit_config(spec, c2c, &r2c)
        })
        .collect()
}

fn emit_config(
    spec: &ZyFwdSlabSpec,
    c2c: &CufftdxFft,
    r2c: &CufftdxFft,
) -> Result<ZyFwdSlabEmit, String> {
    let [nx, ny, nz] = spec.fft_lengths;
    let n = ny;
    let nz_half = nz / 2 + 1;
    let n_grids = spec.grid_in_bufs.len() as u32;
    let per_chunk_bytes = n_grids * ny * nz_half * complex_bytes(spec.precision);

    // R2C and C2C at the same (ept, fpb) must agree on thread geometry for one
    // block to run both phases; bail to the reference path if cuFFTDx ever
    // diverges them.
    if r2c.block_dim() != c2c.block_dim() || r2c.stride() != c2c.stride() {
        return Err(crate::kernel_compiler::infeasibility::infeasible(format!(
            "zy_fwd_slab: R2C/C2C geometry mismatch (r2c bd={:?} stride={}, c2c bd={:?} stride={})",
            r2c.block_dim(),
            r2c.stride(),
            c2c.block_dim(),
            c2c.stride(),
        )));
    }

    let ept = c2c.input_ept() as i64;
    let stride = c2c.stride() as i64;
    let fpb = c2c.ffts_per_block() as i64;
    let block_dim = c2c.block_dim();
    let z_symbol = r2c.symbol_name().to_string();
    let y_symbol = c2c.symbol_name().to_string();
    let fft_scratch_bytes = c2c.shared_mem_bytes();
    let real_t = spec.precision.ctype();
    let complex_t = complex_ctype(spec.precision);
    let ct_ptr = format!("{complex_t}*");
    let rt_ptr = format!("{real_t}*");
    let char_t = "char";
    let char_ptr = "char*";

    let smem_budget = spec
        .max_smem_bytes
        .checked_sub(fft_scratch_bytes)
        .ok_or_else(|| {
            crate::kernel_compiler::infeasibility::infeasible(format!(
                "zy_fwd_slab: FFT scratch ({fft_scratch_bytes}B) exceeds max_smem_bytes ({})",
                spec.max_smem_bytes
            ))
        })?;
    let max_chunk = (smem_budget / per_chunk_bytes).max(1);
    let chunk = spec.inner_size.min(max_chunk);
    let total_staging_bytes = per_chunk_bytes * chunk;
    let shared_bytes = total_staging_bytes + fft_scratch_bytes;
    if shared_bytes > spec.max_smem_bytes {
        return Err(crate::kernel_compiler::infeasibility::infeasible(format!(
            "zy_fwd_slab: needs {shared_bytes}B smem for chunk=1, budget {}",
            spec.max_smem_bytes
        )));
    }

    let total_batch = spec.batch_shape.iter().copied().product::<i64>().max(1);
    let total_slabs = (total_batch * nx as i64) as u32;
    let out_slab_elems = (ny * nz_half * spec.inner_size) as i64;
    let total_staging_bytes_i = total_staging_bytes as i64;
    let total_slabs_i = total_slabs as i64;
    let nx_i = nx as i64;
    let ny_i = ny as i64;
    let n_i = n as i64;
    let nz_half_i = nz_half as i64;
    let inner_size_i = spec.inner_size as i64;
    let needs_batch = !spec.batch_shape.is_empty();
    let zero = zero_complex(complex_t, real_t);

    let mut params: Vec<Param> = Vec::with_capacity(2 * n_grids as usize);
    for j in 0..n_grids {
        params.push(Param::Pointer {
            const_: true,
            restrict: true,
            ctype: real_t.into(),
            name: format!("in_{j}"),
        });
    }
    for j in 0..n_grids {
        params.push(Param::Pointer {
            const_: false,
            restrict: true,
            ctype: complex_t.into(),
            name: format!("out_{j}"),
        });
    }

    let mut prelude: Vec<Stmt> = Vec::new();
    cuda! { prelude =>
        extern "C" __device__ void #z_symbol(#ct_ptr, #char_ptr);
        extern "C" __device__ void #y_symbol(#ct_ptr, #char_ptr);
    }

    let mut body: Vec<Stmt> = Vec::new();
    cuda! { body =>
        extern __shared__ #char_t _shmem[];
        #ct_ptr staging_base = (#ct_ptr)_shmem;
        #char_ptr fft_scratch = _shmem + #total_staging_bytes_i;
        ;
        i32 slab_id = blockIdx.x;
        if (slab_id >= #total_slabs_i) return;
        ;
        i32 x_pos = slab_id % #nx_i;
    }
    if needs_batch {
        cuda! { body => i32 batch_flat = slab_id / #nx_i; }
        push_batch_decompose(&mut body, spec.batch_shape);
    }
    cuda! { body =>
        i64 slab_out = (i64)slab_id * #out_slab_elems;
    }

    let per_grid_staging_complex = ny * nz_half * chunk;
    let slab_in_stride = (ny * n * spec.inner_size) as i64;
    for (jg, buf) in spec.grid_in_bufs.iter().enumerate() {
        let jg_i = jg as i64;
        let per_grid_off = jg_i * per_grid_staging_complex as i64;
        let in_name = format!("in_{jg}");
        let out_name = format!("out_{jg}");
        let grid_batch = batch_offset_expr(buf, spec.batch_shape.len());

        // Wrap per-grid locals in `if (1) { … }` so `staging`,
        // `grid_batch`, `slab_in`, `thread_data` don't redeclare
        // across grid iterations of the surrounding kernel scope.
        let mut grid_body: Vec<Stmt> = Vec::new();
        cuda! { grid_body =>
            #ct_ptr staging = staging_base + #per_grid_off;
            i32 grid_batch = #grid_batch;
            i64 slab_in = ((i64)grid_batch * #nx_i + x_pos) * #slab_in_stride;
            #complex_t thread_data[#ept];
            #rt_ptr z_real = (#rt_ptr)thread_data;
        }

        let mut ic_off: u32 = 0;
        while ic_off < spec.inner_size {
            let cisz = chunk.min(spec.inner_size - ic_off) as i64;
            let ic_off_i = ic_off as i64;
            let n_z_ffts = ny_i * cisz;
            let n_y_ffts = nz_half_i * cisz;
            let nz_half_cisz = nz_half_i * cisz;
            let ny_inner = ny_i * inner_size_i;

            cuda! { grid_body =>
                ;
                for (i32 g = 0; g < #n_z_ffts; g += #fpb) {
                    i32 fft_idx = g + threadIdx.y;
                    i32 y = fft_idx / #cisz;
                    i32 m = fft_idx % #cisz;
                    unroll for (i32 i = 0; i < #ept; i++) {
                        i32 pos = threadIdx.x + i * #stride;
                        if (fft_idx < #n_z_ffts && pos < #n_i) {
                            z_real[i] =
                                #in_name[slab_in + y * #n_i * #inner_size_i + pos * #inner_size_i + #ic_off_i + m];
                        } else {
                            z_real[i] = (#real_t)0;
                        }
                    }
                    #z_symbol(thread_data, fft_scratch);
                    if (fft_idx < #n_z_ffts) {
                        unroll for (i32 i = 0; i < #ept; i++) {
                            i32 pos = threadIdx.x + i * #stride;
                            if (pos < #nz_half_i) {
                                staging[y * #nz_half_cisz + pos * #cisz + m] = thread_data[i];
                            }
                        }
                    }
                }
                __syncthreads();
                ;
                for (i32 g = 0; g < #n_y_ffts; g += #fpb) {
                    i32 fft_idx = g + threadIdx.y;
                    i32 kz = fft_idx / #cisz;
                    i32 m = fft_idx % #cisz;
                    unroll for (i32 i = 0; i < #ept; i++) {
                        i32 pos = threadIdx.x + i * #stride;
                        if (fft_idx < #n_y_ffts && pos < #ny_i) {
                            thread_data[i] = staging[pos * #nz_half_cisz + kz * #cisz + m];
                        } else {
                            thread_data[i] = #zero;
                        }
                    }
                    #y_symbol(thread_data, fft_scratch);
                    if (fft_idx < #n_y_ffts) {
                        unroll for (i32 i = 0; i < #ept; i++) {
                            i32 pos = threadIdx.x + i * #stride;
                            if (pos < #ny_i) {
                                #out_name[slab_out + kz * #ny_inner + pos * #inner_size_i + #ic_off_i + m]
                                    = thread_data[i];
                            }
                        }
                    }
                }
                __syncthreads();
            }
            ic_off += cisz as u32;
        }
        cuda! { body =>
            if (1) {
                splice!(grid_body);
            }
        }
    }

    let kernel_name = format!("{}_e{ept}_f{fpb}", spec.kernel_name);
    let mut module = prelude;
    module.push(Stmt::Kernel {
        name: kernel_name.clone(),
        launch_bounds: String::new(),
        params,
        body,
    });

    Ok(ZyFwdSlabEmit {
        source: module,
        ltoir: vec![r2c.ltoir().to_vec(), c2c.ltoir().to_vec()],
        kernel_name,
        block_dim,
        shared_bytes,
        grid_size: total_slabs,
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

    fn buf(ic: &[i32], ext: &[i64]) -> Buffer {
        Buffer {
            name: "ignored".into(),
            dtype: Dtype::F32,
            ic: ic.to_vec(),
            extents: ext.to_vec(),
            elem_size: 0,
        }
    }

    #[test]
    fn renders_single_grid_no_batch() {
        if !libmathdx_available() {
            return;
        }
        let b = buf(&[], &[]);
        let variants = emit_variants(&ZyFwdSlabSpec {
            batch_shape: &[],
            grid_in_bufs: &[&b],
            inner_size: 1,
            max_smem_bytes: 65536,
            fft_lengths: [16, 16, 16],
            precision: Dtype::F32,
            sm: 120,
            kernel_name: "zy_fwd_test".into(),
            max_candidates:
                crate::kernel_compiler::spectral_map::candidate_budget::DEFAULT_MAX_CANDIDATES,
        })
        .expect("emit_variants");
        let out = variants.into_iter().next().expect("≥1 variant");
        let src = fourierd3_engine::ir::stmt::render_module_string(&out.source);
        assert!(src.contains("extern \"C\" __device__ void cufftdx_execute_"));
        assert!(src.contains("zy_fwd_test"));
        let calls = src.matches("(thread_data, fft_scratch);").count();
        assert!(
            calls >= 2,
            "expected ≥2 FFT execute calls, got {calls}\n{src}"
        );
        assert!(out.block_dim[0] >= 1);
        assert!(out.shared_bytes > 0);
        assert!(!out.ltoir.is_empty());
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
        if sm <= 0 {
            return;
        }
        let b = buf(&[], &[]);
        let variants = emit_variants(&ZyFwdSlabSpec {
            batch_shape: &[],
            grid_in_bufs: &[&b],
            inner_size: 1,
            max_smem_bytes: 65536,
            fft_lengths: [16, 16, 16],
            precision: Dtype::F32,
            sm: sm as u32,
            kernel_name: "zy_fwd_smoke".into(),
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

    #[test]
    fn renders_two_grids_with_inner_size() {
        if !libmathdx_available() {
            return;
        }
        let b0 = buf(&[-1], &[4]);
        let b1 = buf(&[-1], &[4]);
        let variants = emit_variants(&ZyFwdSlabSpec {
            batch_shape: &[4],
            grid_in_bufs: &[&b0, &b1],
            inner_size: 3,
            max_smem_bytes: 5000,
            fft_lengths: [16, 16, 16],
            precision: Dtype::F32,
            sm: 120,
            kernel_name: "zy_fwd_2g".into(),
            max_candidates:
                crate::kernel_compiler::spectral_map::candidate_budget::DEFAULT_MAX_CANDIDATES,
        })
        .expect("emit_variants");
        let out = variants.into_iter().next().expect("≥1 variant");
        let src = fourierd3_engine::ir::stmt::render_module_string(&out.source);
        assert!(src.contains("in_0") && src.contains("in_1"));
        assert!(src.contains("out_0") && src.contains("out_1"));
        let calls = src.matches("(thread_data, fft_scratch);").count();
        assert_eq!(calls, 12, "expected 12 FFT execute calls\n{src}");
    }
}
