// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kernel_compiler::buffer::Buffer;
use crate::kernel_compiler::libmathdx::{CufftdxFft, FftDirection, FftSpec, FftType};
use crate::kernel_compiler::spectral_map::specification::complex_ctype;
use crate::kernel_compiler::spectral_map::{batch_offset_expr, push_batch_decompose, zero_complex};
use fourierd3_engine::cuda;
use fourierd3_engine::dtype::Dtype;
use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::{Param, Stmt};

pub(crate) struct XFusedEmit {
    pub source: Vec<Stmt>,
    pub ltoir_fwd: Vec<u8>,
    pub ltoir_inv: Vec<u8>,
    pub kernel_name: String,
    pub block_dim: [u32; 3],
    pub shared_bytes: u32,
    pub grid_size: u32,
}

pub(crate) struct XFusedSpec<'a> {
    pub fft_lengths: [u32; 3],
    pub precision: Dtype,
    pub sm: u32,
    pub max_smem_bytes: u32,
    pub batch_shape: &'a [i64],
    pub grid_inner_sizes: &'a [u32],
    pub output_inner_sizes: &'a [u32],
    pub input_signs: &'a [i32],
    pub output_signs: &'a [i32],
    pub aux_sizes: &'a [u32],
    pub aux_dtypes: &'a [Dtype],
    pub aux_bufs: &'a [&'a Buffer],
    pub aux_output_sizes: &'a [u32],
    pub aux_output_dtypes: &'a [Dtype],
    pub device_fn_ir: &'a str,
    pub kernel_name: String,
    pub max_candidates: usize,
}

pub(crate) fn emit_variants(spec: &XFusedSpec<'_>) -> Result<Vec<XFusedEmit>, String> {
    let [nx, ny, nz] = spec.fft_lengths;
    if ny != nz {
        return Err(format!("x_fused requires ny == nz, got ny={ny} nz={nz}"));
    }
    let nz_half = nz / 2 + 1;
    let total_batch = spec.batch_shape.iter().copied().product::<i64>().max(1);
    let total_x_lines = (total_batch * ny as i64 * nz_half as i64) as u32;
    let n_grid_in = spec.grid_inner_sizes.len();
    let n_grid_out = spec.output_inner_sizes.len();
    let n_aux = spec.aux_dtypes.len();
    let n_aux_out = spec.aux_output_dtypes.len();
    if spec.input_signs.len() != n_grid_in {
        return Err("input_signs length must equal n_grid_in".into());
    }
    if spec.output_signs.len() != n_grid_out {
        return Err("output_signs length must equal n_grid_out".into());
    }
    if spec.aux_sizes.len() != n_aux || spec.aux_bufs.len() != n_aux {
        return Err("aux_sizes / aux_bufs length must equal n_aux".into());
    }
    if spec.aux_output_sizes.len() != n_aux_out {
        return Err("aux_output_sizes length must equal n_aux_out".into());
    }

    let desired_blocks = 132u32;
    let fpb_cap = (total_x_lines / desired_blocks).max(1);
    let ffts_fwd = CufftdxFft::build_candidates(
        &FftSpec {
            size: nx,
            ty: FftType::C2C,
            direction: FftDirection::Forward,
            precision: spec.precision,
            sm: spec.sm,
            ept: None,
            fpb: None,
        },
        spec.max_smem_bytes,
        fpb_cap,
        crate::kernel_compiler::spectral_map::candidate_budget::X_ORDER,
        spec.max_candidates,
    )?;
    ffts_fwd
        .iter()
        .map(|fft_fwd| emit_config(spec, fft_fwd))
        .collect()
}

fn emit_config(spec: &XFusedSpec<'_>, fft_fwd: &CufftdxFft) -> Result<XFusedEmit, String> {
    let [nx, ny, nz] = spec.fft_lengths;
    let nz_half = nz / 2 + 1;
    let total_batch = spec.batch_shape.iter().copied().product::<i64>().max(1);
    let total_x_lines = (total_batch * ny as i64 * nz_half as i64) as u32;
    let n_grid_in = spec.grid_inner_sizes.len();
    let n_grid_out = spec.output_inner_sizes.len();
    let n_aux = spec.aux_dtypes.len();
    let n_aux_out = spec.aux_output_dtypes.len();

    let fft_inv = CufftdxFft::build(&FftSpec {
        size: nx,
        ty: FftType::C2C,
        direction: FftDirection::Inverse,
        precision: spec.precision,
        sm: spec.sm,
        ept: Some(fft_fwd.input_ept()),
        fpb: Some(fft_fwd.ffts_per_block()),
    })?;
    ensure_matching_fft_layout(fft_fwd, &fft_inv)?;
    let ept = fft_fwd.input_ept();
    let stride = fft_fwd.stride();
    let fpb = fft_fwd.ffts_per_block();
    let block_dim = fft_fwd.block_dim();
    let fft_scratch_bytes = fft_fwd.shared_mem_bytes().max(fft_inv.shared_mem_bytes());
    let sym_fwd = fft_fwd.symbol_name().to_string();
    let sym_inv = fft_inv.symbol_name().to_string();
    let bx = block_dim[0];

    let real_t = spec.precision.ctype();
    let complex_t = complex_ctype(spec.precision);
    let real_bytes = spec.precision.size() as u32;
    let ct_ptr = format!("{complex_t}*");
    let rt_ptr = format!("{real_t}*");
    // `char` is the cuFFTDx workspace byte-pointer type. The macro's type
    // table only knows Rust spellings, so we splice the CUDA literal in via
    // the `#name`-as-type Interp form.
    let char_t = "char";
    let char_ptr = "char*";

    // Aux-output reduce scratch lives right after the FFT scratch so it
    // doesn't trample the inverse-FFT workspace in Phase 4. Layout:
    //   shared_mem: [FFT scratch][_reduce: real_t * fpb * block_x]
    let aux_reduce_bytes = reduction_shared_bytes(n_aux_out, fpb, bx, real_bytes);
    let shared_bytes = fft_scratch_bytes + aux_reduce_bytes;
    let aux_reduce_offset = fft_scratch_bytes as i64;

    let device_fn_n_in = 1 + n_grid_in + n_aux;
    let module = crate::kernel_compiler::llvm::parse_module(spec.device_fn_ir)?;
    let device_fn_name = module.function.name.clone();
    let device_fn_src = crate::kernel_compiler::llvm::emit_cuda(&module, device_fn_n_in);

    let ptr = |const_, ctype: &str, name: String| Param::Pointer {
        const_,
        restrict: true,
        ctype: ctype.into(),
        name,
    };
    let mut params: Vec<Param> = Vec::new();
    params.extend((0..n_grid_in).map(|j| ptr(true, complex_t, format!("in_{j}"))));
    params.extend((0..n_aux).map(|k| ptr(true, spec.aux_dtypes[k].ctype(), format!("aux_{k}"))));
    params.extend((0..n_grid_out).map(|j| ptr(false, complex_t, format!("out_{j}"))));
    params.extend((0..n_aux_out).map(|j| {
        ptr(
            false,
            spec.aux_output_dtypes[j].ctype(),
            format!("aout_{j}"),
        )
    }));

    let mut prelude_stmts: Vec<Stmt> = Vec::new();
    cuda! { prelude_stmts =>
        extern "C" __device__ void #sym_fwd(#ct_ptr, #char_ptr);
        extern "C" __device__ void #sym_inv(#ct_ptr, #char_ptr);
    }

    let mut body: Vec<Stmt> = Vec::new();
    cuda! { body =>
        extern __shared__ #char_t shared_mem[];
    }
    let nx_i = nx as i64;
    let ny_i = ny as i64;
    let nz_half_i = nz_half as i64;
    let ept_i = ept as i64;
    let fpb_i = fpb as i64;
    let total_x_lines_i = total_x_lines as i64;
    let bx_i = bx as i64;
    let ny_half = (ny / 2) as i64;
    let nz_nyq = (nz_half - 1) as i64;
    let needs_batch = !spec.batch_shape.is_empty();

    cuda! { body =>
        #ct_ptr fft_scratch = (#ct_ptr)shared_mem;
    }
    if n_aux_out > 0 {
        cuda! { body =>
            #rt_ptr _reduce = (#rt_ptr)(shared_mem + #aux_reduce_offset);
        }
    }
    cuda! { body =>
        ;
        i32 fft_id = blockIdx.x * #fpb_i + threadIdx.y;
        if (fft_id >= #total_x_lines_i) return;
        ;
        i32 ky = fft_id % #ny_i;
        i32 _fft_rem = fft_id / #ny_i;
        i32 kz = _fft_rem % #nz_half_i;
    }
    if needs_batch {
        cuda! { body => i32 batch_flat = _fft_rem / #nz_half_i; }
        push_batch_decompose(&mut body, spec.batch_shape);
    }
    cuda! { body =>
        ;
        i32 ky_sym = (ky <= #ny_half) ? ky : ky - #ny_i;
        i32 kz_sym = kz;
        ;
        #complex_t thread_data[#ept_i];
    }
    for j in 0..n_grid_in {
        let sz = ept_i * spec.grid_inner_sizes[j] as i64;
        let name = format!("{}{j}", String::from("g_ft_"));
        cuda! { body => #complex_t #name[#sz]; }
    }
    for j in 0..n_grid_out {
        let sz = ept_i * spec.output_inner_sizes[j] as i64;
        let name = format!("{}{j}", String::from("g_out_"));
        cuda! { body => #complex_t #name[#sz]; }
    }
    cuda! { body => ; }

    push_aux_inputs(&mut body, spec);
    push_aux_output_accumulators(&mut body, spec, real_t, nz, nz_nyq);

    let ctx = PhaseCtx {
        nx_i,
        ny_i,
        nz_half_i,
        ept_i,
        stride_i: stride as i64,
        bx_i,
        complex_t,
        real_t,
        sym_fwd: &sym_fwd,
        sym_inv: &sym_inv,
        needs_batch,
    };
    push_x_fwd_phase(&mut body, &ctx, spec);
    push_x_pointwise_phase(&mut body, &ctx, spec, &device_fn_name);
    push_x_aux_reduce_phase(&mut body, &ctx, spec);
    push_x_inv_phase(&mut body, &ctx, spec);

    let mut module = prelude_stmts;
    module.push(Stmt::Raw(device_fn_src));
    let kernel_name = format!("{}_e{ept}_f{fpb}", spec.kernel_name);
    module.push(Stmt::Kernel {
        name: kernel_name.clone(),
        launch_bounds: String::new(),
        params,
        body,
    });

    Ok(XFusedEmit {
        source: module,
        ltoir_fwd: fft_fwd.ltoir().to_vec(),
        ltoir_inv: fft_inv.ltoir().to_vec(),
        kernel_name,
        block_dim,
        shared_bytes,
        grid_size: total_x_lines.div_ceil(fpb),
    })
}

fn ensure_matching_fft_layout(forward: &CufftdxFft, inverse: &CufftdxFft) -> Result<(), String> {
    let forward_layout = (
        forward.block_dim(),
        forward.input_ept(),
        forward.stride(),
        forward.ffts_per_block(),
    );
    let inverse_layout = (
        inverse.block_dim(),
        inverse.input_ept(),
        inverse.stride(),
        inverse.ffts_per_block(),
    );
    if forward_layout == inverse_layout {
        Ok(())
    } else {
        Err("fwd/inv FFT traits diverged for the same size".into())
    }
}

fn reduction_shared_bytes(n_outputs: usize, fpb: u32, block_x: u32, real_bytes: u32) -> u32 {
    if n_outputs == 0 {
        0
    } else {
        fpb * block_x * real_bytes
    }
}

fn push_aux_inputs(body: &mut Vec<Stmt>, spec: &XFusedSpec<'_>) {
    for (k, (&size, dtype)) in spec.aux_sizes.iter().zip(spec.aux_dtypes).enumerate() {
        let size = size as i64;
        let ctype = dtype.ctype();
        let local = format!("_aux_{k}");
        let local_batch = format!("_aux_{k}_batch");
        let source = format!("aux_{k}");
        let offset = batch_offset_expr(spec.aux_bufs[k], spec.batch_shape.len());
        cuda! { body =>
            #ctype #local[#size];
            i32 #local_batch = #offset;
            for (i32 _a = 0; _a < #size; _a++) {
                #local[_a] = #source[#local_batch * #size + _a];
            }
        }
    }
    if !spec.aux_sizes.is_empty() {
        cuda! { body => ; }
    }
}

fn push_aux_output_accumulators(
    body: &mut Vec<Stmt>,
    spec: &XFusedSpec<'_>,
    real_t: &str,
    nz: u32,
    nz_nyq: i64,
) {
    if spec.aux_output_dtypes.is_empty() {
        return;
    }
    for (j, (&size, dtype)) in spec
        .aux_output_sizes
        .iter()
        .zip(spec.aux_output_dtypes)
        .enumerate()
    {
        let size = size as i64;
        let ctype = dtype.ctype();
        let accumulator = format!("_aout_acc_{j}");
        cuda! { body =>
            #ctype #accumulator[#size];
            unroll for (i32 _m = 0; _m < #size; _m++) {
                #accumulator[_m] = (#ctype)0;
            }
        }
    }
    if nz.is_multiple_of(2) {
        cuda! { body =>
            ;
            #real_t _rfft_w = (kz == 0 || kz == #nz_nyq) ? (#real_t)1 : (#real_t)2;
            ;
        }
    } else {
        cuda! { body =>
            ;
            #real_t _rfft_w = (kz == 0) ? (#real_t)1 : (#real_t)2;
            ;
        }
    }
}

pub(crate) struct PhaseCtx<'a> {
    pub(crate) nx_i: i64,
    pub(crate) ny_i: i64,
    pub(crate) nz_half_i: i64,
    pub(crate) ept_i: i64,
    pub(crate) stride_i: i64,
    pub(crate) bx_i: i64,
    pub(crate) complex_t: &'a str,
    pub(crate) real_t: &'a str,
    pub(crate) sym_fwd: &'a str,
    pub(crate) sym_inv: &'a str,
    pub(crate) needs_batch: bool,
}

pub(crate) fn push_x_fwd_phase(body: &mut Vec<Stmt>, ctx: &PhaseCtx<'_>, spec: &XFusedSpec<'_>) {
    let &PhaseCtx {
        nx_i,
        ny_i,
        nz_half_i,
        ept_i,
        stride_i,
        complex_t,
        real_t,
        sym_fwd,
        ..
    } = ctx;
    let mk = format!("make_{complex_t}");

    for j in 0..spec.grid_inner_sizes.len() {
        let isz = spec.grid_inner_sizes[j] as i64;
        let batch_elems = ny_i * nz_half_i * isz * nx_i;
        let x_stride = nz_half_i * ny_i * isz;
        let ny_isz = ny_i * isz;
        let in_j = format!("in_{j}");
        let g_ft_j = format!("{}{j}", String::from("g_ft_"));
        let zero = zero_complex(complex_t, real_t);
        let imag = if spec.input_signs[j] == 1 {
            Expr::sub(
                Expr::call(real_t, vec![Expr::lit(0)]),
                Expr::var(String::from("v.y")),
            )
        } else {
            Expr::var(String::from("v.y"))
        };
        let batch_term = if ctx.needs_batch {
            Expr::mul(
                Expr::cast(
                    String::from("long long"),
                    Expr::var(String::from("batch_flat")),
                ),
                Expr::lit(batch_elems),
            )
        } else {
            Expr::lit(0)
        };
        let load_idx = Expr::add(
            Expr::add(
                Expr::add(
                    Expr::add(
                        batch_term,
                        Expr::mul(Expr::var(String::from("pos")), Expr::lit(x_stride)),
                    ),
                    Expr::mul(Expr::var(String::from("kz")), Expr::lit(ny_isz)),
                ),
                Expr::mul(Expr::var(String::from("ky")), Expr::lit(isz)),
            ),
            Expr::var(String::from("_m")),
        );

        cuda! { body =>
            for (i32 _m = 0; _m < #isz; _m++) {
                unroll for (i32 _i = 0; _i < #ept_i; _i++) {
                    i32 pos = threadIdx.x + _i * #stride_i;
                    if (pos < #nx_i) {
                        thread_data[_i] = #in_j[#load_idx];
                    } else {
                        thread_data[_i] = #zero;
                    }
                }
                #sym_fwd(thread_data, shared_mem);
                unroll for (i32 _i = 0; _i < #ept_i; _i++) {
                    #complex_t v = thread_data[_i];
                    #g_ft_j[_i * #isz + _m] = #mk(v.x, #imag);
                }
            }
            ;
        }
    }
}

fn push_x_pointwise_phase(
    body: &mut Vec<Stmt>,
    ctx: &PhaseCtx<'_>,
    spec: &XFusedSpec<'_>,
    device_fn_name: &str,
) {
    let &PhaseCtx {
        nx_i,
        ept_i,
        stride_i,
        real_t,
        ..
    } = ctx;
    let nx_half = (nx_i) / 2;
    let n_grid_in = spec.grid_inner_sizes.len();
    let n_grid_out = spec.output_inner_sizes.len();
    let n_aux = spec.aux_dtypes.len();
    let n_aux_out = spec.aux_output_dtypes.len();
    let const_rt_ptr = format!("const {real_t}*");
    let rt_ptr = format!("{real_t}*");

    // The user device fn expects complex slots as `real_t*` (RIRI layout),
    // so we cast `complex_t*` → `real_t*` at the call site.
    let fn_args: Vec<Expr> = std::iter::once(Expr::var(String::from("_idx")))
        .chain((0..n_grid_in).map(|j| {
            let isz = spec.grid_inner_sizes[j] as i64;
            Expr::cast(
                const_rt_ptr.clone(),
                Expr::addr(Expr::index(
                    format!("{}{j}", String::from("g_ft_")),
                    Expr::mul(Expr::var(String::from("_i")), Expr::lit(isz)),
                )),
            )
        }))
        .chain((0..n_aux).map(|k| Expr::var(format!("_aux_{k}"))))
        .chain((0..n_grid_out).map(|j| {
            let oisz = spec.output_inner_sizes[j] as i64;
            Expr::cast(
                rt_ptr.clone(),
                Expr::addr(Expr::index(
                    format!("{}{j}", String::from("g_out_")),
                    Expr::mul(Expr::var(String::from("_i")), Expr::lit(oisz)),
                )),
            )
        }))
        .chain((0..n_aux_out).map(|j| Expr::var(format!("{}{j}", String::from("_aout_")))))
        .collect();

    let mut guard_body: Vec<Stmt> = Vec::new();
    cuda! { guard_body =>
        i32 kx_sym = (pos <= #nx_half) ? pos : pos - #nx_i;
        i32 _idx[] = { kx_sym, ky_sym, kz_sym };
    }
    for j in 0..n_aux_out {
        let aoisz = spec.aux_output_sizes[j] as i64;
        let aot = spec.aux_output_dtypes[j].ctype();
        let aout_j = format!("{}{j}", String::from("_aout_"));
        cuda! { guard_body => #aot #aout_j[#aoisz]; }
    }
    guard_body.push(Stmt::Eval(Expr::call(device_fn_name, fn_args)));
    for j in 0..n_aux_out {
        let aoisz = spec.aux_output_sizes[j] as i64;
        let acc = format!("{}{j}", String::from("_aout_acc_"));
        let aout = format!("{}{j}", String::from("_aout_"));
        cuda! { guard_body =>
            unroll for (i32 _m = 0; _m < #aoisz; _m++) {
                #acc[_m] += #aout[_m];
            }
        }
    }

    cuda! { body =>
        unroll for (i32 _i = 0; _i < #ept_i; _i++) {
            i32 pos = threadIdx.x + _i * #stride_i;
            if (pos < #nx_i) {
                splice!(guard_body);
            }
        }
        ;
    }
}

fn push_x_aux_reduce_phase(body: &mut Vec<Stmt>, ctx: &PhaseCtx<'_>, spec: &XFusedSpec<'_>) {
    let n_aux_out = spec.aux_output_dtypes.len();
    if n_aux_out == 0 {
        return;
    }
    let &PhaseCtx { bx_i, real_t, .. } = ctx;

    for j in 0..n_aux_out {
        let aoisz = spec.aux_output_sizes[j] as i64;
        let acc = format!("{}{j}", String::from("_aout_acc_"));
        let aout = format!("aout_{j}");
        cuda! { body =>
            for (i32 _m = 0; _m < #aoisz; _m++) {
                _reduce[threadIdx.y * #bx_i + threadIdx.x] = (#real_t)#acc[_m] * _rfft_w;
                __syncthreads();
                if (threadIdx.x == 0) {
                    #real_t _v = _reduce[threadIdx.y * #bx_i];
                    for (i32 _s = 1; _s < #bx_i; _s++) {
                        _v += _reduce[threadIdx.y * #bx_i + _s];
                    }
                    #aout[fft_id * #aoisz + _m] = _v;
                }
                __syncthreads();
            }
        }
    }
    cuda! { body => ; }
}

pub(crate) fn push_x_inv_phase(body: &mut Vec<Stmt>, ctx: &PhaseCtx<'_>, spec: &XFusedSpec<'_>) {
    let n_grid_out = spec.output_inner_sizes.len();
    if n_grid_out == 0 {
        return;
    }
    let &PhaseCtx {
        nx_i,
        ny_i,
        nz_half_i,
        ept_i,
        stride_i,
        complex_t,
        real_t,
        sym_inv,
        ..
    } = ctx;
    let mk = format!("make_{complex_t}");

    cuda! { body =>
        #real_t inv_nx = (#real_t)1 / (#real_t)#nx_i;
    }
    for j in 0..n_grid_out {
        let oisz = spec.output_inner_sizes[j] as i64;
        let out_j = format!("out_{j}");
        let g_out_j = format!("{}{j}", String::from("g_out_"));
        let batch_elems = ny_i * nz_half_i * oisz * nx_i;
        let x_stride = nz_half_i * ny_i * oisz;
        let imag = if spec.output_signs[j] == 1 {
            Expr::sub(
                Expr::call(real_t, vec![Expr::lit(0)]),
                Expr::var(String::from("v.y")),
            )
        } else {
            Expr::var(String::from("v.y"))
        };
        let batch_term = if ctx.needs_batch {
            Expr::mul(
                Expr::cast(
                    String::from("long long"),
                    Expr::var(String::from("batch_flat")),
                ),
                Expr::lit(batch_elems),
            )
        } else {
            Expr::lit(0)
        };
        let store_idx = Expr::add(
            Expr::add(
                Expr::add(
                    Expr::add(
                        batch_term,
                        Expr::mul(Expr::var(String::from("pos")), Expr::lit(x_stride)),
                    ),
                    Expr::mul(Expr::var(String::from("kz")), Expr::lit(oisz * ny_i)),
                ),
                Expr::mul(Expr::var(String::from("_m")), Expr::lit(ny_i)),
            ),
            Expr::var(String::from("ky")),
        );

        cuda! { body =>
            for (i32 _m = 0; _m < #oisz; _m++) {
                unroll for (i32 _i = 0; _i < #ept_i; _i++) {
                    #complex_t v = #g_out_j[_i * #oisz + _m];
                    thread_data[_i] = #mk(v.x, #imag);
                }
                #sym_inv(thread_data, shared_mem);
                unroll for (i32 _i = 0; _i < #ept_i; _i++) {
                    i32 pos = threadIdx.x + _i * #stride_i;
                    if (pos < #nx_i) {
                        #out_j[#store_idx] = #mk(thread_data[_i].x * inv_nx, thread_data[_i].y * inv_nx);
                    }
                }
            }
            ;
        }
    }
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

    fn passthrough_ir() -> &'static str {
        r#"; ModuleID = "fn"
target triple = "nvptx64-nvidia-cuda"
target datalayout = ""

define void @"fn"(i32* %"_idx", float* %"g0", float* %"out0")
{
entry:
  ret void
}
"#
    }

    #[test]
    fn renders_minimal() {
        if !libmathdx_available() {
            return;
        }
        let variants = emit_variants(&XFusedSpec {
            fft_lengths: [16, 16, 16],
            precision: Dtype::F32,
            sm: 120,
            max_smem_bytes: 200_000,
            batch_shape: &[],
            grid_inner_sizes: &[1],
            output_inner_sizes: &[1],
            input_signs: &[-1],
            output_signs: &[-1],
            aux_sizes: &[],
            aux_dtypes: &[],
            aux_bufs: &[],
            aux_output_sizes: &[],
            aux_output_dtypes: &[],
            device_fn_ir: passthrough_ir(),
            kernel_name: "x_fused_test".into(),
            max_candidates:
                crate::kernel_compiler::spectral_map::candidate_budget::DEFAULT_MAX_CANDIDATES,
        })
        .expect("emit_variants");
        let out = variants.into_iter().next().expect("≥1 variant");
        let src = fourierd3_engine::ir::stmt::render_module_string(&out.source);
        assert!(src.contains("x_fused_test"));
        assert!(src.contains("extern \"C\" __device__ void cufftdx_execute_"));
        let n_externs = src.matches("extern \"C\" __device__").count();
        assert_eq!(n_externs, 2, "expected 2 extern decls\n{src}");
        let n_calls = src.matches("(thread_data, shared_mem);").count();
        assert_eq!(n_calls, 2, "expected 2 FFT execute calls\n{src}");
        assert!(!out.ltoir_fwd.is_empty() && !out.ltoir_inv.is_empty());
        assert!(out.ltoir_fwd != out.ltoir_inv);
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
        let variants = emit_variants(&XFusedSpec {
            fft_lengths: [16, 16, 16],
            precision: Dtype::F32,
            sm: sm as u32,
            max_smem_bytes: 200_000,
            batch_shape: &[],
            grid_inner_sizes: &[1],
            output_inner_sizes: &[1],
            input_signs: &[-1],
            output_signs: &[-1],
            aux_sizes: &[],
            aux_dtypes: &[],
            aux_bufs: &[],
            aux_output_sizes: &[],
            aux_output_dtypes: &[],
            device_fn_ir: passthrough_ir(),
            kernel_name: "x_fused_smoke".into(),
            max_candidates:
                crate::kernel_compiler::spectral_map::candidate_budget::DEFAULT_MAX_CANDIDATES,
        })
        .expect("emit_variants");
        let out = variants.into_iter().next().expect("≥1 variant");
        let src = fourierd3_engine::ir::stmt::render_module_string(&out.source);
        crate::kernel_compiler::cuda_toolchain::compile_cubin(
            src.as_bytes(),
            Some("x_fused_smoke"),
            &[],
            &[&out.ltoir_fwd, &out.ltoir_inv],
        )
        .expect("compile");
    }
}
