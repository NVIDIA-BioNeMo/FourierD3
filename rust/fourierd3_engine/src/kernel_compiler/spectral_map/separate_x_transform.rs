// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kernel_compiler::libmathdx::{CufftdxFft, FftDirection, FftSpec, FftType};
use crate::kernel_compiler::spectral_map::fused_x_transform::{
    PhaseCtx, XFusedSpec, push_x_fwd_phase, push_x_inv_phase,
};
use crate::kernel_compiler::spectral_map::specification::complex_ctype;
use crate::kernel_compiler::spectral_map::{batch_offset_expr, push_batch_decompose, zero_complex};
use fourierd3_engine::cuda;
use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::{Param, Stmt};

pub(crate) struct XDefusedEmit {
    pub source: Vec<Stmt>,
    pub ltoir: Vec<Vec<u8>>,
    pub kernel_name: String,
    pub block_dim: [u32; 3],
    pub shared_bytes: u32,
    pub grid_size: u32,
}

fn pick_fft(nx: u32, dir: FftDirection, spec: &XFusedSpec<'_>) -> Result<CufftdxFft, String> {
    let cands = CufftdxFft::build_candidates(
        &FftSpec {
            size: nx,
            ty: FftType::C2C,
            direction: dir,
            precision: spec.precision,
            sm: spec.sm,
            ept: None,
            fpb: None,
        },
        spec.max_smem_bytes,
        1,
        crate::kernel_compiler::spectral_map::candidate_budget::X_ORDER,
        spec.max_candidates,
    )?;
    cands
        .into_iter()
        .next()
        .ok_or_else(|| "no feasible cuFFTDx config for defused X FFT".into())
}

fn push_line_prologue(body: &mut Vec<Stmt>, spec: &XFusedSpec<'_>, fpb_i: i64) {
    let [_, ny, nz] = spec.fft_lengths;
    let ny_i = ny as i64;
    let nz_half_i = (nz / 2 + 1) as i64;
    let total_x_lines =
        (spec.batch_shape.iter().copied().product::<i64>().max(1)) * ny_i * nz_half_i;
    let char_t = "char";
    cuda! { body =>
        extern __shared__ #char_t shared_mem[];
        i32 fft_id = blockIdx.x * #fpb_i + threadIdx.y;
        if (fft_id >= #total_x_lines) return;
        i32 ky = fft_id % #ny_i;
        i32 _fft_rem = fft_id / #ny_i;
        i32 kz = _fft_rem % #nz_half_i;
    }
    if !spec.batch_shape.is_empty() {
        cuda! { body => i32 batch_flat = _fft_rem / #nz_half_i; }
        push_batch_decompose(body, spec.batch_shape);
    }
}

pub(crate) fn emit_x_fwd(spec: &XFusedSpec<'_>) -> Result<XDefusedEmit, String> {
    let [nx, ny, nz] = spec.fft_lengths;
    let nz_half = nz / 2 + 1;
    let fft = pick_fft(nx, FftDirection::Forward, spec)?;
    let ept = fft.input_ept();
    let stride = fft.stride();
    let fpb = fft.ffts_per_block();
    let block_dim = fft.block_dim();
    let sym = fft.symbol_name().to_string();
    let total_x_lines =
        (spec.batch_shape.iter().copied().product::<i64>().max(1)) * ny as i64 * nz_half as i64;

    let complex_t = complex_ctype(spec.precision);
    let real_t = spec.precision.ctype();
    let ct_ptr = format!("{complex_t}*");
    let char_ptr = "char*";
    let n_grid_in = spec.grid_inner_sizes.len();

    let ptr = |const_, name: String| Param::Pointer {
        const_,
        restrict: true,
        ctype: complex_t.into(),
        name,
    };
    let mut params: Vec<Param> = Vec::new();
    params.extend((0..n_grid_in).map(|j| ptr(true, format!("in_{j}"))));
    params.extend((0..n_grid_in).map(|j| ptr(false, format!("xfreq_{j}"))));

    let mut prelude: Vec<Stmt> = Vec::new();
    cuda! { prelude => extern "C" __device__ void #sym(#ct_ptr, #char_ptr); }

    let ept_i = ept as i64;
    let stride_i = stride as i64;
    let nx_i = nx as i64;
    let mut body: Vec<Stmt> = Vec::new();
    push_line_prologue(&mut body, spec, fpb as i64);
    cuda! { body => #complex_t thread_data[#ept_i]; }
    for j in 0..n_grid_in {
        let sz = ept_i * spec.grid_inner_sizes[j] as i64;
        let g = format!("{}{j}", String::from("g_ft_"));
        cuda! { body => #complex_t #g[#sz]; }
    }
    let ctx = PhaseCtx {
        nx_i,
        ny_i: ny as i64,
        nz_half_i: nz_half as i64,
        ept_i,
        stride_i,
        bx_i: block_dim[0] as i64,
        complex_t,
        real_t,
        sym_fwd: &sym,
        sym_inv: &sym,
        needs_batch: !spec.batch_shape.is_empty(),
    };
    push_x_fwd_phase(&mut body, &ctx, spec);

    for j in 0..n_grid_in {
        let isz = spec.grid_inner_sizes[j] as i64;
        let g = format!("{}{j}", String::from("g_ft_"));
        let xf = format!("xfreq_{j}");
        let line_stride = nx_i * isz;
        cuda! { body =>
            for (i32 _m = 0; _m < #isz; _m++) {
                unroll for (i32 _i = 0; _i < #ept_i; _i++) {
                    i32 pos = threadIdx.x + _i * #stride_i;
                    if (pos < #nx_i) {
                        #xf[fft_id * #line_stride + pos * #isz + _m] = #g[_i * #isz + _m];
                    }
                }
            }
        }
    }

    let mut module = prelude;
    module.push(Stmt::Kernel {
        name: spec.kernel_name.clone(),
        launch_bounds: String::new(),
        params,
        body,
    });
    Ok(XDefusedEmit {
        source: module,
        ltoir: vec![fft.ltoir().to_vec()],
        kernel_name: spec.kernel_name.clone(),
        block_dim,
        shared_bytes: fft.shared_mem_bytes(),
        grid_size: (total_x_lines as u32).div_ceil(fpb),
    })
}

pub(crate) fn emit_x_contract(spec: &XFusedSpec<'_>) -> Result<XDefusedEmit, String> {
    let [nx, ny, nz] = spec.fft_lengths;
    let nz_half = nz / 2 + 1;
    let nx_i = nx as i64;
    let ny_i = ny as i64;
    let nz_half_i = nz_half as i64;
    let nx_half = nx_i / 2;
    let ny_half = ny_i / 2;
    let nz_nyq = (nz_half - 1) as i64;
    let total_x_lines =
        (spec.batch_shape.iter().copied().product::<i64>().max(1)) * ny_i * nz_half_i;
    let total_k = total_x_lines * nx_i;

    let complex_t = complex_ctype(spec.precision);
    let real_t = spec.precision.ctype();
    let const_rt_ptr = format!("const {real_t}*");
    let rt_ptr = format!("{real_t}*");
    let n_grid_in = spec.grid_inner_sizes.len();
    let n_grid_out = spec.output_inner_sizes.len();
    let n_aux = spec.aux_dtypes.len();
    let n_aux_out = spec.aux_output_dtypes.len();

    let mut params: Vec<Param> = Vec::new();
    params.extend((0..n_grid_in).map(|j| Param::Pointer {
        const_: true,
        restrict: true,
        ctype: complex_t.into(),
        name: format!("xfreq_{j}"),
    }));
    params.extend((0..n_aux).map(|k| Param::Pointer {
        const_: true,
        restrict: true,
        ctype: spec.aux_dtypes[k].into(),
        name: format!("aux_{k}"),
    }));
    params.extend((0..n_grid_out).map(|j| Param::Pointer {
        const_: false,
        restrict: true,
        ctype: complex_t.into(),
        name: format!("xfreq_out_{j}"),
    }));
    params.extend((0..n_aux_out).map(|j| Param::Pointer {
        const_: false,
        restrict: true,
        ctype: spec.aux_output_dtypes[j].into(),
        name: format!("aout_{j}"),
    }));

    let device_fn_n_in = 1 + n_grid_in + n_aux;
    let module = crate::kernel_compiler::llvm::parse_module(spec.device_fn_ir)?;
    let device_fn_name = module.function.name.clone();
    let device_fn_src = crate::kernel_compiler::llvm::emit_cuda(&module, device_fn_n_in);

    let mut body: Vec<Stmt> = Vec::new();
    cuda! { body =>
        i64 _gid = (i64)blockIdx.x * blockDim.x + threadIdx.x;
        if (_gid >= #total_k) return;
        i32 kx = _gid % #nx_i;
        i64 _line = _gid / #nx_i;
        i32 ky = _line % #ny_i;
        i64 _fft_rem = _line / #ny_i;
        i32 kz = _fft_rem % #nz_half_i;
    }
    if !spec.batch_shape.is_empty() {
        // `_fft_rem` (not `_rem`) so it survives `push_batch_decompose`, which
        // declares its own `_rem` for the multi-axis decode.
        cuda! { body => i32 batch_flat = _fft_rem / #nz_half_i; }
        push_batch_decompose(&mut body, spec.batch_shape);
    }
    cuda! { body =>
        i32 kx_sym = (kx <= #nx_half) ? kx : kx - #nx_i;
        i32 ky_sym = (ky <= #ny_half) ? ky : ky - #ny_i;
        i32 kz_sym = kz;
        i32 _idx[] = { kx_sym, ky_sym, kz_sym };
    }

    for k in 0..n_aux {
        let sz = spec.aux_sizes[k] as i64;
        let actype = spec.aux_dtypes[k].ctype();
        let local = format!("_aux_{k}");
        let lb = format!("_aux_{k}_batch");
        let src = format!("aux_{k}");
        let aux_offset = batch_offset_expr(spec.aux_bufs[k], spec.batch_shape.len());
        cuda! { body =>
            #actype #local[#sz];
            i32 #lb = #aux_offset;
            for (i32 _a = 0; _a < #sz; _a++) {
                #local[_a] = #src[#lb * #sz + _a];
            }
        }
    }

    for j in 0..n_grid_out {
        let oisz = spec.output_inner_sizes[j] as i64;
        let g = format!("{}{j}", String::from("g_out_"));
        cuda! { body => #complex_t #g[#oisz]; }
    }
    for j in 0..n_aux_out {
        let aoisz = spec.aux_output_sizes[j] as i64;
        let aot = spec.aux_output_dtypes[j].ctype();
        let a = format!("{}{j}", String::from("_aout_"));
        cuda! { body => #aot #a[#aoisz]; }
    }

    let fn_args: Vec<Expr> = std::iter::once(Expr::var(String::from("_idx")))
        .chain((0..n_grid_in).map(|j| {
            let isz = spec.grid_inner_sizes[j] as i64;
            let line_stride = nx_i * isz;
            Expr::cast(
                const_rt_ptr.clone(),
                Expr::addr(Expr::index(
                    format!("xfreq_{j}"),
                    Expr::add(
                        Expr::mul(Expr::var(String::from("_line")), Expr::lit(line_stride)),
                        Expr::mul(Expr::var(String::from("kx")), Expr::lit(isz)),
                    ),
                )),
            )
        }))
        .chain((0..n_aux).map(|k| Expr::var(format!("_aux_{k}"))))
        .chain((0..n_grid_out).map(|j| {
            Expr::cast(
                rt_ptr.clone(),
                Expr::addr(Expr::index(
                    format!("{}{j}", String::from("g_out_")),
                    Expr::lit(0),
                )),
            )
        }))
        .chain((0..n_aux_out).map(|j| Expr::var(format!("{}{j}", String::from("_aout_")))))
        .collect();
    body.push(Stmt::Eval(Expr::call(device_fn_name, fn_args)));

    // rFFT weight: 1 at DC (and Nyquist for even nz), 2 elsewhere.
    if nz % 2 == 0 {
        cuda! { body => #real_t _rfft_w = (kz == 0 || kz == #nz_nyq) ? (#real_t)1 : (#real_t)2; }
    } else {
        cuda! { body => #real_t _rfft_w = (kz == 0) ? (#real_t)1 : (#real_t)2; }
    }

    for j in 0..n_aux_out {
        let aoisz = spec.aux_output_sizes[j] as i64;
        let aot = spec.aux_output_dtypes[j].ctype();
        let a = format!("{}{j}", String::from("_aout_"));
        let aout = format!("aout_{j}");
        cuda! { body =>
            for (i32 _m = 0; _m < #aoisz; _m++) {
                #aot _w = (#aot)((#real_t)#a[_m] * _rfft_w);
                atomicAdd(&#aout[_line * #aoisz + _m], _w);
            }
        }
    }

    for j in 0..n_grid_out {
        let oisz = spec.output_inner_sizes[j] as i64;
        let g = format!("{}{j}", String::from("g_out_"));
        let xf = format!("xfreq_out_{j}");
        let line_stride = nx_i * oisz;
        cuda! { body =>
            for (i32 _m = 0; _m < #oisz; _m++) {
                #xf[_line * #line_stride + kx * #oisz + _m] = #g[_m];
            }
        }
    }

    let module = vec![
        Stmt::Raw(device_fn_src),
        Stmt::Kernel {
            name: spec.kernel_name.clone(),
            launch_bounds: String::new(),
            params,
            body,
        },
    ];
    let block = 256u32;
    Ok(XDefusedEmit {
        source: module,
        ltoir: vec![],
        kernel_name: spec.kernel_name.clone(),
        block_dim: [block, 1, 1],
        shared_bytes: 0,
        grid_size: (total_k as u32).div_ceil(block),
    })
}

pub(crate) fn emit_x_inv(spec: &XFusedSpec<'_>) -> Result<XDefusedEmit, String> {
    let [nx, ny, nz] = spec.fft_lengths;
    let nz_half = nz / 2 + 1;
    let fft = pick_fft(nx, FftDirection::Inverse, spec)?;
    let ept = fft.input_ept();
    let stride = fft.stride();
    let fpb = fft.ffts_per_block();
    let block_dim = fft.block_dim();
    let sym = fft.symbol_name().to_string();
    let total_x_lines =
        (spec.batch_shape.iter().copied().product::<i64>().max(1)) * ny as i64 * nz_half as i64;

    let complex_t = complex_ctype(spec.precision);
    let real_t = spec.precision.ctype();
    let ct_ptr = format!("{complex_t}*");
    let char_ptr = "char*";
    let n_grid_out = spec.output_inner_sizes.len();

    let mut params: Vec<Param> = Vec::new();
    params.extend((0..n_grid_out).map(|j| Param::Pointer {
        const_: true,
        restrict: true,
        ctype: complex_t.into(),
        name: format!("xfreq_out_{j}"),
    }));
    params.extend((0..n_grid_out).map(|j| Param::Pointer {
        const_: false,
        restrict: true,
        ctype: complex_t.into(),
        name: format!("out_{j}"),
    }));

    let mut prelude: Vec<Stmt> = Vec::new();
    cuda! { prelude => extern "C" __device__ void #sym(#ct_ptr, #char_ptr); }

    let ept_i = ept as i64;
    let stride_i = stride as i64;
    let nx_i = nx as i64;
    let mut body: Vec<Stmt> = Vec::new();
    push_line_prologue(&mut body, spec, fpb as i64);
    cuda! { body => #complex_t thread_data[#ept_i]; }
    for j in 0..n_grid_out {
        let sz = ept_i * spec.output_inner_sizes[j] as i64;
        let g = format!("{}{j}", String::from("g_out_"));
        cuda! { body => #complex_t #g[#sz]; }
    }

    let zero = zero_complex(complex_t, real_t);
    for j in 0..n_grid_out {
        let oisz = spec.output_inner_sizes[j] as i64;
        let g = format!("{}{j}", String::from("g_out_"));
        let xf = format!("xfreq_out_{j}");
        let line_stride = nx_i * oisz;
        cuda! { body =>
            for (i32 _m = 0; _m < #oisz; _m++) {
                unroll for (i32 _i = 0; _i < #ept_i; _i++) {
                    i32 pos = threadIdx.x + _i * #stride_i;
                    if (pos < #nx_i) {
                        #g[_i * #oisz + _m] = #xf[fft_id * #line_stride + pos * #oisz + _m];
                    } else {
                        #g[_i * #oisz + _m] = #zero;
                    }
                }
            }
        }
    }

    let ctx = PhaseCtx {
        nx_i,
        ny_i: ny as i64,
        nz_half_i: nz_half as i64,
        ept_i,
        stride_i,
        bx_i: block_dim[0] as i64,
        complex_t,
        real_t,
        sym_fwd: &sym,
        sym_inv: &sym,
        needs_batch: !spec.batch_shape.is_empty(),
    };
    push_x_inv_phase(&mut body, &ctx, spec);

    let mut module = prelude;
    module.push(Stmt::Kernel {
        name: spec.kernel_name.clone(),
        launch_bounds: String::new(),
        params,
        body,
    });
    Ok(XDefusedEmit {
        source: module,
        ltoir: vec![fft.ltoir().to_vec()],
        kernel_name: spec.kernel_name.clone(),
        block_dim,
        shared_bytes: fft.shared_mem_bytes(),
        grid_size: (total_x_lines as u32).div_ceil(fpb),
    })
}
