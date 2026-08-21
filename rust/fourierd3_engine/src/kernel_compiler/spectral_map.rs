// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward FFT, reciprocal-space map, and inverse FFT fused into one kernel
//! set, plus the emit helpers its stages share.

use std::collections::HashMap;

use fourierd3_engine::ir::expr::Expr;
use fourierd3_engine::ir::stmt::Stmt;

use crate::kernel_compiler::buffer::Buffer;

pub(crate) mod candidate_budget;
pub(crate) mod forward_zy_slabs;
pub(crate) mod fused_x_transform;
pub(crate) mod inverse_yz_slabs;
pub(crate) mod pipeline;
pub(crate) mod separate_x_transform;
pub(crate) mod specification;

pub(crate) use pipeline::SpectralMapPipeline;
pub(crate) use specification::SpectralMapSpec;

pub(crate) fn zero_complex(complex_t: &str, real_t: &str) -> Expr {
    Expr::call(
        format!("make_{complex_t}"),
        vec![
            Expr::call(real_t, vec![Expr::lit(0)]),
            Expr::call(real_t, vec![Expr::lit(0)]),
        ],
    )
}

/// Decompose the local `batch_flat` (which the caller is responsible
/// for declaring) into `batch_idx_0`, `batch_idx_1`, …, matching the
/// row-major encoding `_compute_batch_sizes` uses on the JAX side.
pub(crate) fn push_batch_decompose(stmts: &mut Vec<Stmt>, batch_shape: &[i64]) {
    crate::kernel_compiler::batch_indexing::push_batch_decode(
        stmts,
        batch_shape,
        Expr::var(String::from("batch_flat")),
    );
}

pub(crate) fn batch_offset_expr(buf: &Buffer, n_axes: usize) -> Expr {
    for axis in 0..n_axes {
        debug_assert!(
            buf.ic[axis] < 0,
            "spectral_map batch_offset_expr only supports broadcast ICs"
        );
    }
    let idx_offsets = HashMap::new();
    crate::kernel_compiler::batch_indexing::row_major_offset_expr(
        &buf.ic[..n_axes],
        &buf.extents[..n_axes],
        &idx_offsets,
    )
}
