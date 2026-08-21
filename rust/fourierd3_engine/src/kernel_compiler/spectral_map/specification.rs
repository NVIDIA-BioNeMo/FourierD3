// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::dtype::Dtype;

use crate::kernel_compiler::buffer::Buffer;

pub(crate) fn complex_ctype(precision: Dtype) -> &'static str {
    match precision {
        Dtype::F32 => "float2",
        Dtype::F64 => "double2",
        other => panic!("FFT precision must be f32 or f64, got {other:?}"),
    }
}

pub(crate) fn complex_bytes(precision: Dtype) -> u32 {
    2 * precision.size() as u32
}

#[derive(Clone, Debug)]
pub(crate) struct SpectralMapSpec {
    pub fft_lengths: [u32; 3],
    pub precision: Dtype,
    pub sm: u32,

    pub n_grid_in: u32,
    pub n_grid_out: u32,
    pub n_aux: u32,
    pub n_aux_out: u32,

    pub batch_shape: Vec<i64>,

    pub grid_inner_sizes: Vec<u32>,
    pub output_inner_sizes: Vec<u32>,

    pub input_signs: Vec<i32>,
    pub output_signs: Vec<i32>,

    pub grid_in_bufs: Vec<Buffer>,
    pub aux_bufs: Vec<Buffer>,

    pub aux_inner_shapes: Vec<Vec<u32>>,
    pub aux_dtypes: Vec<Dtype>,

    pub aux_output_inner_shapes: Vec<Vec<u32>>,
    pub aux_output_dtypes: Vec<Dtype>,
}

impl SpectralMapSpec {
    pub(crate) fn nx(&self) -> u32 {
        self.fft_lengths[0]
    }
    pub(crate) fn ny(&self) -> u32 {
        self.fft_lengths[1]
    }
    pub(crate) fn nz(&self) -> u32 {
        self.fft_lengths[2]
    }
    pub(crate) fn nz_half(&self) -> u32 {
        self.nz() / 2 + 1
    }
    pub(crate) fn total_batch(&self) -> i64 {
        self.batch_shape.iter().copied().product::<i64>().max(1)
    }
    pub(crate) fn total_slabs(&self) -> i64 {
        self.total_batch() * self.nx() as i64
    }
    pub(crate) fn total_x_lines(&self) -> i64 {
        self.total_batch() * self.ny() as i64 * self.nz_half() as i64
    }
}
