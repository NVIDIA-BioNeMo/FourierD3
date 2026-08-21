// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use fourierd3_engine::dtype::Dtype;

#[derive(Clone, Debug)]
pub(crate) struct Buffer {
    pub name: String,
    pub dtype: Dtype,
    pub ic: Vec<i32>,
    pub extents: Vec<i64>,
    pub elem_size: i64,
}

impl Buffer {
    pub(crate) fn needs_scatter(&self, batch_sizes: &[i64]) -> bool {
        self.ic
            .iter()
            .zip(&self.extents)
            .zip(batch_sizes)
            .any(|((&r, &e), &b)| r >= 0 || e != b)
    }
}

pub(crate) fn buffer_nbytes(buf: &Buffer, volume: i64) -> Result<usize, String> {
    let batch: i64 = buf.extents.iter().product();
    let elems = batch
        .checked_mul(volume)
        .and_then(|n| n.checked_mul(buf.elem_size))
        .ok_or_else(|| format!("buffer {} element count overflow", buf.name))?;
    if elems < 0 {
        return Err(format!(
            "buffer {} has negative element count {elems}",
            buf.name
        ));
    }
    (elems as usize)
        .checked_mul(buf.dtype.size())
        .ok_or_else(|| format!("buffer {} byte size overflow", buf.name))
}
