// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How many cuFFTDx `(elements-per-thread, ffts-per-block)` configurations to
//! build for a given compile budget, and in which order.

/// Per-stage cuFFTDx `(ept, fpb)` order, best-first, that `build_candidates`
/// ranks the sweep by before building only the top [`max_candidates`]. Derived
/// by greedy-submodular coverage of measured autotune sweeps (RTX 5080 + B200 +
/// H100, f32, cubic grids 16..128) — the configs that jointly cover the per-size
/// winners — then padded with the remaining observed winners. cuFFTDx LTOIR
/// generation is serialized and dominates cold compile, so capping the count is
/// what bounds it. Regenerate with `python/scripts/rank_fft.py`.
pub(crate) const FWD_ZY_ORDER: &[(u32, u32)] = &[
    (8, 32),
    (16, 16),
    (8, 16),
    (8, 64),
    (4, 32),
    (2, 32),
    (4, 64),
    (16, 64),
    (4, 8),
    (4, 16),
];
pub(crate) const X_ORDER: &[(u32, u32)] = &[
    (4, 1),
    (16, 16),
    (8, 2),
    (8, 8),
    (8, 1),
    (16, 1),
    (8, 4),
    (8, 16),
    (16, 8),
];
pub(crate) const INV_YZ_ORDER: &[(u32, u32)] = &[
    (4, 32),
    (8, 16),
    (8, 64),
    (2, 32),
    (8, 32),
    (4, 64),
    (4, 8),
    (2, 16),
    (4, 16),
];

/// Default cap on cuFFTDx candidates built per launch: past the
/// greedy-coverage knee yet ~3× fewer than the full `ept × fpb` sweep, so cold
/// compile and first-run autotune drop proportionally. Callers raise the cap
/// through the compile budget when the one-off compile cost is acceptable
/// (e.g. ahead-of-time plan generation).
pub(crate) const DEFAULT_MAX_CANDIDATES: usize = 12;

/// Measured cold cost of building one cuFFTDx candidate, dominated by the
/// serialized LTOIR generation (RTX 5080 host, CUDA 13; ~2.4 s per distinct
/// spec, cached thereafter). Re-measure with the ignored
/// `measure_candidate_cost` test below.
pub(crate) const COMPILE_MS_PER_CANDIDATE: f64 = 2_400.0;

/// Candidates admitted by a predicted-compile-cost budget in milliseconds;
/// `None` selects [`DEFAULT_MAX_CANDIDATES`].
pub(crate) fn candidates_within_budget(compile_budget_ms: Option<f64>) -> usize {
    match compile_budget_ms {
        None => DEFAULT_MAX_CANDIDATES,
        Some(b) => ((b / COMPILE_MS_PER_CANDIDATE) as usize).max(1),
    }
}

#[cfg(test)]
mod tests {
    /// Re-measures [`super::COMPILE_MS_PER_CANDIDATE`]:
    /// `cargo test -p kernel_compiler --lib measure_candidate_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn measure_candidate_cost() {
        use crate::kernel_compiler::libmathdx::{CufftdxFft, FftDirection, FftSpec, FftType};
        crate::kernel_compiler::cuda_toolchain::populate_from_python_for_tests();
        let sm = {
            crate::cuda_driver::ensure_context().expect("CUDA context");
            crate::cuda_driver::Device::current()
                .sm_arch()
                .expect("sm arch") as u32
        };
        for rep in 0..3 {
            for &(size, ept, fpb) in &[(64u32, 8u32, 16u32), (128, 16, 8), (128, 8, 32)] {
                let t = std::time::Instant::now();
                CufftdxFft::build(&FftSpec {
                    size,
                    ty: FftType::C2C,
                    direction: FftDirection::Forward,
                    precision: fourierd3_engine::dtype::Dtype::F32,
                    sm,
                    ept: Some(ept),
                    fpb: Some(fpb),
                })
                .expect("cuFFTDx candidate builds");
                println!(
                    "rep{rep} size {size:4} ept {ept:2} fpb {fpb:2}: {:8.1} ms LTOIR",
                    t.elapsed().as_secs_f64() * 1e3,
                );
            }
        }
    }
}
