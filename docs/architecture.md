<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Architecture

FourierD3 evaluates periodic DFT-D3(BJ) dispersion without a real-space cutoff
on the dispersion sum. Python expresses the differentiable scientific model;
the native engine turns its repeated particle-mesh operations into specialized
CUDA graphs.

## Scientific flow

The energy follows one data path:

```text
positions + lattice
  → coordination numbers
  → environment-dependent low-rank C6 coefficients
  → periodic B-spline scatter
  → Fourier-space interaction
  → interpolation and D3(BJ) damping
  → scalar energy
```

JAX differentiates that scalar with respect to positions and lattice strain.
The public derivative helper returns position and strain gradients; physical
forces are the negative position gradient. Energy, forces, and strain therefore
share one mathematical implementation rather than separate kernels with
independent conventions.

The Python package has three bounded parts:

- `fourierd3.dispersion`: coordination, coefficient interpolation, reciprocal
  kernel, and energy assembly;
- `fourierd3.particle_mesh`: B-splines, periodic scatter, and spectral maps;
- `fourierd3._engine`: tracing, lowering, primitives, and the native boundary.

## Compilation and execution

The engine follows this pipeline:

```text
ClosedJaxpr
  → StableHLO
  → LLVM IR for scalar maps
  → typed CUDA source
  → NVRTC + nvJitLink + cuFFTDx
  → execution plan
  → resident CUDA graph
```

StableHLO lowering is strict: an unsupported operation is an error. Candidate
plans differ in legal tilings, cache layouts, and FFT decompositions, not in
their mathematical result. The first execution benchmarks feasible candidates;
the selected flat plan is then reused.

An execution plan separates compiled payloads from structure. It contains
cubins, workspace declarations, kernel and memory nodes, dependencies, and
optional candidate choices. The bytes are a private, versioned engine detail.

## Native crate boundary

The Rust workspace deliberately has only two crates:

- `fourierd3_engine` owns compilation, CUDA loading, plans, execution, the CUDA
  source IR, XLA FFI, and the Python extension;
- `fourierd3_macros` owns the procedural macros for the CUDA source DSL and XLA
  handler wrapper. Rust requires procedural macros to live in a separate crate.

All runtime modules are private except the CUDA source IR and macro support
surface required by macro expansion. Workspace lints deny dead code and
unreachable public items.

## GPU portability

FourierD3 queries the active device's compute capability and compiles for that
target. The cache key includes the architecture, and unsupported cuFFTDx
transforms are rejected as infeasible candidates. There is no single-GPU
specialization or fixed architecture allow-list in the source.
