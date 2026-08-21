<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Execution plans

FourierD3 compiles GPU work into private, versioned `RCPE` bytes. A plan holds
architecture-specific cubins, workspace declarations, dependency-ordered
kernel launches, memory initialization, and optional autotuning choices. It
contains no CUDA source.

The compiler and executor are the only consumers. The encoding is not a public
API or a cross-language interchange format, and releases may replace it. Plans
must be produced and consumed by the same FourierD3 version and GPU target.

The bounds-checked codec lives in
[`execution_plan/wire.rs`](../rust/fourierd3_engine/src/execution_plan/wire.rs).
Its round-trip tests detect accidental inconsistencies within a release.
