<!-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development

## Environment

FourierD3 requires Python 3.11 or later, Rust, a C/C++ toolchain, and an NVIDIA
driver. CUDA 12 and CUDA 13 dependency groups are provided.

```bash
uv venv
source .venv/bin/activate
uv pip install '.[test,cuda13]'
maturin develop --release
```

Use `cuda12` instead of `cuda13` when that is the installed CUDA family. The
native libraries are resolved at run time, so `LD_LIBRARY_PATH` must include
their wheel or system-library directory when the loader cannot find them by
soname.

## Validation

Run the source checks before committing:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
uvx ruff check .
uvx ruff format --check .
pytest -q
```

The Rust suite covers the CUDA DSL, LLVM lowering, plan codec, autotune
structure, NVRTC/nvJitLink integration, and runtime graph logic. The
Python GPU test covers energy together with position and strain derivatives.
Long or memory-heavy commands should run in a bounded `memcap` scope.

## Native invariants

- The workspace contains `fourierd3_engine` and `fourierd3_macros`, and no
  additional runtime crates.
- Unsupported compiler input fails explicitly.
- `dead_code` and `unreachable_pub` are denied workspace-wide.
- Every execution-plan dependency points to an earlier node.
- Choice candidates implement the same bound-buffer effect.
- GPU architecture is detected at run time and included in compiled-artifact
  cache keys.

## Source release

Build and inspect the source distribution independently of the working tree:

```bash
maturin sdist --out /tmp/fourierd3-sdist
tar -tzf /tmp/fourierd3-sdist/fourierd3-*.tar.gz
```

The archive must contain the Rust sources, Python package, tests, `LICENSE`,
`NOTICE`, and policy documents. It must not contain `.git`, `target`, `.venv`,
cache directories, internal parent-project sources, or generated local
artifacts. Rebuilding a wheel from the extracted archive verifies that the
release is self-contained.
