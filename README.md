# FourierD3

FourierD3 evaluates periodic DFT-D3(BJ) dispersion energy and its position and
strain derivatives on NVIDIA GPUs from JAX. It combines a low-rank representation of the
environment-dependent D3 coefficients with particle-mesh summation so the
long-range contribution scales as O(N log N) without a real-space cutoff on
the dispersion sum.

This repository contains the complete source needed to build `FourierD3-acc`,
the accelerated JAX, Rust, and CUDA implementation described in the paper. Its
release artifacts are immutable source archives generated from reviewed tags.

## Research release

FourierD3 accompanies [*A fast summation method for the DFT-D3 dispersion
correction*](https://arxiv.org/abs/2607.15103v1) by Victoria Valeeva, Cheuk Hin
Ho, Mario Geiger, Franco Pellegrini, Gábor Csányi, Emine Kucukbenli, and
Christoph Ortner.

The separate [`FourierD3-torch`](https://github.com/vicvaleeva/FourierD3)
repository contains the reference implementation and scripts used to reproduce
the paper's results. This repository contains the accelerated implementation.

FourierD3 is a one-time research release and is not actively maintained. The
project is not accepting external contributions; see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Python API

```python
from fourierd3 import dispersion_energy_and_derivatives

energy, position_gradient, strain_gradient = dispersion_energy_and_derivatives(
    positions,
    lattice,
    metadata,
)
```

`positions` has axes `(atom, xyz)`, and `lattice` uses the row-vector convention
`(cell_vector, xyz)`. The returned position gradient has axes `(atom, xyz)` and
the strain gradient has axes `(xyz_out, xyz_in)`. Physical forces are the
negative position gradient.

For compilation, close the metadata over the JIT-compiled function so its grid
and transfer-function structure remain static:

```python
import jax

run = jax.jit(
    lambda positions, lattice: dispersion_energy_and_derivatives(positions, lattice, metadata)
)
```

`metadata` is a mapping with the following contract:

| Key | Shape or type | Meaning |
|---|---|---|
| `species` | `(atom,)` integer | Species index for each atom |
| `n_species` | integer | Number of represented species |
| `rcov` | `(atom,)` | Covalent radius for each atom |
| `cnref` | `(species, reference)` | Reference coordination numbers |
| `v_q` | `(species, reference, rank)` | Low-rank C6 factors |
| `eigs` | `(rank,)` | Low-rank eigenvalues |
| `selfcont` | `(species, rank)` or broadcastable | Reciprocal self contribution |
| `sqrtQz` | `(species,)` | Square root of the D3 Q factor |
| `params` | `(4,)` | `(s6, s8, a1, a2)` damping parameters |
| `grid_size` | three integers | Periodic mesh extents |
| `r_cut` | scalar | Coordination-number cutoff |
| `transfer_fn` | tuple | Transfer function returned by `fourierd3.bspline_transfer` |
| `src`, `dst` | `(edge,)` integer | Directed neighbor-list endpoints |
| `shift` | `(edge, 3)` integer | Periodic image shift in lattice vectors |

All length-valued inputs and damping parameters must use one consistent unit
system. The coefficient tensors and damping parameters are model data; this
package does not bundle a parameter database.

The supported surface is exactly three names:

- `dispersion_energy(strain, positions, lattice, metadata)`;
- `dispersion_energy_and_derivatives(positions, lattice, metadata)`;
- `bspline_transfer(order)`, which builds the `transfer_fn` entry above.

Everything else is an implementation detail. `fourierd3.dispersion` holds the
stages the energy is assembled from, `fourierd3.particle_mesh` holds the mesh
operations, and `fourierd3._engine` holds the tracing, compilation, and
execution machinery; none of them is a stable interface.

## Source layout

```
src/fourierd3/        the Python package
  dispersion/         coordination numbers, C6 interpolation, reciprocal kernel, energy
  particle_mesh/      B-spline transfer functions, periodic scatter, spectral map
  _engine/            jaxpr tracing, StableHLO → LLVM, the JAX primitives, the runtime
rust/
  fourierd3_engine/   complete native engine and Python XLA FFI extension
    kernel_compiler/  LLVM IR → CUDA source → cubin, plus autotune candidates
    execution_plan/   serialized compiler/executor contract
    plan_executor/    one-time loading and CUDA graph replay
    ir/               typed CUDA source construction
    cuda_compiler/    NVRTC and nvJitLink
    cuda_driver/      minimal libcuda bindings used by FourierD3
  fourierd3_macros/   CUDA source DSL and XLA FFI procedural macros
```

The two flows the tree follows are `coordination → coefficients → particle
mesh → spectral interaction` on the science side, and `JAX → StableHLO → LLVM →
CUDA → execution plan → CUDA graph` on the engine side.

See the [architecture](docs/architecture.md) and
[development workflow](docs/development.md) for the native design and its
validation contract.

## Build and test

```bash
uv venv
source .venv/bin/activate
uv pip install '.[test,cuda13]'
maturin develop
pytest
```

Use `cuda12` instead on a CUDA 12 system. A bare install leaves the CUDA JAX,
nvJitLink, and mathDx wheels out for environments that provide compatible
system libraries themselves.

## Supported GPUs and CUDA versions

FourierD3 compiles for the compute capability of the device it runs on. The
engine reads that capability from the driver and passes it to NVRTC and
cuFFTDx; there is no architecture allow-list, and the cubin cache is keyed by
the target so one host may serve several generations. Unsupported cuFFTDx
transforms are rejected as infeasible candidates. The emitted warp collectives
use the Volta-and-later `.sync` forms, which puts the floor at `sm_70`; there is
no fixed ceiling.

The engine resolves `libcuda`, NVRTC, nvJitLink, and libmathdx by soname at
run time and accepts CUDA 12 or 13. A reciprocal-space transform the installed
cuFFTDx does not support for the device is reported as infeasible rather than
mis-compiled.

## License

FourierD3 is licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
