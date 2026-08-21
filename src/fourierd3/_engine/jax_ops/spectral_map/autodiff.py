# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import jax
from jax.interpreters import ad

from fourierd3._engine.indexing import _assemble_cotangents
from fourierd3._engine.jax_ops.spectral_map.derived_jaxprs import (
    derive_jvp_jaxpr,
    derive_transpose_jaxpr,
)
from fourierd3._engine.jax_ops.spectral_map.primitive import spectral_map_p


def _jvp(
    primals,
    tangents,
    *,
    jaxpr,
    n_grid_in,
    n_grid_out,
    fft_lengths,
    is_hermitian,
    input_signs,
    output_signs,
    index_configuration,
    outputs_shape_dtype,
    _force_reference,
):
    ic = index_configuration
    n_aux = len(primals) - n_grid_in

    primal_out = spectral_map_p.bind(
        *primals,
        jaxpr=jaxpr,
        n_grid_in=n_grid_in,
        n_grid_out=n_grid_out,
        fft_lengths=fft_lengths,
        is_hermitian=is_hermitian,
        input_signs=input_signs,
        output_signs=output_signs,
        index_configuration=ic,
        outputs_shape_dtype=outputs_shape_dtype,
        _force_reference=_force_reference,
    )

    d_grids = tangents[:n_grid_in]
    d_auxs = tangents[n_grid_in:]
    nz_grid = [g for g in range(n_grid_in) if not isinstance(d_grids[g], ad.Zero)]
    nz_aux = [k for k in range(n_aux) if not isinstance(d_auxs[k], ad.Zero)]

    if not nz_grid and not nz_aux:
        return primal_out, [ad.Zero.from_primal_value(p) for p in primal_out]

    jvp_ic = tuple(
        [ic[g] for g in range(n_grid_in)]
        + [ic[g] for g in nz_grid]
        + [ic[n_grid_in + k] for k in range(n_aux)]
        + [ic[n_grid_in + k] for k in nz_aux]
        + list(ic[n_grid_in + n_aux :])
    )

    jvp_input_signs = input_signs + tuple(input_signs[g] for g in nz_grid)

    tangent_out = spectral_map_p.bind(
        *primals[:n_grid_in],
        *[d_grids[g] for g in nz_grid],
        *primals[n_grid_in:],
        *[d_auxs[k] for k in nz_aux],
        jaxpr=derive_jvp_jaxpr(jaxpr, n_grid_in, nz_grid, nz_aux),
        n_grid_in=n_grid_in + len(nz_grid),
        n_grid_out=n_grid_out,
        fft_lengths=fft_lengths,
        is_hermitian=is_hermitian,
        input_signs=jvp_input_signs,
        output_signs=output_signs,
        index_configuration=jvp_ic,
        outputs_shape_dtype=outputs_shape_dtype,
        _force_reference=_force_reference,
    )
    return primal_out, tangent_out


def _transpose(
    cts,
    *args,
    jaxpr,
    n_grid_in,
    n_grid_out,
    fft_lengths,
    is_hermitian,
    input_signs,
    output_signs,
    index_configuration,
    outputs_shape_dtype,
    _force_reference,
):
    ic = index_configuration
    n_aux = len(args) - n_grid_in

    grids_or_undef = list(args[:n_grid_in])
    auxs_or_undef = list(args[n_grid_in:])

    undef_grid = [g for g in range(n_grid_in) if ad.is_undefined_primal(grids_or_undef[g])]
    undef_aux = [k for k in range(n_aux) if ad.is_undefined_primal(auxs_or_undef[k])]

    if (not undef_grid and not undef_aux) or all(isinstance(c, ad.Zero) for c in cts):
        result = _assemble_cotangents(grids_or_undef, {})
        result.extend(_assemble_cotangents(auxs_or_undef, {}))
        return tuple(result)

    def_grid = [g for g in range(n_grid_in) if g not in undef_grid]
    def_aux = [k for k in range(n_aux) if k not in undef_aux]
    nonzero_ct_idx = [j for j, c in enumerate(cts) if not isinstance(c, ad.Zero)]
    nz_ct_grid = [j for j in nonzero_ct_idx if j < n_grid_out]
    nz_ct_aux = [j for j in nonzero_ct_idx if j >= n_grid_out]

    transpose_ic = tuple(
        [ic[n_grid_in + n_aux + j] for j in nz_ct_grid]
        + [ic[g] for g in def_grid]
        + [ic[n_grid_in + n_aux + j] for j in nz_ct_aux]
        + [ic[n_grid_in + k] for k in def_aux]
        + [ic[g] for g in undef_grid]
        + [ic[n_grid_in + k] for k in undef_aux]
    )

    def _osd(v):
        return jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)

    transpose_osd = tuple(
        [_osd(grids_or_undef[g]) for g in undef_grid] + [_osd(auxs_or_undef[k]) for k in undef_aux]
    )

    transpose_input_signs = tuple(output_signs[j] * -1 for j in nz_ct_grid) + tuple(
        input_signs[g] for g in def_grid
    )
    transpose_output_signs = tuple(input_signs[g] * -1 for g in undef_grid)

    tr_results = spectral_map_p.bind(
        *[cts[j] for j in nz_ct_grid],
        *[grids_or_undef[g] for g in def_grid],
        *[cts[j] for j in nz_ct_aux],
        *[auxs_or_undef[k] for k in def_aux],
        jaxpr=derive_transpose_jaxpr(
            jaxpr,
            n_grid_in,
            n_grid_out,
            fft_lengths,
            needed_grid_idx=undef_grid,
            needed_aux_idx=undef_aux,
            nonzero_ct_idx=nonzero_ct_idx,
        ),
        n_grid_in=len(nz_ct_grid) + len(def_grid),
        n_grid_out=len(undef_grid),
        fft_lengths=fft_lengths,
        is_hermitian=is_hermitian,
        input_signs=transpose_input_signs,
        output_signs=transpose_output_signs,
        index_configuration=transpose_ic,
        outputs_shape_dtype=transpose_osd,
        _force_reference=_force_reference,
    )

    n_grid_out_tr = len(undef_grid)
    grid_cts = dict(zip(undef_grid, tr_results[:n_grid_out_tr], strict=False))
    aux_cts = dict(zip(undef_aux, tr_results[n_grid_out_tr:], strict=False))

    result = _assemble_cotangents(grids_or_undef, grid_cts)
    result.extend(_assemble_cotangents(auxs_or_undef, aux_cts))
    return tuple(result)
