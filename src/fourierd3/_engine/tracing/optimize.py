# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import string
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core

DEFAULT_PASSES: list[str] = [
    "mul_fusion",
]

# When True, assume inputs are never NaN/Inf, enabling aggressive simplifications
# like x*0 -> 0. Set by `optimize()` via the `fast_math` parameter.
ignore_nan_inf: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ignore_nan_inf", default=False
)


def _new_var(aval: core.AbstractValue) -> core.Var:
    return core.Var(aval)


def _const_select_fold(jaxpr: core.Jaxpr, consts: list) -> tuple[core.Jaxpr, list]:
    const_vals: dict[int, np.ndarray] = {}
    for cv, cval in zip(jaxpr.constvars, consts, strict=False):
        const_vals[id(cv)] = np.asarray(cval)

    alias: dict[int, core.Atom] = {}

    def resolve(atom):
        if isinstance(atom, core.Literal):
            return atom
        key = id(atom)
        while key in alias:
            atom = alias[key]
            if isinstance(atom, core.Literal):
                return atom
            key = id(atom)
        return atom

    new_eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name != "select_n":
            remapped = eqn.replace(invars=[resolve(iv) for iv in eqn.invars])
            new_eqns.append(remapped)
            continue

        cond = resolve(eqn.invars[0])
        cond_val = None
        if isinstance(cond, core.Literal):
            cond_val = np.asarray(cond.val)
        else:
            cond_val = const_vals.get(id(cond))

        if cond_val is None:
            remapped = eqn.replace(invars=[resolve(iv) for iv in eqn.invars])
            new_eqns.append(remapped)
            continue

        idx = int(cond_val)
        selected = resolve(eqn.invars[1 + idx])
        for ov in eqn.outvars:
            if not isinstance(ov, core.DropVar):
                alias[id(ov)] = selected

    new_outvars = [resolve(ov) for ov in jaxpr.outvars]
    return jaxpr.replace(eqns=new_eqns, outvars=new_outvars), consts


def _algebra(jaxpr: core.Jaxpr, consts: list) -> tuple[core.Jaxpr, list]:
    _fast = ignore_nan_inf.get()
    const_vals: dict[int, np.ndarray] = {}
    for cv, cval in zip(jaxpr.constvars, consts, strict=False):
        const_vals[id(cv)] = np.asarray(cval)

    alias: dict[int, core.Atom] = {}
    new_consts = list(consts)
    new_constvars = list(jaxpr.constvars)

    def _add_const(aval, value):
        nv = _new_var(aval)
        new_constvars.append(nv)
        new_consts.append(value)
        const_vals[id(nv)] = np.asarray(value)
        return nv

    def resolve(atom):
        if isinstance(atom, core.Literal):
            return atom
        key = id(atom)
        while key in alias:
            atom = alias[key]
            if isinstance(atom, core.Literal):
                return atom
            key = id(atom)
        return atom

    def _get_scalar_const(atom):
        atom = resolve(atom)
        if isinstance(atom, core.Literal):
            v = np.asarray(atom.val)
        else:
            v = const_vals.get(id(atom))
        if v is None:
            return None
        if v.size != 1:
            return None
        return float(v.flat[0])

    new_eqns = []
    for eqn in jaxpr.eqns:
        prim = eqn.primitive.name
        handled = False

        if prim == "mul" and len(eqn.invars) == 2:
            a_val = _get_scalar_const(eqn.invars[0])
            b_val = _get_scalar_const(eqn.invars[1])
            ov = eqn.outvars[0]
            if not isinstance(ov, core.DropVar):
                if _fast and (a_val == 0.0 or b_val == 0.0):
                    zero = np.zeros(ov.aval.shape, dtype=ov.aval.dtype)
                    alias[id(ov)] = _add_const(ov.aval, zero)
                    handled = True
                elif a_val == 1.0:
                    alias[id(ov)] = resolve(eqn.invars[1])
                    handled = True
                elif b_val == 1.0:
                    alias[id(ov)] = resolve(eqn.invars[0])
                    handled = True

        elif prim in ("add", "add_any") and len(eqn.invars) == 2:
            a_val = _get_scalar_const(eqn.invars[0])
            b_val = _get_scalar_const(eqn.invars[1])
            ov = eqn.outvars[0]
            if not isinstance(ov, core.DropVar):
                if a_val == 0.0:
                    alias[id(ov)] = resolve(eqn.invars[1])
                    handled = True
                elif b_val == 0.0:
                    alias[id(ov)] = resolve(eqn.invars[0])
                    handled = True

        elif prim == "sub" and len(eqn.invars) == 2:
            b_val = _get_scalar_const(eqn.invars[1])
            ov = eqn.outvars[0]
            if not isinstance(ov, core.DropVar) and b_val == 0.0:
                alias[id(ov)] = resolve(eqn.invars[0])
                handled = True

        elif prim == "neg" and len(eqn.invars) == 1:
            a_val = _get_scalar_const(eqn.invars[0])
            ov = eqn.outvars[0]
            if not isinstance(ov, core.DropVar) and a_val == 0.0:
                zero = np.zeros(ov.aval.shape, dtype=ov.aval.dtype)
                alias[id(ov)] = _add_const(ov.aval, zero)
                handled = True

        if not handled:
            remapped = eqn.replace(invars=[resolve(iv) for iv in eqn.invars])
            new_eqns.append(remapped)

    new_outvars = [resolve(ov) for ov in jaxpr.outvars]
    return (
        jaxpr.replace(constvars=new_constvars, eqns=new_eqns, outvars=new_outvars),
        new_consts,
    )


def _dce(jaxpr: core.Jaxpr, consts: list) -> tuple[core.Jaxpr, list]:
    live: set[int] = set()
    for ov in jaxpr.outvars:
        if not isinstance(ov, core.Literal):
            live.add(id(ov))

    kept: list[core.JaxprEqn] = []
    for eqn in reversed(jaxpr.eqns):
        needed = any(id(ov) in live for ov in eqn.outvars if not isinstance(ov, core.DropVar))
        if not needed:
            continue
        kept.append(eqn)
        for iv in eqn.invars:
            if not isinstance(iv, core.Literal):
                live.add(id(iv))

    kept.reverse()

    used_consts = []
    used_constvars = []
    for cv, cval in zip(jaxpr.constvars, consts, strict=False):
        if id(cv) in live:
            used_consts.append(cval)
            used_constvars.append(cv)

    return (
        jaxpr.replace(constvars=used_constvars, eqns=kept),
        used_consts,
    )


def _const_dedup(jaxpr: core.Jaxpr, consts: list) -> tuple[core.Jaxpr, list]:
    canonical: dict[tuple, core.Var] = {}
    alias: dict[int, core.Var] = {}
    keep_consts: list = []
    keep_constvars: list[core.Var] = []

    for cv, cval in zip(jaxpr.constvars, consts, strict=False):
        arr = np.asarray(cval)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key in canonical:
            alias[id(cv)] = canonical[key]
        else:
            canonical[key] = cv
            keep_consts.append(cval)
            keep_constvars.append(cv)

    if not alias:
        return jaxpr, consts

    def resolve(atom):
        if isinstance(atom, core.Literal):
            return atom
        return alias.get(id(atom), atom)

    new_eqns = [eqn.replace(invars=[resolve(iv) for iv in eqn.invars]) for eqn in jaxpr.eqns]
    new_outvars = [resolve(ov) for ov in jaxpr.outvars]
    return (
        jaxpr.replace(constvars=keep_constvars, eqns=new_eqns, outvars=new_outvars),
        keep_consts,
    )


def _eqn_key(eqn: core.JaxprEqn):
    invars_key = []
    for iv in eqn.invars:
        if isinstance(iv, core.Literal):
            v = iv.val
            if hasattr(v, "tobytes"):
                invars_key.append(("lit", v.dtype.str, v.tobytes()))
            else:
                invars_key.append(("lit", type(v).__name__, v))
        else:
            invars_key.append(("var", id(iv)))

    def _freeze(v):
        if isinstance(v, dict):
            return tuple(sorted((_freeze(k), _freeze(val)) for k, val in v.items()))
        if isinstance(v, (list, tuple)):
            return tuple(_freeze(x) for x in v)
        if isinstance(v, np.ndarray):
            return ("np", v.dtype.str, v.tobytes())
        if isinstance(v, core.ClosedJaxpr):
            return id(v)  # sub-jaxprs are identity-compared
        try:
            hash(v)
            return v
        except TypeError:
            return id(v)

    params_key = _freeze(eqn.params)
    return (eqn.primitive.name, params_key, tuple(invars_key))


def _cse(jaxpr: core.Jaxpr, consts: list) -> tuple[core.Jaxpr, list]:
    seen: dict = {}
    alias: dict[int, core.Atom] = {}

    def resolve(atom):
        if isinstance(atom, core.Literal):
            return atom
        key = id(atom)
        while key in alias:
            atom = alias[key]
            if isinstance(atom, core.Literal):
                return atom
            key = id(atom)
        return atom

    new_eqns = []
    for eqn in jaxpr.eqns:
        remapped = eqn.replace(invars=[resolve(iv) for iv in eqn.invars])
        key = _eqn_key(remapped)
        prev = seen.get(key)
        if prev is not None and len(prev) == len(remapped.outvars):
            for ov, pv in zip(remapped.outvars, prev, strict=False):
                if not isinstance(ov, core.DropVar):
                    alias[id(ov)] = pv
        else:
            seen[key] = [ov for ov in remapped.outvars if not isinstance(ov, core.DropVar)]
            new_eqns.append(remapped)

    new_outvars = [resolve(ov) for ov in jaxpr.outvars]
    result = jaxpr.replace(eqns=new_eqns, outvars=new_outvars)
    return _dce(result, consts)


_MIN_CHAIN_LEAVES = 3  # minimum leaf operands to trigger fusion (≥ 2 muls)


def _get_atom_shape(atom: core.Atom) -> tuple[int, ...]:
    if isinstance(atom, core.Literal):
        return np.asarray(atom.val).shape
    return atom.aval.shape


def _mul_fusion(jaxpr: core.Jaxpr, consts: list) -> tuple[core.Jaxpr, list]:
    producer: dict[int, tuple[int, core.JaxprEqn]] = {}
    for i, eqn in enumerate(jaxpr.eqns):
        for ov in eqn.outvars:
            if not isinstance(ov, core.DropVar):
                producer[id(ov)] = (i, eqn)

    use_count: dict[int, int] = {}
    for eqn in jaxpr.eqns:
        for iv in eqn.invars:
            if not isinstance(iv, core.Literal):
                use_count[id(iv)] = use_count.get(id(iv), 0) + 1
    for ov in jaxpr.outvars:
        if not isinstance(ov, core.Literal):
            use_count[id(ov)] = use_count.get(id(ov), 0) + 1

    absorbed_all: set[int] = set()
    replacements: list[tuple[int, list, set]] = []

    def _walk(
        atom: core.Atom,
        dim_to_out: dict[int, int],
        chain_absorbed: set[int],
        leaves: list,
    ) -> None:
        if isinstance(atom, core.Literal):
            leaves.append((atom, dim_to_out))
            return
        vid = id(atom)
        if vid not in producer or use_count.get(vid, 0) != 1:
            leaves.append((atom, dim_to_out))
            return
        idx, peqn = producer[vid]
        if idx in absorbed_all:
            leaves.append((atom, dim_to_out))
            return

        pname = peqn.primitive.name

        if pname == "mul" and len(peqn.invars) == 2:
            chain_absorbed.add(idx)
            peqn_out_shape = peqn.outvars[0].aval.shape
            peqn_ndim = len(peqn_out_shape)
            for iv in peqn.invars:
                iv_shape = _get_atom_shape(iv)
                iv_ndim = len(iv_shape)
                iv_d2o: dict[int, int] = {}
                offset = peqn_ndim - iv_ndim
                for d in range(iv_ndim):
                    pd = offset + d
                    if iv_shape[d] == 1 and peqn_out_shape[pd] != 1:
                        continue  # broadcast dimension
                    if pd in dim_to_out:
                        iv_d2o[d] = dim_to_out[pd]
                _walk(iv, iv_d2o, chain_absorbed, leaves)
            return

        if pname == "broadcast_in_dim":
            bcast_dims: tuple[int, ...] = peqn.params["broadcast_dimensions"]
            bcast_covered = set(bcast_dims)
            if any(d not in bcast_covered for d in dim_to_out):
                leaves.append((atom, dim_to_out))
                return
            chain_absorbed.add(idx)
            iv = peqn.invars[0]
            iv_shape = _get_atom_shape(iv)
            peqn_out_shape = peqn.outvars[0].aval.shape
            iv_d2o: dict[int, int] = {}
            for d_in, d_eqn in enumerate(bcast_dims):
                if iv_shape[d_in] == 1 and peqn_out_shape[d_eqn] != 1:
                    continue
                if d_eqn in dim_to_out:
                    iv_d2o[d_in] = dim_to_out[d_eqn]
            _walk(iv, iv_d2o, chain_absorbed, leaves)
            return

        leaves.append((atom, dim_to_out))

    for root_i in reversed(range(len(jaxpr.eqns))):
        if root_i in absorbed_all:
            continue
        eqn = jaxpr.eqns[root_i]
        if eqn.primitive.name != "mul" or len(eqn.invars) != 2:
            continue
        root_ov = eqn.outvars[0]
        if isinstance(root_ov, core.DropVar):
            continue

        out_shape = root_ov.aval.shape
        out_ndim = len(out_shape)
        chain_absorbed: set[int] = {root_i}
        leaves: list = []

        for iv in eqn.invars:
            iv_shape = _get_atom_shape(iv)
            iv_ndim = len(iv_shape)
            iv_d2o: dict[int, int] = {}
            offset = out_ndim - iv_ndim
            for d in range(iv_ndim):
                out_d = offset + d
                if iv_shape[d] == 1 and out_shape[out_d] != 1:
                    continue
                iv_d2o[d] = out_d
            _walk(iv, iv_d2o, chain_absorbed, leaves)

        if len(leaves) < _MIN_CHAIN_LEAVES:
            continue

        absorbed_all.update(chain_absorbed)
        replacements.append((root_i, leaves, chain_absorbed))

    if not replacements:
        return jaxpr, consts

    alpha = string.ascii_lowercase

    new_consts = list(consts)
    new_constvars = list(jaxpr.constvars)
    replacement_data: dict[int, tuple[list[core.JaxprEqn], core.Atom]] = {}

    for root_idx, leaves, _chain_abs in replacements:
        root_eqn = jaxpr.eqns[root_idx]
        out_var = root_eqn.outvars[0]
        out_shape = out_var.aval.shape
        out_ndim = len(out_shape)
        out_dtype = out_var.aval.dtype
        out_sub = alpha[:out_ndim]

        subs: list[str] = []
        squeeze_info: list[tuple[int, ...] | None] = []
        leaf_atoms: list[core.Atom] = []

        for leaf_atom, d2o in leaves:
            leaf_ndim = len(_get_atom_shape(leaf_atom))
            sub = ""
            keep: list[int] = []
            for d in range(leaf_ndim):
                if d in d2o:
                    sub += alpha[d2o[d]]
                    keep.append(d)
            sq = tuple(d for d in range(leaf_ndim) if d not in keep)
            subs.append(sub)
            squeeze_info.append(sq if sq else None)
            leaf_atoms.append(leaf_atom)

        einsum_str = ",".join(subs) + "->" + out_sub

        def _make_einsum_fn(es: str, si: list[tuple[int, ...] | None]):
            def fn(*args):
                processed = []
                for arg, sq in zip(args, si, strict=False):
                    if sq:
                        arg = jnp.squeeze(arg, axis=sq)
                    processed.append(arg)
                return jnp.einsum(es, *processed, optimize="optimal")

            return fn

        dummy_args = [jnp.zeros(_get_atom_shape(la), dtype=out_dtype) for la in leaf_atoms]
        traced = jax.make_jaxpr(_make_einsum_fn(einsum_str, squeeze_info))(*dummy_args)

        sub_jaxpr = traced.jaxpr
        var_map: dict[int, core.Atom] = {}

        for sv, sc in zip(sub_jaxpr.constvars, traced.consts, strict=False):
            nv = _new_var(sv.aval)
            new_constvars.append(nv)
            new_consts.append(sc)
            var_map[id(sv)] = nv

        for sv, la in zip(sub_jaxpr.invars, leaf_atoms, strict=False):
            var_map[id(sv)] = la

        sub_eqns: list[core.JaxprEqn] = []
        for sub_eqn in sub_jaxpr.eqns:
            new_invars = [
                iv if isinstance(iv, core.Literal) else var_map.get(id(iv), iv)
                for iv in sub_eqn.invars
            ]
            new_outvars = []
            for ov in sub_eqn.outvars:
                if isinstance(ov, core.DropVar):
                    new_outvars.append(ov)
                else:
                    nv = _new_var(ov.aval)
                    var_map[id(ov)] = nv
                    new_outvars.append(nv)
            sub_eqns.append(sub_eqn.replace(invars=new_invars, outvars=new_outvars))

        sub_out = sub_jaxpr.outvars[0]
        mapped_out = var_map.get(id(sub_out), sub_out)
        replacement_data[root_idx] = (sub_eqns, mapped_out)

    alias: dict[int, core.Atom] = {}
    for root_idx, _, _ in replacements:
        root_eqn = jaxpr.eqns[root_idx]
        _, mapped_out = replacement_data[root_idx]
        for ov in root_eqn.outvars:
            if not isinstance(ov, core.DropVar):
                alias[id(ov)] = mapped_out

    def resolve(atom: core.Atom) -> core.Atom:
        if isinstance(atom, core.Literal):
            return atom
        return alias.get(id(atom), atom)

    new_eqns: list[core.JaxprEqn] = []
    for i, eqn in enumerate(jaxpr.eqns):
        if i in absorbed_all:
            if i in replacement_data:
                sub_eqns, _ = replacement_data[i]
                new_eqns.extend(sub_eqns)
            continue
        new_eqns.append(eqn)

    final_eqns = [eqn.replace(invars=[resolve(iv) for iv in eqn.invars]) for eqn in new_eqns]
    new_outvars = [resolve(ov) for ov in jaxpr.outvars]

    return (
        jaxpr.replace(constvars=new_constvars, eqns=final_eqns, outvars=new_outvars),
        new_consts,
    )


_PASS_TABLE: dict[str, callable] = {
    "const_select_fold": _const_select_fold,
    "algebra": _algebra,
    "mul_fusion": _mul_fusion,
    "dce": _dce,
    "const_dedup": _const_dedup,
    "cse": _cse,
}


def optimize(
    closed_jaxpr: core.ClosedJaxpr,
    *,
    passes: Sequence[str] | None = None,
    fast_math: bool = False,
) -> core.ClosedJaxpr:
    if passes is None:
        passes = DEFAULT_PASSES

    jaxpr = closed_jaxpr.jaxpr
    consts = list(closed_jaxpr.consts)

    token = ignore_nan_inf.set(fast_math)
    try:
        for name in passes:
            fn = _PASS_TABLE.get(name)
            if fn is None:
                raise ValueError(
                    f"Unknown optimization pass {name!r}. Available: {sorted(_PASS_TABLE)}"
                )
            jaxpr, consts = fn(jaxpr, consts)
    finally:
        ignore_nan_inf.reset(token)

    return core.ClosedJaxpr(jaxpr, consts)
