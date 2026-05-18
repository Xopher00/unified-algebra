"""Tensor contraction fusion — semantic-level normalize_contracts pass.

Fuses adjacent tensor contractions at the Morphism layer before substrate
decomposition.  After fusion, every remaining ``DomainPrim("tensors", ...)``
is replaced by the substrate composition via ``compile_contract_spec``.

No imports from ``unialg.runtime``, ``unialg.structure``, or ``unialg.main``.
"""
from __future__ import annotations

from collections.abc import Callable

import hydra.names as HydraNames
from hydra.core import Name, TypeVariable, TypePair, PairType, TypeUnit
from hydra.dsl.python import FrozenDict
from hydra.variables import substitute_type_variables

from unialg.objects import BINARY
from unialg.semantics import morphisms as ops
from unialg.semantics.morphisms import Morphism, MorphismError, _copy
from unialg.syntax import expressions as expr

from .notation import Equation
from .primitives import compile_contract_spec
from .semantics import ContractSpec, _collect_semiring_aux, _left_nested_shape


# ---------------------------------------------------------------------------
# Hydra-based label encoding for alpha-renaming
# ---------------------------------------------------------------------------

def _labels_to_type(labels: tuple[str, ...]):
    """Encode label tuple as a Hydra Type tree of TypeVariables."""
    if len(labels) == 0:
        return TypeUnit()
    if len(labels) == 1:
        return TypeVariable(Name(labels[0]))
    result = TypeVariable(Name(labels[0]))
    for lab in labels[1:]:
        result = TypePair(PairType(result, TypeVariable(Name(lab))))
    return result


def _type_to_labels(typ) -> tuple[str, ...]:
    """Decode a Hydra Type tree back to a label tuple."""
    if isinstance(typ, TypeUnit):
        return ()
    if isinstance(typ, TypeVariable):
        return (typ.value.value,)
    if isinstance(typ, TypePair):
        return _type_to_labels(typ.value.first) + _type_to_labels(typ.value.second)
    raise ValueError(f"unexpected type in label decode: {type(typ)}")


def _labels_in_order(inputs: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Return input labels in first-seen order."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for inp in inputs:
        for label in inp:
            if label not in seen_set:
                seen.append(label)
                seen_set.add(label)
    return tuple(seen)


def _reduced_labels(
    inputs: tuple[tuple[str, ...], ...],
    output: tuple[str, ...],
) -> tuple[str, ...]:
    """Derive reduced labels from inputs and output labels."""
    output_set = set(output)
    return tuple(label for label in _labels_in_order(inputs) if label not in output_set)


def _equation_from_inputs(
    inputs: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
    output: tuple[str, ...],
) -> Equation:
    """Build an Equation with reduced labels derived consistently."""
    merged = tuple(inputs)
    return Equation(inputs=merged, output=output, reduced=_reduced_labels(merged, output))


def _rename_reduced_labels(spec: ContractSpec, avoid: set[str]) -> ContractSpec:
    """Alpha-rename spec's reduced labels that collide with avoid set.

    Uses Hydra's substitute_type_variables for the renaming.
    """
    collisions = set(spec.equation.reduced) & avoid
    if not collisions:
        return spec

    all_used: set[str] = set()
    for inp in spec.equation.inputs:
        all_used.update(inp)
    all_used.update(spec.equation.output)
    all_used |= avoid

    rename_map: dict[str, str] = {}
    for label in sorted(collisions):
        fresh = HydraNames.unique_label(frozenset(all_used), label)
        rename_map[label] = fresh
        all_used.add(fresh)

    subst = FrozenDict({Name(old): Name(new) for old, new in rename_map.items()})

    renamed_inputs = tuple(
        _type_to_labels(substitute_type_variables(subst, _labels_to_type(inp)))
        for inp in spec.equation.inputs
    )
    renamed_output = (
        _type_to_labels(substitute_type_variables(subst, _labels_to_type(spec.equation.output)))
        if spec.equation.output else ()
    )

    return ContractSpec(
        semiring=spec.semiring,
        equation=_equation_from_inputs(renamed_inputs, renamed_output),
        adjoint=spec.adjoint,
        shape=spec.shape,
    )


# ---------------------------------------------------------------------------
# Public finalize hook
# ---------------------------------------------------------------------------

def normalize_contracts(m: Morphism, env: dict) -> Morphism:
    """Domain finalize hook: fuse adjacent contractions, then decompose all.

    Called by ``semantics/construct.py`` after each top-level morphism is built.
    ``env["_domain_context"]`` carries the opaque backend context needed by
    ``compile_contract_spec``.
    """
    ctx = env.get("_domain_context")
    original_aux = m.aux_primitives
    fused = _fuse_to_fixpoint(m)
    decomposed = _decompose_all(fused, ctx)
    # Preserve any aux_primitives that may have been lost during tree reconstruction.
    extra = tuple(p for p in original_aux if p not in decomposed.aux_primitives)
    if extra:
        return Morphism(decomposed.node, decomposed.param, decomposed.monad,
                        decomposed.aux_primitives + extra)
    return decomposed


# ---------------------------------------------------------------------------
# Bottom-up traversal helper
# ---------------------------------------------------------------------------

def _child_morphism(node, child, param) -> Morphism:
    return Morphism(node=child, param=param, monad=node.monad, aux_primitives=())


def _rebuild_contextual_binary(node, f_new: Morphism, g_new: Morphism) -> Morphism | None:
    if isinstance(node, (expr.Compose, expr.SharedCompose)):
        return ops.compose(f_new, g_new)
    if isinstance(node, expr.Parallel):
        return ops.par(f_new, g_new)
    if isinstance(node, expr.Pair):
        return ops.pair(f_new, g_new)
    if isinstance(node, expr.Case):
        return ops.case(f_new, g_new)
    return None


def _rewrite_contextual_children(m: Morphism, xf: Callable) -> Morphism:
    node = m.node
    if not isinstance(node, expr.ContextualBinary):
        return m

    if isinstance(node.f, expr.MonadicEmbed) or isinstance(node.g, expr.MonadicEmbed):
        return m

    f_m = _child_morphism(node, node.f, node.f_param)
    g_m = _child_morphism(node, node.g, node.g_param)
    f_new = _rewrite_bottom_up(f_m, xf)
    g_new = _rewrite_bottom_up(g_m, xf)

    if f_new.node is f_m.node and g_new.node is g_m.node:
        return m

    return _rebuild_contextual_binary(node, f_new, g_new) or m


def _rewrite_bottom_up(m: Morphism, xf: Callable) -> Morphism:
    """Apply xf bottom-up, rebuilding ContextualBinary nodes when children change.

    xf receives a Morphism and returns a (possibly new) Morphism.
    The return value is always the result of xf applied to the (possibly
    rebuilt) top-level node.
    """
    return xf(_rewrite_contextual_children(m, xf))


# ---------------------------------------------------------------------------
# Fusion pass
# ---------------------------------------------------------------------------

def _tensor_spec(node: expr.MorphismExpr) -> ContractSpec | None:
    """If node is a DomainPrim("tensors", ContractSpec), return the spec."""
    if (
        isinstance(node, expr.DomainPrim)
        and node.tag == "tensors"
        and isinstance(node.raw, ContractSpec)
    ):
        return node.raw
    return None


def _contract_morphism_from_spec(
    spec: ContractSpec,
    *,
    param,
    monad,
) -> Morphism:
    """Wrap a ContractSpec as a tensor DomainPrim Morphism."""
    return Morphism(
        node=expr.DomainPrim("tensors", spec, spec.dom, spec.cod),
        param=param,
        monad=monad,
        aux_primitives=_collect_semiring_aux(spec.semiring),
    )


_ABSORB_FIELDS = ("inputs", "shape", "pre", "has_opaque", "has_absorbed")


class _AbsorbResult:
    __slots__ = _ABSORB_FIELDS

    def __init__(self, inputs, shape, pre, has_opaque, has_absorbed):
        self.inputs = inputs
        self.shape = shape
        self.pre = pre
        self.has_opaque = has_opaque
        self.has_absorbed = has_absorbed


def _absorb_par_tree(par_node, outer_inputs, outer_spec):
    """Walk a Parallel tree recursively, building equation inputs + shape + pre_map.

    Returns _AbsorbResult or None if fusion is structurally impossible
    (e.g. semiring mismatch, label mismatch).
    """
    if isinstance(par_node, expr.Parallel):
        left = _absorb_par_tree(par_node.f, outer_inputs, outer_spec)
        if left is None:
            return None
        right = _absorb_par_tree(par_node.g, outer_inputs, outer_spec)
        if right is None:
            return None
        return _AbsorbResult(
            inputs=left.inputs + right.inputs,
            shape=expr.Prod(left.shape, right.shape),
            pre=ops.par(left.pre, right.pre),
            has_opaque=left.has_opaque or right.has_opaque,
            has_absorbed=left.has_absorbed or right.has_absorbed,
        )

    if isinstance(par_node, expr.Identity):
        expected = next(outer_inputs)
        return _AbsorbResult(
            inputs=[expected],
            shape=expr.Id(),
            pre=ops.identity(BINARY),
            has_opaque=False,
            has_absorbed=False,
        )

    inner_spec = _tensor_spec(par_node)

    if inner_spec is not None:
        expected = next(outer_inputs)
        if inner_spec.semiring != outer_spec.semiring:
            return None
        if inner_spec.adjoint != outer_spec.adjoint:
            return None
        if inner_spec.equation.output != expected:
            return None
        inner_shape = inner_spec.shape or _left_nested_shape(len(inner_spec.equation.inputs))
        inner_dom = inner_spec.dom
        return _AbsorbResult(
            inputs=list(inner_spec.equation.inputs),
            shape=inner_shape,
            pre=ops.identity(inner_dom),
            has_opaque=False,
            has_absorbed=True,
        )

    # Opaque leaf: not a tensor contraction, not identity.
    # Treat as residue — passthrough label, opaque morphism in pre_map.
    # Note: aux_primitives are not carried here; normalize_contracts
    # restores them from the original morphism's aux_primitives.
    expected = next(outer_inputs)
    opaque_m = Morphism(node=par_node, aux_primitives=())
    return _AbsorbResult(
        inputs=[expected],
        shape=expr.Id(),
        pre=opaque_m,
        has_opaque=True,
        has_absorbed=False,
    )


def _pair_tensor_branch(pair_node) -> tuple[ContractSpec, bool] | None:
    """Return the contract branch and whether it is first in the Pair."""
    left_spec = _tensor_spec(pair_node.f)
    right_spec = _tensor_spec(pair_node.g)

    if (left_spec is None) == (right_spec is None):
        return None

    if left_spec is not None:
        if not isinstance(pair_node.g, expr.Identity):
            return None
        return left_spec, True

    if not isinstance(pair_node.f, expr.Identity):
        return None
    return right_spec, False


def _pair_slot_labels(
    outer_inputs: list[tuple[str, ...]],
    inner_n_inputs: int,
    contract_first: bool,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]] | None:
    """Return outer contract slot labels and shared identity labels."""
    if len(outer_inputs) != 1 + inner_n_inputs:
        return None

    if contract_first:
        return outer_inputs[0], outer_inputs[1:]
    return outer_inputs[inner_n_inputs], outer_inputs[:inner_n_inputs]


def _avoid_labels(
    identity_slot_labels: list[tuple[str, ...]],
    output: tuple[str, ...],
) -> set[str]:
    avoid = set(output)
    for labels in identity_slot_labels:
        avoid.update(labels)
    return avoid


def _try_fuse_pair(m: Morphism, pair_node, outer_spec) -> Morphism | None:
    """Fuse compose(Pair(c1, id), outer) into compose(Copy, fused_contract).

    Only handles the case where exactly one Pair branch is a tensor DomainPrim
    and the other is Identity with matching labels (shared-input correspondence).
    """
    from unialg.objects import TypeUnit as TU

    if m.param != TU() or m.monad is not None:
        return None
    if pair_node.param != TU() or pair_node.monad is not None:
        return None

    branch = _pair_tensor_branch(pair_node)
    if branch is None:
        return None
    inner_spec, contract_first = branch

    if inner_spec.semiring != outer_spec.semiring:
        return None
    if inner_spec.adjoint != outer_spec.adjoint:
        return None

    outer_inputs = list(outer_spec.equation.inputs)
    inner_n_inputs = len(inner_spec.equation.inputs)
    slots = _pair_slot_labels(outer_inputs, inner_n_inputs, contract_first)
    if slots is None:
        return None
    contract_slot_labels, identity_slot_labels = slots

    if inner_spec.equation.output != contract_slot_labels:
        return None
    if tuple(identity_slot_labels) != inner_spec.equation.inputs:
        return None

    avoid = _avoid_labels(identity_slot_labels, outer_spec.equation.output)
    renamed_spec = _rename_reduced_labels(inner_spec, avoid)

    inner_shape = renamed_spec.shape or _left_nested_shape(inner_n_inputs)
    id_shape = inner_spec.shape or _left_nested_shape(inner_n_inputs)

    if contract_first:
        fused_inputs = list(renamed_spec.equation.inputs) + list(identity_slot_labels)
        fused_shape = expr.Prod(inner_shape, id_shape)
    else:
        fused_inputs = list(identity_slot_labels) + list(renamed_spec.equation.inputs)
        fused_shape = expr.Prod(id_shape, inner_shape)

    fused_eq = _equation_from_inputs(fused_inputs, outer_spec.equation.output)
    fused_spec = ContractSpec(
        semiring=inner_spec.semiring,
        equation=fused_eq,
        adjoint=inner_spec.adjoint,
        shape=fused_shape,
    )

    fused_contract = _contract_morphism_from_spec(
        fused_spec,
        param=m.param,
        monad=m.monad,
    )

    shared_dom = pair_node.dom
    copy_m = _copy(shared_dom)

    if copy_m.cod() != fused_contract.dom():
        return None

    return ops.compose(copy_m, fused_contract)


def _try_fuse(m: Morphism) -> Morphism:
    """If m is compose(par_tree, outer_contract), try to fuse into one contract.

    Walks the par-tree recursively (not flattening), building equation inputs
    and polynomial shape together. The shape preserves the tree nesting so
    apply_poly(shape, BINARY) matches the original compose's dom.
    """
    node = m.node
    if not isinstance(node, expr.Compose):
        return m

    outer_spec = _tensor_spec(node.g)
    if outer_spec is None:
        return m

    if isinstance(node.f, expr.Pair):
        pair_result = _try_fuse_pair(m, node.f, outer_spec)
        if pair_result is not None:
            return pair_result

    outer_inputs = iter(outer_spec.equation.inputs)
    result = _absorb_par_tree(node.f, outer_inputs, outer_spec)

    if result is None:
        return m

    remaining = list(outer_inputs)
    if remaining:
        return m

    if result.has_opaque and not result.has_absorbed:
        return m

    fused_eq = _equation_from_inputs(result.inputs, outer_spec.equation.output)
    fused_spec = ContractSpec(
        semiring=outer_spec.semiring,
        equation=fused_eq,
        adjoint=outer_spec.adjoint,
        shape=result.shape,
    )

    fused_contract = _contract_morphism_from_spec(
        fused_spec,
        param=m.param,
        monad=m.monad,
    )

    if result.has_opaque:
        # pre_map >> fused_contract: optic operational form
        pre_map = result.pre
        if pre_map.cod() != fused_contract.dom():
            return m
        return ops.compose(pre_map, fused_contract)

    if fused_spec.dom != m.dom() or fused_spec.cod != m.cod():
        return m
    return fused_contract


def _fuse_one(m: Morphism) -> Morphism:
    return _rewrite_bottom_up(m, _try_fuse)


def _fuse_to_fixpoint(m: Morphism, *, max_rounds: int = 100) -> Morphism:
    for _ in range(max_rounds):
        nxt = _fuse_one(m)
        if nxt.node is m.node:
            return nxt
        m = nxt
    raise MorphismError("contract normalization did not converge")


# ---------------------------------------------------------------------------
# Decompose pass
# ---------------------------------------------------------------------------

def _decompose_leaf(m: Morphism, ctx) -> Morphism:
    """Replace a tensor DomainPrim with its compiled substrate composition."""
    node = m.node
    if not (
        isinstance(node, expr.DomainPrim)
        and node.tag == "tensors"
        and isinstance(node.raw, ContractSpec)
    ):
        return m
    result = compile_contract_spec(node.raw, ctx)
    if result.dom() != m.dom() or result.cod() != m.cod():
        raise MorphismError(
            f"DomainPrim decomposition produced wrong dom/cod for {node.raw.equation!r}"
        )
    return result


def _decompose_all(m: Morphism, ctx) -> Morphism:
    """Replace every DomainPrim("tensors") node with its substrate composition."""
    return _rewrite_bottom_up(m, lambda leaf: _decompose_leaf(leaf, ctx))
