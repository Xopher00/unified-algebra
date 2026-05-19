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
from unialg.semantics.functors import Functor as _Functor
from unialg.semantics.morphisms import Morphism, MorphismError, _copy
from unialg.semantics.optics import Optic as _Optic
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


def _is_all_identity(node) -> bool:
    """True when node is a tree of Identity and Parallel(Identity, ...) nodes."""
    if isinstance(node, expr.Identity):
        return True
    if isinstance(node, expr.Parallel):
        return _is_all_identity(node.f) and _is_all_identity(node.g)
    return False


def _count_slots(space) -> int:
    """Count BINARY factors in a nested TypePair."""
    if isinstance(space, TypePair):
        return _count_slots(space.value.first) + _count_slots(space.value.second)
    return 1


class _LeafData:
    __slots__ = ("inputs", "has_opaque", "has_absorbed")

    def __init__(self, inputs, has_opaque, has_absorbed):
        self.inputs = inputs
        self.has_opaque = has_opaque
        self.has_absorbed = has_absorbed


def _leaf_optic(body, fwd, carrier=BINARY):
    return _Optic(functor=_Functor(name="_", body=body), forward=fwd, backward=fwd, carrier=carrier)


def _par_to_optic(par_node, outer_inputs, outer_spec, avoid):
    """Reduce a Parallel/Pair/Identity/contract subtree to (Optic, _LeafData).

    Polynomial-functor catamorphism whose algebra emits an Optic.
    Returns None on structural mismatch.
    """
    from unialg.objects import TypeUnit as TU

    # Binary nodes: Parallel combines directly; Pair factors through Copy.
    if isinstance(par_node, (expr.Parallel, expr.Pair)):
        is_pair = isinstance(par_node, expr.Pair)
        if is_pair and (par_node.param != TU() or par_node.monad is not None):
            return None
        left = _par_to_optic(par_node.f, outer_inputs, outer_spec, avoid)
        if left is None:
            return None
        l_optic, l_data = left
        right = _par_to_optic(par_node.g, outer_inputs, outer_spec, avoid)
        if right is None:
            return None
        r_optic, r_data = right
        try:
            combined = l_optic.par(r_optic)
        except MorphismError:
            return None
        if is_pair:
            copy_m = _copy(par_node.dom)
            fwd = copy_m if _is_all_identity(combined.forward.node) else ops.compose(copy_m, combined.forward)
            combined = _Optic(functor=combined.functor, forward=fwd, backward=combined.backward, carrier=BINARY)
        return combined, _LeafData(
            l_data.inputs + r_data.inputs,
            l_data.has_opaque or r_data.has_opaque,
            l_data.has_absorbed or r_data.has_absorbed,
        )

    if isinstance(par_node, expr.Identity):
        n = _count_slots(par_node.space)
        consumed = tuple(next(outer_inputs) for _ in range(n))
        avoid.update(l for slot in consumed for l in slot)
        shape = _left_nested_shape(n) if n > 1 else expr.Id()
        return _leaf_optic(shape, ops.identity(par_node.space)), _LeafData(consumed, False, False)

    inner_spec = _tensor_spec(par_node)
    if inner_spec is not None:
        expected = next(outer_inputs)
        if (inner_spec.semiring != outer_spec.semiring
                or inner_spec.adjoint != outer_spec.adjoint
                or inner_spec.equation.output != expected):
            return None
        renamed = _rename_reduced_labels(inner_spec, avoid)
        avoid.update(l for inp in renamed.equation.inputs for l in inp)
        avoid.update(renamed.equation.reduced)
        shape = renamed.shape or _left_nested_shape(len(renamed.equation.inputs))
        return _leaf_optic(shape, ops.identity(renamed.dom)), _LeafData(tuple(renamed.equation.inputs), False, True)

    # Opaque leaf
    expected = next(outer_inputs)
    fwd = Morphism(node=par_node, aux_primitives=())
    optic = _Optic(functor=_Functor(name="_", body=expr.Id()), forward=fwd, backward=ops.identity(BINARY), carrier=None)
    return optic, _LeafData((expected,), True, False)


def _try_fuse(m: Morphism) -> Morphism:
    """Fuse compose(par_tree, outer_contract) into a single contract.

    Reduces the par-tree via _par_to_optic — the polynomial-functor catamorphism
    whose algebra emits an Optic (functor.body=shape, forward=pre-map).  The
    fused contract is built from the absorbed inputs and composed with the
    pre-map when non-trivial.
    """
    node = m.node
    if not isinstance(node, expr.Compose):
        return m

    outer_spec = _tensor_spec(node.g)
    if outer_spec is None:
        return m

    avoid = set(outer_spec.equation.output)
    outer_inputs = iter(outer_spec.equation.inputs)
    result = _par_to_optic(node.f, outer_inputs, outer_spec, avoid)

    if result is None:
        return m

    if list(outer_inputs):
        return m

    inner_optic, leaf = result

    if leaf.has_opaque and not leaf.has_absorbed:
        return m

    fused_eq = _equation_from_inputs(leaf.inputs, outer_spec.equation.output)
    fused_spec = ContractSpec(
        semiring=outer_spec.semiring,
        equation=fused_eq,
        adjoint=outer_spec.adjoint,
        shape=inner_optic.functor.body,
    )

    fused_contract = _contract_morphism_from_spec(
        fused_spec,
        param=m.param,
        monad=m.monad,
    )

    pre = inner_optic.forward
    if pre.dom() != m.dom() or pre.cod() != fused_contract.dom():
        return m
    if fused_spec.cod != m.cod():
        return m

    # Identity-tree forward (pure Parallel with no opaque/Copy routing) — skip wrap.
    if _is_all_identity(pre.node):
        return fused_contract
    # Non-trivial routing (Copy from Pair, opaque pre-map). Only worth wrapping
    # if at least one inner contract was actually absorbed into the fused spec.
    if not leaf.has_absorbed:
        return m
    return ops.compose(pre, fused_contract)


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
