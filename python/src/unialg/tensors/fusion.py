"""Tensor contraction fusion — semantic-level normalize_contracts pass.

Fuses adjacent tensor contractions at the Morphism layer before substrate
decomposition.  After fusion, every remaining ``DomainPrim("tensors", ...)``
is replaced by the substrate composition via ``compile_contract_spec``.

No imports from ``unialg.runtime``, ``unialg.structure``, or ``unialg.main``.
"""
from __future__ import annotations

from collections.abc import Callable

import hydra.names as HydraNames
from hydra.core import Name, TypeVariable, TypePair, TypeUnit
from hydra.dsl.python import FrozenDict
from hydra.variables import substitute_type_variables

from unialg.objects import BINARY
from unialg.semantics import morphisms as ops
from unialg.semantics.functors import Functor as _Functor, id_ as _id_body
from unialg.semantics.morphisms import Morphism, MorphismError, _copy
from unialg.semantics.optics import Optic as _Optic, traversal_optic as _traversal_optic, identity_optic as _identity_optic
from unialg.syntax import expressions as expr

from .notation import Equation
from .primitives import compile_contract_spec
from .semantics import ContractSpec, _collect_semiring_aux


# ---------------------------------------------------------------------------
# Shape-based label extraction and renaming
# ---------------------------------------------------------------------------

def _decode_hydra_labels(t) -> tuple[str, ...]:
    """Decode a Hydra TypeVariable/TypePair tree to label strings."""
    if isinstance(t, TypeVariable):
        inner = t.value
        if isinstance(inner, TypeVariable):
            inner = inner.value
        return (inner.value.removeprefix("idx."),)
    if isinstance(t, TypePair):
        return _decode_hydra_labels(t.value.first) + _decode_hydra_labels(t.value.second)
    if isinstance(t, TypeUnit):
        return ()
    return ()


def _labels_from_base(t: expr.PolyExpr) -> tuple[str, ...]:
    """Extract label strings from a PolyExpr Exp base."""
    if isinstance(t, expr.Const):
        return _decode_hydra_labels(t.space)
    if isinstance(t, expr.Prod):
        return _labels_from_base(t.left) + _labels_from_base(t.right)
    if isinstance(t, (expr.One, expr.Zero)):
        return ()
    return ()


def _extract_labels(shape) -> tuple[tuple[str, ...], ...]:
    """Walk a PolyExpr and extract label tuples from Exp bases."""
    if isinstance(shape, expr.Exp):
        return (_labels_from_base(shape.base),)
    if isinstance(shape, expr.Prod):
        return _extract_labels(shape.left) + _extract_labels(shape.right)
    if isinstance(shape, expr.Id):
        return ((),)
    return ()


def _equation_from_shape(shape, output: tuple[str, ...]) -> Equation:
    """Build Equation from a composite Exp shape + output labels."""
    inputs = _extract_labels(shape)
    all_labels: list[str] = []
    seen: set[str] = set()
    for inp in inputs:
        for l in inp:
            if l not in seen:
                all_labels.append(l)
                seen.add(l)
    output_set = set(output)
    reduced = tuple(l for l in all_labels if l not in output_set)
    return Equation(inputs=inputs, output=output, reduced=reduced)


def _rename_shape_labels(shape, output_labels: tuple[str, ...], avoid: set[str]):
    """Alpha-rename reduced labels in shape's Exp bases that collide with avoid.

    Only renames labels that are NOT in output_labels (i.e., the "reduced" set).
    Non-reduced labels are shared with the outer contract and must not be renamed.
    """
    labels = _extract_labels(shape)
    all_labels = {l for inp in labels for l in inp}
    reduced = all_labels - set(output_labels)
    collisions = reduced & avoid
    if not collisions:
        return shape

    all_used = all_labels | avoid
    rename_map: dict[str, str] = {}
    for label in sorted(collisions):
        fresh = HydraNames.unique_label(frozenset(all_used), label)
        rename_map[label] = fresh
        all_used.add(fresh)

    subst = FrozenDict({Name(f"idx.{old}"): Name(f"idx.{new}") for old, new in rename_map.items()})

    def _sub_poly_base(base: expr.PolyExpr) -> expr.PolyExpr:
        if isinstance(base, expr.Const):
            return expr.Const(substitute_type_variables(subst, base.space))
        if isinstance(base, expr.Prod):
            return expr.Prod(_sub_poly_base(base.left), _sub_poly_base(base.right))
        return base

    def _sub_shape(s):
        if isinstance(s, expr.Exp):
            return expr.Exp(_sub_poly_base(s.base), s.body)
        if isinstance(s, expr.Prod):
            return expr.Prod(_sub_shape(s.left), _sub_shape(s.right))
        return s

    return _sub_shape(shape)


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


def _leaf_optic(body):
    return _identity_optic(name="_", functor=_Functor(name="_", body=body), focus=BINARY)


def _spec_mismatches(inner_spec, outer_spec, expected_labels) -> bool:
    return (inner_spec.semiring != outer_spec.semiring
            or inner_spec.adjoint != outer_spec.adjoint
            or inner_spec.equation.output != expected_labels)


def _optic_pair_wrap(l_optic, combined):
    """Apply Copy-based Pair wrapping to a combined optic. Returns new combined or None."""
    shared_dom = l_optic.forward.dom()
    copy_m = _copy(shared_dom)
    fwd = copy_m if _is_all_identity(combined.forward.node) else ops.compose(copy_m, combined.forward)
    try:
        return _Optic(functor=combined.functor, forward=fwd, backward=combined.backward, carrier=BINARY)
    except (MorphismError, AssertionError):
        return None


def _optic_leaf(par_node, outer_shape, outer_spec, avoid):
    """Handle leaf positions: Identity, contract spec, or opaque morphism."""
    if isinstance(par_node, expr.Identity):
        return _leaf_optic(outer_shape), False, False

    inner_spec = _tensor_spec(par_node)
    if inner_spec is not None:
        expected_labels = _labels_from_base(outer_shape.base) if isinstance(outer_shape, expr.Exp) else ()
        if _spec_mismatches(inner_spec, outer_spec, expected_labels):
            return None
        inner_shape = _rename_shape_labels(inner_spec.shape, inner_spec.equation.output, avoid)
        new_labels = _extract_labels(inner_shape)
        avoid.update(l for inp in new_labels for l in inp)
        avoid.update(l for inp in new_labels for l in inp if l not in set(inner_spec.equation.output))
        return _leaf_optic(inner_shape), False, True

    fwd = Morphism(node=par_node, aux_primitives=())
    optic = _traversal_optic("_", _Functor(name="_", body=_id_body()), fwd, ops.identity(BINARY))
    return optic, True, False


def _pair_node_is_bare(par_node) -> bool:
    from unialg.objects import TypeUnit as TU
    return par_node.param == TU() and par_node.monad is None


def _par_to_optic(par_node, outer_shape, outer_spec, avoid):
    """Shape-driven descent: outer_shape guides which Exp slot each leaf fills.

    Reduces a Parallel/Pair/Identity/contract subtree to (Optic, has_opaque, has_absorbed).
    Returns None on structural mismatch.
    """
    if not isinstance(par_node, (expr.Parallel, expr.Pair)):
        return _optic_leaf(par_node, outer_shape, outer_spec, avoid)
    if not isinstance(outer_shape, expr.Prod):
        return None
    is_pair = isinstance(par_node, expr.Pair)
    if is_pair and not _pair_node_is_bare(par_node):
        return None
    left = _par_to_optic(par_node.f, outer_shape.left, outer_spec, avoid)
    if left is None:
        return None
    l_optic, l_opaque, l_absorbed = left
    right = _par_to_optic(par_node.g, outer_shape.right, outer_spec, avoid)
    if right is None:
        return None
    r_optic, r_opaque, r_absorbed = right
    try:
        combined = l_optic.product(r_optic)
    except MorphismError:
        return None
    if is_pair:
        combined = _optic_pair_wrap(l_optic, combined)
        if combined is None:
            return None
    return combined, bool(l_opaque + r_opaque), bool(l_absorbed + r_absorbed)


def _fill_opaque_labels(composite, outer):
    """Replace bare Id() in composite shape with matching Exp from outer shape.

    Opaque leaves use Id() (BINARY types) but the fused equation needs the
    outer slot's labels.  Walking both shapes in sync recovers them.
    """
    if isinstance(composite, expr.Prod) and isinstance(outer, expr.Prod):
        return expr.Prod(
            _fill_opaque_labels(composite.left, outer.left),
            _fill_opaque_labels(composite.right, outer.right),
        )
    if isinstance(composite, expr.Id) and isinstance(outer, expr.Exp):
        return outer
    return composite


def _strip_pre_map_ids(m: Morphism) -> Morphism:
    """Replace identity(ExpType(...)) with identity(stripped) in a pre-map tree.

    Absorbed contract leaves contribute identity(ExpType_carrier) to the combined
    forward.  The fused contract's dom uses _strip_exp, so the pre-map must also
    use stripped types for ops.compose to accept the boundary.
    """
    from .semantics import _strip_exp
    if isinstance(m.node, expr.Identity):
        return ops.identity(_strip_exp(m.node.space))
    return _rewrite_contextual_children(m, _strip_pre_map_ids)


def _types_compatible(pre: Morphism, fused_contract: Morphism, m: Morphism, fused_spec) -> bool:
    from .semantics import _strip_exp
    return (
        _strip_exp(pre.dom()) == _strip_exp(m.dom())
        and _strip_exp(pre.cod()) == _strip_exp(fused_contract.dom())
        and _strip_exp(fused_spec.cod) == _strip_exp(m.cod())
    )


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
    result = _par_to_optic(node.f, outer_spec.shape, outer_spec, avoid)

    if result is None:
        return m

    inner_optic, has_opaque, has_absorbed = result

    if has_opaque and not has_absorbed:
        return m

    labeled_shape = _fill_opaque_labels(inner_optic.functor.body, outer_spec.shape)
    fused_eq = _equation_from_shape(labeled_shape, outer_spec.equation.output)
    fused_spec = ContractSpec(
        semiring=outer_spec.semiring,
        equation=fused_eq,
        adjoint=outer_spec.adjoint,
        shape=labeled_shape,
    )

    fused_contract = _contract_morphism_from_spec(
        fused_spec,
        param=m.param,
        monad=m.monad,
    )

    pre = inner_optic.forward
    if not _types_compatible(pre, fused_contract, m, fused_spec):
        return m

    # Identity-tree forward (pure Parallel with no opaque/Copy routing) — skip wrap.
    if _is_all_identity(pre.node):
        return fused_contract
    # Non-trivial routing (Copy from Pair, opaque pre-map). Only worth wrapping
    # if at least one inner contract was actually absorbed into the fused spec.
    if not has_absorbed:
        return m
    # Absorbed contract slots contribute identity(ExpType) to the combined forward.
    # Strip to BINARY-level so pre.cod() matches fused_contract.dom().
    if has_opaque:
        pre = _strip_pre_map_ids(pre)
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
    from .semantics import _strip_exp
    result = compile_contract_spec(node.raw, ctx)
    if _strip_exp(result.dom()) != _strip_exp(m.dom()) or _strip_exp(result.cod()) != _strip_exp(m.cod()):
        raise MorphismError(
            f"DomainPrim decomposition produced wrong dom/cod for {node.raw.equation!r}"
        )
    return result


def _decompose_all(m: Morphism, ctx) -> Morphism:
    """Replace every DomainPrim("tensors") node with its substrate composition."""
    return _rewrite_bottom_up(m, lambda leaf: _decompose_leaf(leaf, ctx))
