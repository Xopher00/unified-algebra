"""Tensor extension — semantic layer.

Converts parsed declarations and expressions into typed substrate objects.
No Hydra terms, no runtime calls, no backend-specific code.

Imports from substrate: Morphism and objects/types.
Does NOT import from runtime/, structure/, or any backend library.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from unialg.semantics.functors import apply_poly, _count_id, id_, prod
from unialg.semantics.morphisms import Morphism, MorphismError
from unialg.objects import BINARY, ExpType, ProductType, repeated_product
from unialg.syntax import expressions as expr

import hydra.core as core

from .notation import SemiringDecl, ContractExpr, Equation
from .semirings import Semiring


# ---------------------------------------------------------------------------
# Internal compile data for tensor contraction lowering
# ---------------------------------------------------------------------------

def _left_nested_shape(n: int):
    """Build a left-nested Prod(Id, ...) tree for n inputs."""
    if n <= 0:
        raise ValueError("need at least 1 input for shape")
    if n == 1:
        return id_()
    shape = prod(id_(), id_())
    for _ in range(n - 2):
        shape = prod(shape, id_())
    return shape


def _index_product(labels: tuple[str, ...]):
    """Encode label tuple as a Hydra type: product of TypeVariable(idx.<label>)."""
    if not labels:
        return core.TypeUnit()
    cur = core.TypeVariable(core.Name(f"idx.{labels[0]}"))
    for lab in labels[1:]:
        cur = ProductType(cur, core.TypeVariable(core.Name(f"idx.{lab}")))
    return cur


def _slot_shape(labels: tuple[str, ...]):
    """Polynomial shape for a single tensor slot.

    Empty labels → Id() (scalar).  Non-empty → Exp(index_product, Id()).
    """
    if not labels:
        return expr.Id()
    return expr.Exp(expr.Const(_index_product(labels)), expr.Id())


def _exp_shape_from_equation(eq):
    """Build polynomial shape encoding input labels as Exp bases."""
    shapes = [_slot_shape(inp) for inp in eq.inputs]
    if len(shapes) == 1:
        return shapes[0]
    result = shapes[0]
    for s in shapes[1:]:
        result = expr.Prod(result, s)
    return result


def tensor_type(labels: tuple[str, ...], carrier=BINARY):
    """Type of a tensor indexed by labels: ExpType(index_product, carrier)."""
    if not labels:
        return carrier
    return ExpType(_index_product(labels), carrier)


def _strip_exp(typ):
    """Remove Exp wrappers from a type tree: ExpType(base, carrier) → carrier.

    Keeps dom/cod as BINARY products for substrate compatibility while the
    polynomial shape carries label metadata in Exp bases.
    """
    from hydra.core import TypePair, TypeFunction
    if isinstance(typ, TypePair):
        return ProductType(_strip_exp(typ.value.first), _strip_exp(typ.value.second))
    if isinstance(typ, TypeFunction):
        return typ.value.codomain
    return typ


@dataclass(frozen=True)
class ContractSpec:
    """Internal tensor contraction spec consumed before returning to core semantics."""
    semiring: Semiring
    equation: Equation
    adjoint: bool = False
    shape: object = None
    _domain_tag: str = field(default="tensors", init=False, repr=False)

    @property
    def dom(self):
        if self.shape is not None:
            return _strip_exp(apply_poly(self.shape, BINARY))
        n = len(self.equation.inputs)
        return BINARY if n == 1 else repeated_product(BINARY, n)

    @property
    def cod(self):
        return BINARY


# ---------------------------------------------------------------------------
# Semiring resolution: SemiringDecl → Semiring
# ---------------------------------------------------------------------------

def _check_binary_op(m: Morphism, carrier, label: str) -> None:
    """Raise MorphismError if m is not of type carrier × carrier → carrier."""
    expected_dom = ProductType(carrier, carrier)
    MorphismError.check(None, m.dom(), expected_dom,
                        f"{label} domain must be carrier × carrier")
    MorphismError.check(None, m.cod(), carrier,
                        f"{label} codomain must be carrier")


def resolve_semiring(decl: SemiringDecl, op_morphisms: dict[str, Morphism]) -> Semiring:
    """Resolve a parsed SemiringDecl into a typed Semiring.

    ``op_morphisms`` maps backend op names to already-constructed Morphisms.
    Reduce ops are derived by convention: ``"reduce.{plus}"`` etc.
    """
    def _lookup(name: str, label: str) -> Morphism:
        if name not in op_morphisms:
            raise MorphismError(
                f"algebra '{decl.name}': unknown op '{name}' for {label}"
            )
        return op_morphisms[name]

    m_plus = _lookup(decl.plus, "plus")
    m_times = _lookup(decl.times, "times")
    _check_binary_op(m_plus,  BINARY, f"algebra '{decl.name}': plus")
    _check_binary_op(m_times, BINARY, f"algebra '{decl.name}': times")

    plus_reduce_name = f"reduce.{decl.plus}"
    times_reduce_name = f"reduce.{decl.times}"
    m_plus_reduce = _lookup(plus_reduce_name, "plus_reduce")
    m_times_reduce = (
        _lookup(times_reduce_name, "times_reduce")
        if decl.adjoint
        else op_morphisms.get(times_reduce_name)
    )

    m_adjoint = _lookup(decl.adjoint, "adjoint") if decl.adjoint else None
    if m_adjoint is not None:
        _check_binary_op(m_adjoint, BINARY, f"algebra '{decl.name}': adjoint")

    f_zero = _parse_identity_value(decl.zero)
    f_one = _parse_identity_value(decl.one)

    return Semiring(
        name=decl.name,
        carrier=BINARY,
        plus=m_plus,
        times=m_times,
        zero=f_zero,
        one=f_one,
        plus_reduce=m_plus_reduce,
        times_reduce=m_times_reduce,
        adjoint=m_adjoint,
    )


def _parse_identity_value(value: str | float) -> float:
    """Convert a parsed identity value to a float."""
    if isinstance(value, str):
        if value == "inf":
            return float("inf")
        raise MorphismError(f"unsupported constant name: {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# contract_morphism: equation + semiring → composed Morphism
# ---------------------------------------------------------------------------

def contract_morphism(
    sr: Semiring,
    equation: str | Equation,
    *,
    adjoint: bool = False,
) -> Morphism:
    """Build a lazy contraction Morphism from a semiring and equation string.

    Returns a ``DomainPrim("tensors", ContractSpec)`` node.  The contraction
    is compiled into the substrate composition (align/product/reduce) during
    the ``normalize_contracts`` finalize pass in ``tensors/fusion.py``.
    """
    if adjoint and sr.adjoint is None:
        raise ValueError(f"Semiring '{sr.name}' has no adjoint operation")
    eq = Equation.parse(equation) if isinstance(equation, str) else equation
    spec = ContractSpec(semiring=sr, equation=eq, adjoint=adjoint,
                        shape=_exp_shape_from_equation(eq))
    return Morphism(
        node=expr.DomainPrim("tensors", spec, spec.dom, spec.cod),
        aux_primitives=_collect_semiring_aux(sr),
    )


def _collect_semiring_aux(sr: Semiring) -> tuple:
    """Gather aux_primitives from all Morphisms in a Semiring."""
    parts: list = []
    for m in (sr.plus, sr.times):
        parts.extend(m.aux_primitives)
    for m in (sr.plus_reduce, sr.times_reduce, sr.adjoint):
        if m is not None:
            parts.extend(m.aux_primitives)
    return tuple(parts)


# ---------------------------------------------------------------------------
# Domain protocol: construct / construct_expr / refs
# ---------------------------------------------------------------------------

def construct(declarations: list, env: dict) -> dict:
    """Resolve all SemiringDecl in ``declarations`` against ``env``."""
    semirings: dict[str, Semiring] = {}
    for decl in declarations:
        if isinstance(decl, SemiringDecl):
            semirings[decl.name] = resolve_semiring(decl, env)
    return {"semirings": semirings}


def construct_expr(node, env: dict) -> Morphism:
    """Resolve a ContractExpr into a composed substrate Morphism."""
    if not isinstance(node, ContractExpr):
        raise TypeError(f"tensor construct_expr: unexpected {type(node).__name__!r}")

    domain_data = env.get("_domain_data", {}).get("tensors", {})
    semirings = domain_data.get("semirings", {})

    if node.semiring_name not in semirings:
        raise MorphismError(
            f"contract references unknown algebra '{node.semiring_name}'"
        )
    sr = semirings[node.semiring_name]
    return contract_morphism(
        sr,
        node.equation_str,
        adjoint=node.adjoint,
    )


def refs(node) -> set[str]:
    """Return morphism-namespace refs from a domain expression node."""
    return set()
