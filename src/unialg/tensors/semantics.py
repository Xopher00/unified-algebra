"""Tensor extension — semantic layer.

Converts parsed declarations and expressions into typed substrate objects.
No Hydra terms, no runtime calls, no backend-specific code.

Imports from substrate: Morphism and objects/types.
Does NOT import from runtime/, structure/, or any backend library.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from unialg.semantics.functors import apply_poly
from unialg.semantics.morphisms import Morphism, MorphismError
from unialg.objects import BINARY, repeated_product
from unialg.syntax import expressions as expr

from .notation import SemiringDecl, ContractExpr, Equation
from .semirings import Semiring


# ---------------------------------------------------------------------------
# Internal compile data for tensor contraction lowering
# ---------------------------------------------------------------------------

def _left_nested_shape(n: int):
    """Build a left-nested Prod(Id, ...) tree for n inputs."""
    from unialg.syntax.expressions import Prod, Id
    if n <= 0:
        raise ValueError("need at least 1 input for shape")
    if n == 1:
        return Id()
    shape = Prod(Id(), Id())
    for _ in range(n - 2):
        shape = Prod(shape, Id())
    return shape


def _count_id(shape) -> int:
    """Count Id positions in a PolyExpr shape."""
    from unialg.syntax.expressions import Prod, Id
    if isinstance(shape, Id):
        return 1
    if isinstance(shape, Prod):
        return _count_id(shape.left) + _count_id(shape.right)
    raise ValueError(f"unexpected shape node in _count_id: {type(shape).__name__}")


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
            return apply_poly(self.shape, BINARY)
        n = len(self.equation.inputs)
        return BINARY if n == 1 else repeated_product(BINARY, n)

    @property
    def cod(self):
        return BINARY


# ---------------------------------------------------------------------------
# Semiring resolution: SemiringDecl → Semiring
# ---------------------------------------------------------------------------

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

    plus_reduce_name = f"reduce.{decl.plus}"
    times_reduce_name = f"reduce.{decl.times}"
    m_plus_reduce = _lookup(plus_reduce_name, "plus_reduce") if plus_reduce_name in op_morphisms else None
    m_times_reduce = _lookup(times_reduce_name, "times_reduce") if times_reduce_name in op_morphisms else None

    m_adjoint = _lookup(decl.adjoint, "adjoint") if decl.adjoint else None

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
                        shape=_left_nested_shape(len(eq.inputs)))
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
