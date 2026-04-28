"""Polynomial endofunctors for the assembly layer.

A `Functor` is a polynomial endofunctor `F : C -> C` declared by an expression
in the polynomial sub-language: sums, products, constants, identity (the
recursion variable `X`), and exponentials over a constant base. Together with
a category-of-discourse tag (``set`` | ``poset``), a Functor parameterizes
``AlgebraHomComposition``.

Representation
==============

The polynomial expression is a Hydra **union type** — ``ua.functor.PolyExpr`` —
with seven variants. Each variant is constructed via ``hydra.dsl.terms.inject``
or ``inject_unit``, dispatched on the injected variant name. Payloads use
Hydra's built-in term shapes: ``unit`` for the nullary variants, a Sort term
for ``const``, a ``pair`` for ``sum`` / ``prod``, and a ``pair`` of (Sort,
PolyExpr) for ``exp``. Sort references are carried as typed Sort terms via
``sort_wrap`` — the same pattern used by ``Equation.domain_sort`` and
``Lens.residual_sort`` — never as raw name strings. ``Functor`` itself is a
record view of type ``ua.functor.Functor``, mirroring Sort / Semiring /
Equation / Lens.

This module is the data layer only. Grammar lives in ``parser/_grammar.py``;
runtime execution lives in ``assembly/_algebra_hom.py``.

Examples (constructed from Python; the .ua surface comes later)::

    Functor("F_list",   sum_(one(), prod(const(base_sort), id_())))
    Functor("F_tree",   sum_(const(base_sort), prod(id_(), id_())))
    Functor("F_stream", prod(const(output_sort), id_()))
    Functor("F_mealy",  exp(input_sort, prod(const(output_sort), id_())))
    Functor("F_poset",  id_(), category="poset")
"""
from __future__ import annotations

import hydra.dsl.terms as Terms
from hydra.core import Name

from unialg.terms import _RecordView
from unialg.algebra.sort import Sort, ProductSort, sort_wrap


# ---------------------------------------------------------------------------
# Type & variant names
# ---------------------------------------------------------------------------

POLY_TYPE_NAME = Name("ua.functor.PolyExpr")

_K_ZERO  = Name("zero")
_K_ONE   = Name("one")
_K_ID    = Name("id")
_K_CONST = Name("const")
_K_SUM   = Name("sum")
_K_PROD  = Name("prod")
_K_EXP   = Name("exp")


# ---------------------------------------------------------------------------
# PolyExpr — wrapper over a TermInject of the polynomial union
# ---------------------------------------------------------------------------

class PolyExpr(_RecordView):
    """Polynomial endofunctor expression.

    Wraps a ``TermInject`` of union type ``ua.functor.PolyExpr``. Subclasses
    ``_RecordView`` so the wrapper protocol (``term`` property, ``_unwrap``
    integration) is shared with the rest of the algebra layer; the underlying
    term is a union, not a record, so the descriptor machinery is unused.
    """
    __slots__ = ()

    def __init__(self, term):
        if isinstance(term, PolyExpr):
            term = term._term
        self._term = term

    @property
    def kind(self) -> str:
        """Variant tag: ``zero`` | ``one`` | ``id`` | ``const`` | ``sum`` | ``prod`` | ``exp``."""
        return self._term.value.field.name.value

    @property
    def _payload(self):
        """The injected payload term (unit / sort / pair, depending on kind)."""
        return self._term.value.field.term

    def _expect(self, *kinds: str) -> None:
        if self.kind not in kinds:
            raise AttributeError(
                f"PolyExpr accessor not valid for kind={self.kind!r} "
                f"(expected one of {kinds})"
            )

    @property
    def sort(self) -> Sort | ProductSort:
        """For ``const``: the referenced sort."""
        self._expect("const")
        return sort_wrap(self._payload)

    @property
    def left(self) -> "PolyExpr":
        """For ``sum`` / ``prod``: the left operand."""
        self._expect("sum", "prod")
        return PolyExpr(self._payload.value[0])

    @property
    def right(self) -> "PolyExpr":
        """For ``sum`` / ``prod``: the right operand."""
        self._expect("sum", "prod")
        return PolyExpr(self._payload.value[1])

    @property
    def base_sort(self) -> Sort | ProductSort:
        """For ``exp``: the constant base sort of the exponential."""
        self._expect("exp")
        return sort_wrap(self._payload.value[0])

    @property
    def body(self) -> "PolyExpr":
        """For ``exp``: the body of the exponential."""
        self._expect("exp")
        return PolyExpr(self._payload.value[1])

    def __eq__(self, other) -> bool:
        return isinstance(other, PolyExpr) and self._term == other._term

    def __hash__(self) -> int:
        return hash(self._term)

    def __repr__(self) -> str:
        return f"PolyExpr({pretty(self)})"


# ---------------------------------------------------------------------------
# Constructors — each returns a PolyExpr wrapping a TermInject
# ---------------------------------------------------------------------------

def zero() -> PolyExpr:
    """Initial object: ``F(X) = 0``."""
    return PolyExpr(Terms.inject_unit(POLY_TYPE_NAME, _K_ZERO))


def one() -> PolyExpr:
    """Terminal object: ``F(X) = 1``."""
    return PolyExpr(Terms.inject_unit(POLY_TYPE_NAME, _K_ONE))


def id_() -> PolyExpr:
    """Identity functor: ``F(X) = X``."""
    return PolyExpr(Terms.inject_unit(POLY_TYPE_NAME, _K_ID))


def const(sort: Sort | ProductSort) -> PolyExpr:
    """Constant functor: ``F(X) = sort``."""
    return PolyExpr(Terms.inject(POLY_TYPE_NAME, _K_CONST, _RecordView._unwrap(sort)))


def sum_(left: PolyExpr, right: PolyExpr) -> PolyExpr:
    """Coproduct: ``F(X) + G(X)``."""
    payload = Terms.pair(_RecordView._unwrap(left), _RecordView._unwrap(right))
    return PolyExpr(Terms.inject(POLY_TYPE_NAME, _K_SUM, payload))


def prod(left: PolyExpr, right: PolyExpr) -> PolyExpr:
    """Product: ``F(X) * G(X)``."""
    payload = Terms.pair(_RecordView._unwrap(left), _RecordView._unwrap(right))
    return PolyExpr(Terms.inject(POLY_TYPE_NAME, _K_PROD, payload))


def exp(base_sort: Sort | ProductSort, body: PolyExpr) -> PolyExpr:
    """Exponential ``A -> F(X)`` with constant base sort ``A``."""
    payload = Terms.pair(_RecordView._unwrap(base_sort), _RecordView._unwrap(body))
    return PolyExpr(Terms.inject(POLY_TYPE_NAME, _K_EXP, payload))


# ---------------------------------------------------------------------------
# Functor
# ---------------------------------------------------------------------------

class Functor(_RecordView):
    """A polynomial endofunctor with a category-of-discourse tag.

    ``category="set"`` is the default and covers Para over Set/Vect — the
    setting for catamorphisms / anamorphisms. ``category="poset"`` switches
    the ambient category to the thin poset of the carrier's semiring ordering;
    the body must be ``id_()`` and the induced (co)algebra hom is the Tarski
    least fixpoint of the cell morphism.
    """

    _type_name = Name("ua.functor.Functor")

    name     = _RecordView.Scalar(str)
    body     = _RecordView.Term(coerce=PolyExpr)
    category = _RecordView.Scalar(str, default="set")

    def summands(self) -> tuple[PolyExpr, ...]:
        """Flatten nested ``sum`` variants into a tuple of summands."""
        return _flatten_sum(self.body)

    def x_arity(self) -> int:
        """Total count of ``X`` occurrences across all summands."""
        return sum(_x_arity(s) for s in self.summands())

    def is_recursive(self) -> bool:
        """True iff some summand contains a recursive position (``X``)."""
        return self.x_arity() > 0

    def consts(self) -> tuple[Sort | ProductSort, ...]:
        """All sorts referenced by ``const`` or ``exp`` subterms (insertion order)."""
        return tuple(_consts(self.body))

    def validate(self) -> None:
        """Check category-of-discourse constraints. Raises ``ValueError`` on violations.

        Sort references are typed Sort terms by construction — there is no
        name lookup to validate. Polynomial well-formedness is enforced by the
        constructors (you cannot build an ill-typed PolyExpr).
        """
        if self.category == "poset":
            if self.body.kind != "id":
                raise ValueError(
                    f"Functor {self.name!r}: category=poset requires body=X "
                    f"(identity), got {pretty(self.body)}"
                )
        elif self.category != "set":
            raise ValueError(
                f"Functor {self.name!r}: category must be 'set' or 'poset', "
                f"got {self.category!r}"
            )

    def __repr__(self) -> str:
        cat = "" if self.category == "set" else f" [category={self.category}]"
        return f"functor {self.name} : {pretty(self.body)}{cat}"


# ---------------------------------------------------------------------------
# Structural helpers — operate on PolyExpr via .kind dispatch
# ---------------------------------------------------------------------------

def _flatten_sum(expr: PolyExpr) -> tuple[PolyExpr, ...]:
    if expr.kind == "sum":
        return _flatten_sum(expr.left) + _flatten_sum(expr.right)
    return (expr,)


def _x_arity(expr: PolyExpr) -> int:
    k = expr.kind
    if k == "id":
        return 1
    if k in ("sum", "prod"):
        return _x_arity(expr.left) + _x_arity(expr.right)
    if k == "exp":
        return _x_arity(expr.body)
    return 0


def _consts(expr: PolyExpr) -> list[Sort | ProductSort]:
    """Return all sorts referenced by ``const`` / ``exp`` subterms, in order."""
    k = expr.kind
    if k == "const":
        return [expr.sort]
    if k in ("sum", "prod"):
        return _consts(expr.left) + _consts(expr.right)
    if k == "exp":
        return [expr.base_sort] + _consts(expr.body)
    return []


def pretty(expr: PolyExpr) -> str:
    """Render a PolyExpr in the polynomial sub-language surface form."""
    k = expr.kind
    if k == "zero":
        return "0"
    if k == "one":
        return "1"
    if k == "id":
        return "X"
    if k == "const":
        return expr.sort.name
    if k == "sum":
        return f"{pretty(expr.left)} + {pretty(expr.right)}"
    if k == "prod":
        ls = f"({pretty(expr.left)})" if expr.left.kind == "sum" else pretty(expr.left)
        rs = f"({pretty(expr.right)})" if expr.right.kind == "sum" else pretty(expr.right)
        return f"{ls} * {rs}"
    if k == "exp":
        bs = pretty(expr.body)
        if expr.body.kind in ("sum", "prod"):
            bs = f"({bs})"
        return f"{expr.base_sort.name} -> {bs}"
    raise ValueError(f"pretty: unknown kind {k!r}")
