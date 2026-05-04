"""E2E tests for fold catamorphism via DSL syntax.

Tests the Maybe functor (1 + X) end-to-end through reduce_term.
The List functor (1 + Const & X) requires ProductSort domain ops which
have additional limitations (scalar contraction engine doesn't handle
rank-0 einsums). Maybe fold is the clean test path.

Program.__call__ cannot dispatch fold entry points because TermCoder
only handles tensor arrays, not Hydra Maybe/List terms. Tests use
reduce_term directly.
"""

import sys
import numpy as np
import pytest

import hydra.core as core
import hydra.dsl.terms as T
from hydra.dsl.python import Right
from hydra.reduction import reduce_term

from unialg import NumpyBackend, parse_ua
from unialg.terms import _literal_value


_MAYBE_PROG = """\
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec embed(real)

op double : embed -> embed
  nonlinearity = square

functor F_maybe : 1 + X

cell handle_maybe : embed -> embed = fold[F_maybe](0.0, double)
"""


def _reduce(prog, entry, arg_term):
    """Apply a morphism entry point to a Hydra term via reduce_term."""
    var = core.TermVariable(core.Name(f"ua.morphism.{entry}"))
    app = core.TermApplication(core.Application(var, arg_term))
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 5000))
    try:
        result = reduce_term(prog._cx, prog._graph, True, app)
    finally:
        sys.setrecursionlimit(old_limit)
    match result:
        case Right(value=v):
            return v
        case _:
            pytest.fail(f"reduce_term failed: {result}")


def _decode_result(backend, term):
    """Decode a reduced Hydra term to a Python float."""
    if isinstance(term, core.TermLiteral):
        lit = term.value
        if isinstance(lit, core.LiteralBinary):
            arr = backend.from_wire(lit.value)
            return float(arr.item()) if hasattr(arr, 'item') else float(arr)
        return float(_literal_value(term))
    pytest.fail(f"Expected TermLiteral, got {type(term).__name__}: {term}")


class TestMaybeFoldParsesAndCompiles:

    def test_fold_entry_point_registered(self):
        prog = parse_ua(_MAYBE_PROG)
        assert "handle_maybe" in prog.entry_points()

    def test_equation_also_available(self):
        prog = parse_ua(_MAYBE_PROG)
        assert "double" in prog.entry_points()

    def test_symbolic_cata_syntax(self):
        text = _MAYBE_PROG.replace("fold[F_maybe]", ">[F_maybe]")
        prog = parse_ua(text)
        assert "handle_maybe" in prog.entry_points()


class TestMaybeFoldExecution:

    def test_just_applies_step(self):
        prog = parse_ua(_MAYBE_PROG)
        backend = NumpyBackend()
        val = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(5.0)))
        result = _reduce(prog, "handle_maybe", T.just(val))
        assert _decode_result(backend, result) == pytest.approx(25.0)

    def test_nothing_returns_init(self):
        prog = parse_ua(_MAYBE_PROG)
        result = _reduce(prog, "handle_maybe", T.nothing())
        assert float(_literal_value(result)) == pytest.approx(0.0)

    def test_just_negative(self):
        prog = parse_ua(_MAYBE_PROG)
        backend = NumpyBackend()
        val = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(-3.0)))
        result = _reduce(prog, "handle_maybe", T.just(val))
        assert _decode_result(backend, result) == pytest.approx(9.0)
