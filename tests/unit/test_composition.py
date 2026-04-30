"""Composition tests: path (sequential) and fan (parallel) composition via Hydra lambda terms."""

import pytest

import hydra.core as core
from hydra.core import Name

from unialg.assembly.legacy.compositions import PathComposition, FanComposition


# ---------------------------------------------------------------------------
# Path: lambda term structure
# ---------------------------------------------------------------------------

class TestPathStructure:

    def test_path_returns_name_and_lambda(self, hidden):
        name, term = PathComposition("act", ["relu"]).to_lambda()
        assert name == Name("ua.path.act")
        assert isinstance(term.value, core.Lambda)

    def test_path_name_prefix(self, hidden):
        name, _ = PathComposition("ffn", ["a", "b", "c"]).to_lambda()
        assert name.value == "ua.path.ffn"

    def test_path_empty_raises(self, hidden):
        with pytest.raises(ValueError, match="at least one equation"):
            PathComposition("bad", [])

    def test_path_single_step(self, hidden):
        """Single-equation path should be lambda x. eq(x)."""
        _, term = PathComposition("single", ["relu"]).to_lambda()
        # The body should be an application
        body = term.value.body
        assert isinstance(body.value, core.Application)

    def test_path_two_step(self, hidden):
        """Two-equation path: lambda x. b(a(x))."""
        _, term = PathComposition("two", ["a", "b"]).to_lambda()
        body = term.value.body
        # outer: apply(var("ua.equation.b"), ...)
        assert isinstance(body.value, core.Application)
        func = body.value.function
        assert isinstance(func, core.TermVariable)
        assert func.value.value == "ua.equation.b"

    def test_from_term_has_params_default(self):
        """PathComposition.from_term() must not crash on _params access."""
        pc = PathComposition("act", ["relu"])
        pc2 = PathComposition.from_term(pc._term)
        assert pc2._params is None
        name, term = pc2.to_lambda()
        assert name.value == "ua.path.act"


# ---------------------------------------------------------------------------
# Fan: lambda term structure
# ---------------------------------------------------------------------------

class TestFanStructure:

    def test_fan_returns_name_and_lambda(self, hidden):
        name, term = FanComposition("f", ["a", "b"], "m").to_lambda()
        assert name == Name("ua.fan.f")
        assert isinstance(term.value, core.Lambda)

    def test_fan_empty_branches_raises(self, hidden):
        with pytest.raises(ValueError, match="at least one branch"):
            FanComposition("bad", [], "m")

    def test_fan_many_branches_allowed(self, hidden):
        """Fan arity is unbounded — list-based merge handles any branch count."""
        _, term = FanComposition("wide", ["a", "b", "c", "d", "e"], "m").to_lambda()
        # Should not raise — produces a valid lambda term
        assert term is not None
