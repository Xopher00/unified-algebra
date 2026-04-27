"""Composition tests: path (sequential) and fan (parallel) composition via Hydra lambda terms."""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    build_graph, assemble_graph, Equation,
    PathSpec, FanSpec,
)
from unialg.assembly.compositions import PathComposition, FanComposition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def tropical_sr():
    return Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


@pytest.fixture
def tropical_sort(tropical_sr):
    return Sort("tropic", tropical_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    result = coder.decode(None, arr)
    assert isinstance(result, Right)
    return result.value


def decode_term(coder, term):
    result = coder.encode(None, None, term)
    assert isinstance(result, Right)
    return result.value


def assert_reduce_ok(cx, graph, term):
    result = reduce_term(cx, graph, True, term)
    assert isinstance(result, Right), f"reduce_term returned Left: {result}"
    return result.value


def _schema(eq_by_name, extra_sorts=()):
    from unialg.algebra.sort import sort_wrap
    schema = {}
    for eq in eq_by_name.values():
        eq.register_sorts(schema)
    for s in extra_sorts:
        sort_wrap(s).register_schema(schema)
    return FrozenDict(schema)


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
