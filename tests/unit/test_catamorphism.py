"""Catamorphism unit tests: fold and unfold lambda term structure."""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var, list_
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    Equation,
    build_graph, assemble_graph,
    PathSpec, FoldSpec, UnfoldSpec,
)
from unialg.assembly.compositions import FoldComposition, UnfoldComposition
from unialg.assembly import unfold_n_primitive


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
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


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


def make_graph_with_stdlib(primitives=None, bound_terms=None):
    """Build a graph with Hydra standard library primitives included."""
    from hydra.sources.libraries import standard_library
    all_prims = dict(standard_library())
    if primitives:
        all_prims.update(primitives)
    return build_graph([], primitives=all_prims, bound_terms=bound_terms)


def _schema(eq_by_name, extra_sorts=()):
    from unialg.algebra.sort import sort_wrap
    schema = {}
    for eq in eq_by_name.values():
        eq.register_sorts(schema)
    for s in extra_sorts:
        sort_wrap(s).register_schema(schema)
    return FrozenDict(schema)


# ---------------------------------------------------------------------------
# Fold: structure
# ---------------------------------------------------------------------------

class TestFoldStructure:

    def test_fold_returns_name_and_lambda(self, hidden, coder):
        init = encode_array(coder, np.zeros(3))
        name, term = FoldComposition("rnn", "step", init).to_lambda()
        assert name == Name("ua.fold.rnn")
        assert isinstance(term.value, core.Lambda)

    def test_fold_name_prefix(self, hidden, coder):
        init = encode_array(coder, np.zeros(3))
        name, _ = FoldComposition("test", "step", init).to_lambda()
        assert name.value == "ua.fold.test"


# ---------------------------------------------------------------------------
# Unfold: structure
# ---------------------------------------------------------------------------

class TestUnfoldStructure:

    def test_unfold_returns_name_and_lambda(self, hidden):
        name, term = UnfoldComposition("stream", "step", 3).to_lambda()
        assert name == Name("ua.unfold.stream")
        assert isinstance(term.value, core.Lambda)
