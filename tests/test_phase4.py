"""Phase 4 tests: sort junction validation and graph assembly."""

import numpy as np
import pytest

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unified_algebra.backend import numpy_backend
from unified_algebra.semiring import semiring
from unified_algebra.sort import sort, tensor_coder
from unified_algebra.morphism import equation
from unified_algebra.graph import validate_pipeline, assemble_graph


@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def coder():
    return tensor_coder()


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


def enc(coder, arr):
    return coder.decode(None, arr).value


def dec(coder, term):
    return coder.encode(None, None, term).value


# ---------------------------------------------------------------------------
# Sort name mismatch
# ---------------------------------------------------------------------------

class TestSortNameMismatch:

    def test_different_sort_names_rejected(self):
        real = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = sort("hidden", real)
        output = sort("output", real)
        eq1 = equation("eq1", "ij,j->i", hidden, hidden, real)
        eq2 = equation("eq2", "ij,j->i", output, output, real)
        with pytest.raises(TypeError, match="Sort junction error"):
            validate_pipeline([eq1, eq2])


# ---------------------------------------------------------------------------
# Semiring mismatch
# ---------------------------------------------------------------------------

class TestSemiringMismatch:

    def test_different_semiring_rejected(self):
        real = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        tropical = semiring("tropical", plus="minimum", times="add",
                            zero=float("inf"), one=0.0)
        hidden_real = sort("hidden", real)
        hidden_trop = sort("hidden", tropical)
        eq1 = equation("real_eq", "ij,j->i", hidden_real, hidden_real, real)
        eq2 = equation("trop_eq", "ij,j->i", hidden_trop, hidden_trop, tropical)
        with pytest.raises(TypeError, match="Sort junction error"):
            validate_pipeline([eq1, eq2])


# ---------------------------------------------------------------------------
# Valid composition
# ---------------------------------------------------------------------------

class TestValidComposition:

    def test_single_equation_trivially_valid(self):
        real = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = sort("hidden", real)
        eq = equation("linear", "ij,j->i", hidden, hidden, real)
        validate_pipeline([eq])  # no error

    def test_three_equation_pipeline(self, backend):
        real = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = sort("hidden", real)
        eqs = [
            equation("linear1", "ij,j->i", hidden, hidden, real),
            equation("relu", None, hidden, hidden, nonlinearity="relu"),
            equation("linear2", "ij,j->i", hidden, hidden, real),
        ]
        validate_pipeline(eqs)  # no error
        graph = assemble_graph(eqs, backend)
        assert Name("ua.equation.linear1") in graph.primitives
        assert Name("ua.equation.relu") in graph.primitives
        assert Name("ua.equation.linear2") in graph.primitives

    def test_sorts_registered_in_schema(self, backend):
        real = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = sort("hidden", real)
        eqs = [equation("linear", "ij,j->i", hidden, hidden, real)]
        graph = assemble_graph(eqs, backend)
        assert Name("ua.sort.hidden:real") in graph.schema_types


# ---------------------------------------------------------------------------
# Assemble and run
# ---------------------------------------------------------------------------

class TestAssembleAndRun:

    def test_two_layer_network(self, backend, cx, coder):
        real = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = sort("hidden", real)
        eqs = [
            equation("linear", "ij,j->i", hidden, hidden, real),
            equation("act", None, hidden, hidden, nonlinearity="relu"),
        ]
        graph = assemble_graph(eqs, backend)

        W1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
        W2 = np.array([[2.0, 0.0], [0.0, 2.0]])
        x = np.array([3.0, 1.0])
        expected = W2 @ np.maximum(0, W1 @ x)

        h = apply(var("ua.equation.act"),
                  apply(apply(var("ua.equation.linear"),
                              enc(coder, W1)), enc(coder, x)))
        y = apply(apply(var("ua.equation.linear"),
                        enc(coder, W2)), h)

        result = reduce_term(cx, graph, True, y)
        assert isinstance(result, Right)
        np.testing.assert_allclose(dec(coder, result.value), expected)

    def test_combined_equation(self, backend, cx, coder):
        """Single equation with contraction + nonlinearity: relu(W @ x)."""
        real = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = sort("hidden", real)
        eqs = [equation("lr", "ij,j->i", hidden, hidden, real, nonlinearity="relu")]
        graph = assemble_graph(eqs, backend)

        W = np.array([[1.0, -2.0], [-3.0, 4.0]])
        x = np.array([1.0, 1.0])
        expected = np.maximum(0, W @ x)

        term = apply(apply(var("ua.equation.lr"),
                           enc(coder, W)), enc(coder, x))
        result = reduce_term(cx, graph, True, term)
        assert isinstance(result, Right)
        np.testing.assert_allclose(dec(coder, result.value), expected)
