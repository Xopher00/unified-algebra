"""Phase 3 tests: equations registered as Hydra Primitives, called via reduce_term."""

import numpy as np
import pytest

import unified_algebra._hydra_setup  # noqa: F401
import hydra.graph
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right, Left
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unified_algebra.backend import numpy_backend
from unified_algebra.semiring import semiring
from unified_algebra.sort import sort, tensor_coder, build_graph
from unified_algebra.morphism import equation, resolve_equation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def real_sr():
    return semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def tropical_sr():
    return semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def fuzzy_sr():
    return semiring("fuzzy", plus="maximum", times="minimum", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return sort("hidden", real_sr)


@pytest.fixture
def coder():
    return tensor_coder()


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


def make_graph(primitives=None):
    return build_graph([], primitives=primitives)


def assert_reduce_ok(cx, graph, term):
    result = reduce_term(cx, graph, True, term)
    assert isinstance(result, Right), f"reduce_term returned Left: {result}"
    return result.value


# ---------------------------------------------------------------------------
# Parametric equations (contraction over semiring)
# ---------------------------------------------------------------------------

class TestParametricEquation:

    def test_is_primitive(self, real_sr, hidden, backend):
        eq = equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim = resolve_equation(eq, backend)
        assert isinstance(prim, hydra.graph.Primitive)

    def test_name(self, real_sr, hidden, backend):
        eq = equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim = resolve_equation(eq, backend)
        assert prim.name == Name("ua.equation.linear")

    def test_direct_call(self, cx, real_sr, hidden, backend, coder):
        eq = equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim = resolve_equation(eq, backend)
        graph = make_graph()

        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])

        result = prim.implementation(cx, graph, (
            encode_array(coder, W), encode_array(coder, x),
        ))
        assert isinstance(result, Right)
        np.testing.assert_allclose(decode_term(coder, result.value), W @ x)

    def test_via_reduce_term(self, cx, real_sr, hidden, backend, coder):
        eq = equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim = resolve_equation(eq, backend)
        graph = make_graph(primitives={prim.name: prim})

        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])

        term = apply(apply(var("ua.equation.linear"),
                           encode_array(coder, W)),
                     encode_array(coder, x))
        out = decode_term(coder, assert_reduce_ok(cx, graph, term))
        np.testing.assert_allclose(out, W @ x)

    def test_non_square(self, cx, real_sr, hidden, backend, coder):
        eq = equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim = resolve_equation(eq, backend)
        graph = make_graph()

        W = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
        x = np.array([1.0, 2.0, 3.0])

        result = prim.implementation(cx, graph, (
            encode_array(coder, W), encode_array(coder, x),
        ))
        assert isinstance(result, Right)
        np.testing.assert_allclose(decode_term(coder, result.value), W @ x)


# ---------------------------------------------------------------------------
# Pointwise equations (nonlinearity, no contraction)
# ---------------------------------------------------------------------------

class TestPointwiseEquation:

    def test_is_primitive(self, hidden, backend):
        eq = equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim = resolve_equation(eq, backend)
        assert isinstance(prim, hydra.graph.Primitive)

    def test_name(self, hidden, backend):
        eq = equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim = resolve_equation(eq, backend)
        assert prim.name == Name("ua.equation.relu")

    def test_relu(self, cx, hidden, backend, coder):
        eq = equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim = resolve_equation(eq, backend)
        graph = make_graph()

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = prim.implementation(cx, graph, (encode_array(coder, x),))
        assert isinstance(result, Right)
        np.testing.assert_allclose(decode_term(coder, result.value), np.maximum(0, x))

    def test_via_reduce_term(self, cx, hidden, backend, coder):
        eq = equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim = resolve_equation(eq, backend)
        graph = make_graph(primitives={prim.name: prim})

        x = np.array([-1.0, 0.0, 1.0, 2.0])
        term = apply(var("ua.equation.relu"), encode_array(coder, x))
        out = decode_term(coder, assert_reduce_ok(cx, graph, term))
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_sigmoid(self, cx, hidden, backend, coder):
        eq = equation("sigmoid", None, hidden, hidden, nonlinearity="sigmoid")
        prim = resolve_equation(eq, backend)
        graph = make_graph()

        x = np.array([-2.0, 0.0, 2.0])
        result = prim.implementation(cx, graph, (encode_array(coder, x),))
        assert isinstance(result, Right)
        np.testing.assert_allclose(
            decode_term(coder, result.value),
            1.0 / (1.0 + np.exp(-x)), rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Combined: contraction + nonlinearity in one equation
# ---------------------------------------------------------------------------

class TestCombinedEquation:

    def test_linear_relu(self, cx, real_sr, hidden, backend, coder):
        """Y[i] = relu(W[i,j] X[j]) — one equation, not two."""
        eq = equation("linear_relu", "ij,j->i", hidden, hidden, real_sr, nonlinearity="relu")
        prim = resolve_equation(eq, backend)
        graph = make_graph(primitives={prim.name: prim})

        W = np.array([[1.0, -2.0], [-3.0, 4.0]])
        x = np.array([1.0, 1.0])
        expected = np.maximum(0, W @ x)

        term = apply(apply(var("ua.equation.linear_relu"),
                           encode_array(coder, W)),
                     encode_array(coder, x))
        out = decode_term(coder, assert_reduce_ok(cx, graph, term))
        np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Semiring variants
# ---------------------------------------------------------------------------

class TestSemiringVariants:

    def test_tropical(self, cx, backend, coder):
        trop_sr = semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)
        hidden = sort("hidden", trop_sr)
        eq = equation("trop_linear", "ij,j->i", hidden, hidden, trop_sr)
        prim = resolve_equation(eq, backend)
        graph = make_graph(primitives={prim.name: prim})

        W = np.array([[1.0, 3.0], [2.0, 0.0]])
        x = np.array([1.0, 2.0])

        term = apply(apply(var("ua.equation.trop_linear"),
                           encode_array(coder, W)),
                     encode_array(coder, x))
        out = decode_term(coder, assert_reduce_ok(cx, graph, term))
        np.testing.assert_allclose(out, np.array([2.0, 2.0]))

    def test_fuzzy(self, cx, backend, coder):
        fuz_sr = semiring("fuzzy", plus="maximum", times="minimum", zero=0.0, one=1.0)
        hidden = sort("hidden", fuz_sr)
        eq = equation("fuz_linear", "ij,j->i", hidden, hidden, fuz_sr)
        prim = resolve_equation(eq, backend)
        graph = make_graph(primitives={prim.name: prim})

        W = np.array([[0.8, 0.3], [0.2, 0.9]])
        x = np.array([0.6, 0.7])

        term = apply(apply(var("ua.equation.fuz_linear"),
                           encode_array(coder, W)),
                     encode_array(coder, x))
        out = decode_term(coder, assert_reduce_ok(cx, graph, term))
        np.testing.assert_allclose(out, np.array([0.6, 0.7]))


# ---------------------------------------------------------------------------
# End-to-end composition
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_chain(self, cx, real_sr, hidden, backend, coder):
        """relu(W @ x) — two equations composed through reduce_term."""
        linear = resolve_equation(
            equation("linear", "ij,j->i", hidden, hidden, real_sr), backend)
        relu = resolve_equation(
            equation("relu", None, hidden, hidden, nonlinearity="relu"), backend)

        graph = make_graph(primitives={linear.name: linear, relu.name: relu})

        W = np.array([[1.0, -2.0], [-3.0, 4.0]])
        x = np.array([1.0, 1.0])
        expected = np.maximum(0, W @ x)

        inner = apply(apply(var("ua.equation.linear"),
                            encode_array(coder, W)),
                      encode_array(coder, x))
        outer = apply(var("ua.equation.relu"), inner)

        out = decode_term(coder, assert_reduce_ok(cx, graph, outer))
        np.testing.assert_allclose(out, expected)

    def test_two_layers(self, cx, real_sr, hidden, backend, coder):
        """y = W2 @ relu(W1 @ x)"""
        linear = resolve_equation(
            equation("linear", "ij,j->i", hidden, hidden, real_sr), backend)
        relu = resolve_equation(
            equation("relu", None, hidden, hidden, nonlinearity="relu"), backend)

        graph = make_graph(primitives={linear.name: linear, relu.name: relu})

        W1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
        W2 = np.array([[2.0, 0.0], [0.0, 2.0]])
        x = np.array([3.0, 1.0])
        expected = W2 @ np.maximum(0, W1 @ x)

        h = apply(var("ua.equation.relu"),
                  apply(apply(var("ua.equation.linear"),
                              encode_array(coder, W1)),
                        encode_array(coder, x)))
        y = apply(apply(var("ua.equation.linear"), encode_array(coder, W2)), h)

        out = decode_term(coder, assert_reduce_ok(cx, graph, y))
        np.testing.assert_allclose(out, expected)
