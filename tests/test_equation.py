"""Equation tests: equations registered as Hydra Primitives, called via reduce_term."""

import numpy as np
import pytest

import hydra.graph
from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right, Left
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    build_graph, Equation,
    resolve_equation,
)


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
def fuzzy_sr():
    return Semiring("fuzzy", plus="maximum", times="minimum", zero=0.0, one=1.0,
                    bottom=0.0, top=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


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
        eq = Equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim, _ = resolve_equation(eq, backend)
        assert isinstance(prim, hydra.graph.Primitive)

    def test_name(self, real_sr, hidden, backend):
        eq = Equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim, _ = resolve_equation(eq, backend)
        assert prim.name == Name("ua.equation.linear")

    def test_direct_call(self, cx, real_sr, hidden, backend, coder):
        eq = Equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim, _ = resolve_equation(eq, backend)
        graph = make_graph()

        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])

        result = prim.implementation(cx, graph, (
            encode_array(coder, W), encode_array(coder, x),
        ))
        assert isinstance(result, Right)
        np.testing.assert_allclose(decode_term(coder, result.value), W @ x)

    def test_via_reduce_term(self, cx, real_sr, hidden, backend, coder):
        eq = Equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim, _ = resolve_equation(eq, backend)
        graph = make_graph(primitives={prim.name: prim})

        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])

        term = apply(apply(var("ua.equation.linear"),
                           encode_array(coder, W)),
                     encode_array(coder, x))
        out = decode_term(coder, assert_reduce_ok(cx, graph, term))
        np.testing.assert_allclose(out, W @ x)

    def test_non_square(self, cx, real_sr, hidden, backend, coder):
        eq = Equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim, _ = resolve_equation(eq, backend)
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
        eq = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim, _ = resolve_equation(eq, backend)
        assert isinstance(prim, hydra.graph.Primitive)

    def test_name(self, hidden, backend):
        eq = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim, _ = resolve_equation(eq, backend)
        assert prim.name == Name("ua.equation.relu")

    def test_relu(self, cx, hidden, backend, coder):
        eq = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim, _ = resolve_equation(eq, backend)
        graph = make_graph()

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = prim.implementation(cx, graph, (encode_array(coder, x),))
        assert isinstance(result, Right)
        np.testing.assert_allclose(decode_term(coder, result.value), np.maximum(0, x))

    def test_via_reduce_term(self, cx, hidden, backend, coder):
        eq = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        prim, _ = resolve_equation(eq, backend)
        graph = make_graph(primitives={prim.name: prim})

        x = np.array([-1.0, 0.0, 1.0, 2.0])
        term = apply(var("ua.equation.relu"), encode_array(coder, x))
        out = decode_term(coder, assert_reduce_ok(cx, graph, term))
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_sigmoid(self, cx, hidden, backend, coder):
        eq = Equation("sigmoid", None, hidden, hidden, nonlinearity="sigmoid")
        prim, _ = resolve_equation(eq, backend)
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
        eq = Equation("linear_relu", "ij,j->i", hidden, hidden, real_sr, nonlinearity="relu")
        prim, _ = resolve_equation(eq, backend)
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
        trop_sr = Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)
        hidden = Sort("hidden", trop_sr)
        eq = Equation("trop_linear", "ij,j->i", hidden, hidden, trop_sr)
        prim, _ = resolve_equation(eq, backend)
        graph = make_graph(primitives={prim.name: prim})

        W = np.array([[1.0, 3.0], [2.0, 0.0]])
        x = np.array([1.0, 2.0])

        term = apply(apply(var("ua.equation.trop_linear"),
                           encode_array(coder, W)),
                     encode_array(coder, x))
        out = decode_term(coder, assert_reduce_ok(cx, graph, term))
        np.testing.assert_allclose(out, np.array([2.0, 2.0]))

    def test_fuzzy(self, cx, backend, coder):
        fuz_sr = Semiring("fuzzy", plus="maximum", times="minimum", zero=0.0, one=1.0,
                          bottom=0.0, top=1.0)
        hidden = Sort("hidden", fuz_sr)
        eq = Equation("fuz_linear", "ij,j->i", hidden, hidden, fuz_sr)
        prim, _ = resolve_equation(eq, backend)
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
        linear, _ = resolve_equation(Equation("linear", "ij,j->i", hidden, hidden, real_sr), backend)
        relu, _ = resolve_equation(Equation("relu", None, hidden, hidden, nonlinearity="relu"), backend)

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
        linear, _ = resolve_equation(Equation("linear", "ij,j->i", hidden, hidden, real_sr), backend)
        relu, _ = resolve_equation(Equation("relu", None, hidden, hidden, nonlinearity="relu"), backend)

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


# ---------------------------------------------------------------------------
# Arity > 3: list-packing kicks in
# ---------------------------------------------------------------------------

class TestArityPacking:
    """Equations with n_params + n_inputs > 3 list-pack into Hydra prim slots
    while keeping the native callable variadic."""

    def test_four_tensor_inputs_native(self, real_sr, hidden, backend):
        """Einsum 'i,i,i,i->i' (arity 4): native_fn stays variadic."""
        eq = Equation("e4_native", "i,i,i,i->i", hidden, hidden, real_sr)
        prim, native_fn = resolve_equation(eq, backend)
        # Hydra prim is packed to a single list_-coded slot
        assert prim.name.value == "ua.equation.e4_native"
        result = native_fn(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]),
            np.array([5.0, 6.0]), np.array([7.0, 8.0]),
        )
        np.testing.assert_allclose(result, [1*3*5*7, 2*4*6*8])

    def test_four_tensor_inputs_via_program(self, real_sr, hidden, backend):
        """Full pipeline: compile_program + variadic call for arity 4."""
        from unialg import compile_program
        eq = Equation("e4_prog", "i,i,i,i->i", hidden, hidden, real_sr)
        prog = compile_program([eq], backend=backend)
        result = prog(
            "e4_prog",
            np.array([1.0, 2.0]), np.array([3.0, 4.0]),
            np.array([5.0, 6.0]), np.array([7.0, 8.0]),
        )
        np.testing.assert_allclose(result, [105.0, 384.0])

    def test_three_inputs_with_two_params(self, real_sr, hidden, backend):
        """Arity 5: 3 tensor inputs + 2 scalar param_slots. Native stays variadic."""
        backend.unary_ops["scaled_add"] = lambda x, a, b: a * x + b
        eq = Equation("e5", "i,i,i->i", hidden, hidden, real_sr,
                      nonlinearity="scaled_add", param_slots=("a", "b"))
        prim, native_fn = resolve_equation(eq, backend)
        # Variadic: 2 params then 3 tensors
        result = native_fn(2.0, 1.0, np.array([1.0]), np.array([2.0]), np.array([3.0]))
        # product 1*2*3=6; 2*6+1=13
        np.testing.assert_allclose(result, [13.0])
