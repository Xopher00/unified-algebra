"""Architecture end-to-end tests: assemble_graph execution across network topologies."""

import numpy as np
import pytest

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import NumpyBackend, Semiring, Sort, Equation
from unialg.terms import tensor_coder
from unialg.assembly.graph import assemble_graph


@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


@pytest.fixture
def real():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real):
    return Sort("hidden", real)


def enc(coder, arr):
    return coder.decode(None, arr).value


def dec(coder, term):
    return coder.encode(None, None, term).value


# ---------------------------------------------------------------------------
# Assemble and run
# ---------------------------------------------------------------------------

class TestAssembleAndRun:

    def test_two_layer_network(self, backend, cx, coder):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = Sort("hidden", real)
        eqs = [
            Equation("linear", "ij,j->i", hidden, hidden, real),
            Equation("act", None, hidden, hidden, nonlinearity="relu", inputs=("linear",)),
        ]
        graph, *_ = assemble_graph(eqs, backend)

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
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = Sort("hidden", real)
        eqs = [Equation("lr", "ij,j->i", hidden, hidden, real, nonlinearity="relu")]
        graph, *_ = assemble_graph(eqs, backend)

        W = np.array([[1.0, -2.0], [-3.0, 4.0]])
        x = np.array([1.0, 1.0])
        expected = np.maximum(0, W @ x)

        term = apply(apply(var("ua.equation.lr"),
                           enc(coder, W)), enc(coder, x))
        result = reduce_term(cx, graph, True, term)
        assert isinstance(result, Right)
        np.testing.assert_allclose(dec(coder, result.value), expected)


# ---------------------------------------------------------------------------
# DAG assembly and execution
# ---------------------------------------------------------------------------

class TestDAGAssemble:

    def test_fan_out_assembled(self, real, hidden, backend):
        a = Equation("A", "ij,j->i", hidden, hidden, real)
        b = Equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        c = Equation("C", None, hidden, hidden, nonlinearity="sigmoid", inputs=("A",))
        graph, *_ = assemble_graph([a, b, c], backend)
        assert Name("ua.equation.A") in graph.primitives
        assert Name("ua.equation.B") in graph.primitives
        assert Name("ua.equation.C") in graph.primitives

    def test_fan_out_executes(self, real, hidden, backend, cx, coder):
        """Two equations reading the same input — fan-out."""
        a = Equation("linear", "ij,j->i", hidden, hidden, real)
        b = Equation("relu_out", None, hidden, hidden, nonlinearity="relu", inputs=("linear",))
        c = Equation("sig_out", None, hidden, hidden, nonlinearity="sigmoid", inputs=("linear",))
        graph, *_ = assemble_graph([a, b, c], backend)

        W = np.array([[1.0, -2.0], [-3.0, 4.0]])
        x = np.array([1.0, 1.0])
        h = W @ x  # [-1.0, 1.0]

        W_term = enc(coder, W)
        x_term = enc(coder, x)
        h_term = apply(apply(var("ua.equation.linear"), W_term), x_term)

        relu_term = apply(var("ua.equation.relu_out"), h_term)
        sig_term = apply(var("ua.equation.sig_out"), h_term)

        relu_result = reduce_term(cx, graph, True, relu_term)
        sig_result = reduce_term(cx, graph, True, sig_term)

        assert isinstance(relu_result, Right)
        assert isinstance(sig_result, Right)
        np.testing.assert_allclose(dec(coder, relu_result.value), np.maximum(0, h))
        from scipy.special import expit
        np.testing.assert_allclose(dec(coder, sig_result.value), expit(h), rtol=1e-6)

    def test_diamond_executes(self, real, hidden, backend, cx, coder):
        """Diamond: A → B, A → C, then D reads B and C."""
        a = Equation("A", "ij,j->i", hidden, hidden, real)
        b = Equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        c = Equation("C", None, hidden, hidden, nonlinearity="tanh", inputs=("A",))
        graph, *_ = assemble_graph([a, b, c], backend)

        W = np.array([[1.0, -1.0], [-1.0, 1.0]])
        x = np.array([2.0, 1.0])
        h = W @ x

        W_term = enc(coder, W)
        x_term = enc(coder, x)
        h_term = apply(apply(var("ua.equation.A"), W_term), x_term)

        b_out = reduce_term(cx, graph, True, apply(var("ua.equation.B"), h_term))
        c_out = reduce_term(cx, graph, True, apply(var("ua.equation.C"), h_term))

        assert isinstance(b_out, Right)
        assert isinstance(c_out, Right)
        np.testing.assert_allclose(dec(coder, b_out.value), np.maximum(0, h))
        np.testing.assert_allclose(dec(coder, c_out.value), np.tanh(h))
