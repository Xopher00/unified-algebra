"""Phase 14: Feedforward neural network expressed as a path of equations.

Demonstrates that a multi-layer feedforward network is a sequential composition
of linear morphisms and nonlinear pointwise equations, semiring-polymorphic.
"""

import numpy as np
import pytest

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unified_algebra import (
    numpy_backend, semiring, sort, tensor_coder, sort_coder,
    equation, assemble_graph, PathSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()

@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))

@pytest.fixture
def real_sr():
    return semiring("real14ff", plus="add", times="multiply", zero=0.0, one=1.0)

@pytest.fixture
def hidden(real_sr):
    return sort("h14ff", real_sr)

@pytest.fixture
def coder():
    return tensor_coder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    result = coder.decode(None, np.ascontiguousarray(arr, dtype=np.float64))
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


# ---------------------------------------------------------------------------
# Architecture: 3-layer feedforward (linear → relu → linear → relu → linear)
# ---------------------------------------------------------------------------

def build_ffn_equations(hidden, real_sr):
    """5 equations: 3 linear layers + 2 relu activations, wired with inputs=."""
    return [
        equation("ffn_linear1", "ij,j->i", hidden, hidden, real_sr),
        equation("ffn_relu1",   None, hidden, hidden,
                 nonlinearity="relu", inputs=("ffn_linear1",)),
        equation("ffn_linear2", "ij,j->i", hidden, hidden, real_sr,
                 inputs=("ffn_relu1",)),
        equation("ffn_relu2",   None, hidden, hidden,
                 nonlinearity="relu", inputs=("ffn_linear2",)),
        equation("ffn_linear3", "ij,j->i", hidden, hidden, real_sr,
                 inputs=("ffn_relu2",)),
    ]


class TestFeedforward:

    def test_graph_has_all_primitives(self, hidden, real_sr, backend):
        """assemble_graph produces primitives for all 5 equations and the path bound term."""
        eqs = build_ffn_equations(hidden, real_sr)
        graph = assemble_graph(
            eqs, backend,
            specs=[PathSpec("ffn", ["ffn_linear1", "ffn_relu1",
                                    "ffn_linear2", "ffn_relu2",
                                    "ffn_linear3"], hidden, hidden)],
        )
        for name in ("ffn_linear1", "ffn_relu1", "ffn_linear2",
                     "ffn_relu2", "ffn_linear3"):
            assert Name(f"ua.equation.{name}") in graph.primitives
        assert Name("ua.path.ffn") in graph.bound_terms

    def test_path_produces_correct_output(self, hidden, real_sr, backend, cx, coder):
        """Running the path produces W3 @ relu(W2 @ relu(W1 @ x))."""
        eqs = build_ffn_equations(hidden, real_sr)
        graph = assemble_graph(
            eqs, backend,
            specs=[PathSpec("ffn", ["ffn_linear1", "ffn_relu1",
                                    "ffn_linear2", "ffn_relu2",
                                    "ffn_linear3"], hidden, hidden)],
        )

        W1 = np.array([[1.0, -1.0], [-1.0,  1.0]])
        W2 = np.array([[0.5,  0.5], [ 0.5, -0.5]])
        W3 = np.array([[2.0,  0.0], [ 0.0,  2.0]])
        x  = np.array([3.0, 1.0])

        # Encode weights as bound terms by running individual equations
        sc = sort_coder(hidden, backend)
        x_enc = encode_array(coder, x)

        # Build a graph with weights bound so the equations resolve correctly
        from unified_algebra import build_graph
        from hydra.graph import Graph
        from hydra.dsl.python import FrozenDict as FD

        # Run step-by-step using individual equation primitives
        prim1 = graph.primitives[Name("ua.equation.ffn_linear1")]
        prim2 = graph.primitives[Name("ua.equation.ffn_relu1")]
        prim3 = graph.primitives[Name("ua.equation.ffn_linear2")]
        prim4 = graph.primitives[Name("ua.equation.ffn_relu2")]
        prim5 = graph.primitives[Name("ua.equation.ffn_linear3")]

        w1_enc = encode_array(coder, W1)
        w2_enc = encode_array(coder, W2)
        w3_enc = encode_array(coder, W3)

        # Each linear prim takes (weight, input)
        h1 = decode_term(coder, assert_reduce_ok(cx, graph,
                apply(apply(var("ua.equation.ffn_linear1"), w1_enc), x_enc)))
        h2 = decode_term(coder, assert_reduce_ok(cx, graph,
                apply(var("ua.equation.ffn_relu1"), encode_array(coder, h1))))
        h3 = decode_term(coder, assert_reduce_ok(cx, graph,
                apply(apply(var("ua.equation.ffn_linear2"), w2_enc), encode_array(coder, h2))))
        h4 = decode_term(coder, assert_reduce_ok(cx, graph,
                apply(var("ua.equation.ffn_relu2"), encode_array(coder, h3))))
        out = decode_term(coder, assert_reduce_ok(cx, graph,
                apply(apply(var("ua.equation.ffn_linear3"), w3_enc), encode_array(coder, h4))))

        expected = W3 @ np.maximum(0, W2 @ np.maximum(0, W1 @ x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_semiring_swap_tropical(self, backend):
        """Same 5-equation architecture with tropical semiring assembles without error."""
        trop = semiring("tropical14ff", plus="minimum", times="add",
                        zero=float("inf"), one=0.0)
        h = sort("h14ff_t", trop)
        eqs = [
            equation("tffn_linear1", "ij,j->i", h, h, trop),
            equation("tffn_relu1",   None, h, h,
                     nonlinearity="relu", inputs=("tffn_linear1",)),
            equation("tffn_linear2", "ij,j->i", h, h, trop, inputs=("tffn_relu1",)),
            equation("tffn_relu2",   None, h, h,
                     nonlinearity="relu", inputs=("tffn_linear2",)),
            equation("tffn_linear3", "ij,j->i", h, h, trop, inputs=("tffn_relu2",)),
        ]
        graph = assemble_graph(
            eqs, backend,
            specs=[PathSpec("tffn", ["tffn_linear1", "tffn_relu1",
                                     "tffn_linear2", "tffn_relu2",
                                     "tffn_linear3"], h, h)],
        )
        assert Name("ua.path.tffn") in graph.bound_terms
        for name in ("tffn_linear1", "tffn_relu1", "tffn_linear2",
                     "tffn_relu2", "tffn_linear3"):
            assert Name(f"ua.equation.{name}") in graph.primitives
