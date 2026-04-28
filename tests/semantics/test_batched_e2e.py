"""Batched end-to-end tests: correctness via reduce_term for batched sorts, paths, and fans."""

import numpy as np
import pytest

from hydra.context import Context
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
    PathSpec, FanSpec,
)
from unialg.terms import tensor_coder
from unialg.assembly.graph import build_graph, assemble_graph
from unialg.assembly.compositions import PathComposition, FanComposition
from unialg.assembly._equation_resolution import resolve_equation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


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


# ===========================================================================
# End-to-end correctness via reduce_term
# ===========================================================================

class TestBatchedEndToEnd:
    """Verify correct numerical output for batched sorts through reduce_term."""

    def test_batched_pointwise(self, cx, real_sr, backend, coder):
        """Relu on a batch of vectors produces elementwise relu."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        prim, *_ = resolve_equation(eq, backend)

        from hydra.sources.libraries import standard_library
        primitives = dict(standard_library())
        primitives[prim.name] = prim
        graph = build_graph([], primitives=primitives)

        # Batch of 3 vectors, each of length 4
        x = np.array([[-1.0, 0.5, -0.3, 2.0],
                      [0.0, -1.5, 1.0, -0.5],
                      [3.0, -2.0, 0.0, 0.1]])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.relu_b"), x_enc)
        ))
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_batched_unary_einsum(self, cx, real_sr, backend, coder):
        """Unary einsum 'i->i' on a batched sort sums nothing — becomes 'bi->bi'."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        # "i->i" is a trace/copy — with real semiring it's just identity copy
        eq = Equation("identity_b", "i->i", hidden_b, hidden_b, real_sr)
        prim, *_ = resolve_equation(eq, backend)

        from hydra.sources.libraries import standard_library
        primitives = dict(standard_library())
        primitives[prim.name] = prim
        graph = build_graph([], primitives=primitives)

        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.equation.identity_b"), x_enc)
        ))
        np.testing.assert_allclose(out, x)

    def test_batched_matmul_tiled_weight(self, cx, real_sr, backend, coder):
        """Batched matmul: each sample gets its own weight matrix (W tiled).

        Equation declared as 'ij,j->i'; resolved as 'bij,bj->bi'.
        W is tiled across the batch axis: shape (B, out, in).
        """
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq = Equation("linear_b", "ij,j->i", hidden_b, hidden_b, real_sr)
        prim, *_ = resolve_equation(eq, backend)

        from hydra.sources.libraries import standard_library
        primitives = dict(standard_library())
        primitives[prim.name] = prim
        graph = build_graph([], primitives=primitives)

        B, out_dim, in_dim = 4, 3, 2
        W_single = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        W_batch = np.tile(W_single[np.newaxis], (B, 1, 1))           # (4, 3, 2)
        X = np.random.default_rng(42).standard_normal((B, in_dim))   # (4, 2)

        W_enc = encode_array(coder, W_batch)
        X_enc = encode_array(coder, X)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.linear_b"), W_enc), X_enc)
        ))

        # Oracle: each sample i: W_single @ X[i]
        expected = X @ W_single.T   # (4, 3)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_batched_linear_relu(self, cx, real_sr, backend, coder):
        """Combined batched equation: batched matmul + relu in one equation."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq = Equation("linear_relu_b", "ij,j->i", hidden_b, hidden_b,
                      real_sr, nonlinearity="relu")
        prim, *_ = resolve_equation(eq, backend)

        from hydra.sources.libraries import standard_library
        primitives = dict(standard_library())
        primitives[prim.name] = prim
        graph = build_graph([], primitives=primitives)

        B, out_dim, in_dim = 3, 2, 3
        W_single = np.array([[1.0, -1.0, 0.5], [-0.5, 1.0, -1.0]])  # (2, 3)
        W_batch = np.tile(W_single[np.newaxis], (B, 1, 1))            # (3, 2, 3)
        X = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.5, 0.0, -0.5]])

        W_enc = encode_array(coder, W_batch)
        X_enc = encode_array(coder, X)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.linear_relu_b"), W_enc), X_enc)
        ))

        expected = np.maximum(0, X @ W_single.T)
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ===========================================================================
# Batched path composition
# ===========================================================================

class TestBatchedPath:
    """Sequential composition of batched equations."""

    def test_batched_path_structure(self, real_sr, backend):
        """PathComposition on batched equations builds the same lambda structure as unbatched."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq1 = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq2 = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh")
        p = PathComposition("b_pipe", ["relu_b", "tanh_b"]).to_lambda()
        # PathComposition().to_lambda() returns a (Name, Term) tuple
        assert p is not None

    def test_batched_path_end_to_end(self, cx, real_sr, backend, coder):
        """Two-step batched path: relu then tanh applied to a batch."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq_relu = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq_tanh = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh")

        graph, *_ = assemble_graph(
            [eq_relu, eq_tanh], backend,
            specs=[PathSpec("b_pipe", ["relu_b", "tanh_b"], hidden_b, hidden_b)],
        )

        x = np.array([[-1.0, 0.5, 2.0],
                      [0.0, -0.5, 1.5]])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.b_pipe"), x_enc)
        ))
        expected = np.tanh(np.maximum(0, x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_batched_three_step_path(self, cx, real_sr, backend, coder):
        """Three-step batched path: relu -> tanh -> relu."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq_relu = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq_tanh = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh")
        eq_relu2 = Equation("relu_b2", None, hidden_b, hidden_b, nonlinearity="relu")

        graph, *_ = assemble_graph(
            [eq_relu, eq_tanh, eq_relu2], backend,
            specs=[PathSpec("b_pipe3", ["relu_b", "tanh_b", "relu_b2"], hidden_b, hidden_b)],
        )

        x = np.array([[-2.0, 1.0], [0.5, -0.5], [0.0, 3.0]])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.path.b_pipe3"), x_enc)
        ))
        expected = np.maximum(0, np.tanh(np.maximum(0, x)))
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ===========================================================================
# Batched fan composition
# ===========================================================================

class TestBatchedFan:
    """Parallel composition of batched equations."""

    def test_batched_fan_two_branches(self, cx, real_sr, backend, coder):
        """Two-branch fan over a batch: relu and tanh, merged by multiply."""
        hidden_b = Sort("hidden", real_sr, batched=True)
        eq_relu = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq_tanh = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh")
        eq_merge = Equation("merge_b", "i,i->i", hidden_b, hidden_b, real_sr)

        graph, *_ = assemble_graph(
            [eq_relu, eq_tanh, eq_merge], backend,
            specs=[FanSpec("b_fan", ["relu_b", "tanh_b"], ["merge_b"], hidden_b, hidden_b)],
        )

        x = np.array([[-1.0, 0.5, 0.0, 2.0],
                      [0.3, -0.3, 1.0, -1.0]])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.b_fan"), x_enc)
        ))
        expected = np.maximum(0, x) * np.tanh(x)
        np.testing.assert_allclose(out, expected, rtol=1e-6)
