"""Phase 3 tests: morphisms registered as Hydra Primitives, called via reduce_term."""

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
from unified_algebra.semiring import semiring, resolve_semiring
from unified_algebra.sort import tensor_coder, build_graph
from unified_algebra.morphism import parametric_morphism, pointwise_morphism


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def real_sr(backend):
    return resolve_semiring(
        semiring("real", plus="add", times="multiply", zero=0.0, one=1.0),
        backend,
    )


@pytest.fixture
def tropical_sr(backend):
    return resolve_semiring(
        semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0),
        backend,
    )


@pytest.fixture
def fuzzy_sr(backend):
    return resolve_semiring(
        semiring("fuzzy", plus="maximum", times="minimum", zero=0.0, one=1.0),
        backend,
    )


@pytest.fixture
def coder():
    return tensor_coder()


@pytest.fixture
def cx():
    """Minimal Hydra execution context."""
    return Context(
        trace=(),
        messages=(),
        other=FrozenDict({}),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    """Encode a numpy array to a Hydra Term (unwrap Right)."""
    result = coder.decode(None, arr)
    assert isinstance(result, Right), f"encode_array failed: {result}"
    return result.value


def decode_term(coder, term):
    """Decode a Hydra Term back to a numpy array (unwrap Right)."""
    result = coder.encode(None, None, term)
    assert isinstance(result, Right), f"decode_term failed: {result}"
    return result.value


def make_minimal_graph(primitives=None):
    """Build a Hydra Graph with no sorts and optional primitives."""
    return build_graph([], primitives=primitives)


def assert_reduce_ok(cx, graph, term):
    """Call reduce_term and assert it returned Right; return the inner term."""
    result = reduce_term(cx, graph, True, term)
    assert isinstance(result, Right), f"reduce_term returned Left: {result}"
    return result.value


# ---------------------------------------------------------------------------
# TestParametricMorphism
# ---------------------------------------------------------------------------

class TestParametricMorphism:

    def test_is_primitive(self, real_sr, backend):
        prim = parametric_morphism("linear", "ij,j->i", real_sr, backend)
        assert isinstance(prim, hydra.graph.Primitive)

    def test_primitive_has_correct_name(self, real_sr, backend):
        prim = parametric_morphism("linear", "ij,j->i", real_sr, backend)
        assert prim.name == Name("ua.morphism.linear")

    def test_direct_call(self, cx, real_sr, backend, coder):
        """Call implementation directly without reduce_term; verify W @ x."""
        prim = parametric_morphism("linear", "ij,j->i", real_sr, backend)
        graph = make_minimal_graph()

        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])
        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        result = prim.implementation(cx, graph, (x_term, W_term))
        assert isinstance(result, Right), f"implementation returned Left: {result}"
        out = decode_term(coder, result.value)
        np.testing.assert_allclose(out, W @ x)

    def test_via_reduce_term(self, cx, real_sr, backend, coder):
        """Register primitive in graph and call through reduce_term."""
        prim = parametric_morphism("linear", "ij,j->i", real_sr, backend)
        prim_name = Name("ua.morphism.linear")
        graph = make_minimal_graph(primitives={prim_name: prim})

        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 1.0])
        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        # apply(apply(var("ua.morphism.linear"), x_term), W_term)
        term = apply(apply(var("ua.morphism.linear"), x_term), W_term)
        result_term = assert_reduce_ok(cx, graph, term)
        out = decode_term(coder, result_term)
        np.testing.assert_allclose(out, W @ x)

    def test_direct_call_non_square(self, cx, real_sr, backend, coder):
        """Test with non-square weight matrix."""
        prim = parametric_morphism("linear", "ij,j->i", real_sr, backend)
        graph = make_minimal_graph()

        W = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])  # (2, 3)
        x = np.array([1.0, 2.0, 3.0])                      # (3,)
        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        result = prim.implementation(cx, graph, (x_term, W_term))
        assert isinstance(result, Right)
        out = decode_term(coder, result.value)
        np.testing.assert_allclose(out, W @ x)


# ---------------------------------------------------------------------------
# TestPointwiseMorphism
# ---------------------------------------------------------------------------

class TestPointwiseMorphism:

    def test_is_primitive(self, backend):
        prim = pointwise_morphism("relu", "relu", backend)
        assert isinstance(prim, hydra.graph.Primitive)

    def test_primitive_has_correct_name(self, backend):
        prim = pointwise_morphism("relu", "relu", backend)
        assert prim.name == Name("ua.pointwise.relu")

    def test_direct_call_relu(self, cx, backend, coder):
        """ReLU zeros out negatives, passes through positives."""
        prim = pointwise_morphism("relu", "relu", backend)
        graph = make_minimal_graph()

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        x_term = encode_array(coder, x)

        result = prim.implementation(cx, graph, (x_term,))
        assert isinstance(result, Right), f"implementation returned Left: {result}"
        out = decode_term(coder, result.value)
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_direct_call_relu_all_negative(self, cx, backend, coder):
        """All-negative input should yield all zeros."""
        prim = pointwise_morphism("relu", "relu", backend)
        graph = make_minimal_graph()

        x = np.array([-5.0, -3.0, -0.1])
        x_term = encode_array(coder, x)

        result = prim.implementation(cx, graph, (x_term,))
        assert isinstance(result, Right)
        out = decode_term(coder, result.value)
        np.testing.assert_allclose(out, np.zeros(3))

    def test_via_reduce_term(self, cx, backend, coder):
        """Register relu primitive and call through reduce_term."""
        prim = pointwise_morphism("relu", "relu", backend)
        prim_name = Name("ua.pointwise.relu")
        graph = make_minimal_graph(primitives={prim_name: prim})

        x = np.array([-1.0, 0.0, 1.0, 2.0])
        x_term = encode_array(coder, x)

        term = apply(var("ua.pointwise.relu"), x_term)
        result_term = assert_reduce_ok(cx, graph, term)
        out = decode_term(coder, result_term)
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_direct_call_sigmoid(self, cx, backend, coder):
        """Sigmoid maps any input to (0, 1)."""
        prim = pointwise_morphism("sigmoid", "sigmoid", backend)
        graph = make_minimal_graph()

        x = np.array([-2.0, 0.0, 2.0])
        x_term = encode_array(coder, x)

        result = prim.implementation(cx, graph, (x_term,))
        assert isinstance(result, Right)
        out = decode_term(coder, result.value)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestSemiringVariants
# ---------------------------------------------------------------------------

class TestSemiringVariants:

    def test_tropical_morphism(self, cx, tropical_sr, backend, coder):
        """Tropical (min-plus) semiring: Y_i = min_j(W_ij + x_j)."""
        prim = parametric_morphism("tropical_linear", "ij,j->i", tropical_sr, backend)
        graph = make_minimal_graph()

        W = np.array([[1.0, 3.0], [2.0, 0.0]])
        x = np.array([1.0, 2.0])
        # Y_0 = min(1+1, 3+2) = min(2, 5) = 2
        # Y_1 = min(2+1, 0+2) = min(3, 2) = 2
        expected = np.array([2.0, 2.0])

        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        result = prim.implementation(cx, graph, (x_term, W_term))
        assert isinstance(result, Right)
        out = decode_term(coder, result.value)
        np.testing.assert_allclose(out, expected)

    def test_tropical_via_reduce_term(self, cx, tropical_sr, backend, coder):
        """Tropical morphism callable via reduce_term."""
        prim = parametric_morphism("tropical_linear", "ij,j->i", tropical_sr, backend)
        prim_name = Name("ua.morphism.tropical_linear")
        graph = make_minimal_graph(primitives={prim_name: prim})

        W = np.array([[1.0, 3.0], [2.0, 0.0]])
        x = np.array([1.0, 2.0])
        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        term = apply(apply(var("ua.morphism.tropical_linear"), x_term), W_term)
        result_term = assert_reduce_ok(cx, graph, term)
        out = decode_term(coder, result_term)
        np.testing.assert_allclose(out, np.array([2.0, 2.0]))

    def test_fuzzy_morphism(self, cx, fuzzy_sr, backend, coder):
        """Fuzzy (max-min) semiring: Y_i = max_j(min(W_ij, x_j))."""
        prim = parametric_morphism("fuzzy_linear", "ij,j->i", fuzzy_sr, backend)
        graph = make_minimal_graph()

        W = np.array([[0.8, 0.3], [0.2, 0.9]])
        x = np.array([0.6, 0.7])
        # Y_0 = max(min(0.8,0.6), min(0.3,0.7)) = max(0.6, 0.3) = 0.6
        # Y_1 = max(min(0.2,0.6), min(0.9,0.7)) = max(0.2, 0.7) = 0.7
        expected = np.array([0.6, 0.7])

        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        result = prim.implementation(cx, graph, (x_term, W_term))
        assert isinstance(result, Right)
        out = decode_term(coder, result.value)
        np.testing.assert_allclose(out, expected)

    def test_fuzzy_via_reduce_term(self, cx, fuzzy_sr, backend, coder):
        """Fuzzy morphism callable via reduce_term."""
        prim = parametric_morphism("fuzzy_linear", "ij,j->i", fuzzy_sr, backend)
        prim_name = Name("ua.morphism.fuzzy_linear")
        graph = make_minimal_graph(primitives={prim_name: prim})

        W = np.array([[0.8, 0.3], [0.2, 0.9]])
        x = np.array([0.6, 0.7])
        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        term = apply(apply(var("ua.morphism.fuzzy_linear"), x_term), W_term)
        result_term = assert_reduce_ok(cx, graph, term)
        out = decode_term(coder, result_term)
        np.testing.assert_allclose(out, np.array([0.6, 0.7]))


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_chain_linear_then_relu(self, cx, real_sr, backend, coder):
        """Chain linear and relu in the same graph: relu(W @ x)."""
        linear_prim = parametric_morphism("linear", "ij,j->i", real_sr, backend)
        relu_prim = pointwise_morphism("relu", "relu", backend)

        linear_name = Name("ua.morphism.linear")
        relu_name = Name("ua.pointwise.relu")

        graph = make_minimal_graph(primitives={
            linear_name: linear_prim,
            relu_name: relu_prim,
        })

        W = np.array([[1.0, -2.0], [-3.0, 4.0]])
        x = np.array([1.0, 1.0])
        # W @ x = [-1.0, 1.0]; relu => [0.0, 1.0]
        expected = np.maximum(0, W @ x)

        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        # apply(var("ua.pointwise.relu"), apply(apply(var("ua.morphism.linear"), x_term), W_term))
        inner = apply(apply(var("ua.morphism.linear"), x_term), W_term)
        outer = apply(var("ua.pointwise.relu"), inner)

        result_term = assert_reduce_ok(cx, graph, outer)
        out = decode_term(coder, result_term)
        np.testing.assert_allclose(out, expected)

    def test_chain_linear_then_relu_all_negative(self, cx, real_sr, backend, coder):
        """Chain where W @ x is all negative → relu clamps to zeros."""
        linear_prim = parametric_morphism("linear", "ij,j->i", real_sr, backend)
        relu_prim = pointwise_morphism("relu", "relu", backend)

        linear_name = Name("ua.morphism.linear")
        relu_name = Name("ua.pointwise.relu")

        graph = make_minimal_graph(primitives={
            linear_name: linear_prim,
            relu_name: relu_prim,
        })

        W = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        x = np.array([1.0, 1.0])
        # W @ x = [-3.0, -7.0]; relu => [0.0, 0.0]
        expected = np.zeros(2)

        W_term = encode_array(coder, W)
        x_term = encode_array(coder, x)

        inner = apply(apply(var("ua.morphism.linear"), x_term), W_term)
        outer = apply(var("ua.pointwise.relu"), inner)

        result_term = assert_reduce_ok(cx, graph, outer)
        out = decode_term(coder, result_term)
        np.testing.assert_allclose(out, expected)

    def test_two_layers(self, cx, real_sr, backend, coder):
        """Two linear layers: y = W2 @ relu(W1 @ x)."""
        linear_prim = parametric_morphism("linear", "ij,j->i", real_sr, backend)
        relu_prim = pointwise_morphism("relu", "relu", backend)

        linear_name = Name("ua.morphism.linear")
        relu_name = Name("ua.pointwise.relu")

        graph = make_minimal_graph(primitives={
            linear_name: linear_prim,
            relu_name: relu_prim,
        })

        W1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
        W2 = np.array([[2.0, 0.0], [0.0, 2.0]])
        x = np.array([3.0, 1.0])
        expected = W2 @ np.maximum(0, W1 @ x)

        W1_term = encode_array(coder, W1)
        W2_term = encode_array(coder, W2)
        x_term = encode_array(coder, x)

        h = apply(var("ua.pointwise.relu"), apply(apply(var("ua.morphism.linear"), x_term), W1_term))
        y = apply(apply(var("ua.morphism.linear"), h), W2_term)

        result_term = assert_reduce_ok(cx, graph, y)
        out = decode_term(coder, result_term)
        np.testing.assert_allclose(out, expected)
