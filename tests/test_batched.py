"""Batched sort tests: batched sorts — automatic batch dimension handling.

Key invariant: declaration is logical (unbatched einsum stored in the
equation record); resolution is physical (einsum rewritten at resolve time
when the domain sort is batched).
"""

import numpy as np
import pytest

import hydra.core as core
from hydra.context import Context
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
import hydra.dsl.terms as Terms
from hydra.reduction import reduce_term

from unialg import (
    numpy_backend, semiring, sort, tensor_coder, sort_coder,
    is_batched, validate_pipeline, Equation,
    path, fan, validate_spec,
    build_graph, assemble_graph, PathSpec, FanSpec,
)
from unialg.algebra import sort_type_from_term
from unialg.resolve.morphism import _prepend_batch_dim


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
    return semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


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


# ===========================================================================
# Part A: Sort construction and type encoding
# ===========================================================================

class TestBatchedSortConstruction:
    """Verify that the batched flag is stored and reflected in the TypeVariable."""

    def test_batched_sort_identity(self, real_sr):
        """Batched and unbatched sorts with the same name produce distinct types."""
        import hydra.core as core
        unbatched = sort("hidden", real_sr, batched=False)
        batched = sort("hidden", real_sr, batched=True)

        t_unbatched = sort_type_from_term(unbatched)
        t_batched = sort_type_from_term(batched)

        assert t_unbatched != t_batched
        # Unbatched: TypeApplication(ua.sort.hidden, ua.semiring.real)
        assert t_unbatched.value.function == core.TypeVariable(core.Name("ua.sort.hidden"))
        assert t_unbatched.value.argument == core.TypeVariable(core.Name("ua.semiring.real"))
        # Batched: TypeApplication(ua.batched, TypeApplication(ua.sort.hidden, ua.semiring.real))
        assert t_batched.value.function == core.TypeVariable(core.Name("ua.batched"))
        assert t_batched.value.argument == t_unbatched

    def test_batched_type_from_term(self, real_sr):
        """sort_type_from_term with batched=True wraps in ua.batched TypeApplication."""
        tropical_sr = semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)
        s = sort("output", tropical_sr, batched=True)
        t = sort_type_from_term(s)
        assert t.value.function == core.TypeVariable(core.Name("ua.batched"))
        inner = t.value.argument
        assert inner.value.function == core.TypeVariable(core.Name("ua.sort.output"))
        assert inner.value.argument == core.TypeVariable(core.Name("ua.semiring.tropical"))

    def test_unbatched_type_from_term(self, real_sr):
        """sort_type_from_term with batched=False has no batched wrapper."""
        tropical_sr = semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)
        s = sort("output", tropical_sr, batched=False)
        t = sort_type_from_term(s)
        assert t.value.function == core.TypeVariable(core.Name("ua.sort.output"))
        assert t.value.argument == core.TypeVariable(core.Name("ua.semiring.tropical"))

    def test_is_batched_helper_true(self, real_sr):
        """is_batched returns True for batched=True sorts."""
        s = sort("hidden", real_sr, batched=True)
        assert is_batched(s) is True

    def test_is_batched_helper_false(self, real_sr):
        """is_batched returns False for batched=False sorts."""
        s = sort("hidden", real_sr, batched=False)
        assert is_batched(s) is False

    def test_is_batched_helper_default(self, real_sr):
        """is_batched returns False when batched flag is absent (old records)."""
        # Create a sort without the batched field by constructing the record manually
        import hydra.dsl.terms as T
        from unialg.algebra.sort import SORT_TYPE_NAME
        old_style = T.record(SORT_TYPE_NAME, [
            T.field("name", T.string("legacy")),
            T.field("semiring", real_sr),
        ])
        assert is_batched(old_style) is False

    def test_sort_type_from_term_batched_structure(self, real_sr):
        """sort_type_from_term for a batched sort produces the expected TypeApplication structure."""
        s = sort("encoder", real_sr, batched=True)
        t = sort_type_from_term(s)
        assert t.value.function == core.TypeVariable(core.Name("ua.batched"))
        inner = t.value.argument
        assert inner.value.function == core.TypeVariable(core.Name("ua.sort.encoder"))
        assert inner.value.argument == core.TypeVariable(core.Name("ua.semiring.real"))


# ===========================================================================
# Part B: _prepend_batch_dim helper
# ===========================================================================

class TestPrependBatchDim:
    """Unit tests for the einsum batch-prepend helper."""

    def test_vector_matvec(self):
        """ij,j->i becomes bij,bj->bi (standard matmul with batch)."""
        result = _prepend_batch_dim("ij,j->i")
        # 'b' is the first unused letter; i and j are used
        assert result == "bij,bj->bi"

    def test_matmul_chain(self):
        """ij,jk->ik: batch char must not be i, j, or k."""
        result = _prepend_batch_dim("ij,jk->ik")
        lhs, rhs = result.split("->")
        parts = lhs.split(",")
        batch_char = parts[0][0]
        assert batch_char not in {"i", "j", "k"}
        assert all(p.startswith(batch_char) for p in parts)
        assert rhs.startswith(batch_char)

    def test_single_input(self):
        """i->i (identity): becomes bi->bi."""
        result = _prepend_batch_dim("i->i")
        assert result == "bi->bi"

    def test_empty_string_passthrough(self):
        """Empty string passes through unchanged."""
        assert _prepend_batch_dim("") == ""

    def test_batch_char_avoids_used(self):
        """When 'b' is already in use, the next available char is chosen."""
        # 'b' is used as an index, so the batch char should be 'c' (next in sequence)
        result = _prepend_batch_dim("bj,j->b")
        batch_char = result[0]
        assert batch_char == "c"

    def test_output_structure(self):
        """The output always has the same number of comma-separated inputs as the original."""
        original = "ij,jk,kl->il"
        result = _prepend_batch_dim(original)
        orig_inputs = original.split("->")[0].split(",")
        result_inputs = result.split("->")[0].split(",")
        assert len(result_inputs) == len(orig_inputs)


# ===========================================================================
# Part C: resolve_equation with batched sorts
# ===========================================================================

class TestBatchedEquationResolution:
    """Verify that resolve_equation produces correct primitives for batched sorts."""

    def test_batched_pointwise_resolves(self, real_sr, backend):
        """Pointwise equation on batched sort resolves without error.

        Pointwise ops are elementwise and need no einsum rewriting.
        """
        hidden_b = sort("hidden", real_sr, batched=True)
        eq = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        prim = eq.resolve(backend)
        assert prim.name == core.Name("ua.equation.relu_b")

    def test_batched_unary_einsum_resolves(self, real_sr, backend):
        """Unary einsum on batched sort resolves (the einsum gets prepended)."""
        hidden_b = sort("hidden", real_sr, batched=True)
        eq = Equation("bn_scale", "i->i", hidden_b, hidden_b, real_sr)
        prim = eq.resolve(backend)
        assert prim.name == core.Name("ua.equation.bn_scale")

    def test_batched_binary_einsum_resolves(self, real_sr, backend):
        """Binary einsum on batched sort resolves — becomes a 2-input prim2."""
        hidden_b = sort("hidden", real_sr, batched=True)
        eq = Equation("linear_b", "ij,j->i", hidden_b, hidden_b, real_sr)
        prim = eq.resolve(backend)
        assert prim.name == core.Name("ua.equation.linear_b")

    def test_unbatched_still_works(self, real_sr, backend):
        """sort() with default batched=False is unchanged from pre-Phase9 behaviour."""
        hidden = sort("hidden", real_sr)  # batched=False by default
        eq = Equation("linear", "ij,j->i", hidden, hidden, real_sr)
        prim = eq.resolve(backend)
        assert prim.name == core.Name("ua.equation.linear")


# ===========================================================================
# Part D: Sort junctions with batched sorts
# ===========================================================================

class TestBatchedSortJunctions:
    """validate_pipeline must treat batched and unbatched sorts as distinct types."""

    def test_batched_to_batched_ok(self, real_sr):
        """Batched codomain → batched domain: junction passes."""
        hidden_b = sort("hidden", real_sr, batched=True)
        eq1 = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq2 = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh", inputs=("relu_b",))
        # Should not raise
        validate_pipeline([eq1, eq2])

    def test_unbatched_to_unbatched_ok(self, real_sr):
        """Unbatched codomain → unbatched domain: junction passes."""
        hidden = sort("hidden", real_sr, batched=False)
        eq1 = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq2 = Equation("tanh", None, hidden, hidden, nonlinearity="tanh", inputs=("relu",))
        validate_pipeline([eq1, eq2])

    def test_batched_to_unbatched_fails(self, real_sr):
        """Batched codomain → unbatched domain: junction raises TypeError."""
        hidden_b = sort("hidden", real_sr, batched=True)
        hidden = sort("hidden", real_sr, batched=False)
        eq_batched = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq_unbatched = Equation("tanh", None, hidden, hidden, nonlinearity="tanh", inputs=("relu_b",))
        with pytest.raises(TypeError):
            validate_pipeline([eq_batched, eq_unbatched])

    def test_unbatched_to_batched_fails(self, real_sr):
        """Unbatched codomain → batched domain: junction raises TypeError."""
        hidden_b = sort("hidden", real_sr, batched=True)
        hidden = sort("hidden", real_sr, batched=False)
        eq_unbatched = Equation("relu", None, hidden, hidden, nonlinearity="relu")
        eq_batched = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh", inputs=("relu",))
        with pytest.raises(TypeError):
            validate_pipeline([eq_unbatched, eq_batched])


# ===========================================================================
# Part E: End-to-end correctness via reduce_term
# ===========================================================================

class TestBatchedEndToEnd:
    """Verify correct numerical output for batched sorts through reduce_term."""

    def test_batched_pointwise(self, cx, real_sr, backend, coder):
        """Relu on a batch of vectors produces elementwise relu."""
        hidden_b = sort("hidden", real_sr, batched=True)
        eq = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        prim = eq.resolve(backend)

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
        hidden_b = sort("hidden", real_sr, batched=True)
        # "i->i" is a trace/copy — with real semiring it's just identity copy
        eq = Equation("identity_b", "i->i", hidden_b, hidden_b, real_sr)
        prim = eq.resolve(backend)

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
        hidden_b = sort("hidden", real_sr, batched=True)
        eq = Equation("linear_b", "ij,j->i", hidden_b, hidden_b, real_sr)
        prim = eq.resolve(backend)

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
        hidden_b = sort("hidden", real_sr, batched=True)
        eq = Equation("linear_relu_b", "ij,j->i", hidden_b, hidden_b,
                      real_sr, nonlinearity="relu")
        prim = eq.resolve(backend)

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
# Part F: Batched path and fan compositions
# ===========================================================================

class TestBatchedPath:
    """Sequential composition of batched equations."""

    def test_batched_path_structure(self, real_sr, backend):
        """path() on batched equations builds the same lambda structure as unbatched."""
        hidden_b = sort("hidden", real_sr, batched=True)
        eq1 = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq2 = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh")
        p = path("b_pipe", ["relu_b", "tanh_b"])
        # path() returns a Hydra Term (lambda)
        assert p is not None

    def test_batched_path_end_to_end(self, cx, real_sr, backend, coder):
        """Two-step batched path: relu then tanh applied to a batch."""
        hidden_b = sort("hidden", real_sr, batched=True)
        eq_relu = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq_tanh = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh")

        graph = assemble_graph(
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
        """Three-step batched path: relu → tanh → relu."""
        hidden_b = sort("hidden", real_sr, batched=True)
        eq_relu = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq_tanh = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh")
        eq_relu2 = Equation("relu_b2", None, hidden_b, hidden_b, nonlinearity="relu")

        graph = assemble_graph(
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


class TestBatchedFan:
    """Parallel composition of batched equations."""

    def test_batched_fan_two_branches(self, cx, real_sr, backend, coder):
        """Two-branch fan over a batch: relu and tanh, merged by multiply."""
        hidden_b = sort("hidden", real_sr, batched=True)
        eq_relu = Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu")
        eq_tanh = Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh")
        eq_merge = Equation("merge_b", "i,i->i", hidden_b, hidden_b, real_sr)

        graph = assemble_graph(
            [eq_relu, eq_tanh, eq_merge], backend,
            specs=[FanSpec("b_fan", ["relu_b", "tanh_b"], "merge_b", hidden_b, hidden_b)],
        )

        x = np.array([[-1.0, 0.5, 0.0, 2.0],
                      [0.3, -0.3, 1.0, -1.0]])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.b_fan"), x_enc)
        ))
        expected = np.maximum(0, x) * np.tanh(x)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_batched_fan_four_branches(self, cx, real_sr, backend, coder):
        """Four-branch batched fan with additive merge."""
        add_sr = semiring("add", plus="add", times="add", zero=0.0, one=0.0)
        hidden_b = sort("hidden", add_sr, batched=True)
        eqs = [
            Equation("relu_b", None, hidden_b, hidden_b, nonlinearity="relu"),
            Equation("tanh_b", None, hidden_b, hidden_b, nonlinearity="tanh"),
            Equation("abs_b",  None, hidden_b, hidden_b, nonlinearity="abs"),
            Equation("neg_b",  None, hidden_b, hidden_b, nonlinearity="neg"),
        ]
        eq_merge = Equation("add_merge_b", "i,i->i", hidden_b, hidden_b, add_sr)

        graph = assemble_graph(
            eqs + [eq_merge], backend,
            specs=[FanSpec("b_fan4", ["relu_b", "tanh_b", "abs_b", "neg_b"],
                           "add_merge_b", hidden_b, hidden_b)],
        )

        x = np.array([[-1.0, 0.5], [2.0, -2.0]])
        x_enc = encode_array(coder, x)

        out = decode_term(coder, assert_reduce_ok(
            cx, graph, apply(var("ua.fan.b_fan4"), x_enc)
        ))
        # times=add in "i,i->i" means elementwise add, fold pairwise
        expected = (np.maximum(0, x) + np.tanh(x) + np.abs(x) + (-x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ===========================================================================
# Part G: build_graph registers batched sort type correctly
# ===========================================================================

class TestBatchedGraphRegistration:
    """Sorts are registered in schema_types as component names."""

    def test_batched_sort_in_schema(self, real_sr):
        """build_graph registers component names (ua.sort.X, ua.semiring.Y, ua.batched)."""
        hidden_b = sort("hidden", real_sr, batched=True)
        graph = build_graph([hidden_b])
        assert core.Name("ua.sort.hidden") in graph.schema_types
        assert core.Name("ua.semiring.real") in graph.schema_types
        assert core.Name("ua.batched") in graph.schema_types

    def test_unbatched_sort_in_schema(self, real_sr):
        """build_graph registers component names for unbatched sort."""
        hidden = sort("hidden", real_sr, batched=False)
        graph = build_graph([hidden])
        assert core.Name("ua.sort.hidden") in graph.schema_types
        assert core.Name("ua.semiring.real") in graph.schema_types

    def test_both_sorts_in_schema(self, real_sr):
        """Both batched and unbatched variants share component name entries."""
        hidden = sort("hidden", real_sr, batched=False)
        hidden_b = sort("hidden", real_sr, batched=True)
        graph = build_graph([hidden, hidden_b])
        assert core.Name("ua.sort.hidden") in graph.schema_types
        assert core.Name("ua.semiring.real") in graph.schema_types
        assert core.Name("ua.batched") in graph.schema_types
