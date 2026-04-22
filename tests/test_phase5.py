"""Phase 5 tests: DAG-based equation wiring and rank checking."""

import numpy as np
import pytest

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import (
    numpy_backend, semiring, sort, tensor_coder,
    equation, topo_edges, validate_pipeline, assemble_graph,
)


@pytest.fixture
def backend():
    return numpy_backend()

@pytest.fixture
def coder():
    return tensor_coder()

@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))

@pytest.fixture
def real():
    return semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)

@pytest.fixture
def hidden(real):
    return sort("hidden", real)


def enc(coder, arr):
    return coder.decode(None, arr).value

def dec(coder, term):
    return coder.encode(None, None, term).value


# ---------------------------------------------------------------------------
# DAG resolution
# ---------------------------------------------------------------------------

class TestDAGResolution:

    def test_linear_with_inputs(self, real, hidden):
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        edges = topo_edges([a, b])
        assert len(edges) == 1
        assert edges[0][2] == 0  # slot 0

    def test_fan_out(self, real, hidden):
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", "ij,j->i", hidden, hidden, real, inputs=("A",))
        c = equation("C", "ij,j->i", hidden, hidden, real, inputs=("A",))
        edges = topo_edges([a, b, c])
        assert len(edges) == 2  # A→B and A→C

    def test_fan_in(self, real, hidden):
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", "ij,j->i", hidden, hidden, real)
        c = equation("C", "ij,jk->ik", hidden, hidden, real, inputs=("A", "B"))
        edges = topo_edges([a, b, c])
        assert len(edges) == 2  # A→C slot 0, B→C slot 1

    def test_diamond(self, real, hidden):
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        c = equation("C", None, hidden, hidden, nonlinearity="tanh", inputs=("A",))
        d = equation("D", "ij,jk->ik", hidden, hidden, real, inputs=("B", "C"))
        edges = topo_edges([a, b, c, d])
        assert len(edges) == 4  # A→B, A→C, B→D, C→D

    def test_external_inputs_ignored(self, real, hidden):
        a = equation("A", "ij,j->i", hidden, hidden, real, inputs=("X",))
        edges = topo_edges([a])
        assert len(edges) == 0  # X is external, no edge

    def test_cycle_raises(self, real, hidden):
        a = equation("A", None, hidden, hidden, nonlinearity="relu", inputs=("B",))
        b = equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        with pytest.raises(ValueError, match="Cycle"):
            topo_edges([a, b])


# ---------------------------------------------------------------------------
# Rank checking
# ---------------------------------------------------------------------------

class TestRankChecking:

    def test_matching_ranks_pass(self, real, hidden):
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", "ij,j->i", hidden, hidden, real, inputs=("A",))
        # A outputs rank 1, B slot 0 expects rank 2 (the W), slot 1 expects rank 1
        # But A feeds into slot 0 of B — need A output rank == B input rank at slot 0
        # Actually: A output is rank 1 ("i"), B expects "ij" at slot 0 (rank 2)
        # This SHOULD fail — let's test the mismatch case separately
        # For a match: A outputs rank 2, B expects rank 2 at slot 0
        a2 = equation("A2", "ij,jk->ik", hidden, hidden, real)
        b2 = equation("B2", "ij,jk->ik", hidden, hidden, real, inputs=("A2",))
        validate_pipeline([a2, b2])  # rank 2 → rank 2 at slot 0, should pass

    def test_rank_mismatch_raises(self, real, hidden):
        # A outputs rank 1 ("i"), B expects rank 2 at slot 0 ("ij")
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", "ij,j->i", hidden, hidden, real, inputs=("A",))
        with pytest.raises(TypeError, match="Rank mismatch"):
            validate_pipeline([a, b])

    def test_pointwise_skips_rank(self, real, hidden):
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        validate_pipeline([a, b])  # pointwise has no einsum, skip rank check

    def test_rank_match_pointwise_to_parametric(self, real, hidden):
        # Pointwise → parametric: pointwise has no rank, skip check
        a = equation("A", None, hidden, hidden, nonlinearity="relu")
        b = equation("B", "ij,j->i", hidden, hidden, real, inputs=("A",))
        validate_pipeline([a, b])  # should pass — can't check rank on pointwise


# ---------------------------------------------------------------------------
# DAG assembly and execution
# ---------------------------------------------------------------------------

class TestDAGAssemble:

    def test_fan_out_assembled(self, real, hidden, backend):
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        c = equation("C", None, hidden, hidden, nonlinearity="sigmoid", inputs=("A",))
        graph = assemble_graph([a, b, c], backend)
        assert Name("ua.equation.A") in graph.primitives
        assert Name("ua.equation.B") in graph.primitives
        assert Name("ua.equation.C") in graph.primitives

    def test_fan_out_executes(self, real, hidden, backend, cx, coder):
        """Two equations reading the same input — fan-out."""
        a = equation("linear", "ij,j->i", hidden, hidden, real)
        b = equation("relu_out", None, hidden, hidden, nonlinearity="relu", inputs=("linear",))
        c = equation("sig_out", None, hidden, hidden, nonlinearity="sigmoid", inputs=("linear",))
        graph = assemble_graph([a, b, c], backend)

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
        a = equation("A", "ij,j->i", hidden, hidden, real)
        b = equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        c = equation("C", None, hidden, hidden, nonlinearity="tanh", inputs=("A",))
        # D is pointwise — it will just pass through one of them
        # For a real diamond merge we'd need a multi-input equation
        # For now just verify the DAG validates and both branches execute
        graph = assemble_graph([a, b, c], backend)

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
