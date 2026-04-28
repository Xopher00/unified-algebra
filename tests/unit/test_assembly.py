"""Unit tests for assembly: sort junction validation, DAG resolution, rank checking."""

import numpy as np
import pytest

from hydra.core import Name

from unialg import NumpyBackend, Semiring, Sort, Equation
from unialg.assembly.graph import topo_edges, validate_pipeline, assemble_graph


@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real):
    return Sort("hidden", real)


# ---------------------------------------------------------------------------
# Valid composition
# ---------------------------------------------------------------------------

class TestValidComposition:

    def test_single_equation_trivially_valid(self):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = Sort("hidden", real)
        eq = Equation("linear", "ij,j->i", hidden, hidden, real)
        validate_pipeline([eq])  # no error

    def test_three_equation_pipeline(self, backend):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = Sort("hidden", real)
        eqs = [
            Equation("linear1", "ij,j->i", hidden, hidden, real),
            Equation("relu", None, hidden, hidden, nonlinearity="relu", inputs=("linear1",)),
            Equation("linear2", "ij,j->i", hidden, hidden, real, inputs=("relu",)),
        ]
        validate_pipeline(eqs)  # no error
        graph, *_ = assemble_graph(eqs, backend)
        assert Name("ua.equation.linear1") in graph.primitives
        assert Name("ua.equation.relu") in graph.primitives
        assert Name("ua.equation.linear2") in graph.primitives

    def test_sorts_registered_in_schema(self, backend):
        real = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
        hidden = Sort("hidden", real)
        eqs = [Equation("linear", "ij,j->i", hidden, hidden, real)]
        graph, *_ = assemble_graph(eqs, backend)
        assert Name("ua.sort.hidden") in graph.schema_types
        assert Name("ua.semiring.real") in graph.schema_types


# ---------------------------------------------------------------------------
# DAG resolution
# ---------------------------------------------------------------------------

class TestDAGResolution:

    def test_linear_with_inputs(self, real, hidden):
        a = Equation("A", "ij,j->i", hidden, hidden, real)
        b = Equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        edges = topo_edges([a, b])
        assert len(edges) == 1
        assert edges[0][2] == 0  # slot 0

    def test_fan_out(self, real, hidden):
        a = Equation("A", "ij,j->i", hidden, hidden, real)
        b = Equation("B", "ij,j->i", hidden, hidden, real, inputs=("A",))
        c = Equation("C", "ij,j->i", hidden, hidden, real, inputs=("A",))
        edges = topo_edges([a, b, c])
        assert len(edges) == 2  # A→B and A→C

    def test_fan_in(self, real, hidden):
        a = Equation("A", "ij,j->i", hidden, hidden, real)
        b = Equation("B", "ij,j->i", hidden, hidden, real)
        c = Equation("C", "ij,jk->ik", hidden, hidden, real, inputs=("A", "B"))
        edges = topo_edges([a, b, c])
        assert len(edges) == 2  # A→C slot 0, B→C slot 1

    def test_diamond(self, real, hidden):
        a = Equation("A", "ij,j->i", hidden, hidden, real)
        b = Equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        c = Equation("C", None, hidden, hidden, nonlinearity="tanh", inputs=("A",))
        d = Equation("D", "ij,jk->ik", hidden, hidden, real, inputs=("B", "C"))
        edges = topo_edges([a, b, c, d])
        assert len(edges) == 4  # A→B, A→C, B→D, C→D

    def test_external_inputs_ignored(self, real, hidden):
        a = Equation("A", "ij,j->i", hidden, hidden, real, inputs=("X",))
        edges = topo_edges([a])
        assert len(edges) == 0  # X is external, no edge

    def test_cycle_raises(self, real, hidden):
        a = Equation("A", None, hidden, hidden, nonlinearity="relu", inputs=("B",))
        b = Equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        with pytest.raises(ValueError, match="Cycle"):
            topo_edges([a, b])


# ---------------------------------------------------------------------------
# Rank checking
# ---------------------------------------------------------------------------

class TestRankChecking:

    def test_matching_ranks_pass(self, real, hidden):
        a2 = Equation("A2", "ij,jk->ik", hidden, hidden, real)
        b2 = Equation("B2", "ij,jk->ik", hidden, hidden, real, inputs=("A2",))
        validate_pipeline([a2, b2])  # rank 2 → rank 2 at slot 0, should pass

    def test_rank_mismatch_raises(self, real, hidden):
        # A outputs rank 1 ("i"), B expects rank 2 at slot 0 ("ij")
        a = Equation("A", "ij,j->i", hidden, hidden, real)
        b = Equation("B", "ij,j->i", hidden, hidden, real, inputs=("A",))
        with pytest.raises(TypeError, match="Rank mismatch"):
            validate_pipeline([a, b])

    def test_pointwise_skips_rank(self, real, hidden):
        a = Equation("A", "ij,j->i", hidden, hidden, real)
        b = Equation("B", None, hidden, hidden, nonlinearity="relu", inputs=("A",))
        validate_pipeline([a, b])  # pointwise has no einsum, skip rank check

    def test_rank_match_pointwise_to_parametric(self, real, hidden):
        # Pointwise → parametric: pointwise has no rank, skip check
        a = Equation("A", None, hidden, hidden, nonlinearity="relu")
        b = Equation("B", "ij,j->i", hidden, hidden, real, inputs=("A",))
        validate_pipeline([a, b])  # should pass — can't check rank on pointwise
