"""Message-passing architecture tests: aggregation, layer composition, multi-hop propagation.

Neighbourhood aggregation expressed as semiring-parameterised morphisms.

Under the real semiring (plus=add, times=multiply), the equation "ij,j->i" computes:

    y_i = sum_j A_ij * x_j

Under the tropical semiring (plus=minimum, times=add), the same equation computes:

    y_i = min_j(A_ij + x_j)

which is one Bellman-Ford relaxation.
"""

import numpy as np
import pytest

from hydra.core import Name

from unialg import NumpyBackend, Semiring, Sort, Equation, compile_program
from conftest import encode_array


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def real_sr():
    return Semiring("real_gnn", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def tropical_sr():
    return Semiring("tropical_gnn", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def node_sort(real_sr):
    return Sort("node_gnn", real_sr)


@pytest.fixture
def node_sort_trop(tropical_sr):
    return Sort("node_gnn_trop", tropical_sr)


A3 = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


class TestMessagePassing:
    """Single aggregation step: y = A @ x."""

    def test_output_matches_matmul(self, node_sort, real_sr, backend, coder):
        eq = Equation("gnn_agg_basic", "ij,j->i", node_sort, node_sort, real_sr)
        x = np.array([1.0, 2.0, 3.0])
        prog = compile_program([eq], backend=backend)
        out = prog("gnn_agg_basic", A3, x)
        np.testing.assert_allclose(out, A3 @ x, rtol=1e-6)

    def test_entry_points_includes_equation(self, node_sort, real_sr, backend):
        eq = Equation("gnn_agg_ep", "ij,j->i", node_sort, node_sort, real_sr)
        prog = compile_program([eq], backend=backend)
        assert "gnn_agg_ep" in prog.entry_points()

    def test_different_features(self, node_sort, real_sr, backend, coder):
        eq = Equation("gnn_agg_feat", "ij,j->i", node_sort, node_sort, real_sr)
        x = np.array([0.5, -1.0, 2.0])
        prog = compile_program([eq], backend=backend)
        out = prog("gnn_agg_feat", A3, x)
        np.testing.assert_allclose(out, A3 @ x, rtol=1e-6)

    def test_graph_has_primitive(self, node_sort, real_sr, backend):
        eq = Equation("gnn_agg_prim", "ij,j->i", node_sort, node_sort, real_sr)
        prog = compile_program([eq], backend=backend)
        assert Name("ua.equation.gnn_agg_prim") in prog.graph.primitives


class TestAggregateLayer:
    """Full aggregation layer: aggregate (A) -> linear (W) -> relu.

    Oracle: relu(W @ (A @ x))
    """

    A4 = np.array([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)

    W4 = np.array([
        [0.5, -0.5,  0.0,  0.5],
        [0.0,  1.0, -0.5,  0.0],
        [0.3,  0.0,  0.7, -0.3],
        [-0.2, 0.4,  0.0,  0.6],
    ], dtype=np.float64)

    def _build_eqs(self, node_sort, real_sr):
        return [
            Equation("gnn_l_agg",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_l_lin",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_l_relu", None, node_sort, node_sort, nonlinearity="relu"),
        ]

    def test_layer_output_correct(self, node_sort, real_sr, backend, coder):
        x = np.array([1.0, 0.0, -1.0, 0.5])
        prog = compile_program(self._build_eqs(node_sort, real_sr), backend=backend)
        h1 = prog("gnn_l_agg", self.A4, x)
        h2 = prog("gnn_l_lin", self.W4, h1)
        out = prog("gnn_l_relu", h2)
        expected = np.maximum(0.0, self.W4 @ (self.A4 @ x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_layer_entry_point_listed(self, node_sort, real_sr, backend, coder):
        prog = compile_program(self._build_eqs(node_sort, real_sr), backend=backend)
        assert "gnn_l_agg" in prog.entry_points()

    def test_layer_relu_clips_negative(self, node_sort, real_sr, backend, coder):
        x = np.array([-2.0, -2.0, -2.0, -2.0])
        prog = compile_program(self._build_eqs(node_sort, real_sr), backend=backend)
        h1 = prog("gnn_l_agg", self.A4, x)
        h2 = prog("gnn_l_lin", self.W4, h1)
        out = prog("gnn_l_relu", h2)
        assert np.all(out >= 0.0), "ReLU should produce non-negative outputs"
        expected = np.maximum(0.0, self.W4 @ (self.A4 @ x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)


class TestTwoLayerPipeline:
    """Two aggregation layers chained: agg->lin1->relu->agg->lin2->relu."""

    A4 = np.array([
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ], dtype=np.float64)

    W1 = np.array([
        [ 0.4, -0.2,  0.1,  0.3],
        [-0.1,  0.5,  0.2, -0.4],
        [ 0.2,  0.1, -0.3,  0.5],
        [ 0.3, -0.3,  0.4,  0.1],
    ], dtype=np.float64)

    W2 = np.array([
        [ 0.1,  0.3, -0.1,  0.2],
        [ 0.4,  0.0,  0.3, -0.2],
        [-0.3,  0.2,  0.5,  0.1],
        [ 0.2, -0.4,  0.0,  0.3],
    ], dtype=np.float64)

    def _build_eqs(self, node_sort, real_sr):
        return [
            Equation("gnn_2l_agg",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_2l_lin1", "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_2l_lin2", "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_2l_relu", None, node_sort, node_sort, nonlinearity="relu"),
        ]

    def test_two_layer_output_correct(self, node_sort, real_sr, backend, coder):
        x = np.array([1.0, -0.5, 0.5, 2.0])
        prog = compile_program(self._build_eqs(node_sort, real_sr), backend=backend)
        h1 = prog("gnn_2l_agg", self.A4, x)
        h2 = prog("gnn_2l_lin1", self.W1, h1)
        h3 = prog("gnn_2l_relu", h2)
        h4 = prog("gnn_2l_agg", self.A4, h3)
        h5 = prog("gnn_2l_lin2", self.W2, h4)
        out = prog("gnn_2l_relu", h5)
        expected = np.maximum(0.0, self.W2 @ (self.A4 @ np.maximum(0.0, self.W1 @ (self.A4 @ x))))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_two_layer_all_primitives_present(self, node_sort, real_sr, backend, coder):
        prog = compile_program(self._build_eqs(node_sort, real_sr), backend=backend)
        for name in ("gnn_2l_agg", "gnn_2l_lin1", "gnn_2l_lin2", "gnn_2l_relu"):
            assert Name(f"ua.equation.{name}") in prog.graph.primitives

    def test_two_layer_output_nonneg(self, node_sort, real_sr, backend, coder):
        x = np.array([-3.0, -1.0, -2.0, -0.5])
        prog = compile_program(self._build_eqs(node_sort, real_sr), backend=backend)
        h1 = prog("gnn_2l_agg", self.A4, x)
        h2 = prog("gnn_2l_lin1", self.W1, h1)
        h3 = prog("gnn_2l_relu", h2)
        h4 = prog("gnn_2l_agg", self.A4, h3)
        h5 = prog("gnn_2l_lin2", self.W2, h4)
        out = prog("gnn_2l_relu", h5)
        assert np.all(out >= 0.0)


class TestMultiHopPropagation:
    """K-hop message propagation via repeated equation calls."""

    A3_chain = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    def _build_prog(self, node_sort, real_sr, backend):
        eq = Equation("gnn_khop_agg", "ij,j->i", node_sort, node_sort, real_sr)
        return compile_program([eq], backend=backend)

    def test_k1_matches_single_hop(self, node_sort, real_sr, backend, coder):
        x = np.array([1.0, 0.0, 0.0])
        prog = self._build_prog(node_sort, real_sr, backend)
        out = prog("gnn_khop_agg", self.A3_chain, x)
        np.testing.assert_allclose(out, self.A3_chain @ x, rtol=1e-6)

    def test_k2_matches_two_hops(self, node_sort, real_sr, backend, coder):
        x = np.array([1.0, 0.0, 0.0])
        prog = self._build_prog(node_sort, real_sr, backend)
        h1 = prog("gnn_khop_agg", self.A3_chain, x)
        out = prog("gnn_khop_agg", self.A3_chain, h1)
        expected = self.A3_chain @ (self.A3_chain @ x)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_k3_matches_three_hops(self, node_sort, real_sr, backend, coder):
        x = np.array([1.0, 0.0, 0.0])
        prog = self._build_prog(node_sort, real_sr, backend)
        h1 = prog("gnn_khop_agg", self.A3_chain, x)
        h2 = prog("gnn_khop_agg", self.A3_chain, h1)
        out = prog("gnn_khop_agg", self.A3_chain, h2)
        expected = self.A3_chain @ (self.A3_chain @ (self.A3_chain @ x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_k2_different_start_node(self, node_sort, real_sr, backend, coder):
        x = np.array([0.0, 0.0, 1.0])
        prog = self._build_prog(node_sort, real_sr, backend)
        h1 = prog("gnn_khop_agg", self.A3_chain, x)
        out = prog("gnn_khop_agg", self.A3_chain, h1)
        expected = self.A3_chain @ (self.A3_chain @ x)
        np.testing.assert_allclose(out, expected, rtol=1e-6)


class TestTropicalSemiring:
    """Tropical message passing: Bellman-Ford as semiring-parameterised aggregation."""

    INF = float("inf")

    W_trop = np.array([
        [INF, INF, INF],
        [2.0, INF, INF],
        [5.0, 1.0, INF],
    ], dtype=np.float64)

    x0 = np.array([0.0, INF, INF], dtype=np.float64)

    EXPECTED_1HOP = np.array([INF, 2.0, 5.0], dtype=np.float64)
    EXPECTED_2HOP = np.array([INF, INF, 3.0], dtype=np.float64)

    def test_single_hop_bellman_ford(self, node_sort_trop, tropical_sr, backend, coder):
        eq = Equation("gnn_trop_agg1", "ij,j->i",
                      node_sort_trop, node_sort_trop, tropical_sr)
        prog = compile_program([eq], backend=backend)
        out = prog("gnn_trop_agg1", self.W_trop, self.x0)
        np.testing.assert_allclose(out, self.EXPECTED_1HOP, rtol=1e-6)

    def test_two_hop_bellman_ford(self, node_sort_trop, tropical_sr, backend, coder):
        eq = Equation("gnn_trop_agg2", "ij,j->i",
                      node_sort_trop, node_sort_trop, tropical_sr)
        prog = compile_program([eq], backend=backend)
        h1 = prog("gnn_trop_agg2", self.W_trop, self.x0)
        out = prog("gnn_trop_agg2", self.W_trop, h1)
        np.testing.assert_allclose(out, self.EXPECTED_2HOP, rtol=1e-6)

    def test_tropical_entry_points(self, node_sort_trop, tropical_sr, backend, coder):
        eq = Equation("gnn_trop_ep", "ij,j->i",
                      node_sort_trop, node_sort_trop, tropical_sr)
        prog = compile_program([eq], backend=backend)
        assert "gnn_trop_ep" in prog.entry_points()

    def test_tropical_vs_real_differ(self, node_sort_trop, tropical_sr,
                                     real_sr, backend, coder):
        """Tropical and real semirings produce different results on the same graph."""
        node_sort_real = Sort("node_gnn_trop_vs_real", real_sr)
        eq_trop = Equation("gnn_tvr_trop", "ij,j->i",
                           node_sort_trop, node_sort_trop, tropical_sr)
        eq_real = Equation("gnn_tvr_real", "ij,j->i",
                           node_sort_real, node_sort_real, real_sr)
        prog_trop = compile_program([eq_trop], backend=backend)
        prog_real = compile_program([eq_real], backend=backend)
        x_real = np.array([1.0, 1.0, 1.0])
        out_trop = prog_trop("gnn_tvr_trop", self.W_trop, self.x0)
        out_real = prog_real("gnn_tvr_real", self.W_trop, x_real)
        np.testing.assert_allclose(out_trop, self.EXPECTED_1HOP, rtol=1e-6)
        assert not np.array_equal(out_trop, out_real), \
            "Tropical and real outputs should differ"
