"""GNN architecture tests: message passing, layer composition, multi-hop propagation.

Graph Neural Networks expressed as semiring-parameterised morphisms.

Under the real semiring (⊕ = add, ⊗ = multiply), the equation "ij,j->i" computes:

    y_i = sum_j A_ij * x_j

which is exactly one step of neighbourhood aggregation (matrix-vector product
with the adjacency matrix). Chaining multiple steps gives multi-hop propagation.

Under the tropical semiring (⊕ = min, ⊗ = add), the same equation computes:

    y_i = min_j(A_ij + x_j)

which is one Bellman-Ford relaxation — multi-hop tropical GNN = Bellman-Ford.

Architecture coverage:
  - Single-step message passing (TestGNNMessagePassing)
  - Full GNN layer: aggregate → linear → relu (TestGNNLayer)
  - Two GNN layers chained (TestGNNTwoLayer)
  - K-hop propagation via repeated path steps (TestGNNKHops)
  - Tropical GNN: Bellman-Ford as message passing (TestGNNTropicalSemiring)
"""

import numpy as np
import pytest

from hydra.core import Name

from unialg import (
    numpy_backend, Semiring, Sort, tensor_coder,
    Equation, compile_program, PathSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()


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


@pytest.fixture
def coder():
    return tensor_coder()


# ---------------------------------------------------------------------------
# Helper: encode a numpy array as a Hydra term
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    from hydra.dsl.python import Right
    result = coder.decode(None, np.ascontiguousarray(arr, dtype=np.float64))
    assert isinstance(result, Right)
    return result.value


# ---------------------------------------------------------------------------
# Reference graph: 3-node undirected-ish graph
#
#   A = [[0, 1, 0],    node 0 has neighbour 1
#        [1, 0, 1],    node 1 has neighbours 0 and 2
#        [0, 1, 0]]    node 2 has neighbour 1
#
# A @ x gives each node the sum of its neighbours' features.
# ---------------------------------------------------------------------------

A3 = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# TestGNNMessagePassing
# ---------------------------------------------------------------------------

class TestGNNMessagePassing:
    """Single aggregation step: y = A @ x."""

    def test_output_matches_matmul(self, node_sort, real_sr, backend, coder):
        """GNN aggregation 'ij,j->i' with A produces A @ x."""
        eq = Equation("gnn_agg_basic", "ij,j->i", node_sort, node_sort, real_sr)
        x = np.array([1.0, 2.0, 3.0])

        a_enc = encode_array(coder, A3)
        prog = compile_program(
            [eq], backend=backend,
            specs=[PathSpec("gnn_agg_basic_path", ["gnn_agg_basic"],
                            node_sort, node_sort,
                            params={"gnn_agg_basic": [a_enc]})],
        )

        out = prog("gnn_agg_basic_path", x)
        expected = A3 @ x
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_entry_points_includes_equation(self, node_sort, real_sr, backend):
        """compile_program makes the equation accessible as an entry point."""
        eq = Equation("gnn_agg_ep", "ij,j->i", node_sort, node_sort, real_sr)
        prog = compile_program([eq], backend=backend)
        assert "gnn_agg_ep" in prog.entry_points()

    def test_different_features(self, node_sort, real_sr, backend, coder):
        """Aggregation is correct for different initial feature vectors."""
        eq = Equation("gnn_agg_feat", "ij,j->i", node_sort, node_sort, real_sr)
        x = np.array([0.5, -1.0, 2.0])

        a_enc = encode_array(coder, A3)
        prog = compile_program(
            [eq], backend=backend,
            specs=[PathSpec("gnn_agg_feat_path", ["gnn_agg_feat"],
                            node_sort, node_sort,
                            params={"gnn_agg_feat": [a_enc]})],
        )

        out = prog("gnn_agg_feat_path", x)
        np.testing.assert_allclose(out, A3 @ x, rtol=1e-6)

    def test_graph_has_primitive(self, node_sort, real_sr, backend):
        """The equation primitive appears in the compiled graph."""
        eq = Equation("gnn_agg_prim", "ij,j->i", node_sort, node_sort, real_sr)
        prog = compile_program([eq], backend=backend)
        assert Name("ua.equation.gnn_agg_prim") in prog.graph.primitives


# ---------------------------------------------------------------------------
# TestGNNLayer
# ---------------------------------------------------------------------------

class TestGNNLayer:
    """Full GNN layer: aggregate (A) → linear transform (W) → relu.

    4-node graph with raw adjacency A (4×4) and learnable weight matrix W (4×4).
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

    def build_layer_program(self, node_sort, real_sr, backend, coder):
        A_enc = encode_array(coder, self.A4)
        W_enc = encode_array(coder, self.W4)
        # No inputs= wiring between equations: PathSpec handles sequencing.
        # inputs= would require rank compatibility at the DAG slot level, which
        # breaks when a rank-1 output feeds slot 0 of a rank-2 einsum.
        eqs = [
            Equation("gnn_l_agg",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_l_lin",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_l_relu", None, node_sort, node_sort,
                     nonlinearity="relu"),
        ]
        return compile_program(
            eqs, backend=backend,
            specs=[PathSpec(
                "gnn_layer",
                ["gnn_l_agg", "gnn_l_lin", "gnn_l_relu"],
                node_sort, node_sort,
                params={"gnn_l_agg": [A_enc], "gnn_l_lin": [W_enc]},
            )],
        )

    def test_layer_output_correct(self, node_sort, real_sr, backend, coder):
        """GNN layer produces relu(W @ (A @ x))."""
        x = np.array([1.0, 0.0, -1.0, 0.5])
        prog = self.build_layer_program(node_sort, real_sr, backend, coder)
        out = prog("gnn_layer", x)
        expected = np.maximum(0.0, self.W4 @ (self.A4 @ x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_layer_entry_point_listed(self, node_sort, real_sr, backend, coder):
        """'gnn_layer' appears in entry_points() after compilation."""
        prog = self.build_layer_program(node_sort, real_sr, backend, coder)
        assert "gnn_layer" in prog.entry_points()

    def test_layer_bound_term_in_graph(self, node_sort, real_sr, backend, coder):
        """PathSpec creates a bound_term for the composed layer."""
        prog = self.build_layer_program(node_sort, real_sr, backend, coder)
        assert Name("ua.path.gnn_layer") in prog.graph.bound_terms

    def test_layer_relu_clips_negative(self, node_sort, real_sr, backend, coder):
        """ReLU zeros out negative post-aggregation values."""
        # Choose x so that W @ (A @ x) has at least one negative component.
        x = np.array([-2.0, -2.0, -2.0, -2.0])
        prog = self.build_layer_program(node_sort, real_sr, backend, coder)
        out = prog("gnn_layer", x)
        assert np.all(out >= 0.0), "ReLU should produce non-negative outputs"
        expected = np.maximum(0.0, self.W4 @ (self.A4 @ x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestGNNTwoLayer
# ---------------------------------------------------------------------------

class TestGNNTwoLayer:
    """Two GNN layers chained: agg1 → lin1 → relu1 → agg2 → lin2 → relu2.

    Both aggregations use the same adjacency A; each linear layer has its own W.
    Oracle: relu(W2 @ (A @ relu(W1 @ (A @ x))))
    """

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

    def build_two_layer_program(self, node_sort, real_sr, backend, coder):
        A_enc  = encode_array(coder, self.A4)
        W1_enc = encode_array(coder, self.W1)
        W2_enc = encode_array(coder, self.W2)
        # No inputs= wiring: PathSpec handles sequencing without rank-check
        # conflicts between agg (rank-1 output) and lin (rank-2 slot 0).
        eqs = [
            Equation("gnn_2l_agg1",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_2l_lin1",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_2l_relu1", None, node_sort, node_sort,
                     nonlinearity="relu"),
            Equation("gnn_2l_agg2",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_2l_lin2",  "ij,j->i", node_sort, node_sort, real_sr),
            Equation("gnn_2l_relu2", None, node_sort, node_sort,
                     nonlinearity="relu"),
        ]
        return compile_program(
            eqs, backend=backend,
            specs=[PathSpec(
                "gnn_two_layer",
                ["gnn_2l_agg1", "gnn_2l_lin1", "gnn_2l_relu1",
                 "gnn_2l_agg2", "gnn_2l_lin2", "gnn_2l_relu2"],
                node_sort, node_sort,
                params={
                    "gnn_2l_agg1": [A_enc],
                    "gnn_2l_lin1": [W1_enc],
                    "gnn_2l_agg2": [A_enc],
                    "gnn_2l_lin2": [W2_enc],
                },
            )],
        )

    def test_two_layer_output_correct(self, node_sort, real_sr, backend, coder):
        """Two-layer GNN produces relu(W2 @ (A @ relu(W1 @ (A @ x))))."""
        x = np.array([1.0, -0.5, 0.5, 2.0])
        prog = self.build_two_layer_program(node_sort, real_sr, backend, coder)
        out = prog("gnn_two_layer", x)
        expected = np.maximum(0.0, self.W2 @ (self.A4 @ np.maximum(0.0, self.W1 @ (self.A4 @ x))))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_two_layer_entry_point(self, node_sort, real_sr, backend, coder):
        """'gnn_two_layer' is a reachable entry point."""
        prog = self.build_two_layer_program(node_sort, real_sr, backend, coder)
        assert "gnn_two_layer" in prog.entry_points()

    def test_two_layer_all_primitives_present(self, node_sort, real_sr, backend, coder):
        """All six equation primitives appear in the compiled graph."""
        prog = self.build_two_layer_program(node_sort, real_sr, backend, coder)
        for name in ("gnn_2l_agg1", "gnn_2l_lin1", "gnn_2l_relu1",
                     "gnn_2l_agg2", "gnn_2l_lin2", "gnn_2l_relu2"):
            assert Name(f"ua.equation.{name}") in prog.graph.primitives

    def test_two_layer_output_nonneg(self, node_sort, real_sr, backend, coder):
        """Final relu guarantees non-negative output regardless of input."""
        x = np.array([-3.0, -1.0, -2.0, -0.5])
        prog = self.build_two_layer_program(node_sort, real_sr, backend, coder)
        out = prog("gnn_two_layer", x)
        assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# TestGNNKHops
# ---------------------------------------------------------------------------

class TestGNNKHops:
    """K-hop message propagation via a repeated aggregation PathSpec.

    Aggregation equation "ij,j->i" with A repeated K times gives A^K @ x
    (standard matrix power applied iteratively).
    """

    # 3-node undirected chain: 0 -- 1 -- 2
    A3_chain = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    def _build_khop_program(self, k, node_sort, real_sr, backend, coder):
        """Build a program that applies A k times in sequence via PathSpec."""
        A_enc = encode_array(coder, self.A3_chain)
        eq_names = [f"gnn_khop_agg_{i}" for i in range(k)]
        # No inputs= wiring: PathSpec sequences equations without rank checks
        # across the agg→agg boundary (rank-1 output to rank-2 slot 0).
        eqs = [Equation(name, "ij,j->i", node_sort, node_sort, real_sr)
               for name in eq_names]
        params = {name: [A_enc] for name in eq_names}
        return compile_program(
            eqs, backend=backend,
            specs=[PathSpec(f"gnn_khop_{k}", eq_names, node_sort, node_sort,
                            params=params)],
        )

    def test_k1_matches_single_hop(self, node_sort, real_sr, backend, coder):
        """1-hop propagation: y = A @ x."""
        x = np.array([1.0, 0.0, 0.0])
        prog = self._build_khop_program(1, node_sort, real_sr, backend, coder)
        out = prog("gnn_khop_1", x)
        expected = self.A3_chain @ x
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_k2_matches_two_hops(self, node_sort, real_sr, backend, coder):
        """2-hop propagation: y = A @ (A @ x) = A^2 @ x."""
        x = np.array([1.0, 0.0, 0.0])
        prog = self._build_khop_program(2, node_sort, real_sr, backend, coder)
        out = prog("gnn_khop_2", x)
        expected = self.A3_chain @ (self.A3_chain @ x)
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_k3_matches_three_hops(self, node_sort, real_sr, backend, coder):
        """3-hop propagation: y = A^3 @ x."""
        x = np.array([1.0, 0.0, 0.0])
        prog = self._build_khop_program(3, node_sort, real_sr, backend, coder)
        out = prog("gnn_khop_3", x)
        expected = self.A3_chain @ (self.A3_chain @ (self.A3_chain @ x))
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_khop_entry_points(self, node_sort, real_sr, backend, coder):
        """Each k-hop path registers as a distinct entry point."""
        prog2 = self._build_khop_program(2, node_sort, real_sr, backend, coder)
        assert "gnn_khop_2" in prog2.entry_points()

    def test_k2_different_start_node(self, node_sort, real_sr, backend, coder):
        """2-hop propagation from node 2 gives correct result."""
        x = np.array([0.0, 0.0, 1.0])
        prog = self._build_khop_program(2, node_sort, real_sr, backend, coder)
        out = prog("gnn_khop_2", x)
        expected = self.A3_chain @ (self.A3_chain @ x)
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestGNNTropicalSemiring
# ---------------------------------------------------------------------------

class TestGNNTropicalSemiring:
    """Tropical GNN: Bellman-Ford as semiring-parameterised message passing.

    Under (min, add), "ij,j->i" computes:
        y_i = min_j(W_ij + x_j)

    This is one Bellman-Ford relaxation. Chaining two steps gives 2-hop
    shortest distances.

    3-node DAG:
        node 0 --2--> node 1
        node 0 --5--> node 2
        node 1 --1--> node 2

    W[i, j] = cost of direct edge j -> i (inf if no edge, inf on diagonal).

    Starting from node 0 (x = [0, inf, inf]):
        After 1 hop: [inf, 2, 5]
        After 2 hops: [inf, inf, 3]   (0->1->2 costs 2+1=3, beats 0->2 at 5)
    """

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
        """One tropical aggregation step gives 1-hop shortest distances."""
        eq = Equation("gnn_trop_agg1", "ij,j->i",
                      node_sort_trop, node_sort_trop, tropical_sr)
        w_enc = encode_array(coder, self.W_trop)
        prog = compile_program(
            [eq], backend=backend,
            specs=[PathSpec("gnn_trop_1hop", ["gnn_trop_agg1"],
                            node_sort_trop, node_sort_trop,
                            params={"gnn_trop_agg1": [w_enc]})],
        )
        out = prog("gnn_trop_1hop", self.x0)
        np.testing.assert_allclose(out, self.EXPECTED_1HOP, rtol=1e-6)

    def test_two_hop_bellman_ford(self, node_sort_trop, tropical_sr, backend, coder):
        """Two tropical aggregation steps give 2-hop shortest distances."""
        w_enc = encode_array(coder, self.W_trop)
        # No inputs= wiring: PathSpec handles sequencing without rank checks.
        eqs = [
            Equation("gnn_trop_agg2a", "ij,j->i",
                     node_sort_trop, node_sort_trop, tropical_sr),
            Equation("gnn_trop_agg2b", "ij,j->i",
                     node_sort_trop, node_sort_trop, tropical_sr),
        ]
        prog = compile_program(
            eqs, backend=backend,
            specs=[PathSpec("gnn_trop_2hop",
                            ["gnn_trop_agg2a", "gnn_trop_agg2b"],
                            node_sort_trop, node_sort_trop,
                            params={"gnn_trop_agg2a": [w_enc],
                                    "gnn_trop_agg2b": [w_enc]})],
        )
        out = prog("gnn_trop_2hop", self.x0)
        np.testing.assert_allclose(out, self.EXPECTED_2HOP, rtol=1e-6)

    def test_tropical_entry_points(self, node_sort_trop, tropical_sr, backend, coder):
        """The 2-hop path entry point is reachable after compilation."""
        w_enc = encode_array(coder, self.W_trop)
        eqs = [
            Equation("gnn_trop_ep_agg1", "ij,j->i",
                     node_sort_trop, node_sort_trop, tropical_sr),
            Equation("gnn_trop_ep_agg2", "ij,j->i",
                     node_sort_trop, node_sort_trop, tropical_sr),
        ]
        prog = compile_program(
            eqs, backend=backend,
            specs=[PathSpec("gnn_trop_ep_2hop",
                            ["gnn_trop_ep_agg1", "gnn_trop_ep_agg2"],
                            node_sort_trop, node_sort_trop,
                            params={"gnn_trop_ep_agg1": [w_enc],
                                    "gnn_trop_ep_agg2": [w_enc]})],
        )
        eps = prog.entry_points()
        assert "gnn_trop_ep_2hop" in eps

    def test_tropical_vs_real_differ(self, node_sort_trop, tropical_sr,
                                     real_sr, backend, coder):
        """Tropical and real semirings produce different results on the same graph.

        Demonstrates semiring polymorphism: swapping ⊕/⊗ changes the semantics
        without changing the equation structure.
        """
        node_sort_real = Sort("node_gnn_trop_vs_real", real_sr)
        w_enc = encode_array(coder, self.W_trop)

        # Tropical equation
        eq_trop = Equation("gnn_tvr_trop", "ij,j->i",
                           node_sort_trop, node_sort_trop, tropical_sr)
        prog_trop = compile_program(
            [eq_trop], backend=backend,
            specs=[PathSpec("gnn_tvr_trop_path", ["gnn_tvr_trop"],
                            node_sort_trop, node_sort_trop,
                            params={"gnn_tvr_trop": [w_enc]})],
        )

        # Real equation (matrix-multiply semantics)
        x_real = np.array([1.0, 1.0, 1.0])
        eq_real = Equation("gnn_tvr_real", "ij,j->i",
                           node_sort_real, node_sort_real, real_sr)
        prog_real = compile_program(
            [eq_real], backend=backend,
            specs=[PathSpec("gnn_tvr_real_path", ["gnn_tvr_real"],
                            node_sort_real, node_sort_real,
                            params={"gnn_tvr_real": [w_enc]})],
        )

        out_trop = prog_trop("gnn_tvr_trop_path", self.x0)
        out_real = prog_real("gnn_tvr_real_path", x_real)

        # Tropical output: min-plus of W_trop with x0 — has inf values
        np.testing.assert_allclose(out_trop, self.EXPECTED_1HOP, rtol=1e-6)

        # Real output: W_trop @ ones — row sums (replacing inf with inf stays inf
        # but column with 2.0+5.0+1.0 sums normally). The key check is they differ.
        # We just verify shapes match and outputs are numerically distinct.
        assert not np.array_equal(out_trop, out_real), \
            "Tropical and real outputs should differ — semiring is not respected"
