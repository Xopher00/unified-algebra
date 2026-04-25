"""Shortest path tests: shortest-path computation via the tropical semiring.

Key insight:  Under the tropical semiring (⊕ = min, ⊗ = add), the equation
"ij,j->i" computes:

    h_i = min_j(W_ij + x_j)

This is exactly one Bellman-Ford relaxation step: W_ij is the edge weight from
node j to node i, and x_j is the current best distance to node j.  Chaining
two such steps gives distances reachable via two hops.

The tests verify:
  1. A single tropical equation produces the correct one-hop distances.
  2. Two tropical equations chained via PathSpec produce correct two-hop
     distances.
  3. Swapping to the real semiring with the identical equation structure gives
     ordinary matrix multiplication — demonstrating semiring polymorphism.
  4. Nodes unreachable from the source keep infinite distance throughout.
"""

import numpy as np
import pytest

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import (
    NumpyBackend, Semiring, Sort, tensor_coder,
    Equation, path,
    assemble_graph, build_graph, PathSpec,
    resolve_equation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def tropical_sr():
    return Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def node_sort(tropical_sr):
    """A sort for node distance vectors under the tropical semiring."""
    return Sort("node", tropical_sr)


@pytest.fixture
def node_sort_real(real_sr):
    """The same 'node' sort name but under the real semiring."""
    return Sort("node_real", real_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


# ---------------------------------------------------------------------------
# Helper: encode / decode arrays via the tensor coder
# ---------------------------------------------------------------------------

def enc(coder, arr):
    result = coder.decode(None, arr)
    assert isinstance(result, Right), f"encode failed: {result}"
    return result.value


def dec(coder, term):
    result = coder.encode(None, None, term)
    assert isinstance(result, Right), f"decode failed: {result}"
    return result.value


def assert_reduce_ok(cx, graph, term):
    result = reduce_term(cx, graph, True, term)
    assert isinstance(result, Right), f"reduce_term returned Left: {result}"
    return result.value


# ---------------------------------------------------------------------------
# Reference graph used in most tests
#
# 3-node graph:          W[i, j] = cost of the direct edge  j -> i
#
#   node 0  --2-->  node 1
#   node 0  --5-->  node 2
#   node 1  --1-->  node 2
#   (no edge from node 2 anywhere; self-loops via infinity)
#
# Edge weight matrix (row = destination, col = source):
#
#          src 0   src 1   src 2
#   dst 0  [inf,   inf,    inf  ]   (nobody reaches node 0)
#   dst 1  [2.0,   inf,    inf  ]   (only node 0 reaches node 1 at cost 2)
#   dst 2  [5.0,   1.0,    inf  ]   (node 0 at cost 5, node 1 at cost 1)
#
# Starting from node 0 with x = [0, inf, inf]:
#   After 1 hop:  h_i = min_j(W[i,j] + x[j])
#     h_0 = min(inf+0, inf+inf, inf+inf) = inf
#     h_1 = min(2+0,   inf+inf, inf+inf) = 2
#     h_2 = min(5+0,   1+inf,   inf+inf) = 5
#
#   After 2 hops (apply the same step to h):
#     h2_0 = min(inf+inf, inf+2,  inf+5)  = inf
#     h2_1 = min(2+inf,   inf+2,  inf+5)  = inf   (no edge brings us to 1 from {1,2})
#     h2_2 = min(5+inf,   1+2,    inf+5)  = 3     (node 1 -> node 2, total cost 3)
# ---------------------------------------------------------------------------

INF = float("inf")

W = np.array([
    [INF, INF, INF],
    [2.0, INF, INF],
    [5.0, 1.0, INF],
], dtype=float)

x0 = np.array([0.0, INF, INF], dtype=float)   # start at node 0

EXPECTED_1HOP = np.array([INF, 2.0, 5.0], dtype=float)
EXPECTED_2HOP = np.array([INF, INF, 3.0], dtype=float)


# ---------------------------------------------------------------------------
# Manual min-plus reference (numpy, no DSL)
# ---------------------------------------------------------------------------

def minplus_matvec(W, x):
    """Compute min_j(W[i,j] + x[j]) for each i — one Bellman-Ford step."""
    return np.min(W + x[np.newaxis, :], axis=1)


class TestShortestPath:

    # -----------------------------------------------------------------------
    # Test 1: single tropical equation == one Bellman-Ford step
    # -----------------------------------------------------------------------

    def test_tropical_equation_is_bellman_ford_step(self, tropical_sr, node_sort, backend, cx, coder):
        """A single "ij,j->i" equation under the tropical semiring computes
        min_j(W[i,j] + x[j]) — one hop of Bellman-Ford — and matches the
        numpy reference exactly.
        """
        eq = Equation("hop1", "ij,j->i", node_sort, node_sort, tropical_sr)
        graph, _ = assemble_graph([eq], backend)

        W_enc = enc(coder, W)
        x_enc = enc(coder, x0)

        result_term = assert_reduce_ok(
            cx, graph,
            apply(apply(var("ua.equation.hop1"), W_enc), x_enc),
        )
        result = dec(coder, result_term)

        expected = minplus_matvec(W, x0)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, EXPECTED_1HOP)

    # -----------------------------------------------------------------------
    # Test 2: two tropical equations chained via PathSpec == two hops
    # -----------------------------------------------------------------------

    def test_two_hop_path(self, tropical_sr, node_sort, backend, cx, coder):
        """Two tropical equations wired sequentially via PathSpec compute
        distances reachable in exactly two edge traversals.

        PathSpec wires: hop2(hop1(x)) — each step is one Bellman-Ford
        relaxation, so the composition covers paths of length 2.
        """
        eq1 = Equation("sp1", "ij,j->i", node_sort, node_sort, tropical_sr)
        eq2 = Equation("sp2", "ij,j->i", node_sort, node_sort, tropical_sr)

        graph, _ = assemble_graph(
            [eq1, eq2], backend,
            specs=[PathSpec("two_hop", ["sp1", "sp2"], node_sort, node_sort)],
        )

        W_enc = enc(coder, W)
        x_enc = enc(coder, x0)

        # Pre-bind W into each hop so the path only takes x as input.
        p_name, p_term = path(
            "two_hop_bound",
            ["sp1", "sp2"],
            params={"sp1": [W_enc], "sp2": [W_enc]},
        )

        prim_sp1, _ = resolve_equation(eq1, backend)
        prim_sp2, _ = resolve_equation(eq2, backend)
        graph2 = build_graph(
            [],
            primitives={
                Name("ua.equation.sp1"): prim_sp1,
                Name("ua.equation.sp2"): prim_sp2,
            },
            bound_terms={p_name: p_term},
        )

        result_term = assert_reduce_ok(
            cx, graph2,
            apply(var("ua.path.two_hop_bound"), x_enc),
        )
        result = dec(coder, result_term)

        expected = minplus_matvec(W, minplus_matvec(W, x0))
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, EXPECTED_2HOP)

    # -----------------------------------------------------------------------
    # Test 3: same architecture, real semiring == standard matrix multiply
    # -----------------------------------------------------------------------

    def test_real_vs_tropical_same_structure(self, real_sr, node_sort_real, backend, cx, coder):
        """The equation "ij,j->i" under the real semiring is ordinary matrix-
        vector multiplication.  Wiring two such equations (PathSpec) is
        equivalent to applying the matrix twice: W @ (W @ x).

        This test uses a concrete weight matrix with finite values and a
        simple non-negative starting vector so the real/tropical results
        are clearly different, demonstrating semiring polymorphism.
        """
        W_real = np.array([
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.5],
        ], dtype=float)
        x_real = np.array([1.0, 0.0, 0.0], dtype=float)

        eq1 = Equation("real1", "ij,j->i", node_sort_real, node_sort_real, real_sr)
        eq2 = Equation("real2", "ij,j->i", node_sort_real, node_sort_real, real_sr)

        W_enc = enc(coder, W_real)
        x_enc = enc(coder, x_real)

        p_name, p_term = path(
            "real_two_hop",
            ["real1", "real2"],
            params={"real1": [W_enc], "real2": [W_enc]},
        )

        prim_real1, _ = resolve_equation(eq1, backend)
        prim_real2, _ = resolve_equation(eq2, backend)
        graph = build_graph(
            [],
            primitives={
                Name("ua.equation.real1"): prim_real1,
                Name("ua.equation.real2"): prim_real2,
            },
            bound_terms={p_name: p_term},
        )

        result_term = assert_reduce_ok(
            cx, graph,
            apply(var("ua.path.real_two_hop"), x_enc),
        )
        result = dec(coder, result_term)

        expected = W_real @ (W_real @ x_real)
        np.testing.assert_allclose(result, expected)

        # Confirm real and tropical give structurally different answers on the
        # same shaped inputs (semiring swap changes semantics, not structure).
        tropical_result = minplus_matvec(W, minplus_matvec(W, x0))
        # The real result has no infinities; the tropical result does.
        assert not np.any(np.isinf(result)), "real semiring result should be finite"
        assert np.any(np.isinf(tropical_result)), "tropical result should contain inf"

    # -----------------------------------------------------------------------
    # Test 4: unreachable nodes keep inf throughout
    # -----------------------------------------------------------------------

    def test_unreachable_nodes_stay_inf(self, tropical_sr, node_sort, backend, cx, coder):
        """Node 0 is unreachable from any other node in our graph (no incoming
        edges).  After any number of Bellman-Ford steps starting from node 0,
        the distance to node 0 should remain infinite.

        We verify both after one hop and after two hops.
        """
        eq = Equation("bfstep", "ij,j->i", node_sort, node_sort, tropical_sr)
        prim, _ = resolve_equation(eq, backend)

        W_enc = enc(coder, W)
        x_enc = enc(coder, x0)

        # One hop
        h1_term = assert_reduce_ok(
            cx,
            build_graph([], primitives={prim.name: prim}, bound_terms={}),
            apply(apply(var("ua.equation.bfstep"), W_enc), x_enc),
        )
        h1 = dec(coder, h1_term)

        assert np.isinf(h1[0]), (
            f"Node 0 should remain unreachable after 1 hop, got distance {h1[0]}"
        )

        # Two hops: feed h1 back as the distance vector
        h1_enc = enc(coder, h1)
        h2_term = assert_reduce_ok(
            cx,
            build_graph([], primitives={prim.name: prim}, bound_terms={}),
            apply(apply(var("ua.equation.bfstep"), W_enc), h1_enc),
        )
        h2 = dec(coder, h2_term)

        assert np.isinf(h2[0]), (
            f"Node 0 should remain unreachable after 2 hops, got distance {h2[0]}"
        )

        # Node 1 also becomes unreachable after the second hop (no back-edge).
        assert np.isinf(h2[1]), (
            f"Node 1 should be unreachable in 2 hops from node 0 via this graph, "
            f"got distance {h2[1]}"
        )

        # Node 2 should be reachable in 2 hops (0->1->2 at cost 3).
        assert np.isfinite(h2[2]), (
            f"Node 2 should be reachable in 2 hops, got distance {h2[2]}"
        )
        np.testing.assert_allclose(h2[2], 3.0)
