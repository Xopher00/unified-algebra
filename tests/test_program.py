"""Program tests: compile_program() and Program wrapper."""

import numpy as np
import pytest

from unialg import (
    compile_program, Program,
    semiring, sort, equation, numpy_backend, tensor_coder,
    path, fan,
    PathSpec, FanSpec, FoldSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()

@pytest.fixture
def real_sr():
    return semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)

@pytest.fixture
def tropical_sr():
    return semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)

@pytest.fixture
def hidden(real_sr):
    return sort("hidden", real_sr)

@pytest.fixture
def coder():
    return tensor_coder()


# ---------------------------------------------------------------------------
# Test 1: single equation callable
# ---------------------------------------------------------------------------

class TestSingleEquation:

    def test_single_linear_equation(self, hidden, real_sr, backend):
        """compile_program on a single equation; call by equation name."""
        eq = equation("t1_linear", "ij,j->i", hidden, hidden, real_sr)
        prog = compile_program([eq], backend=backend)

        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([1.0, 0.0])
        out = prog("t1_linear", W, x)
        np.testing.assert_allclose(out, W @ x, rtol=1e-6)

    def test_single_equation_result_matches_direct_reduce(self, hidden, real_sr, backend, coder):
        """Program output matches direct reduce_term output for the same equation."""
        from hydra.context import Context
        from hydra.dsl.python import FrozenDict, Right
        from hydra.dsl.terms import apply, var
        from hydra.reduction import reduce_term
        from unialg import assemble_graph

        eq = equation("t1_relu", None, hidden, hidden, nonlinearity="relu")
        graph = assemble_graph([eq], backend)
        cx = Context(trace=(), messages=(), other=FrozenDict({}))

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        x_enc = coder.decode(None, x).value

        direct = reduce_term(cx, graph, True, apply(var("ua.equation.t1_relu"), x_enc))
        assert isinstance(direct, Right)
        expected = coder.encode(None, None, direct.value).value

        prog = compile_program([eq], backend=backend)
        out = prog("t1_relu", x)
        np.testing.assert_allclose(out, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: path composition roundtrip
# ---------------------------------------------------------------------------

class TestPathRoundtrip:

    def test_two_step_path(self, hidden, real_sr, backend, coder):
        """Path: linear → relu. compile_program result matches numpy oracle."""
        eq_lin = equation("t2_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = equation("t2_relu", None, hidden, hidden, nonlinearity="relu")

        W = np.array([[1.0, -1.0], [-1.0, 1.0]])
        x = np.array([2.0, 1.0])
        w_enc = coder.decode(None, W).value

        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("t2_net", ["t2_linear", "t2_relu"], hidden, hidden,
                            params={"t2_linear": [w_enc]})],
        )
        out = prog("t2_net", x)
        np.testing.assert_allclose(out, np.maximum(0, W @ x), rtol=1e-6)

    def test_program_is_program_instance(self, hidden, real_sr, backend):
        """compile_program returns a Program."""
        eq = equation("t2b_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)
        assert isinstance(prog, Program)


# ---------------------------------------------------------------------------
# Test 3: tropical semiring
# ---------------------------------------------------------------------------

class TestTropicalSemiring:

    def test_bellman_ford_one_hop(self, tropical_sr, backend, coder):
        """Tropical 'ij,j->i' computes min-plus: h_i = min_j(W_ij + x_j)."""
        trop_sort = sort("t3_node", tropical_sr)
        eq = equation("t3_sp", "ij,j->i", trop_sort, trop_sort, tropical_sr)

        # W[i,j] = cost to reach node i from node j (incoming-edge convention)
        # 3-node DAG: 0→1 cost 2, 0→2 cost 5, 1→2 cost 1
        inf = float("inf")
        W = np.array([[inf, inf, inf],
                      [2.0, inf, inf],
                      [5.0, 1.0, inf]], dtype=float)
        x = np.array([0.0, inf, inf], dtype=float)  # source at node 0

        prog = compile_program([eq], backend=backend)
        out = prog("t3_sp", W, x)

        # After 1 hop: node0=inf, node1=2.0, node2=5.0
        np.testing.assert_allclose(out, np.array([inf, 2.0, 5.0]), rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: rebind returns a fresh Program
# ---------------------------------------------------------------------------

class TestRebind:

    def test_rebind_returns_new_program(self, hidden, real_sr, backend, coder):
        """rebind() returns a new Program; original is not mutated."""
        import hydra.core as core
        from hydra.dsl.terms import var

        eq_relu = equation("t4_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = equation("t4_tanh", None, hidden, hidden, nonlinearity="tanh")

        # hyperparams creates a ua.param.dummy bound_term (not used in path,
        # just verifying the rebind machinery round-trips)
        dummy_val = core.TermLiteral(value=core.LiteralFloat(value=1.0))
        prog = compile_program(
            [eq_relu, eq_tanh], backend=backend,
            specs=[PathSpec("t4_path", ["t4_relu", "t4_tanh"], hidden, hidden)],
            hyperparams={"dummy": dummy_val},
        )

        new_val = core.TermLiteral(value=core.LiteralFloat(value=2.0))
        prog2 = prog.rebind(dummy=new_val)

        # rebind returns a new Program instance
        assert isinstance(prog2, Program)
        assert prog2 is not prog

        # Both programs still produce correct output (path is unaffected by dummy rebind)
        x = np.array([-1.0, 0.5, 2.0])
        expected = np.tanh(np.maximum(0.0, x))
        np.testing.assert_allclose(prog("t4_path", x), expected, rtol=1e-6)
        np.testing.assert_allclose(prog2("t4_path", x), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: entry_points enumeration
# ---------------------------------------------------------------------------

class TestEntryPoints:

    def test_entry_points_include_path_and_equations(self, hidden, real_sr, backend, coder):
        """entry_points lists path + equations; not internal ua.prim names."""
        eq1 = equation("t5_lin", "ij,j->i", hidden, hidden, real_sr)
        eq2 = equation("t5_relu", None, hidden, hidden, nonlinearity="relu")
        W = coder.decode(None, np.eye(2)).value

        prog = compile_program(
            [eq1, eq2], backend=backend,
            specs=[PathSpec("t5_path", ["t5_lin", "t5_relu"], hidden, hidden,
                            params={"t5_lin": [W]})],
        )
        eps = prog.entry_points()
        assert "t5_path" in eps
        assert "t5_lin" in eps
        assert "t5_relu" in eps

    def test_entry_points_excludes_stdlib_primitives(self, hidden, real_sr, backend):
        """entry_points does not expose Hydra stdlib names."""
        eq = equation("t5b_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)
        for ep in prog.entry_points():
            assert not ep.startswith("hydra.")


# ---------------------------------------------------------------------------
# Test 6: error path
# ---------------------------------------------------------------------------

class TestErrorPath:

    def test_unknown_entry_point_raises_valueerror(self, hidden, real_sr, backend):
        """Invoking an unknown entry point raises ValueError naming the entry."""
        eq = equation("t6_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)

        with pytest.raises(ValueError, match="nonexistent"):
            prog("nonexistent", np.array([1.0, 2.0]))

    def test_error_message_lists_available(self, hidden, real_sr, backend):
        """The ValueError message lists available entry points."""
        eq = equation("t6b_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)

        with pytest.raises(ValueError, match="t6b_eq"):
            prog("wrong_name", np.array([1.0]))
