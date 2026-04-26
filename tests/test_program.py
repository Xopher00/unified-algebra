"""Program tests: compile_program() and Program wrapper."""

import numpy as np
import pytest

from unialg import (
    compile_program, Program,
    Semiring, Sort, Equation, NumpyBackend, tensor_coder,
    PathSpec, FanSpec, FoldSpec,
)
from unialg.assembly.compositions import PathComposition, FanComposition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()

@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)

@pytest.fixture
def tropical_sr():
    return Semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)

@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)

@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


# ---------------------------------------------------------------------------
# Test 1: single equation callable
# ---------------------------------------------------------------------------

class TestSingleEquation:

    def test_single_linear_equation(self, hidden, real_sr, backend):
        """compile_program on a single equation; call by equation name."""
        eq = Equation("t1_linear", "ij,j->i", hidden, hidden, real_sr)
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

        eq = Equation("t1_relu", None, hidden, hidden, nonlinearity="relu")
        graph, *_ = assemble_graph([eq], backend)
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
        eq_lin = Equation("t2_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("t2_relu", None, hidden, hidden, nonlinearity="relu")

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
        eq = Equation("t2b_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)
        assert isinstance(prog, Program)


# ---------------------------------------------------------------------------
# Test 3: tropical semiring
# ---------------------------------------------------------------------------

class TestTropicalSemiring:

    def test_bellman_ford_one_hop(self, tropical_sr, backend, coder):
        """Tropical 'ij,j->i' computes min-plus: h_i = min_j(W_ij + x_j)."""
        trop_sort = Sort("t3_node", tropical_sr)
        eq = Equation("t3_sp", "ij,j->i", trop_sort, trop_sort, tropical_sr)

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

        eq_relu = Equation("t4_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("t4_tanh", None, hidden, hidden, nonlinearity="tanh")

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
        eq1 = Equation("t5_lin", "ij,j->i", hidden, hidden, real_sr)
        eq2 = Equation("t5_relu", None, hidden, hidden, nonlinearity="relu")
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
        eq = Equation("t5b_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)
        for ep in prog.entry_points():
            assert not ep.startswith("hydra.")


# ---------------------------------------------------------------------------
# Test 6: compiled fast path
# ---------------------------------------------------------------------------

class TestCompiledFastPath:

    def test_parametrised_path_in_compiled_fns(self, hidden, real_sr, backend, coder):
        """Parameterised paths (inline weight literals) must appear in compiled_fns,
        not fall back to reduce_term. This is the whole point of the compiler."""
        eq_lin = Equation("t6c_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("t6c_relu", None, hidden, hidden, nonlinearity="relu")

        W = np.array([[1.0, -1.0], [-1.0, 1.0]])
        w_enc = coder.decode(None, W).value

        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("t6c_net", ["t6c_linear", "t6c_relu"], hidden, hidden,
                            params={"t6c_linear": [w_enc]})],
        )
        assert "t6c_net" in prog._compiled_fns, (
            "parameterised path missing from compiled_fns — fell back to reduce_term"
        )

    def test_parametrised_path_correct_output(self, hidden, real_sr, backend, coder):
        """Compiled parameterised path produces the same result as the numpy oracle."""
        eq_lin = Equation("t6d_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("t6d_relu", None, hidden, hidden, nonlinearity="relu")

        W = np.array([[1.0, -1.0], [-1.0, 1.0]])
        x = np.array([3.0, 1.0])
        w_enc = coder.decode(None, W).value

        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("t6d_net", ["t6d_linear", "t6d_relu"], hidden, hidden,
                            params={"t6d_linear": [w_enc]})],
        )
        out = prog("t6d_net", x)
        np.testing.assert_allclose(out, np.maximum(0, W @ x), rtol=1e-6)

    def test_residual_path_in_compiled_fns(self, hidden, real_sr, backend):
        """Residual paths must be statically compiled — plus closure registered via _primitives."""
        eq_lin = Equation("t6e_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("t6e_relu", None, hidden, hidden, nonlinearity="relu")

        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("t6e_skip", ["t6e_relu"], hidden, hidden,
                            residual=True, residual_semiring="real")],
        )
        assert "t6e_skip" in prog._compiled_fns, (
            "residual path missing from compiled_fns"
        )

    def test_residual_path_correct_output(self, hidden, real_sr, backend):
        """Compiled residual path computes relu(x) + x."""
        eq_lin = Equation("t6h_linear", "ij,j->i", hidden, hidden, real_sr)
        eq_relu = Equation("t6h_relu", None, hidden, hidden, nonlinearity="relu")

        prog = compile_program(
            [eq_lin, eq_relu], backend=backend,
            specs=[PathSpec("t6h_skip", ["t6h_relu"], hidden, hidden,
                            residual=True, residual_semiring="real")],
        )
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = prog("t6h_skip", x)
        np.testing.assert_allclose(out, np.maximum(0, x) + x, rtol=1e-6)

    def test_jit_called_for_each_compiled_path(self, hidden, real_sr, coder):
        """backend.jit is applied to every compiled closure."""
        jit_calls = []
        def tracking_jit(fn):
            jit_calls.append(fn)
            return fn

        backend = NumpyBackend(jit=tracking_jit)

        eq_relu = Equation("t6f_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("t6f_tanh", None, hidden, hidden, nonlinearity="tanh")

        compile_program(
            [eq_relu, eq_tanh], backend=backend,
            specs=[PathSpec("t6f_path", ["t6f_relu", "t6f_tanh"], hidden, hidden)],
        )
        assert len(jit_calls) >= 1

    def test_jit_wrapped_path_correct_output(self, hidden, real_sr, coder):
        """A jit-wrapped path produces correct output when jit is an identity."""
        backend = NumpyBackend(jit=lambda fn: fn)

        eq_relu = Equation("t6g_relu", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program(
            [eq_relu], backend=backend,
            specs=[PathSpec("t6g_path", ["t6g_relu"], hidden, hidden)],
        )
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = prog("t6g_path", x)
        np.testing.assert_allclose(out, np.maximum(0, x), rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: error path
# ---------------------------------------------------------------------------

class TestErrorPath:

    def test_unknown_entry_point_raises_valueerror(self, hidden, real_sr, backend):
        """Invoking an unknown entry point raises ValueError naming the entry."""
        eq = Equation("t7_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)

        with pytest.raises(ValueError, match="nonexistent"):
            prog("nonexistent", np.array([1.0, 2.0]))

    def test_error_message_lists_available(self, hidden, real_sr, backend):
        """The ValueError message lists available entry points."""
        eq = Equation("t7b_eq", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq], backend=backend)

        with pytest.raises(ValueError, match="t7b_eq"):
            prog("wrong_name", np.array([1.0]))
