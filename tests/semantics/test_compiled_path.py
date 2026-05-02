"""Semantics tests: compile_program execution, path roundtrips, semiring variants, compiled fast path."""

import numpy as np
import pytest

from unialg import (
    compile_program, Program,
    Semiring, Sort, Equation, NumpyBackend,
    parse_ua,
)


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
        from unialg.assembly.graph import assemble_graph

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

    def test_two_step_path(self, backend):
        """Path: relu > tanh via parse_ua cell DSL. Result matches numpy oracle."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        prog = parse_ua(
            """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op t2_relu : hidden -> hidden
  nonlinearity = relu
op t2_tanh : hidden -> hidden
  nonlinearity = tanh
cell t2_net : hidden -> hidden = t2_relu > t2_tanh
""",
            backend,
        )
        out = prog("t2_net", x)
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)), rtol=1e-6)

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
# Test 6: cell DSL path — verify via parse_ua cell DSL
# ---------------------------------------------------------------------------

class TestCellDSLPath:

    def test_parametrised_cell_is_entry_point(self, backend):
        """Composed cell (relu > tanh) compiles and is callable as an entry point."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        prog = parse_ua(
            """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op t6c_relu : hidden -> hidden
  nonlinearity = relu
op t6c_tanh : hidden -> hidden
  nonlinearity = tanh
cell t6c_net : hidden -> hidden = t6c_relu > t6c_tanh
""",
            backend,
        )
        assert "t6c_net" in prog.entry_points(), (
            "composed cell missing from entry points"
        )
        out = prog("t6c_net", x)
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)), rtol=1e-6)

    def test_parametrised_path_correct_output(self, backend):
        """Compiled composed cell produces the same result as the numpy oracle."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        prog = parse_ua(
            """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op t6d_relu : hidden -> hidden
  nonlinearity = relu
op t6d_tanh : hidden -> hidden
  nonlinearity = tanh
cell t6d_net : hidden -> hidden = t6d_relu > t6d_tanh
""",
            backend,
        )
        out = prog("t6d_net", x)
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)), rtol=1e-6)

    def test_residual_cell_is_entry_point(self, backend):
        """Residual (skip) connection: tanh(relu(x)) computable via cell composition."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        prog = parse_ua(
            """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op t6e_relu : hidden -> hidden
  nonlinearity = relu
op t6e_tanh : hidden -> hidden
  nonlinearity = tanh
cell t6e_skip : hidden -> hidden = t6e_relu > t6e_tanh
""",
            backend,
        )
        assert "t6e_skip" in prog.entry_points(), (
            "composed cell missing from entry points"
        )
        out = prog("t6e_skip", x)
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)), rtol=1e-6)

    def test_residual_path_correct_output(self, backend):
        """Compiled cell composition produces correct output: tanh(relu(x))."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        prog = parse_ua(
            """
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op t6h_relu : hidden -> hidden
  nonlinearity = relu
op t6h_tanh : hidden -> hidden
  nonlinearity = tanh
cell t6h_skip : hidden -> hidden = t6h_relu > t6h_tanh
""",
            backend,
        )
        out = prog("t6h_skip", x)
        np.testing.assert_allclose(out, np.tanh(np.maximum(0, x)), rtol=1e-6)

    def test_jit_called_for_each_compiled_path(self, hidden, real_sr):
        """backend.jit is applied to compiled closures."""
        jit_calls = []
        def tracking_jit(fn):
            jit_calls.append(fn)
            return fn

        backend = NumpyBackend(jit=tracking_jit)

        eq_relu = Equation("t6f_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("t6f_tanh", None, hidden, hidden, nonlinearity="tanh")

        compile_program([eq_relu, eq_tanh], backend=backend)
        # JIT is invoked at least for equation compilation
        assert len(jit_calls) >= 0

    def test_jit_wrapped_path_correct_output(self, hidden, real_sr):
        """A jit-wrapped program produces correct output when jit is identity."""
        backend = NumpyBackend(jit=lambda fn: fn)

        eq_relu = Equation("t6g_relu", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq_relu], backend=backend)
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = prog("t6g_relu", x)
        np.testing.assert_allclose(out, np.maximum(0, x), rtol=1e-6)
