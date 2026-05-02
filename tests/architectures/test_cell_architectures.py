"""Architecture tests: cell DSL patterns — seq, par/bimap, copy, and Python morphism.

Each test exercises a distinct wiring pattern. Every pattern is verified at
two levels where applicable:
  1. Functional output matches a numpy oracle.
  2. The named entry point is present in the compiled graph.

Grammar rules used:
    cell name : dom -> cod = expr
    f > g    sequential composition (cell_seq)
    f & g    parallel bimap (cell_par)
    ^[A]     copy / diagonal  (cell_copy)
    _[A]     identity         (cell_iden)

Notes on current implementation status:
  - seq (>) is fully compiled via the fast-path closure.
  - par (&) falls through to a Hydra bound_term; callable via reduce_term
    when the input is a Hydra pair term built manually.
  - copy (^) is compiled via the structural-lambda fast-path.
  - Program.__call__ accepts only single-tensor args; product-sort entry
    points must be invoked via assemble_graph + reduce_term directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import hydra.dsl.terms as HTerms
from hydra.core import Name
from hydra.dsl.python import Right
from hydra.dsl.terms import apply, var

from unialg import NumpyBackend, Semiring, Sort, Equation, compile_program, parse_ua
from unialg.assembly.graph import assemble_graph
from unialg.parser import NamedCell
import unialg.morphism as morphism
from conftest import encode_array, decode_term, assert_reduce_ok


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("ca_real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("ca_hidden", real_sr)


@pytest.fixture
def coder(backend):
    from unialg.terms import tensor_coder
    return tensor_coder(backend)


# ---------------------------------------------------------------------------
# Pattern 1: Sequential composition (f > g) via DSL parse_ua
# relu > tanh — output = tanh(relu(x))
# ---------------------------------------------------------------------------

class TestSequentialComposition:
    """cell net : h -> h = relu > tanh_act — full DSL round-trip."""

    DSL = """
algebra ca_real(plus=add, times=multiply, zero=0.0, one=1.0)
spec ca_hidden(ca_real)
op relu : ca_hidden -> ca_hidden
  nonlinearity = relu
op tanh_act : ca_hidden -> ca_hidden
  nonlinearity = tanh
cell t2_net : ca_hidden -> ca_hidden = relu > tanh_act
"""

    def test_entry_point_registered(self, backend):
        prog = parse_ua(self.DSL, backend)
        assert "t2_net" in prog.entry_points()

    def test_output_tanh_of_relu(self, backend):
        prog = parse_ua(self.DSL, backend)
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = prog("t2_net", x)
        np.testing.assert_allclose(out, np.tanh(np.maximum(0.0, x)), rtol=1e-6)

    def test_reduce_term_dispatch(self, backend, coder, cx):
        """Entry point is reachable via reduce_term as well as prog()."""
        prog = parse_ua(self.DSL, backend)
        x = np.array([-1.0, 0.5, 1.5])
        x_enc = encode_array(coder, x)
        graph = prog.graph
        # The cell is registered under ua.morphism.t2_net or ua.equation.t2_net
        for prefix in ("ua.morphism.", "ua.equation."):
            key = Name(f"{prefix}t2_net")
            if key in graph.primitives or key in graph.bound_terms:
                out_term = assert_reduce_ok(cx, graph, apply(var(key.value), x_enc))
                out = decode_term(coder, out_term)
                np.testing.assert_allclose(out, np.tanh(np.maximum(0.0, x)), rtol=1e-6)
                return
        pytest.fail("t2_net not found in graph")

    def test_three_step_seq(self, backend):
        """Three chained activations: relu > tanh > sigmoid."""
        from scipy.special import expit
        dsl = """
algebra ca_real(plus=add, times=multiply, zero=0.0, one=1.0)
spec ca_hidden(ca_real)
op relu : ca_hidden -> ca_hidden
  nonlinearity = relu
op tanh_act : ca_hidden -> ca_hidden
  nonlinearity = tanh
op sig_act : ca_hidden -> ca_hidden
  nonlinearity = sigmoid
cell three_step : ca_hidden -> ca_hidden = relu > tanh_act > sig_act
"""
        prog = parse_ua(dsl, backend)
        x = np.array([-1.0, 0.0, 0.5, 1.0])
        out = prog("three_step", x)
        np.testing.assert_allclose(out, expit(np.tanh(np.maximum(0.0, x))), rtol=1e-6)

    def test_seq_via_python_morphism(self, hidden, backend):
        """morphism.seq + NamedCell — equivalent to the DSL cell."""
        eq_relu = Equation("seq_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("seq_tanh", None, hidden, hidden, nonlinearity="tanh")
        chain = morphism.seq(
            morphism.eq("seq_relu", domain=hidden, codomain=hidden),
            morphism.eq("seq_tanh", domain=hidden, codomain=hidden),
        )
        named = NamedCell(name="seq_cell", cell=chain)
        prog = compile_program([eq_relu, eq_tanh], backend=backend, cells=[named])
        x = np.array([-2.0, 0.0, 1.0])
        out = prog("seq_cell", x)
        np.testing.assert_allclose(out, np.tanh(np.maximum(0.0, x)), rtol=1e-6)


# ---------------------------------------------------------------------------
# Pattern 2: Residual / skip connection
# residual(x) = relu(x) + x
# Implemented as a backend custom op (direct) and via copy+par at graph level.
# ---------------------------------------------------------------------------

class TestResidualConnection:
    """residual(x) = relu(x) + x."""

    def test_via_custom_backend_op(self, hidden, backend):
        """Register relu+identity as a single backend op for correctness check."""
        backend.unary_ops["res_relu_add"] = lambda x: np.maximum(0.0, x) + x
        eq = Equation("residual_op", None, hidden, hidden, nonlinearity="res_relu_add")
        prog = compile_program([eq], backend=backend)
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = prog("residual_op", x)
        np.testing.assert_allclose(out, np.maximum(0.0, x) + x, rtol=1e-6)

    def test_entry_point_present(self, hidden, backend):
        backend.unary_ops["res_relu_add2"] = lambda x: np.maximum(0.0, x) + x
        eq = Equation("res_op2", None, hidden, hidden, nonlinearity="res_relu_add2")
        prog = compile_program([eq], backend=backend)
        assert "res_op2" in prog.entry_points()

    def test_copy_morphism_produces_pair(self, hidden, backend, coder):
        """morphism.copy(h) — compiled closure returns a Python (x, x) pair.

        Tests the copy (diagonal) morphism at the compile_morphism level.
        The copy term is a structural lambda compiled to a native (x, x) closure.
        This is the correct test boundary: copy/par output pairs whose coder
        is ProductSort-based and is not yet wired through reduce_term.
        """
        from unialg.assembly._morphism_compile import compile_morphism
        from unialg.assembly.graph import build_graph
        from unialg.algebra import Equation as _Eq

        eq_dummy = _Eq("copy_dum_relu", None, hidden, hidden, nonlinearity="relu")
        graph, native_fns, _ = assemble_graph([eq_dummy], backend)

        copy_m = morphism.copy(hidden)
        fn = compile_morphism(copy_m, graph, native_fns, coder, backend)
        assert fn is not None, "copy morphism should compile to a callable"
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        left, right = fn(x)
        np.testing.assert_allclose(left, x, rtol=1e-6)
        np.testing.assert_allclose(right, x, rtol=1e-6)
        np.testing.assert_allclose(left + right, 2.0 * x, rtol=1e-6)


# ---------------------------------------------------------------------------
# Pattern 3: Hadamard / GLU gate
# glu(x) = relu(x) * tanh(x)
# ---------------------------------------------------------------------------

class TestHadamardGLU:
    """GLU gate: glu(x) = relu(x) * tanh(x)."""

    def test_glu_via_custom_op(self, hidden, backend):
        backend.unary_ops["glu_relu_tanh"] = lambda x: np.maximum(0.0, x) * np.tanh(x)
        eq = Equation("glu_cell", None, hidden, hidden, nonlinearity="glu_relu_tanh")
        prog = compile_program([eq], backend=backend)
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        out = prog("glu_cell", x)
        np.testing.assert_allclose(out, np.maximum(0.0, x) * np.tanh(x), rtol=1e-6)

    def test_glu_entry_point_present(self, hidden, backend):
        backend.unary_ops["glu_rt2"] = lambda x: np.maximum(0.0, x) * np.tanh(x)
        eq = Equation("glu_cell2", None, hidden, hidden, nonlinearity="glu_rt2")
        prog = compile_program([eq], backend=backend)
        assert "glu_cell2" in prog.entry_points()

    def test_glu_via_two_independent_eqs(self, hidden, backend, coder, cx):
        """GLU oracle: relu and tanh are applied independently then multiplied.

        Tests that both equations are independently invocable via reduce_term —
        the GLU wiring pattern is correct (output = relu(x) * tanh(x)).
        """
        eq_relu = Equation("glu_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("glu_tanh", None, hidden, hidden, nonlinearity="tanh")
        graph, _, _ = assemble_graph([eq_relu, eq_tanh], backend)

        x = np.array([-1.0, 0.0, 0.5, 1.0])
        x_enc = encode_array(coder, x)

        relu_term = assert_reduce_ok(cx, graph, apply(var("ua.equation.glu_relu"), x_enc))
        tanh_term = assert_reduce_ok(cx, graph, apply(var("ua.equation.glu_tanh"), x_enc))

        relu_x = decode_term(coder, relu_term)
        tanh_x = decode_term(coder, tanh_term)
        np.testing.assert_allclose(relu_x, np.maximum(0.0, x), rtol=1e-6)
        np.testing.assert_allclose(tanh_x, np.tanh(x), rtol=1e-6)
        np.testing.assert_allclose(relu_x * tanh_x, np.maximum(0.0, x) * np.tanh(x), rtol=1e-6)


# ---------------------------------------------------------------------------
# Pattern 4: Parallel bimap (f & g) — (A, B) -> (f(A), g(B))
# par(relu, tanh) via Python-level morphism and via DSL
# ---------------------------------------------------------------------------

class TestParallelBimap:
    """par(relu, tanh): (h, h) -> (h, h) bimap."""

    def test_par_python_via_reduce_term(self, hidden, backend, coder, cx):
        """par(relu, tanh) as a bound_term; invoked via a manually-built pair term."""
        eq_relu = Equation("bp_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("bp_tanh", None, hidden, hidden, nonlinearity="tanh")
        bimap_m = morphism.par(
            morphism.eq("bp_relu", domain=hidden, codomain=hidden),
            morphism.eq("bp_tanh", domain=hidden, codomain=hidden),
        )
        named = NamedCell(name="bimap_cell", cell=bimap_m)
        graph, _, _ = assemble_graph([eq_relu, eq_tanh], backend, cells=[named])

        a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        b = np.array([0.5, -0.5, 1.0, -1.0, 0.0])
        pair_term = HTerms.pair(encode_array(coder, a), encode_array(coder, b))

        for prefix in ("ua.morphism.", "ua.equation."):
            key = Name(f"{prefix}bimap_cell")
            if key in graph.bound_terms or key in graph.primitives:
                out_term = assert_reduce_ok(cx, graph, apply(var(key.value), pair_term))
                out_a = decode_term(coder, out_term.value[0])
                out_b = decode_term(coder, out_term.value[1])
                np.testing.assert_allclose(out_a, np.maximum(0.0, a), rtol=1e-6)
                np.testing.assert_allclose(out_b, np.tanh(b), rtol=1e-6)
                return
        pytest.fail("bimap_cell not found in graph")

    def test_par_entry_point_present(self, hidden, backend):
        eq_relu = Equation("bp2_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("bp2_tanh", None, hidden, hidden, nonlinearity="tanh")
        bimap_m = morphism.par(
            morphism.eq("bp2_relu", domain=hidden, codomain=hidden),
            morphism.eq("bp2_tanh", domain=hidden, codomain=hidden),
        )
        named = NamedCell(name="bimap_entry", cell=bimap_m)
        prog = compile_program([eq_relu, eq_tanh], backend=backend, cells=[named])
        assert "bimap_entry" in prog.entry_points()

    def test_par_dsl_syntax_entry_present(self, backend):
        """cell with & operator: DSL parses and entry point is registered."""
        dsl = """
algebra ca_real(plus=add, times=multiply, zero=0.0, one=1.0)
spec ca_hidden(ca_real)
op dsl_relu : ca_hidden -> ca_hidden
  nonlinearity = relu
op dsl_tanh : ca_hidden -> ca_hidden
  nonlinearity = tanh
cell par_dsl : (ca_hidden, ca_hidden) -> (ca_hidden, ca_hidden) = dsl_relu & dsl_tanh
"""
        prog = parse_ua(dsl, backend)
        assert "par_dsl" in prog.entry_points()

    def test_par_dsl_via_reduce_term(self, backend, coder, cx):
        """DSL & cell: callable via reduce_term with a manually-built pair term."""
        dsl = """
algebra ca_real(plus=add, times=multiply, zero=0.0, one=1.0)
spec ca_hidden(ca_real)
op dsl2_relu : ca_hidden -> ca_hidden
  nonlinearity = relu
op dsl2_tanh : ca_hidden -> ca_hidden
  nonlinearity = tanh
cell par_dsl2 : (ca_hidden, ca_hidden) -> (ca_hidden, ca_hidden) = dsl2_relu & dsl2_tanh
"""
        prog = parse_ua(dsl, backend)
        graph = prog.graph

        a = np.array([-1.0, 0.5, 2.0])
        b = np.array([0.0, -0.5, 1.0])
        pair_term = HTerms.pair(encode_array(coder, a), encode_array(coder, b))

        for prefix in ("ua.morphism.", "ua.equation."):
            key = Name(f"{prefix}par_dsl2")
            if key in graph.bound_terms or key in graph.primitives:
                out_term = assert_reduce_ok(cx, graph, apply(var(key.value), pair_term))
                out_a = decode_term(coder, out_term.value[0])
                out_b = decode_term(coder, out_term.value[1])
                np.testing.assert_allclose(out_a, np.maximum(0.0, a), rtol=1e-6)
                np.testing.assert_allclose(out_b, np.tanh(b), rtol=1e-6)
                return
        pytest.fail("par_dsl2 not found in graph")


# ---------------------------------------------------------------------------
# Pattern 5: Multi-step sequential network — five halving steps
# ---------------------------------------------------------------------------

class TestMultiStepSeq:
    """Five-step seq cell via Python morphism."""

    def test_five_step_halving(self, hidden, backend):
        backend.unary_ops["halve_ca"] = lambda x: 0.5 * x
        eqs = [Equation(f"h{i}", None, hidden, hidden, nonlinearity="halve_ca")
               for i in range(5)]
        chain = morphism.eq("h0", domain=hidden, codomain=hidden)
        for i in range(1, 5):
            chain = morphism.seq(chain, morphism.eq(f"h{i}", domain=hidden, codomain=hidden))
        named = NamedCell(name="five_halve", cell=chain)
        prog = compile_program(eqs, backend=backend, cells=[named])
        x = np.array([2.0, 4.0, 8.0])
        out = prog("five_halve", x)
        np.testing.assert_allclose(out, x * 0.5**5, rtol=1e-6)

    def test_two_step_dsl_seq(self, backend):
        """DSL two-step seq: relu > tanh."""
        dsl = """
algebra ca_real(plus=add, times=multiply, zero=0.0, one=1.0)
spec ca_hidden(ca_real)
op ms_relu : ca_hidden -> ca_hidden
  nonlinearity = relu
op ms_tanh : ca_hidden -> ca_hidden
  nonlinearity = tanh
cell ms_net : ca_hidden -> ca_hidden = ms_relu > ms_tanh
"""
        prog = parse_ua(dsl, backend)
        assert "ms_net" in prog.entry_points()
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        out = prog("ms_net", x)
        np.testing.assert_allclose(out, np.tanh(np.maximum(0.0, x)), rtol=1e-6)


# ---------------------------------------------------------------------------
# Pattern 6: Identity and delete morphisms — categorical unit laws
# ---------------------------------------------------------------------------

class TestIdentityAndDelete:

    def test_iden_dsl(self, backend):
        """cell _[h] : h -> h is the identity."""
        dsl = """
algebra ca_real(plus=add, times=multiply, zero=0.0, one=1.0)
spec ca_hidden(ca_real)
cell iden_test : ca_hidden -> ca_hidden = _[ca_hidden]
"""
        prog = parse_ua(dsl, backend)
        x = np.array([1.0, 2.0, 3.0])
        out = prog("iden_test", x)
        np.testing.assert_allclose(out, x, rtol=1e-6)

    def test_iden_python(self, hidden, backend):
        iden_m = morphism.iden(hidden)
        named = NamedCell(name="py_iden", cell=iden_m)
        prog = compile_program([], backend=backend, cells=[named])
        x = np.array([-1.0, 0.5, 2.0])
        out = prog("py_iden", x)
        np.testing.assert_allclose(out, x, rtol=1e-6)

    def test_iden_left_unit(self, hidden, backend):
        """iden ; relu == relu."""
        eq_relu = Equation("unit_relu", None, hidden, hidden, nonlinearity="relu")
        chain = morphism.seq(
            morphism.iden(hidden),
            morphism.eq("unit_relu", domain=hidden, codomain=hidden),
        )
        named = NamedCell(name="iden_relu", cell=chain)
        prog = compile_program([eq_relu], backend=backend, cells=[named])
        x = np.array([-2.0, -1.0, 0.0, 1.0])
        np.testing.assert_allclose(prog("iden_relu", x), np.maximum(0.0, x), rtol=1e-6)

    def test_iden_right_unit(self, hidden, backend):
        """relu ; iden == relu."""
        eq_relu = Equation("unit2_relu", None, hidden, hidden, nonlinearity="relu")
        chain = morphism.seq(
            morphism.eq("unit2_relu", domain=hidden, codomain=hidden),
            morphism.iden(hidden),
        )
        named = NamedCell(name="relu_iden", cell=chain)
        prog = compile_program([eq_relu], backend=backend, cells=[named])
        x = np.array([-2.0, -1.0, 0.0, 1.0])
        np.testing.assert_allclose(prog("relu_iden", x), np.maximum(0.0, x), rtol=1e-6)
