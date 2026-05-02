"""End-to-end lens tests: reduce_term dispatch, semiring polymorphism.

TestLensEndToEnd: simple forward/backward equation pairs executed via reduce_term.
TestLensSemiringPolymorphism: same structure with tropical semiring.

The optic threading tests (TestOpticLensValidation, TestMultiOpticEndToEnd,
TestOpticSemiringPolymorphism) relied on PathSpec.has_residual which was removed
with the legacy assembly layer. Those tests have been deleted.
"""

import numpy as np
import pytest

from hydra.core import Name
from hydra.dsl.terms import apply, var

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
    compile_program,
)
from unialg.assembly.graph import assemble_graph

from conftest import encode_array, decode_term, assert_reduce_ok


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tropic_sort(tropical_sr):
    return Sort("tropic", tropical_sr)


# ===========================================================================
# Part D: Lens end-to-end via reduce_term
# ===========================================================================

class TestLensEndToEnd:
    """Verify both forward and backward paths execute correctly via reduce_term.

    Without the legacy PathSpec.bwd_eq_names, we test each direction as
    independent equation calls through compile_program.
    """

    def test_forward_executes(self, cx, hidden, backend, coder):
        """Forward path: relu applied to input via reduce_term."""
        eq_fwd = Equation("relu_fwd", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq_fwd], backend=backend)

        x = np.array([-1.0, 0.5, -0.3, 2.0])
        out = prog("relu_fwd", x)
        np.testing.assert_allclose(out, np.maximum(0, x))

    def test_backward_executes(self, cx, hidden, backend, coder):
        """Backward path: tanh applied to input."""
        eq_bwd = Equation("tanh_bwd", None, hidden, hidden, nonlinearity="tanh")
        prog = compile_program([eq_bwd], backend=backend)

        x = np.array([1.0, -1.0, 0.0, 2.0])
        out = prog("tanh_bwd", x)
        np.testing.assert_allclose(out, np.tanh(x))

    def test_fwd_bwd_independent(self, cx, hidden, backend, coder):
        """Forward and backward are independent equations callable separately."""
        eq_fwd = Equation("asym_fwd", None, hidden, hidden, nonlinearity="relu")
        eq_bwd = Equation("asym_bwd", None, hidden, hidden, nonlinearity="tanh")
        prog = compile_program([eq_fwd, eq_bwd], backend=backend)

        x = np.array([-1.0, 0.5, 0.0, 2.0])
        out_fwd = prog("asym_fwd", x)
        out_bwd = prog("asym_bwd", x)
        np.testing.assert_allclose(out_fwd, np.maximum(0, x))
        np.testing.assert_allclose(out_bwd, np.tanh(x))

    def test_two_step_forward(self, cx, hidden, backend, coder):
        """Two equations composed: abs -> relu."""
        eq_abs = Equation("abs_f2", None, hidden, hidden, nonlinearity="abs")
        eq_relu = Equation("relu_f2", None, hidden, hidden, nonlinearity="relu")
        prog = compile_program([eq_abs, eq_relu], backend=backend)

        x = np.array([1.0, -1.0, 0.5])
        abs_out = prog("abs_f2", x)
        out = prog("relu_f2", abs_out)
        np.testing.assert_allclose(out, np.maximum(0, np.abs(x)))

    def test_reverse_composition(self, cx, hidden, backend, coder):
        """Applying bwd then fwd equations is just independent calls."""
        eq_abs = Equation("abs_b2", None, hidden, hidden, nonlinearity="abs")
        eq_tanh = Equation("tanh_b2", None, hidden, hidden, nonlinearity="tanh")
        prog = compile_program([eq_abs, eq_tanh], backend=backend)

        x = np.array([1.0, -1.0, 0.5])
        neg_out = prog("abs_b2", x)
        out = prog("tanh_b2", neg_out)
        np.testing.assert_allclose(out, np.tanh(np.abs(x)))


# ===========================================================================
# Part G: Semiring polymorphism
# ===========================================================================

class TestLensSemiringPolymorphism:
    """Demonstrate that forward/backward equations are semiring-agnostic."""

    def test_tropical_lens_declaration(self, tropical_sr, tropic_sort):
        eq_fwd = Equation("tp_fwd", "i->j", tropic_sort, tropic_sort, tropical_sr)
        eq_bwd = Equation("tp_bwd", "j->i", tropic_sort, tropic_sort, tropical_sr)
        assert eq_fwd.name == "tp_fwd"
        assert eq_bwd.name == "tp_bwd"

    def test_tropical_equation_compiles(self, tropical_sr, tropic_sort, backend, cx, coder):
        eq_fwd = Equation("tp2_fwd", None, tropic_sort, tropic_sort, nonlinearity="relu")
        eq_bwd = Equation("tp2_bwd", None, tropic_sort, tropic_sort, nonlinearity="relu")
        prog = compile_program([eq_fwd, eq_bwd], backend=backend)
        assert "tp2_fwd" in prog.entry_points()
        assert "tp2_bwd" in prog.entry_points()

    def test_tropical_equation_end_to_end(self, cx, tropical_sr, tropic_sort, backend, coder):
        """Tropical identity equation: output == input."""
        eq_fwd = Equation("trp_fwd", "i->i", tropic_sort, tropic_sort, tropical_sr)
        eq_bwd = Equation("trp_bwd", "i->i", tropic_sort, tropic_sort, tropical_sr)
        prog = compile_program([eq_fwd, eq_bwd], backend=backend)

        x = np.array([1.0, 3.0, 2.0])
        out_fwd = prog("trp_fwd", x)
        np.testing.assert_allclose(out_fwd, x)

        out_bwd = prog("trp_bwd", x)
        np.testing.assert_allclose(out_bwd, x)

    def test_real_and_tropical_independent(self, cx, real_sr, tropical_sr, backend, coder):
        """Real and tropical equations can coexist in separate programs."""
        real_sort = Sort("real_s", real_sr)
        trop_sort = Sort("trop_s", tropical_sr)

        eq_real = Equation("real_relu", None, real_sort, real_sort, nonlinearity="relu")
        eq_trop = Equation("trop_relu", None, trop_sort, trop_sort, nonlinearity="relu")

        prog_real = compile_program([eq_real], backend=backend)
        prog_trop = compile_program([eq_trop], backend=backend)

        x = np.array([-1.0, 0.5, 2.0])
        out_real = prog_real("real_relu", x)
        out_trop = prog_trop("trop_relu", x)

        np.testing.assert_allclose(out_real, np.maximum(0, x))
        np.testing.assert_allclose(out_trop, np.maximum(0, x))
