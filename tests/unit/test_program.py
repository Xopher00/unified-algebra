"""Unit tests for Program: rebind and entry_points."""

import numpy as np
import pytest

from unialg import (
    compile_program, Program,
    Equation,
    parse_ua, NumpyBackend,
)


# ---------------------------------------------------------------------------
# Test 4: rebind returns a fresh Program
# ---------------------------------------------------------------------------

class TestRebind:

    def test_rebind_returns_new_program(self, hidden, real_sr, backend, coder):
        """rebind() returns a new Program; original is not mutated."""
        import hydra.core as core

        eq_relu = Equation("t4_relu", None, hidden, hidden, nonlinearity="relu")
        eq_tanh = Equation("t4_tanh", None, hidden, hidden, nonlinearity="tanh")

        # params creates a ua.param.dummy bound_term (not used in equations,
        # just verifying the rebind machinery round-trips)
        dummy_val = core.TermLiteral(value=core.LiteralFloat(value=1.0))
        prog = compile_program(
            [eq_relu, eq_tanh], backend=backend,
            params={"dummy": dummy_val},
        )

        new_val = core.TermLiteral(value=core.LiteralFloat(value=2.0))
        prog2 = prog.rebind(dummy=new_val)

        # rebind returns a new Program instance
        assert isinstance(prog2, Program)
        assert prog2 is not prog

        # Both programs still produce correct output (relu/tanh unaffected by dummy rebind)
        x = np.array([-1.0, 0.5, 2.0])
        expected_relu = np.maximum(0.0, x)
        np.testing.assert_allclose(prog("t4_relu", x), expected_relu, rtol=1e-6)
        np.testing.assert_allclose(prog2("t4_relu", x), expected_relu, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: entry_points enumeration
# ---------------------------------------------------------------------------

class TestEntryPoints:

    def test_entry_points_include_cell_and_equations(self, backend):
        """entry_points lists cell composition + equations; not internal ua.prim names."""
        prog = parse_ua(
            """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op t5_lin : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
op t5_relu : hidden -> hidden
  nonlinearity = relu
cell t5_path : hidden -> hidden = t5_lin > t5_relu
""",
            backend,
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
