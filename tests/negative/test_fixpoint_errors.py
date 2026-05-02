"""Fixpoint error tests: parse-time rejection of invalid fixpoint declarations.

FixpointSpec has been removed. Validation now occurs via the parser's
name-resolution pass and TypedMorphism type constraints.
These tests check that referencing undeclared ops in cell expressions
raises errors at parse/compile time.
"""

import pytest

from unialg import parse_ua, NumpyBackend


def test_cell_references_unknown_step_raises():
    """Referencing an undeclared step op in a cell raises ValueError."""
    with pytest.raises((ValueError, SyntaxError)):
        parse_ua(
            """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
cell bad_fixpoint : hidden -> hidden = nonexistent_step
""",
            NumpyBackend(),
        )


def test_cell_seq_wrong_sort_raises():
    """Sequential composition with incompatible sort raises TypeError."""
    with pytest.raises(TypeError):
        from unialg import Semiring, Sort, Equation
        import unialg.morphism as morphism
        from unialg.parser import NamedCell
        from unialg.assembly.graph import assemble_graph

        real = Semiring("real_fp_err", plus="add", times="multiply", zero=0.0, one=1.0)
        h1 = Sort("fp_err_h1", real)
        h2 = Sort("fp_err_h2", real)

        eq1 = Equation("fp_err_eq1", None, h1, h1, nonlinearity="relu")
        eq2 = Equation("fp_err_eq2", None, h2, h2, nonlinearity="relu")

        # h1 != h2 so seq should raise TypeError on sort mismatch
        morphism.seq(
            morphism.eq("fp_err_eq1", domain=h1, codomain=h1),
            morphism.eq("fp_err_eq2", domain=h2, codomain=h2),
        )
