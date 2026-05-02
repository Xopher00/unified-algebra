"""Catamorphism error tests: unknown op name raises ValueError.

FoldSpec and UnfoldSpec have been removed. Validation now occurs via the
TypedMorphism type system and the parser's name-resolution pass.
These tests check that referencing an undeclared op in a cell expression
raises ValueError at parse time.
"""

import pytest

from unialg import parse_ua, NumpyBackend


def test_cell_references_unknown_op_raises():
    """Referencing an undeclared op in a cell expression raises ValueError."""
    with pytest.raises((ValueError, SyntaxError)):
        parse_ua(
            """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
cell bad_cell : hidden -> hidden = nonexistent_op
""",
            NumpyBackend(),
        )


def test_cell_references_missing_functor_for_cata_raises():
    """Referencing an undeclared functor in fold[F](...) raises ValueError."""
    with pytest.raises((ValueError, SyntaxError)):
        parse_ua(
            """
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)
op step : hidden -> hidden
  nonlinearity = relu
cell bad_cata : hidden -> hidden = >[MissingFunctor](step)
""",
            NumpyBackend(),
        )
