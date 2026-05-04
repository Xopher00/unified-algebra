"""E2E tests for structural cell morphisms: identity, copy, delete.

Identity (_[S]) is testable through Program.__call__.
Copy and delete produce product/unit types that Program.__call__ cannot
decode, so they are tested via reduce_term on the Hydra graph directly.
"""

import numpy as np
import pytest

from unialg import NumpyBackend, parse_ua


_BASE = """\
import numpy
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec embed(real)

op relu : embed -> embed
  nonlinearity = relu
"""


class TestIdentityCell:

    def test_identity_passthrough(self):
        text = _BASE + "\ncell pass_through : embed -> embed = _[embed]\n"
        prog = parse_ua(text)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(prog("pass_through", x), x)

    def test_identity_in_seq_is_transparent(self):
        text = _BASE + "\ncell chain : embed -> embed = _[embed] > relu\n"
        prog = parse_ua(text)
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(prog("chain", x), np.maximum(0, x))

    def test_identity_after_op_is_transparent(self):
        text = _BASE + "\ncell chain : embed -> embed = relu > _[embed]\n"
        prog = parse_ua(text)
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(prog("chain", x), np.maximum(0, x))

    def test_id_alias(self):
        text = _BASE + "\ncell pass_through : embed -> embed = id[embed]\n"
        prog = parse_ua(text)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(prog("pass_through", x), x)
