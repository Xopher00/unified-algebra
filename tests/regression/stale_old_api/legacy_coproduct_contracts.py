"""Behavioral contract tests for sum/coproduct spaces and morphisms.

Contracts:
  - SumSpace is structurally distinct from Space with the same display string
  - SumSpace equality is structural and order-sensitive
  - inl(AB).dom == AB.left, inl(AB).cod == AB
  - inr(AB).dom == AB.right, inr(AB).cod == AB
  - case(f, g).dom == SumSpace(f.dom, g.dom), case(f, g).cod == f.cod
  - case raises CaseError when f.cod != g.cod, message names both codomains
  - case semantics: case(f, g)(Left x) == f(x) and case(f, g)(Right x) == g(x)
"""

import unittest

import hydra.dsl.meta.phantoms as P
import hydra.lexical as L
import hydra.sources.libraries as Libs
from hydra.core import Name

from unialg import CaseError, Morphism, Space, case, inl, inr, run, plain
from unialg.typing import SumSpace

A = Space("A")
B = Space("B")
C = Space("C")

_EITHER = Name("hydra.lib.either.Either")

_graph = None
_ctx = None


def setUpModule():
    global _graph, _ctx
    all_prims = []
    for attr in dir(Libs):
        if attr.startswith("register_") and attr.endswith("_primitives"):
            all_prims.extend(getattr(Libs, attr)().values())
    _graph = L.graph_with_primitives(all_prims, ())
    _ctx = L.empty_context()


def _add1():
    return P.lam("x", P.primitive2(Name("hydra.lib.math.add"), P.var("x"), P.int32(1))).value


def _mul2():
    return P.lam("x", P.primitive2(Name("hydra.lib.math.mul"), P.var("x"), P.int32(2))).value


def _int_val(term):
    return term.value.value.value


# ---------------------------------------------------------------------------
# SumSpace structural equality
# ---------------------------------------------------------------------------

class TestSumSpaceEquality(unittest.TestCase):

    def test_sum_space_not_equal_to_atomic_space(self):
        self.assertNotEqual(SumSpace(A, B), Space("A + B"))

    def test_sum_space_equal_to_itself(self):
        self.assertEqual(SumSpace(A, B), SumSpace(A, B))

    def test_sum_space_not_symmetric(self):
        self.assertNotEqual(SumSpace(A, B), SumSpace(B, A))

    def test_sum_space_name_property(self):
        self.assertEqual(SumSpace(A, B).name, "A + B")


# ---------------------------------------------------------------------------
# inl / inr types
# ---------------------------------------------------------------------------

class TestInjectionTypes(unittest.TestCase):

    def setUp(self):
        self.ab = SumSpace(A, B)

    def test_inl_dom(self):
        self.assertEqual(inl(plain, self.ab).dom, A)

    def test_inl_cod(self):
        self.assertEqual(inl(plain, self.ab).cod, self.ab)

    def test_inr_dom(self):
        self.assertEqual(inr(plain, self.ab).dom, B)

    def test_inr_cod(self):
        self.assertEqual(inr(plain, self.ab).cod, self.ab)


# ---------------------------------------------------------------------------
# case types and error handling
# ---------------------------------------------------------------------------

class TestCaseTypes(unittest.TestCase):

    def test_case_dom(self):
        f = Morphism(A, C, _add1())
        g = Morphism(B, C, _mul2())
        self.assertEqual(case(f, g).dom, SumSpace(A, B))

    def test_case_cod(self):
        f = Morphism(A, C, _add1())
        g = Morphism(B, C, _mul2())
        self.assertEqual(case(f, g).cod, C)

    def test_case_raises_on_cod_mismatch(self):
        f = Morphism(A, C, _add1())
        g = Morphism(B, A, _mul2())
        with self.assertRaises(CaseError):
            case(f, g)

    def test_case_error_names_both_codomains(self):
        f = Morphism(A, C, _add1())
        g = Morphism(B, A, _mul2())
        try:
            case(f, g)
            self.fail("Expected CaseError")
        except CaseError as e:
            msg = str(e)
            self.assertIn("C", msg)
            self.assertIn("A", msg)


# ---------------------------------------------------------------------------
# case semantics
# ---------------------------------------------------------------------------

class TestCaseSemantics(unittest.TestCase):

    def _run(self, morphism, term):
        return run(morphism, term, _ctx, _graph)

    def test_case_left_applies_f(self):
        # case(add1, mul2)(Left 5) == 6
        f = Morphism(A, C, _add1())
        g = Morphism(B, C, _mul2())
        c = case(f, g)
        left5 = P.inject(_EITHER, Name("left"), P.int32(5)).value
        result = self._run(c, left5)
        self.assertEqual(_int_val(result), 6)

    def test_case_right_applies_g(self):
        # case(add1, mul2)(Right 5) == 10
        f = Morphism(A, C, _add1())
        g = Morphism(B, C, _mul2())
        c = case(f, g)
        right5 = P.inject(_EITHER, Name("right"), P.int32(5)).value
        result = self._run(c, right5)
        self.assertEqual(_int_val(result), 10)

    def test_case_left_matches_direct_f(self):
        # case(f, g)(Left x) == f(x) for any x
        f = Morphism(A, C, _add1())
        g = Morphism(B, C, _mul2())
        c = case(f, g)
        for n in [0, 1, 7, 42]:
            left_n = P.inject(_EITHER, Name("left"), P.int32(n)).value
            via_case = _int_val(self._run(c, left_n))
            direct = _int_val(self._run(f, P.int32(n).value))
            self.assertEqual(via_case, direct, f"n={n}")

    def test_case_right_matches_direct_g(self):
        f = Morphism(A, C, _add1())
        g = Morphism(B, C, _mul2())
        c = case(f, g)
        for n in [0, 1, 7, 42]:
            right_n = P.inject(_EITHER, Name("right"), P.int32(n)).value
            via_case = _int_val(self._run(c, right_n))
            direct = _int_val(self._run(g, P.int32(n).value))
            self.assertEqual(via_case, direct, f"n={n}")


if __name__ == "__main__":
    unittest.main()
