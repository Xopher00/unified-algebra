"""Behavioral contract tests for the typed morphism core.

These cover contracts that pynguin could not reach due to Hydra initialization:
  - correct dom/cod after composition
  - CompositionError message names the mismatched spaces
  - lower() returns the internal term
  - run() evaluates a morphism against a concrete Hydra argument
"""

import unittest

import hydra.dsl.meta.phantoms as P
import hydra.lexical as L
import hydra.sources.libraries as Libs
from hydra.core import Name

from unialg import CompositionError, Morphism, Space, compose, identity, lower, run, plain

A = Space("A")
B = Space("B")
C = Space("C")
D = Space("D")

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


class TestComposeTypeStructure(unittest.TestCase):
    def test_dom_after_compose(self):
        h = compose(Morphism(A, B, _add1()), Morphism(B, C, _mul2()))
        self.assertEqual(h.dom, A)

    def test_cod_after_compose(self):
        h = compose(Morphism(A, B, _add1()), Morphism(B, C, _mul2()))
        self.assertEqual(h.cod, C)

    def test_chained_compose_dom_cod(self):
        f = Morphism(A, B, _add1())
        g = Morphism(B, C, _mul2())
        k = Morphism(C, D, _add1())
        h = compose(compose(f, g), k)
        self.assertEqual(h.dom, A)
        self.assertEqual(h.cod, D)


class TestCompositionError(unittest.TestCase):
    def test_mismatch_raises(self):
        f = Morphism(A, B, _add1())
        g = Morphism(C, D, _mul2())
        with self.assertRaises(CompositionError):
            compose(f, g)

    def test_error_names_both_mismatched_spaces(self):
        f = Morphism(A, B, _add1())
        g = Morphism(C, D, _mul2())
        try:
            compose(f, g)
            self.fail("Expected CompositionError")
        except CompositionError as e:
            msg = str(e)
            self.assertIn("B", msg)
            self.assertIn("C", msg)


class TestLower(unittest.TestCase):
    def test_lower_returns_internal_term(self):
        m = Morphism(A, B, _add1())
        self.assertIs(lower(m), m.term)

    def test_lower_identity_not_none(self):
        self.assertIsNotNone(lower(identity(plain, A)))


class TestRun(unittest.TestCase):
    def _run(self, m, n):
        return run(m, P.int32(n).value, _ctx, _graph)

    def test_run_concrete_morphism(self):
        f = Morphism(A, B, _add1())
        result = self._run(f, 5)
        self.assertEqual(result.value.value.value, 6)

    def test_run_identity_transparent(self):
        result = self._run(identity(plain, A), 42)
        self.assertEqual(result.value.value.value, 42)

    def test_run_composed_morphism(self):
        # (3 + 1) * 2 = 8
        f = compose(Morphism(A, B, _add1()), Morphism(B, C, _mul2()))
        result = self._run(f, 3)
        self.assertEqual(result.value.value.value, 8)


if __name__ == "__main__":
    unittest.main()
