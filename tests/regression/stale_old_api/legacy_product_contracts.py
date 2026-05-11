"""Behavioral contract tests for ProductSpace and product morphisms.

Covers:
  - ProductSpace structural equality and non-equality with atomic Space
  - fst and snd domain/codomain
  - par domain/codomain
  - par(id_A, id_B) as semantic identity on A × B
  - par functor law: par(f2∘f1, g2∘g1) = par(f2,g2) ∘ par(f1,g1)
"""

import unittest

import hydra.dsl.meta.phantoms as P
import hydra.lexical as L
import hydra.sources.libraries as Libs
from hydra.core import Name

from unialg import (
    ProductSpace,
    Space,
    Morphism,
    compose,
    fst,
    identity,
    par,
    run,
    snd,
    plain,
)

A = Space("A")
B = Space("B")
C = Space("C")
D = Space("D")
E = Space("E")
F = Space("F")

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


def _run_on_pair(m, left_val, right_val):
    """Apply morphism m to a pair (int32, int32) and return (left_int, right_int)."""
    pair_arg = P.pair(P.int32(left_val), P.int32(right_val)).value
    result = run(m, pair_arg, _ctx, _graph)
    return (
        result.value[0].value.value.value,
        result.value[1].value.value.value,
    )


# ---------------------------------------------------------------------------
# ProductSpace equality
# ---------------------------------------------------------------------------

class TestProductSpaceEquality(unittest.TestCase):

    def test_equal_when_components_match(self):
        self.assertEqual(ProductSpace(A, B), ProductSpace(A, B))

    def test_not_equal_when_order_swapped(self):
        # Non-commutativity: A×B ≠ B×A
        self.assertNotEqual(ProductSpace(A, B), ProductSpace(B, A))

    def test_not_equal_to_atomic_space_with_same_display(self):
        # ProductSpace(A, B).name == "A × B" but must not equal Space("A × B")
        ab = ProductSpace(A, B)
        self.assertNotEqual(ab, Space(ab.name))
        self.assertNotEqual(Space(ab.name), ab)

    def test_different_left_are_unequal(self):
        self.assertNotEqual(ProductSpace(A, B), ProductSpace(C, B))

    def test_different_right_are_unequal(self):
        self.assertNotEqual(ProductSpace(A, B), ProductSpace(A, C))

    def test_nested_product_equality(self):
        self.assertEqual(ProductSpace(ProductSpace(A, B), C),
                         ProductSpace(ProductSpace(A, B), C))

    def test_nested_product_unequal_to_flat(self):
        self.assertNotEqual(ProductSpace(ProductSpace(A, B), C),
                            ProductSpace(A, ProductSpace(B, C)))

    def test_hashable_and_consistent(self):
        self.assertEqual(hash(ProductSpace(A, B)), hash(ProductSpace(A, B)))


# ---------------------------------------------------------------------------
# fst and snd domain/codomain
# ---------------------------------------------------------------------------

class TestProjectionTypes(unittest.TestCase):

    def test_fst_type(self):
        # fst(A × B) : A × B → A
        ab = ProductSpace(A, B)
        m = fst(plain, ab)
        self.assertEqual(m.dom, ab)
        self.assertEqual(m.cod, A)

    def test_snd_type(self):
        # snd(A × B) : A × B → B
        ab = ProductSpace(A, B)
        m = snd(plain, ab)
        self.assertEqual(m.dom, ab)
        self.assertEqual(m.cod, B)

    def test_fst_cod_is_left_not_right(self):
        cd = ProductSpace(C, D)
        self.assertEqual(fst(plain, cd).cod, C)
        self.assertNotEqual(fst(plain, cd).cod, D)

    def test_snd_cod_is_right_not_left(self):
        cd = ProductSpace(C, D)
        self.assertEqual(snd(plain, cd).cod, D)
        self.assertNotEqual(snd(plain, cd).cod, C)


# ---------------------------------------------------------------------------
# par domain/codomain
# ---------------------------------------------------------------------------

class TestParTypes(unittest.TestCase):

    def test_par_type(self):
        # par(f, g) : A × C → B × D  given f : A → B, g : C → D
        f = Morphism(A, B, _add1())
        g = Morphism(C, D, _mul2())
        h = par(f, g)
        self.assertEqual(h.dom, ProductSpace(A, C))
        self.assertEqual(h.cod, ProductSpace(B, D))

    def test_par_dom_not_reversed(self):
        f = Morphism(A, B, _add1())
        g = Morphism(C, D, _mul2())
        self.assertNotEqual(par(f, g).dom, ProductSpace(C, A))

    def test_par_identity_type(self):
        h = par(identity(plain, A), identity(plain, B))
        self.assertEqual(h.dom, ProductSpace(A, B))
        self.assertEqual(h.cod, ProductSpace(A, B))


# ---------------------------------------------------------------------------
# par semantic evaluation
# ---------------------------------------------------------------------------

class TestParEvaluation(unittest.TestCase):

    def test_par_applies_f_to_left_g_to_right(self):
        # add1 on left, mul2 on right; (3, 5) → (4, 10)
        f = Morphism(A, B, _add1())
        g = Morphism(C, D, _mul2())
        self.assertEqual(_run_on_pair(par(f, g), 3, 5), (4, 10))

    def test_par_applies_independently(self):
        # mul2 on left, add1 on right; (7, 2) → (14, 3)
        f = Morphism(A, B, _mul2())
        g = Morphism(C, D, _add1())
        self.assertEqual(_run_on_pair(par(f, g), 7, 2), (14, 3))

    def test_par_identity_behaves_as_identity(self):
        # par(id_A, id_B) on (5, 9) should leave both components unchanged
        h = par(identity(plain, A), identity(plain, B))
        self.assertEqual(_run_on_pair(h, 5, 9), (5, 9))

    def test_par_functor_law(self):
        # par(f2∘f1, g2∘g1) = par(f2, g2) ∘ par(f1, g1)
        #
        # f1 : A → B  (add1)    f2 : B → C  (mul2)
        # g1 : D → E  (mul2)    g2 : E → F  (add1)
        #
        # Applied to (3, 5):
        #   left:  (3+1)*2 = 8
        #   right: 5*2+1   = 11
        f1 = Morphism(A, B, _add1())
        f2 = Morphism(B, C, _mul2())
        g1 = Morphism(D, E, _mul2())
        g2 = Morphism(E, F, _add1())

        lhs = par(compose(f1, f2), compose(g1, g2))
        rhs = compose(par(f1, g1), par(f2, g2))

        lhs_out = _run_on_pair(lhs, 3, 5)
        rhs_out = _run_on_pair(rhs, 3, 5)

        self.assertEqual(lhs_out, (8, 11))
        self.assertEqual(rhs_out, (8, 11))
        self.assertEqual(lhs_out, rhs_out)


if __name__ == "__main__":
    unittest.main()
