"""Contract tests for the polynomial functor data layer.

Contracts:
  - Each constructor returns the correct PolyExpr subclass
  - Structural equality is class-sensitive (Zero != One, Const(A) != Const(B))
  - Functor.summands() flattens the top-level Sum tree left-to-right
  - Functor.x_arity() counts Id nodes in the expression tree
  - Functor.is_recursive() is True iff x_arity > 0
  - Functor.consts() returns SpaceT values from Const and Exp.base, depth-first
  - pretty() produces correct strings with correct precedence parenthesisation
  - Functor.__post_init__ rejects category='poset' with non-Id body
  - Id is not a SpaceT subclass (sanity: recursion variable ≠ concrete space)
"""

import unittest

from unialg import Space
from unialg.typing import UnitSpace
from unialg.morphism.functor import (
    PolyExpr, Zero, One, Id, Const, Sum, Prod, Exp, Functor,
    zero, one, id_, const, sum_, prod, exp, pretty,
)

A = Space("A")
B = Space("B")
C = Space("C")


# ---------------------------------------------------------------------------
# Constructor subclass correctness
# ---------------------------------------------------------------------------

class TestConstructors(unittest.TestCase):

    def test_zero_returns_zero(self):
        self.assertIsInstance(zero(), Zero)

    def test_one_returns_one(self):
        self.assertIsInstance(one(), One)

    def test_id_returns_id(self):
        self.assertIsInstance(id_(), Id)

    def test_const_returns_const(self):
        self.assertIsInstance(const(A), Const)

    def test_sum_returns_sum(self):
        self.assertIsInstance(sum_(one(), id_()), Sum)

    def test_prod_returns_prod(self):
        self.assertIsInstance(prod(const(A), id_()), Prod)

    def test_exp_returns_exp(self):
        self.assertIsInstance(exp(A, id_()), Exp)

    def test_all_are_polyexpr(self):
        for expr in [zero(), one(), id_(), const(A), sum_(one(), id_()), prod(one(), id_()), exp(A, id_())]:
            self.assertIsInstance(expr, PolyExpr)


# ---------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------

class TestEquality(unittest.TestCase):

    def test_zero_equals_zero(self):
        self.assertEqual(zero(), zero())

    def test_one_equals_one(self):
        self.assertEqual(one(), one())

    def test_zero_ne_one(self):
        self.assertNotEqual(zero(), one())

    def test_const_eq_same_space(self):
        self.assertEqual(const(A), const(A))

    def test_const_ne_different_space(self):
        self.assertNotEqual(const(A), const(B))

    def test_sum_order_matters(self):
        self.assertNotEqual(sum_(one(), id_()), sum_(id_(), one()))

    def test_prod_order_matters(self):
        self.assertNotEqual(prod(const(A), id_()), prod(id_(), const(A)))

    def test_nested_equality(self):
        f = sum_(one(), prod(const(A), id_()))
        g = sum_(one(), prod(const(A), id_()))
        self.assertEqual(f, g)


# ---------------------------------------------------------------------------
# Functor.summands
# ---------------------------------------------------------------------------

class TestSummands(unittest.TestCase):

    def _functor(self, body):
        return Functor("F", body)

    def test_non_sum_is_single_summand(self):
        self.assertEqual(self._functor(one()).summands(), (one(),))

    def test_flat_sum(self):
        body = sum_(one(), id_())
        self.assertEqual(self._functor(body).summands(), (one(), id_()))

    def test_left_skewed_sum(self):
        # (one + id) + const(A)
        body = sum_(sum_(one(), id_()), const(A))
        self.assertEqual(self._functor(body).summands(), (one(), id_(), const(A)))

    def test_right_skewed_sum(self):
        # one + (id + const(A))
        body = sum_(one(), sum_(id_(), const(A)))
        self.assertEqual(self._functor(body).summands(), (one(), id_(), const(A)))

    def test_three_summands_associativity(self):
        left = sum_(sum_(one(), const(A)), id_())
        right = sum_(one(), sum_(const(A), id_()))
        f, g = Functor("F", left), Functor("G", right)
        self.assertEqual(f.summands(), g.summands())


# ---------------------------------------------------------------------------
# Functor.x_arity
# ---------------------------------------------------------------------------

class TestXArity(unittest.TestCase):

    def test_zero_has_no_id(self):
        self.assertEqual(Functor("F", zero()).x_arity(), 0)

    def test_one_has_no_id(self):
        self.assertEqual(Functor("F", one()).x_arity(), 0)

    def test_const_has_no_id(self):
        self.assertEqual(Functor("F", const(A)).x_arity(), 0)

    def test_id_itself(self):
        self.assertEqual(Functor("F", id_()).x_arity(), 1)

    def test_maybe_shape(self):
        # F = 1 + X
        body = sum_(one(), id_())
        self.assertEqual(Functor("F", body).x_arity(), 1)

    def test_list_shape(self):
        # F = 1 + A × X
        body = sum_(one(), prod(const(A), id_()))
        self.assertEqual(Functor("F", body).x_arity(), 1)

    def test_pair_functor(self):
        # F = X × X
        body = prod(id_(), id_())
        self.assertEqual(Functor("F", body).x_arity(), 2)

    def test_exp_counts_id_in_body(self):
        # F = A → X
        body = exp(A, id_())
        self.assertEqual(Functor("F", body).x_arity(), 1)

    def test_exp_no_id_in_body(self):
        # F = A → B (constant, no recursion)
        body = exp(A, const(B))
        self.assertEqual(Functor("F", body).x_arity(), 0)


# ---------------------------------------------------------------------------
# Functor.is_recursive
# ---------------------------------------------------------------------------

class TestIsRecursive(unittest.TestCase):

    def test_constant_functor_not_recursive(self):
        self.assertFalse(Functor("F", const(A)).is_recursive())

    def test_maybe_is_recursive(self):
        self.assertTrue(Functor("F", sum_(one(), id_())).is_recursive())

    def test_zero_not_recursive(self):
        self.assertFalse(Functor("F", zero()).is_recursive())


# ---------------------------------------------------------------------------
# Functor.consts
# ---------------------------------------------------------------------------

class TestConsts(unittest.TestCase):

    def test_zero_no_consts(self):
        self.assertEqual(Functor("F", zero()).consts(), [])

    def test_one_no_consts(self):
        self.assertEqual(Functor("F", one()).consts(), [])

    def test_id_no_consts(self):
        self.assertEqual(Functor("F", id_()).consts(), [])

    def test_const_returns_space(self):
        self.assertEqual(Functor("F", const(A)).consts(), [A])

    def test_sum_collects_left_then_right(self):
        body = sum_(const(A), const(B))
        self.assertEqual(Functor("F", body).consts(), [A, B])

    def test_prod_collects_left_then_right(self):
        body = prod(const(A), const(B))
        self.assertEqual(Functor("F", body).consts(), [A, B])

    def test_exp_collects_base_then_body(self):
        # F = A → B  — base=A, body=const(B)
        body = exp(A, const(B))
        self.assertEqual(Functor("F", body).consts(), [A, B])

    def test_exp_base_only(self):
        # F = A → X  — only base contributes
        body = exp(A, id_())
        self.assertEqual(Functor("F", body).consts(), [A])

    def test_list_shape(self):
        # F = 1 + A × X — one const (A)
        body = sum_(one(), prod(const(A), id_()))
        self.assertEqual(Functor("F", body).consts(), [A])


# ---------------------------------------------------------------------------
# pretty()
# ---------------------------------------------------------------------------

class TestPretty(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(pretty(zero()), "0")

    def test_one(self):
        self.assertEqual(pretty(one()), "1")

    def test_id(self):
        self.assertEqual(pretty(id_()), "X")

    def test_const(self):
        self.assertEqual(pretty(const(A)), "A")

    def test_sum(self):
        self.assertEqual(pretty(sum_(one(), id_())), "1 + X")

    def test_prod(self):
        self.assertEqual(pretty(prod(const(A), id_())), "A * X")

    def test_prod_of_sums_has_parens(self):
        # (1 + X) * (A + X)
        result = pretty(prod(sum_(one(), id_()), sum_(const(A), id_())))
        self.assertEqual(result, "(1 + X) * (A + X)")

    def test_prod_of_sum_left_only(self):
        result = pretty(prod(sum_(one(), id_()), id_()))
        self.assertEqual(result, "(1 + X) * X")

    def test_prod_of_sum_right_only(self):
        result = pretty(prod(id_(), sum_(one(), id_())))
        self.assertEqual(result, "X * (1 + X)")

    def test_exp_simple(self):
        self.assertEqual(pretty(exp(A, id_())), "A -> X")

    def test_exp_of_sum_has_parens(self):
        result = pretty(exp(A, sum_(one(), id_())))
        self.assertEqual(result, "A -> (1 + X)")

    def test_exp_of_prod_has_parens(self):
        result = pretty(exp(A, prod(const(B), id_())))
        self.assertEqual(result, "A -> (B * X)")

    def test_exp_of_const_no_parens(self):
        result = pretty(exp(A, const(B)))
        self.assertEqual(result, "A -> B")


# ---------------------------------------------------------------------------
# Functor validation
# ---------------------------------------------------------------------------

class TestFunctorValidation(unittest.TestCase):

    def test_poset_with_id_ok(self):
        f = Functor("F", id_(), category="poset")
        self.assertEqual(f.category, "poset")

    def test_poset_with_non_id_raises(self):
        with self.assertRaises(ValueError):
            Functor("F", sum_(one(), id_()), category="poset")

    def test_set_category_default(self):
        f = Functor("F", sum_(one(), id_()))
        self.assertEqual(f.category, "set")

    def test_set_category_with_non_id_ok(self):
        f = Functor("F", sum_(one(), id_()), category="set")
        self.assertIsNotNone(f)


# ---------------------------------------------------------------------------
# Sanity: Id is not a SpaceT
# ---------------------------------------------------------------------------

class TestIdNotSpaceT(unittest.TestCase):

    def test_id_not_space(self):
        from unialg.typing import Space
        self.assertNotIsInstance(id_(), Space)

    def test_id_not_unit_space(self):
        self.assertNotIsInstance(id_(), UnitSpace)

    def test_id_not_product_space(self):
        from unialg.typing import ProductSpace
        self.assertNotIsInstance(id_(), ProductSpace)

    def test_id_is_polyexpr(self):
        self.assertIsInstance(id_(), PolyExpr)


if __name__ == "__main__":
    unittest.main()
