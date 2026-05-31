"""Behavioral contract tests for Lax Para (LaxParaMorphism, compose, embed).

Contracts:
  - MonadSpace equality is structural (tag + base)
  - Monad descriptor holds tag, bind_name, pure_name; MAYBE and LIST are correct
  - LaxParaMorphism.__post_init__ rejects bad underlying.dom or underlying.cod
  - embed sets param=UnitSpace, preserves dom/cod, cod wraps in MonadSpace
  - compose type structure: param=Q×P, dom/cod correct, monad preserved
  - compose raises on cod≠dom, and on mismatched monads
  - lower(LaxParaMorphism) returns underlying.term
  - run(embed(f, MAYBE), (unit, n)) produces TermMaybe(Just(f(n)))
  - compose(embed(f), embed(g)) semantically matches compose(f, g) wrapped in Just
"""

import unittest

import hydra.dsl.meta.phantoms as P
import hydra.lexical as L
import hydra.sources.libraries as Libs
from hydra.core import Name

from unialg import (
    Morphism, ProductSpace, Space, compose,
    lower, run,
    Monad, MAYBE, LIST,
    LaxParaMorphism, LaxParaCompositionError, embed,
)
from unialg.typing import MonadSpace, UnitSpace

A = Space("A")
B = Space("B")
C = Space("C")
P_space = Space("P")
Q_space = Space("Q")

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


def _just_int(term):
    """Extract int from TermMaybe(Just(TermLiteral(LiteralInteger(IntegerValueInt32(n)))))."""
    return term.value.value.value.value.value


# ---------------------------------------------------------------------------
# MonadSpace
# ---------------------------------------------------------------------------

class TestMonadSpace(unittest.TestCase):

    def test_equality_same_tag_and_base(self):
        self.assertEqual(MonadSpace("Maybe", A), MonadSpace("Maybe", A))

    def test_inequality_different_tag(self):
        self.assertNotEqual(MonadSpace("Maybe", A), MonadSpace("List", A))

    def test_inequality_different_base(self):
        self.assertNotEqual(MonadSpace("Maybe", A), MonadSpace("Maybe", B))

    def test_name_property(self):
        self.assertEqual(MonadSpace("Maybe", A).name, "Maybe(A)")

    def test_distinct_from_space(self):
        self.assertNotEqual(MonadSpace("Maybe", A), Space("Maybe(A)"))


# ---------------------------------------------------------------------------
# Monad descriptor — MAYBE and LIST
# ---------------------------------------------------------------------------

class TestMonadDescriptor(unittest.TestCase):

    def test_maybe_tag(self):
        self.assertEqual(MAYBE.tag, "Maybe")

    def test_maybe_bind_name(self):
        self.assertEqual(MAYBE.bind_name, Name("hydra.lib.maybes.bind"))

    def test_maybe_pure_name(self):
        self.assertEqual(MAYBE.pure_name, Name("hydra.lib.maybes.pure"))

    def test_list_tag(self):
        self.assertEqual(LIST.tag, "List")

    def test_list_bind_name(self):
        self.assertEqual(LIST.bind_name, Name("hydra.lib.lists.bind"))

    def test_list_pure_name(self):
        self.assertEqual(LIST.pure_name, Name("hydra.lib.lists.pure"))

    def test_monad_equality_structural(self):
        m1 = Monad("Maybe", Name("hydra.lib.maybes.bind"), Name("hydra.lib.maybes.pure"))
        self.assertEqual(m1, MAYBE)

    def test_monad_inequality_different_tag(self):
        self.assertNotEqual(MAYBE, LIST)


# ---------------------------------------------------------------------------
# LaxParaMorphism validation
# ---------------------------------------------------------------------------

class TestLaxParaMorphismValidation(unittest.TestCase):

    def _good_underlying(self, param, dom, cod, monad):
        """Build a trivially valid underlying morphism for LaxParaMorphism."""
        pa = ProductSpace(param, dom)
        ta = MonadSpace(monad.tag, cod)
        # snd(p, a) then pure — just needs the right types, not semantic correctness
        x = P.var("v")
        body = P.apply(P.primitive(monad.pure_name), P.second(x))
        term = P.lam("v", body).value
        return Morphism(dom=pa, cod=ta, term=term)

    def test_accepts_correct_underlying(self):
        u = self._good_underlying(P_space, A, B, MAYBE)
        lpm = LaxParaMorphism(param=P_space, dom=A, cod=B, monad=MAYBE, underlying=u)
        self.assertEqual(lpm.param, P_space)
        self.assertEqual(lpm.dom, A)
        self.assertEqual(lpm.cod, B)
        self.assertEqual(lpm.monad, MAYBE)

    def test_rejects_wrong_underlying_dom(self):
        # underlying.dom must be ProductSpace(param, dom); using just A is wrong
        ta = MonadSpace(MAYBE.tag, B)
        bad_term = P.lam("v", P.apply(P.primitive(MAYBE.pure_name), P.var("v"))).value
        u = Morphism(dom=A, cod=ta, term=bad_term)
        with self.assertRaises(ValueError):
            LaxParaMorphism(param=P_space, dom=A, cod=B, monad=MAYBE, underlying=u)

    def test_rejects_wrong_underlying_cod(self):
        # underlying.cod must be MonadSpace(monad.tag, cod); using plain B is wrong
        pa = ProductSpace(P_space, A)
        bad_term = P.lam("v", P.second(P.var("v"))).value
        u = Morphism(dom=pa, cod=B, term=bad_term)
        with self.assertRaises(ValueError):
            LaxParaMorphism(param=P_space, dom=A, cod=B, monad=MAYBE, underlying=u)

    def test_rejects_wrong_monad_tag_in_underlying_cod(self):
        # MonadSpace tag must match monad.tag
        pa = ProductSpace(P_space, A)
        wrong_cod = MonadSpace("List", B)  # LIST tag instead of MAYBE
        bad_term = P.lam("v", P.second(P.var("v"))).value
        u = Morphism(dom=pa, cod=wrong_cod, term=bad_term)
        with self.assertRaises(ValueError):
            LaxParaMorphism(param=P_space, dom=A, cod=B, monad=MAYBE, underlying=u)


# ---------------------------------------------------------------------------
# lax_embed type structure
# ---------------------------------------------------------------------------

class TestLaxEmbed(unittest.TestCase):

    def test_param_is_unit(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f, MAYBE).param, UnitSpace())

    def test_dom_preserved(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f, MAYBE).dom, A)

    def test_cod_preserved(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f, MAYBE).cod, B)

    def test_monad_recorded(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f, MAYBE).monad, MAYBE)

    def test_underlying_dom_is_unit_times_dom(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f, MAYBE).underlying.dom, ProductSpace(UnitSpace(), A))

    def test_underlying_cod_is_monad_space(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f, MAYBE).underlying.cod, MonadSpace("Maybe", B))

    def test_list_embed_underlying_cod(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f, LIST).underlying.cod, MonadSpace("List", B))


# ---------------------------------------------------------------------------
# lax_para_compose type structure
# ---------------------------------------------------------------------------

class TestLaxParaComposeTypes(unittest.TestCase):

    def test_dom(self):
        lpc = compose(embed(Morphism(A, B, _add1()), MAYBE),
                               embed(Morphism(B, C, _mul2()), MAYBE))
        self.assertEqual(lpc.dom, A)

    def test_cod(self):
        lpc = compose(embed(Morphism(A, B, _add1()), MAYBE),
                               embed(Morphism(B, C, _mul2()), MAYBE))
        self.assertEqual(lpc.cod, C)

    def test_param_is_q_times_p(self):
        f_underlying = Morphism(ProductSpace(P_space, A), MonadSpace("Maybe", B),
                                _add1())  # term doesn't matter for type check
        g_underlying = Morphism(ProductSpace(Q_space, B), MonadSpace("Maybe", C),
                                _mul2())
        lpm1 = LaxParaMorphism(param=P_space, dom=A, cod=B, monad=MAYBE, underlying=f_underlying)
        lpm2 = LaxParaMorphism(param=Q_space, dom=B, cod=C, monad=MAYBE, underlying=g_underlying)
        lpc = compose(lpm1, lpm2)
        self.assertEqual(lpc.param, ProductSpace(Q_space, P_space))

    def test_monad_preserved(self):
        lpc = compose(embed(Morphism(A, B, _add1()), MAYBE),
                               embed(Morphism(B, C, _mul2()), MAYBE))
        self.assertEqual(lpc.monad, MAYBE)

    def test_underlying_dom_is_qp_times_a(self):
        lpc = compose(embed(Morphism(A, B, _add1()), MAYBE),
                               embed(Morphism(B, C, _mul2()), MAYBE))
        expected = ProductSpace(lpc.param, A)
        self.assertEqual(lpc.underlying.dom, expected)

    def test_underlying_cod_is_monad_space_c(self):
        lpc = compose(embed(Morphism(A, B, _add1()), MAYBE),
                               embed(Morphism(B, C, _mul2()), MAYBE))
        self.assertEqual(lpc.underlying.cod, MonadSpace("Maybe", C))


# ---------------------------------------------------------------------------
# lax_para_compose error handling
# ---------------------------------------------------------------------------

class TestLaxParaComposeErrors(unittest.TestCase):

    def test_raises_on_cod_dom_mismatch(self):
        f = embed(Morphism(A, B, _add1()), MAYBE)
        g = embed(Morphism(C, B, _mul2()), MAYBE)  # dom=C ≠ cod=B
        with self.assertRaises(LaxParaCompositionError):
            compose(f, g)

    def test_error_message_names_mismatched_spaces(self):
        f = embed(Morphism(A, B, _add1()), MAYBE)
        g = embed(Morphism(C, B, _mul2()), MAYBE)
        try:
            compose(f, g)
            self.fail("Expected LaxParaCompositionError")
        except LaxParaCompositionError as e:
            self.assertIn("B", str(e))
            self.assertIn("C", str(e))

    def test_raises_on_mismatched_monads(self):
        f = embed(Morphism(A, B, _add1()), MAYBE)
        g = embed(Morphism(B, C, _mul2()), LIST)
        with self.assertRaises(LaxParaCompositionError):
            compose(f, g)

    def test_monad_mismatch_error_names_both_tags(self):
        f = embed(Morphism(A, B, _add1()), MAYBE)
        g = embed(Morphism(B, C, _mul2()), LIST)
        try:
            compose(f, g)
            self.fail("Expected LaxParaCompositionError")
        except LaxParaCompositionError as e:
            self.assertIn("Maybe", str(e))
            self.assertIn("List", str(e))


# ---------------------------------------------------------------------------
# lower for LaxParaMorphism
# ---------------------------------------------------------------------------

class TestLaxParaLowering(unittest.TestCase):

    def test_lower_returns_underlying_term(self):
        lpm = embed(Morphism(A, B, _add1()), MAYBE)
        self.assertIs(lower(lpm), lpm.underlying.term)

    def test_lower_composed_returns_underlying_term(self):
        lpc = compose(embed(Morphism(A, B, _add1()), MAYBE),
                               embed(Morphism(B, C, _mul2()), MAYBE))
        self.assertIs(lower(lpc), lpc.underlying.term)


# ---------------------------------------------------------------------------
# run semantics
# ---------------------------------------------------------------------------

class TestLaxParaRunSemantics(unittest.TestCase):

    def _run_embed(self, f, monad, n):
        lpm = embed(f, monad)
        arg = P.pair(P.unit(), P.int32(n)).value
        return run(lpm, arg, _ctx, _graph)

    def test_lax_embed_maybe_produces_just(self):
        # lax_embed wraps result in Just
        result = self._run_embed(Morphism(A, B, _add1()), MAYBE, 5)
        # TermMaybe(Just(int_term)) — Just has a non-None .value
        self.assertIsNotNone(result.value.value)

    def test_lax_embed_maybe_inner_value_correct(self):
        # run(embed(add1, MAYBE), (unit, 5)) = Just(6)
        result = self._run_embed(Morphism(A, B, _add1()), MAYBE, 5)
        self.assertEqual(_just_int(result), 6)

    def test_lax_embed_maybe_mul2(self):
        result = self._run_embed(Morphism(A, B, _mul2()), MAYBE, 7)
        self.assertEqual(_just_int(result), 14)

    def test_lax_compose_maybe_semantics(self):
        # compose(embed(add1), embed(mul2)) applied to 5
        # = bind(Just(6))(λb. Just(b*2)) = Just(12)
        lpc = compose(
            embed(Morphism(A, B, _add1()), MAYBE),
            embed(Morphism(B, C, _mul2()), MAYBE),
        )
        # param = (1 × 1); arg = ((unit, unit), 5)
        arg = P.pair(P.pair(P.unit(), P.unit()), P.int32(5)).value
        result = run(lpc, arg, _ctx, _graph)
        self.assertEqual(_just_int(result), 12)

    def test_lax_compose_matches_ordinary_compose_value(self):
        # The inner value of the lax compose should match the ordinary compose
        f = Morphism(A, B, _add1())
        g = Morphism(B, C, _mul2())
        lpc = compose(embed(f, MAYBE), embed(g, MAYBE))
        lax_arg = P.pair(P.pair(P.unit(), P.unit()), P.int32(5)).value
        lax_result = _just_int(run(lpc, lax_arg, _ctx, _graph))
        plain_result = run(compose(f, g), P.int32(5).value, _ctx, _graph)
        plain_int = plain_result.value.value.value
        self.assertEqual(lax_result, plain_int)


if __name__ == "__main__":
    unittest.main()
