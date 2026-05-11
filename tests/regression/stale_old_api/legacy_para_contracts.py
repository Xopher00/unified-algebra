"""Behavioral contract tests for the Para construction.

Contracts:
  - ParaMorphism.__post_init__ rejects mismatched underlying.dom or underlying.cod
  - embed(f) preserves dom/cod, sets param to UnitSpace
  - para_compose type structure: param = Q × P, dom/cod correct
  - para_compose raises ParaCompositionError on type mismatch
  - para_compose on embedded morphisms is semantically equivalent to ordinary compose
"""

import unittest

import hydra.dsl.meta.phantoms as P
import hydra.lexical as L
import hydra.sources.libraries as Libs
from hydra.core import Name

from unialg import Morphism, ProductSpace, Space, ParaCompositionError, ParaMorphism, embed, compose, lower, run
from unialg.typing import UnitSpace

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


# ---------------------------------------------------------------------------
# ParaMorphism validation
# ---------------------------------------------------------------------------

class TestParaMorphismValidation(unittest.TestCase):

    def test_rejects_wrong_underlying_dom(self):
        # underlying.dom must be ProductSpace(param, dom), here it is just A
        f = Morphism(A, B, _add1())
        with self.assertRaises(ValueError):
            ParaMorphism(param=A, dom=A, cod=B, underlying=f)

    def test_rejects_wrong_underlying_cod(self):
        # underlying.cod is A but declared cod is C
        ab = ProductSpace(A, B)
        fst_term = P.lam("p", P.first(P.var("p"))).value
        f = Morphism(ab, A, fst_term)
        with self.assertRaises(ValueError):
            ParaMorphism(param=A, dom=B, cod=C, underlying=f)

    def test_accepts_correct_underlying(self):
        ab = ProductSpace(A, B)
        snd_term = P.lam("p", P.second(P.var("p"))).value
        f = Morphism(ab, B, snd_term)
        pm = ParaMorphism(param=A, dom=B, cod=B, underlying=f)
        self.assertEqual(pm.param, A)
        self.assertEqual(pm.dom, B)
        self.assertEqual(pm.cod, B)


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

class TestEmbed(unittest.TestCase):

    def test_embed_preserves_dom(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f).dom, A)

    def test_embed_preserves_cod(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f).cod, B)

    def test_embed_param_is_unit(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f).param, UnitSpace())

    def test_embed_underlying_dom_is_unit_times_dom(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f).underlying.dom, ProductSpace(UnitSpace(), A))

    def test_embed_underlying_cod_matches_cod(self):
        f = Morphism(A, B, _add1())
        self.assertEqual(embed(f).underlying.cod, B)


# ---------------------------------------------------------------------------
# para_compose type structure
# ---------------------------------------------------------------------------

class TestParaComposeTypes(unittest.TestCase):

    def test_para_compose_dom(self):
        pc = compose(embed(Morphism(A, B, _add1())), embed(Morphism(B, C, _mul2())))
        self.assertEqual(pc.dom, A)

    def test_para_compose_cod(self):
        pc = compose(embed(Morphism(A, B, _add1())), embed(Morphism(B, C, _mul2())))
        self.assertEqual(pc.cod, C)

    def test_para_compose_param_is_q_times_p(self):
        # Given pf with param P_space and pg with param Q_space,
        # result param must be ProductSpace(Q_space, P_space) — Q first, then P.
        pf_underlying = Morphism(ProductSpace(P_space, A), B, _add1())
        pg_underlying = Morphism(ProductSpace(Q_space, B), C, _mul2())
        pf = ParaMorphism(param=P_space, dom=A, cod=B, underlying=pf_underlying)
        pg = ParaMorphism(param=Q_space, dom=B, cod=C, underlying=pg_underlying)
        pc = compose(pf, pg)
        self.assertEqual(pc.param, ProductSpace(Q_space, P_space))

    def test_para_compose_embedded_param_is_unit_times_unit(self):
        pc = compose(embed(Morphism(A, B, _add1())), embed(Morphism(B, C, _mul2())))
        self.assertEqual(pc.param, ProductSpace(UnitSpace(), UnitSpace()))


# ---------------------------------------------------------------------------
# para_compose error handling
# ---------------------------------------------------------------------------

class TestParaComposeError(unittest.TestCase):

    def test_raises_on_type_mismatch(self):
        # embed(f).cod == B, embed(g).dom == C — not composable
        f = Morphism(A, B, _add1())
        g = Morphism(C, B, _mul2())
        with self.assertRaises(ParaCompositionError):
            compose(embed(f), embed(g))

    def test_error_names_mismatched_spaces(self):
        f = Morphism(A, B, _add1())
        g = Morphism(C, B, _mul2())
        try:
            compose(embed(f), embed(g))
            self.fail("Expected ParaCompositionError")
        except ParaCompositionError as e:
            msg = str(e)
            self.assertIn("B", msg)
            self.assertIn("C", msg)


# ---------------------------------------------------------------------------
# semantic equivalence: compose(embed(f), embed(g)) ≅ compose(f, g)
# ---------------------------------------------------------------------------

class TestParaComposeSemantics(unittest.TestCase):

    def _run_para(self, pc, n):
        arg = P.pair(P.pair(P.unit(), P.unit()), P.int32(n)).value
        return run(pc, arg, _ctx, _graph)

    def _int_val(self, term):
        return term.value.value.value

    def test_embedded_compose_matches_ordinary_compose(self):
        # (5 + 1) * 2 = 12 via both paths
        f = Morphism(A, B, _add1())
        g = Morphism(B, C, _mul2())

        para_result = self._int_val(self._run_para(compose(embed(f), embed(g)), 5))
        ordinary_result = self._int_val(run(compose(f, g), P.int32(5).value, _ctx, _graph))

        self.assertEqual(para_result, 12)
        self.assertEqual(para_result, ordinary_result)


# ---------------------------------------------------------------------------
# lower / run for ParaMorphism
# ---------------------------------------------------------------------------

class TestParaLowering(unittest.TestCase):

    def test_lower_embed_is_underlying_term(self):
        f = Morphism(A, B, _add1())
        pm = embed(f)
        self.assertIs(lower(pm), pm.underlying.term)

    def test_lower_explicit_para_is_underlying_term(self):
        underlying = Morphism(ProductSpace(P_space, A), B, _add1())
        pm = ParaMorphism(param=P_space, dom=A, cod=B, underlying=underlying)
        self.assertIs(lower(pm), pm.underlying.term)

    def test_lower_unsupported_raises_type_error(self):
        with self.assertRaises(TypeError):
            lower("not a morphism")

    def test_run_unsupported_raises_type_error(self):
        with self.assertRaises(TypeError):
            run("not a morphism", None, _ctx, _graph)

    def test_run_embed_equivalent_to_run_morphism(self):
        # run(embed(f), (unit, n)) must equal run(f, n)
        f = Morphism(A, B, _add1())
        pm = embed(f)
        paired_arg = P.pair(P.unit(), P.int32(5)).value
        para_result = run(pm, paired_arg, _ctx, _graph)
        morph_result = run(f, P.int32(5).value, _ctx, _graph)
        self.assertEqual(para_result.value.value.value, 6)
        self.assertEqual(para_result.value.value.value, morph_result.value.value.value)


if __name__ == "__main__":
    unittest.main()
