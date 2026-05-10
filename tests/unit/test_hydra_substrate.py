"""
Hydra API verification tests.

These tests confirm what the installed Hydra substrate actually provides.
They are API probes, not design authority — do not mistake passing behavior
for architectural contracts of this project.

Known bugs in the installed version:
- T.record() and T.wrap() crash: `AttributeError: module 'hydra.constants'
  has no attribute 'placeholder_name'`. Use T.record_with_name(Name('_'), ...)
  and Terms.record(name, fields) instead.
- P.lambda_() does not exist. Use P.lam().
- Primitive names use dot notation: 'hydra.lib.math.add', not slash notation.
"""

import pytest
import hydra.lexical as L
import hydra.reduction as R
import hydra.dsl.meta.phantoms as P
import hydra.dsl.terms as Terms
import hydra.sources.libraries as Libs
from hydra.core import Name, Field as CoreField
from hydra.dsl.python import Right, Nothing
from hydra.phantoms import TTerm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    all_prims = []
    for attr in dir(Libs):
        if attr.startswith("register_") and attr.endswith("_primitives"):
            all_prims.extend(getattr(Libs, attr)().values())
    return L.graph_with_primitives(all_prims, ())


@pytest.fixture(scope="module")
def ctx():
    return L.empty_context()


def reduce(ctx, graph, term):
    t = term.value if hasattr(term, "value") else term
    r = R.reduce_term(ctx, graph, True, t)
    assert isinstance(r, Right), f"Reduction failed: {r}"
    return r.value


# ---------------------------------------------------------------------------
# Identity and composition
# ---------------------------------------------------------------------------

class TestIdentityAndCompose:
    def test_identity_int(self, ctx, graph):
        result = reduce(ctx, graph, P.apply(P.identity(), P.int32(42)))
        assert result.value.value.value == 42

    def test_compose_left_identity(self, ctx, graph):
        """id ∘ f = f"""
        add1 = P.lam("x", P.primitive2(Name("hydra.lib.math.add"), P.var("x"), P.int32(1)))
        lhs = P.compose(P.identity(), add1)
        rhs = add1
        for v in [0, 5, -3]:
            l = reduce(ctx, graph, P.apply(lhs, P.int32(v)))
            r = reduce(ctx, graph, P.apply(rhs, P.int32(v)))
            assert l == r

    def test_compose_right_identity(self, ctx, graph):
        """f ∘ id = f"""
        add1 = P.lam("x", P.primitive2(Name("hydra.lib.math.add"), P.var("x"), P.int32(1)))
        lhs = P.compose(add1, P.identity())
        for v in [0, 5]:
            l = reduce(ctx, graph, P.apply(lhs, P.int32(v)))
            r = reduce(ctx, graph, P.apply(add1, P.int32(v)))
            assert l == r

    def test_compose_associativity(self, ctx, graph):
        """(h ∘ g) ∘ f = h ∘ (g ∘ f)"""
        add1 = P.lam("x", P.primitive2(Name("hydra.lib.math.add"), P.var("x"), P.int32(1)))
        mul2 = P.lam("x", P.primitive2(Name("hydra.lib.math.mul"), P.var("x"), P.int32(2)))
        neg = P.lam("x", P.primitive1(Name("hydra.lib.math.neg"), P.var("x")))
        lhs = P.compose(P.compose(neg, mul2), add1)
        rhs = P.compose(neg, P.compose(mul2, add1))
        for v in [3, 7]:
            l = reduce(ctx, graph, P.apply(lhs, P.int32(v)))
            r = reduce(ctx, graph, P.apply(rhs, P.int32(v)))
            assert l == r


# ---------------------------------------------------------------------------
# Constants and application
# ---------------------------------------------------------------------------

class TestConstantAndApply:
    def test_constant_ignores_input(self, ctx, graph):
        """const k x = k for any x"""
        k = P.constant(P.string("fixed"))
        for v in [P.int32(1), P.string("anything"), P.boolean(True)]:
            result = reduce(ctx, graph, P.apply(k, v))
            assert result.value.value == "fixed"

    def test_lambda_apply(self, ctx, graph):
        double = P.lam("x", P.primitive2(Name("hydra.lib.math.mul"), P.var("x"), P.int32(2)))
        result = reduce(ctx, graph, P.apply(double, P.int32(6)))
        assert result.value.value.value == 12


# ---------------------------------------------------------------------------
# Products: pair, fst, snd
# ---------------------------------------------------------------------------

class TestProducts:
    def test_fst_law(self, ctx, graph):
        """fst(pair(a, b)) = a"""
        p = P.pair(P.int32(10), P.string("b"))
        result = reduce(ctx, graph, P.first(p))
        assert result.value.value.value == 10

    def test_snd_law(self, ctx, graph):
        """snd(pair(a, b)) = b"""
        p = P.pair(P.int32(10), P.string("b"))
        result = reduce(ctx, graph, P.second(p))
        assert result.value.value == "b"

    def test_pair_roundtrip(self, ctx, graph):
        p = P.pair(P.boolean(True), P.float64(3.14))
        fst = reduce(ctx, graph, P.first(p))
        snd = reduce(ctx, graph, P.second(p))
        assert fst.value.value == True
        assert abs(snd.value.value.value - 3.14) < 1e-9


# ---------------------------------------------------------------------------
# Coproducts: inject and cases
# ---------------------------------------------------------------------------

class TestCoproducts:
    def _shape_type(self):
        return Name("Shape"), Name("circle"), Name("rect")

    def test_inject_circle_matches_circle_branch(self, ctx, graph):
        sname, cname, rname = self._shape_type()
        injected = Terms.inject(sname, cname, Terms.float64(3.14))
        match_fn = Terms.match(sname, Nothing(), [
            CoreField(cname, P.lam("r", P.string("is-circle")).value),
            CoreField(rname, P.lam("d", P.string("is-rect")).value),
        ])
        result = reduce(ctx, graph, TTerm(Terms.apply(match_fn, injected)))
        assert result.value.value == "is-circle"

    def test_inject_rect_matches_rect_branch(self, ctx, graph):
        sname, cname, rname = self._shape_type()
        injected = Terms.inject(sname, rname, Terms.string("big"))
        match_fn = Terms.match(sname, Nothing(), [
            CoreField(cname, P.lam("r", P.string("is-circle")).value),
            CoreField(rname, P.lam("d", P.string("is-rect")).value),
        ])
        result = reduce(ctx, graph, TTerm(Terms.apply(match_fn, injected)))
        assert result.value.value == "is-rect"


# ---------------------------------------------------------------------------
# Records: wrap, unwrap, project
# ---------------------------------------------------------------------------

class TestRecords:
    def _point_record(self):
        name = Name("Point")
        rec = Terms.record(name, [
            CoreField(Name("x"), Terms.float64(1.0)),
            CoreField(Name("y"), Terms.float64(2.0)),
        ])
        return name, rec

    def test_project_x(self, ctx, graph):
        name, rec = self._point_record()
        proj = Terms.project(name, Name("x"))
        result = reduce(ctx, graph, TTerm(Terms.apply(proj, rec)))
        assert abs(result.value.value.value - 1.0) < 1e-9

    def test_project_y(self, ctx, graph):
        name, rec = self._point_record()
        proj = Terms.project(name, Name("y"))
        result = reduce(ctx, graph, TTerm(Terms.apply(proj, rec)))
        assert abs(result.value.value.value - 2.0) < 1e-9

    def test_wrap_unwrap_roundtrip(self, ctx, graph):
        name, rec = self._point_record()
        wrapped = Terms.wrap(name, rec)
        unwrap_fn = Terms.unwrap(name)
        result = reduce(ctx, graph, TTerm(Terms.apply(unwrap_fn, wrapped)))
        assert result == rec


# ---------------------------------------------------------------------------
# Primitives: math, strings, lists
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_math_add(self, ctx, graph):
        result = reduce(ctx, graph, P.primitive2(Name("hydra.lib.math.add"), P.int32(3), P.int32(4)))
        assert result.value.value.value == 7

    def test_math_mul(self, ctx, graph):
        result = reduce(ctx, graph, P.primitive2(Name("hydra.lib.math.mul"), P.int32(6), P.int32(7)))
        assert result.value.value.value == 42

    def test_string_cat2(self, ctx, graph):
        result = reduce(ctx, graph, P.primitive2(Name("hydra.lib.strings.cat2"), P.string("foo"), P.string("bar")))
        assert result.value.value == "foobar"

    def test_list_length(self, ctx, graph):
        lst = P.list_([P.int32(1), P.int32(2), P.int32(3)])
        result = reduce(ctx, graph, P.primitive1(Name("hydra.lib.lists.length"), lst))
        assert result.value.value.value == 3
