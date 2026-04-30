"""Equivalence tests between morphism.X smart constructors and _para.X Cell builders.

These tests pin the behavioral contract that must survive the Cell collapse
migration. The migrated constructors return ``TypedMorphism`` wrappers; the
Hydra term shape lives at ``.term`` and the boundary types live on the wrapper.

Per the plan, two assertions per variant:
  1. Construction — for migrated variants: term shape; for legacy variants:
     Cell kinds match.
  2. Behavioral — compile_morphism(...) and compile_cell(...) produce
     closures that behave identically on representative inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

import hydra.core as core
from hydra.core import Name
from hydra.lexical import empty_graph

from unialg import NumpyBackend, Semiring, Sort, ProductSort, Equation
from unialg.terms import tensor_coder
from unialg.assembly.functor import Functor, sum_, prod, one, id_, const
from unialg.assembly._equation_resolution import resolve_equation
from unialg.assembly._typed_morphism import TypedMorphism
from unialg.assembly.para._para import (
    eq as _para_eq,
    lit as _para_lit,
    iden as _para_iden,
    copy as _para_copy,
    delete as _para_delete,
    seq as _para_seq,
    par as _para_par,
    algebra_hom as _para_algebra_hom,
    lens as _para_lens,
    compile_cell,
    CompiledLens as ParaCompiledLens,
)
import unialg.assembly.morphism as morphism
from unialg.assembly._morphism_compile import (
    CompiledLens as MorphismCompiledLens,
    compile_morphism,
)


# ---------------------------------------------------------------------------
# Fixtures (per-file, following testing.md convention)
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return NumpyBackend()


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def base_sort(real_sr):
    return Sort("base", real_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _native_fns(backend, *eqs):
    """Build the native_fns dict expected by compile_cell / compile_morphism."""
    fns = {}
    for eq_obj in eqs:
        _prim, native_fn, *_ = resolve_equation(eq_obj, backend)
        fns[Name(f"ua.equation.{eq_obj.name}")] = native_fn
    return fns


def encode_array(coder, arr):
    from hydra.dsl.python import Right
    result = coder.decode(None, np.ascontiguousarray(arr, dtype=np.float64))
    assert isinstance(result, Right)
    return result.value


def _assert_var(term, name: str):
    assert isinstance(term, core.TermVariable)
    assert term.value.value == name


def _assert_typed(term, domain=None, codomain=None):
    assert isinstance(term, TypedMorphism)
    if domain is not None:
        assert term.domain == domain
    if codomain is not None:
        assert term.codomain == codomain
    return term.term


def _assert_seq_shape(term, f_name: str, g_name: str):
    """Assert lambda x. g(f(x)) and return the parameter name."""
    assert isinstance(term, core.TermLambda)
    param = term.value.parameter.value
    outer = term.value.body
    assert isinstance(outer, core.TermApplication)
    _assert_var(outer.value.function, f"ua.equation.{g_name}")
    inner = outer.value.argument
    assert isinstance(inner, core.TermApplication)
    _assert_var(inner.value.function, f"ua.equation.{f_name}")
    _assert_var(inner.value.argument, param)
    return param


def _assert_par_shape(term, f_name: str, g_name: str):
    """Assert hydra.lib.pairs.bimap f g."""
    assert isinstance(term, core.TermApplication)
    _assert_var(term.value.argument, f"ua.equation.{g_name}")
    inner = term.value.function
    assert isinstance(inner, core.TermApplication)
    _assert_var(inner.value.function, "hydra.lib.pairs.bimap")
    _assert_var(inner.value.argument, f"ua.equation.{f_name}")


# ===========================================================================
# TestEqEquivalence
# ===========================================================================

class TestEqEquivalence:
    """morphism.eq / _para.eq — reference to a named Equation."""

    def test_morphism_eq_returns_typed_term_variable(self, hidden):
        t = morphism.eq("step", domain=hidden, codomain=hidden)
        t = _assert_typed(t, hidden, hidden)
        assert isinstance(t, core.TermVariable)
        assert t.value.value == "ua.equation.step"

    def test_para_eq_returns_cell(self):
        """The Cell path is unchanged."""
        c = _para_eq("step")
        assert c.kind == "eq"
        assert c.equation_name == "step"

    def test_compiled_behavior_equivalent(self, hidden, real_sr, backend, coder):
        eq_step = Equation("teq_step", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_step)

        fn_morph = compile_morphism(
            morphism.eq("teq_step", domain=hidden, codomain=hidden),
            empty_graph(), native_fns, coder, backend,
        )
        fn_para  = compile_cell  (_para_eq("teq_step"),     native_fns, coder, backend)
        assert fn_morph is not None
        assert fn_para  is not None

        x = np.array([2.0, 3.0])
        w = np.array([1.5, 0.5])
        np.testing.assert_allclose(fn_morph(x, w), fn_para(x, w))


# ===========================================================================
# TestLitEquivalence
# ===========================================================================

class TestLitEquivalence:
    """morphism.lit / _para.lit — 0-ary constant morphism."""

    def test_morphism_lit_returns_typed_constant_lambda(self, hidden):
        v = core.TermLiteral(value=core.LiteralFloat(value=2.5))
        t = morphism.lit(v, hidden)
        body = _assert_typed(t, t.unit(), hidden)
        assert isinstance(body, core.TermLambda)
        assert body.value.parameter.value == "_"
        assert body.value.body is v

    def test_para_lit_returns_cell(self):
        """The Cell path wraps the value term in an Inject(lit, ...)."""
        v = core.TermLiteral(value=core.LiteralFloat(value=2.5))
        assert _para_lit(v).kind == "lit"

    def test_compiled_behavior_equivalent(self, backend, coder):
        v = core.TermLiteral(value=core.LiteralFloat(value=7.0))
        fn_morph = compile_morphism(
            morphism.lit(v, core.TypeVariable(Name("ua.test.scalar"))),
            empty_graph(), {}, coder, backend,
        )
        fn_para  = compile_cell  (_para_lit(v),      {}, coder, backend)
        assert fn_morph is not None
        assert fn_para  is not None
        # lit returns the constant regardless of call-time args
        assert float(fn_morph()) == float(fn_para())
        assert float(fn_morph()) == 7.0


# ===========================================================================
# TestIdenEquivalence
# ===========================================================================

class TestIdenEquivalence:
    """morphism.iden / _para.iden — identity morphism id_A : A -> A."""

    def test_morphism_iden_returns_typed_lambda(self, hidden):
        t = morphism.iden(hidden)
        body = _assert_typed(t, hidden, hidden)
        assert isinstance(body, core.TermLambda)
        assert body.value.parameter.value == "x_"
        _assert_var(body.value.body, "x_")

    def test_morphism_iden_validates_sort_arg(self):
        """Constructor validation: non-Sort raises TypeError."""
        with pytest.raises(TypeError, match="Sort, ProductSort, or core.Type"):
            morphism.iden("not a sort")

    def test_para_iden_returns_cell(self, hidden):
        assert _para_iden(hidden).kind == "iden"

    def test_compiled_behavior_equivalent(self, hidden, backend, coder):
        fn_morph = compile_morphism(
            morphism.iden(hidden), empty_graph(), {}, coder, backend
        )
        fn_para  = compile_cell  (_para_iden(hidden),      {}, coder, backend)
        assert fn_morph is not None
        assert fn_para  is not None

        x = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(fn_morph(x), fn_para(x))
        np.testing.assert_allclose(fn_morph(x), x)   # identity leaves input unchanged


# ===========================================================================
# TestCopyEquivalence
# ===========================================================================

class TestCopyEquivalence:
    """morphism.copy / _para.copy — comonoid copy Delta_A : A -> A x A."""

    def test_morphism_copy_returns_term_variable(self, hidden):
        t = morphism.copy(hidden)
        body = _assert_typed(t, hidden)
        assert t.codomain_type == ProductSort([hidden, hidden]).type_
        assert isinstance(body, core.TermLambda)
        assert isinstance(body.value.body, core.TermPair)

    def test_para_copy_returns_cell(self, hidden):
        assert _para_copy(hidden).kind == "copy"

    def test_compiled_behavior_equivalent(self, hidden, backend, coder):
        fn_morph = compile_morphism(
            morphism.copy(hidden), empty_graph(), {}, coder, backend
        )
        fn_para  = compile_cell  (_para_copy(hidden),      {}, coder, backend)
        assert fn_morph is not None
        assert fn_para  is not None

        x = np.array([5.0, 6.0])
        a_m, b_m = fn_morph(x)
        a_p, b_p = fn_para(x)
        np.testing.assert_allclose(a_m, a_p)
        np.testing.assert_allclose(b_m, b_p)
        np.testing.assert_allclose(a_m, x)   # both copies equal the input
        np.testing.assert_allclose(b_m, x)


# ===========================================================================
# TestDeleteEquivalence
# ===========================================================================

class TestDeleteEquivalence:
    """morphism.delete / _para.delete — comonoid delete !_A : A -> 1."""

    def test_morphism_delete_returns_term_variable(self, hidden):
        t = morphism.delete(hidden)
        body = _assert_typed(t, hidden)
        assert t.codomain_type == t.unit()
        assert isinstance(body, core.TermLambda)
        assert isinstance(body.value.body, core.TermUnit)

    def test_para_delete_returns_cell(self, hidden):
        assert _para_delete(hidden).kind == "delete"

    def test_compiled_behavior_equivalent(self, hidden, backend, coder):
        fn_morph = compile_morphism(
            morphism.delete(hidden), empty_graph(), {}, coder, backend
        )
        fn_para  = compile_cell  (_para_delete(hidden),      {}, coder, backend)
        assert fn_morph is not None
        assert fn_para  is not None

        x = np.array([1.0, 2.0, 3.0])
        assert fn_morph(x) is None
        assert fn_para(x) is None


# ===========================================================================
# TestSeqEquivalence
# ===========================================================================

class TestSeqEquivalence:
    """morphism.seq / _para.seq — sequential composition f ; g."""

    def test_morphism_seq_returns_typed_lambda_shape(self, hidden):
        c_morph = morphism.seq(
            morphism.eq("f", domain=hidden, codomain=hidden),
            morphism.eq("g", domain=hidden, codomain=hidden),
        )
        _assert_typed(c_morph, hidden, hidden)
        _assert_seq_shape(c_morph.term, "f", "g")

    def test_para_seq_still_returns_cell(self):
        c_para  = _para_seq(_para_eq("f"), _para_eq("g"))
        assert c_para.kind == "seq"
        assert c_para.left.kind == "eq"
        assert c_para.right.kind == "eq"

    def test_compiled_behavior_equivalent(self, hidden, backend, coder):
        # f: halve, g: double — seq is identity on values
        backend.unary_ops["seq_halve"]  = lambda x: 0.5 * x
        backend.unary_ops["seq_double"] = lambda x: 2.0 * x
        eq_h = Equation("seq_h", None, hidden, hidden, nonlinearity="seq_halve")
        eq_d = Equation("seq_d", None, hidden, hidden, nonlinearity="seq_double")
        native_fns = _native_fns(backend, eq_h, eq_d)

        morph_cell = morphism.seq(
            morphism.eq("seq_h", domain=hidden, codomain=hidden),
            morphism.eq("seq_d", domain=hidden, codomain=hidden),
        )
        para_cell  = _para_seq(_para_eq("seq_h"), _para_eq("seq_d"))

        fn_morph = compile_morphism(morph_cell, empty_graph(), native_fns, coder, backend)
        fn_para  = compile_cell  (para_cell,   native_fns, coder, backend)
        assert fn_morph is not None
        assert fn_para  is not None

        x = np.array([4.0, 8.0])
        np.testing.assert_allclose(fn_morph(x), fn_para(x))
        np.testing.assert_allclose(fn_morph(x), x)  # halve then double = identity


# ===========================================================================
# TestParEquivalence
# ===========================================================================

class TestParEquivalence:
    """morphism.par / _para.par — monoidal product f ⊗ g on pairs."""

    def test_morphism_par_returns_typed_bimap_application(self, hidden):
        c_morph = morphism.par(
            morphism.eq("f", domain=hidden, codomain=hidden),
            morphism.eq("g", domain=hidden, codomain=hidden),
        )
        assert c_morph.domain_type == ProductSort([hidden, hidden]).type_
        assert c_morph.codomain_type == ProductSort([hidden, hidden]).type_
        _assert_par_shape(c_morph.term, "f", "g")

    def test_para_par_still_returns_cell(self):
        c_para  = _para_par(_para_eq("f"), _para_eq("g"))
        assert c_para.kind == "par"
        assert c_para.left.kind == "eq"
        assert c_para.right.kind == "eq"

    def test_compiled_behavior_equivalent(self, hidden, backend, coder):
        backend.unary_ops["par_neg"]  = lambda x: -x
        backend.unary_ops["par_sq"]   = lambda x: x ** 2
        eq_neg = Equation("par_neg_eq", None, hidden, hidden, nonlinearity="par_neg")
        eq_sq  = Equation("par_sq_eq",  None, hidden, hidden, nonlinearity="par_sq")
        native_fns = _native_fns(backend, eq_neg, eq_sq)

        morph_cell = morphism.par(
            morphism.eq("par_neg_eq", domain=hidden, codomain=hidden),
            morphism.eq("par_sq_eq", domain=hidden, codomain=hidden),
        )
        para_cell  = _para_par(_para_eq("par_neg_eq"), _para_eq("par_sq_eq"))

        fn_morph = compile_morphism(morph_cell, empty_graph(), native_fns, coder, backend)
        fn_para  = compile_cell  (para_cell,   native_fns, coder, backend)
        assert fn_morph is not None
        assert fn_para  is not None

        a = np.array([3.0])
        b = np.array([4.0])
        a_m, b_m = fn_morph((a, b))
        a_p, b_p = fn_para((a, b))
        np.testing.assert_allclose(a_m, a_p)
        np.testing.assert_allclose(b_m, b_p)
        np.testing.assert_allclose(a_m, np.array([-3.0]))
        np.testing.assert_allclose(b_m, np.array([16.0]))


# ===========================================================================
# TestAlgebraHomEquivalence
# ===========================================================================

def _list_matcher(seq_value):
    """Decompose a Python list: empty → (0, [], []); non-empty → (1, [tail], [head])."""
    if not seq_value:
        return (0, [], [])
    head, *tail = seq_value
    return (1, [tail], [head])


class TestAlgebraHomEquivalence:
    """morphism.algebra_hom / _para.algebra_hom — F-driven (co)algebra hom.

    Uses F = 1 + base × X (list functor) with an inductive walker.
    """

    def test_construction_kinds_match(self, base_sort):
        f = Functor("F_list_me", sum_(one(), prod(const(base_sort), id_())))
        v = core.TermLiteral(value=core.LiteralFloat(value=1.0))
        c_morph = morphism.algebra_hom(
            f, "algebra",
            [
                morphism.lit(v, base_sort),
                morphism.eq(
                    "step",
                    domain=ProductSort([base_sort, base_sort]),
                    codomain=base_sort,
                ),
            ],
        )
        c_para  = _para_algebra_hom(f, "algebra", [_para_lit(v), _para_eq("step")])
        assert isinstance(c_morph, TypedMorphism)
        assert isinstance(c_morph.term, core.TermLambda)
        assert c_para.kind == "algebraHom"
        assert c_para.functor.name == "F_list_me"
        assert c_para.direction == "algebra"
        assert len(c_para.cells) == 2

    @pytest.mark.xfail(reason="typed algebra_hom runtime still needs list boundary coder")
    def test_compiled_behavior_equivalent(self, hidden, real_sr, backend, coder):
        eq_step = Equation("alg_step", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_step)

        f = Functor("F_list_eq", sum_(one(), prod(const(hidden), id_())))
        init_term = encode_array(coder, np.array([1.0]))

        morph_cell = morphism.algebra_hom(
            f, "algebra",
            [
                morphism.lit(init_term, hidden),
                morphism.eq(
                    "alg_step",
                    domain=ProductSort([hidden, hidden]),
                    codomain=hidden,
                ),
            ],
        )
        para_cell = _para_algebra_hom(
            f, "algebra",
            [_para_lit(init_term), _para_eq("alg_step")],
        )

        fn_morph = compile_morphism(
            morph_cell, empty_graph(), native_fns, coder, backend,
            matchers={"F_list_eq": _list_matcher},
        )
        fn_para = compile_cell(
            para_cell, native_fns, coder, backend,
            matchers={"F_list_eq": _list_matcher},
        )
        assert fn_morph is not None
        assert fn_para  is not None

        # Foldr on [2, 3, 4]: 2 * (3 * (4 * 1)) = 24 in real semiring
        xs = [np.array([2.0]), np.array([3.0]), np.array([4.0])]
        out_morph = fn_morph(xs)
        out_para  = fn_para(xs)
        np.testing.assert_allclose(out_morph, out_para)
        np.testing.assert_allclose(out_morph, np.array([24.0]))


# ===========================================================================
# TestLensEquivalence
# ===========================================================================

class TestLensEquivalence:
    """morphism.lens / _para.lens — bidirectional optic (height-1 and height-2)."""

    def _lens_fields(self, morphism_term):
        term = _assert_typed(morphism_term)
        assert isinstance(term, core.TermRecord)
        assert term.value.type_name.value == "ua.morphism.Lens"
        return {field.name.value: field.term for field in term.value.fields}

    def test_construction_shape_height2(self, hidden, base_sort):
        pair = ProductSort([base_sort, hidden])
        c_morph = morphism.lens(
            morphism.eq("get", domain=hidden, codomain=pair),
            morphism.eq("put", domain=pair, codomain=hidden),
        )
        c_para = _para_lens(_para_eq("get"), _para_eq("put"), residual=base_sort)
        fields = self._lens_fields(c_morph)
        _assert_var(fields["forward"], "ua.equation.get")
        _assert_var(fields["backward"], "ua.equation.put")
        assert set(fields) == {"forward", "backward"}
        assert c_para.residual_sort is not None

    def test_plain_equations_rejected(self, hidden):
        with pytest.raises(TypeError, match="ProductSort"):
            morphism.lens(
                morphism.eq("get", domain=hidden, codomain=hidden),
                morphism.eq("put", domain=hidden, codomain=hidden),
            )

    def test_compiled_behavior_equivalent(self, hidden, backend, coder):
        backend.unary_ops["lns_eq_fwd"] = lambda x: (x, x)
        backend.unary_ops["lns_eq_bwd"] = lambda p: -p[1]
        pair = ProductSort([hidden, hidden])
        eq_fwd = Equation("lns_eq_fwd", None, hidden, pair, nonlinearity="lns_eq_fwd")
        eq_bwd = Equation("lns_eq_bwd", None, pair, hidden, nonlinearity="lns_eq_bwd")
        native_fns = _native_fns(backend, eq_fwd, eq_bwd)

        morph_cell = morphism.lens(
            morphism.eq("lns_eq_fwd", domain=hidden, codomain=pair),
            morphism.eq("lns_eq_bwd", domain=pair, codomain=hidden),
        )
        para_cell = _para_lens(_para_eq("lns_eq_fwd"), _para_eq("lns_eq_bwd"))

        cl_morph = compile_morphism(morph_cell, empty_graph(), native_fns, coder, backend)
        cl_para  = compile_cell  (para_cell,   native_fns, coder, backend)
        assert isinstance(cl_morph, MorphismCompiledLens)
        assert isinstance(cl_para,  ParaCompiledLens)
        assert cl_morph.residual_sort is None
        assert cl_para.residual_sort  is None

        x = np.array([5.0])
        np.testing.assert_allclose(cl_morph.forward(x),  cl_para.forward(x))
        np.testing.assert_allclose(cl_morph.backward((x, x)), cl_para.backward((x, x)))

    def test_compiled_behavior_equivalent_height2(self, hidden, base_sort, backend, coder):
        # height-2: forward: A -> R x B, backward: R x B' -> A'
        backend.unary_ops["h2_eq_fwd"] = lambda x: (x + 10, x * 2)  # A -> (R, B)
        backend.unary_ops["h2_eq_bwd"] = lambda p: p[0] - 10 + p[1]  # (R, B') -> A'
        pair = ProductSort([base_sort, hidden])
        eq_fwd = Equation("h2_eq_fwd", None, hidden, pair, nonlinearity="h2_eq_fwd")
        eq_bwd = Equation("h2_eq_bwd", None, pair, hidden, nonlinearity="h2_eq_bwd")
        native_fns = _native_fns(backend, eq_fwd, eq_bwd)

        morph_cell = morphism.lens(
            morphism.eq("h2_eq_fwd", domain=hidden, codomain=pair),
            morphism.eq("h2_eq_bwd", domain=pair, codomain=hidden),
        )
        para_cell = _para_lens(
            _para_eq("h2_eq_fwd"), _para_eq("h2_eq_bwd"), residual=base_sort
        )

        cl_morph = compile_morphism(morph_cell, empty_graph(), native_fns, coder, backend)
        cl_para  = compile_cell  (para_cell,   native_fns, coder, backend)
        assert isinstance(cl_morph, MorphismCompiledLens)
        assert isinstance(cl_para,  ParaCompiledLens)
        assert cl_morph.residual_sort is None
        assert cl_para.residual_sort  is not None

        a = 3
        r_m, b_m = cl_morph.forward(a)
        r_p, b_p = cl_para.forward(a)
        assert (r_m, b_m) == (r_p, b_p)
        assert r_m == 13
        assert b_m == 6

        out_m = cl_morph.backward((r_m, b_m))
        out_p = cl_para.backward((r_p, b_p))
        assert out_m == out_p
