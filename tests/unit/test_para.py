"""2-category Para's 1-cell algebra and interpreter.

Cell is a Hydra union over the seven 1-cell constructors (eq, lit, seq, par,
copy, delete, algebra_hom). The interpreter compiles eq, lit, and algebra_hom
into runnable morphisms; seq/par/copy/delete parse and store but the
interpreter raises NotImplementedError until the migration of Path /
Parallel / Fan into the Cell representation lands.
"""
import numpy as np
import pytest

from hydra.core import Name

from unialg import NumpyBackend, Sort, Equation
from unialg.assembly.functor import (
    Functor, sum_, prod, one, id_, const,
)
from unialg.assembly._para import (
    Cell, CELL_TYPE_NAME,
    eq, lit, seq, par, copy, delete, algebra_hom, lens,
    pretty, compile_cell, CompiledLens,
)
from unialg.assembly._equation_resolution import resolve_equation

import hydra.core as core


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_sort(real_sr):
    return Sort("base", real_sr)


def _native_fns(backend, *eqs):
    fns = {}
    for eq_obj in eqs:
        prim, native_fn, *_ = resolve_equation(eq_obj, backend)
        fns[Name(f"ua.equation.{eq_obj.name}")] = native_fn
    return fns


# ---------------------------------------------------------------------------
# Variant kinds & accessors
# ---------------------------------------------------------------------------

class TestCellKinds:

    def test_eq(self):
        c = eq("step")
        assert c.kind == "eq"
        assert c.equation_name == "step"

    def test_lit(self):
        v = core.TermLiteral(value=core.LiteralFloat(value=3.14))
        c = lit(v)
        assert c.kind == "lit"
        assert c.value_term == v

    def test_seq(self):
        c = seq(eq("f"), eq("g"))
        assert c.kind == "seq"
        assert c.left.equation_name == "f"
        assert c.right.equation_name == "g"

    def test_par(self):
        c = par(eq("f"), eq("g"))
        assert c.kind == "par"
        assert c.left.equation_name == "f"
        assert c.right.equation_name == "g"

    def test_copy(self, base_sort):
        c = copy(base_sort)
        assert c.kind == "copy"
        assert c.sort.name == "base"

    def test_delete(self, base_sort):
        c = delete(base_sort)
        assert c.kind == "delete"
        assert c.sort.name == "base"

    def test_algebra_hom(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        v = core.TermLiteral(value=core.LiteralFloat(value=1.0))
        c = algebra_hom(f, "algebra", [lit(v), eq("step")])
        assert c.kind == "algebraHom"
        assert c.functor.name == "F_list"
        assert c.direction == "algebra"
        cs = c.cells
        assert len(cs) == 2
        assert cs[0].kind == "lit"
        assert cs[1].kind == "eq"

    def test_algebra_hom_rejects_bad_direction(self, base_sort):
        f = Functor("F", id_())
        with pytest.raises(ValueError, match="direction must be"):
            algebra_hom(f, "wrongway", [eq("c")])


# ---------------------------------------------------------------------------
# Wrong-kind accessors
# ---------------------------------------------------------------------------

class TestAccessorErrors:

    def test_equation_name_only_for_eq(self):
        with pytest.raises(AttributeError, match="kind='lit'"):
            _ = lit(core.TermLiteral(value=core.LiteralFloat(value=0.0))).equation_name

    def test_left_only_for_seq_par(self):
        with pytest.raises(AttributeError, match="kind='eq'"):
            _ = eq("a").left

    def test_sort_only_for_copy_delete(self):
        with pytest.raises(AttributeError, match="kind='eq'"):
            _ = eq("a").sort

    def test_functor_only_for_algebra_hom(self):
        with pytest.raises(AttributeError, match="kind='eq'"):
            _ = eq("a").functor


# ---------------------------------------------------------------------------
# Hydra encoding
# ---------------------------------------------------------------------------

class TestHydraEncoding:

    def test_term_is_term_inject(self):
        c = eq("foo")
        assert isinstance(c.term, core.TermInject)
        assert c.term.value.type_name == CELL_TYPE_NAME
        assert c.term.value.field.name.value == "eq"

    def test_seq_payload_is_pair(self):
        c = seq(eq("a"), eq("b"))
        assert isinstance(c._payload, core.TermPair)


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

class TestPretty:

    def test_atomic(self):
        assert pretty(eq("step")) == "step"

    def test_seq(self):
        assert pretty(seq(eq("f"), eq("g"))) == "(f ; g)"

    def test_nested(self):
        c = seq(seq(eq("a"), eq("b")), eq("c"))
        assert pretty(c) == "((a ; b) ; c)"

    def test_par(self):
        assert pretty(par(eq("f"), eq("g"))) == "(f ⊗ g)"

    def test_algebra_hom(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        v = core.TermLiteral(value=core.LiteralFloat(value=1.0))
        c = algebra_hom(f, "algebra", [lit(v), eq("step")])
        assert pretty(c) == "hom[F_list, algebra](lit(_), step)"


# ---------------------------------------------------------------------------
# Interpreter: eq + lit
# ---------------------------------------------------------------------------

class TestInterpreterAtomic:

    def test_eq_compiles_to_native_fn(self, hidden, real_sr, backend, coder):
        eq_step = Equation("ti_step", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_step)
        fn = compile_cell(eq("ti_step"), native_fns, coder, backend)
        assert fn is not None
        # Run it: 2 * 3 elementwise
        out = fn(np.array([2.0]), np.array([3.0]))
        np.testing.assert_allclose(out, np.array([6.0]))

    def test_eq_missing_returns_none(self, backend, coder):
        fn = compile_cell(eq("nope"), {}, coder, backend)
        assert fn is None

    def test_lit_returns_value(self, backend, coder):
        v = core.TermLiteral(value=core.LiteralFloat(value=42.0))
        fn = compile_cell(lit(v), {}, coder, backend)
        assert fn is not None
        # lit is 0-arg in spirit; calling with no args returns the value.
        # Implementations may broadcast — check the encoded value is what
        # was decoded, modulo scalar/array shape.
        result = fn()
        assert float(result) == 42.0


# ---------------------------------------------------------------------------
# Interpreter: algebra_hom — list-cata via F = 1 + base × X
# ---------------------------------------------------------------------------

def _list_matcher(seq_value):
    if not seq_value:
        return (0, [], [])
    head, *tail = seq_value
    return (1, [tail], [head])


class TestAlgebraHomListCata:

    def test_product_of_list(self, real_sr, hidden, backend, coder):
        from conftest import encode_array
        # step : (elem, state) -> state (elementwise multiply on real)
        eq_step = Equation("la_step", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_step)

        f = Functor("F_list", sum_(one(), prod(const(hidden), id_())))
        init_term = encode_array(coder, np.array([1.0]))
        cell = algebra_hom(f, "algebra", [lit(init_term), eq("la_step")])

        fn = compile_cell(cell, native_fns, coder, backend,
                          matchers={"F_list": _list_matcher})
        assert fn is not None

        out = fn([np.array([2.0]), np.array([3.0]), np.array([4.0])])
        # Foldr: 2 * (3 * (4 * 1)) = 24
        np.testing.assert_allclose(out, np.array([24.0]))


# ---------------------------------------------------------------------------
# Interpreter: algebra_hom — tree-cata via F = base + X²
# ---------------------------------------------------------------------------

class _Tree: pass
class _Leaf(_Tree):
    def __init__(self, value): self.value = value
class _Node(_Tree):
    def __init__(self, left, right): self.left, self.right = left, right


def _tree_matcher(t):
    if isinstance(t, _Leaf):
        return (0, [], [t.value])
    return (1, [t.left, t.right], [])


class TestAlgebraHomTreeCata:
    """Same algebra_hom code as list-cata; only F + matcher change."""

    def test_tree_product(self, real_sr, hidden, backend, coder):
        backend.unary_ops["t_id"] = lambda x: x
        eq_leaf = Equation("ta_leaf", None, hidden, hidden, nonlinearity="t_id")
        eq_node = Equation("ta_node", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_leaf, eq_node)

        f = Functor("F_tree", sum_(const(hidden), prod(id_(), id_())))
        cell = algebra_hom(f, "algebra", [eq("ta_leaf"), eq("ta_node")])

        fn = compile_cell(cell, native_fns, coder, backend,
                          matchers={"F_tree": _tree_matcher})
        assert fn is not None

        # Tree of [2, 3, 4] -> product 24 via real semiring elementwise multiply.
        tree = _Node(
            _Leaf(np.array([2.0])),
            _Node(_Leaf(np.array([3.0])), _Leaf(np.array([4.0]))),
        )
        out = fn(tree)
        np.testing.assert_allclose(out, np.array([24.0]))


# ---------------------------------------------------------------------------
# Interpreter: algebra_hom validation
# ---------------------------------------------------------------------------

class TestAlgebraHomValidation:

    def test_cells_count_must_match_summands(self, base_sort, backend, coder):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        # Only one cell — F has two summands.
        cell = algebra_hom(f, "algebra", [eq("only_one")])
        with pytest.raises(ValueError, match=r"cells length 1"):
            compile_cell(cell, {}, coder, backend, matchers={"F_list": _list_matcher})

    def test_no_matcher_inductive_rejected(self, base_sort, backend, coder):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        v = core.TermLiteral(value=core.LiteralFloat(value=0.0))
        cell = algebra_hom(f, "algebra", [lit(v), eq("step")])
        with pytest.raises(ValueError, match="no matcher registered"):
            compile_cell(cell, {}, coder, backend)


# ---------------------------------------------------------------------------
# Interpreter: coalgebra direction with F = X (closure-style)
# ---------------------------------------------------------------------------

class TestCoalgebraIdentity:
    """direction='coalgebra' with F=X compiles to a step closure.

    The result is a single-step morphism X -> X. Iteration / truncation is
    a separate concern handled by downstream primitives.
    """

    def test_step_closure(self, hidden, backend, coder):
        backend.unary_ops["co_halve"] = lambda x: 0.5 * x
        eq_step = Equation("co_step", None, hidden, hidden, nonlinearity="co_halve")
        native_fns = _native_fns(backend, eq_step)

        f = Functor("F_iter", id_())
        cell = algebra_hom(f, "coalgebra", [eq("co_step")])

        fn = compile_cell(cell, native_fns, coder, backend, matchers={})
        assert fn is not None
        out = fn(np.array([8.0, 16.0]))
        np.testing.assert_allclose(out, np.array([4.0, 8.0]))


# ---------------------------------------------------------------------------
# Interpreter: structural 1-cells (seq, par, copy, delete)
# ---------------------------------------------------------------------------

class TestSeq:
    """f ; g — sequential composition. Result on input x is g(f(x))."""

    def test_compose_two_unaries(self, hidden, backend, coder):
        backend.unary_ops["pa_halve"]  = lambda x: 0.5 * x
        backend.unary_ops["pa_double"] = lambda x: 2.0 * x
        eq_h = Equation("pa_h", None, hidden, hidden, nonlinearity="pa_halve")
        eq_d = Equation("pa_d", None, hidden, hidden, nonlinearity="pa_double")
        native_fns = _native_fns(backend, eq_h, eq_d)

        # seq(halve, double) on x = double(halve(x)) = x
        fn = compile_cell(seq(eq("pa_h"), eq("pa_d")), native_fns, coder, backend)
        assert fn is not None
        np.testing.assert_allclose(fn(np.array([4.0, 8.0])), np.array([4.0, 8.0]))

    def test_compose_three_left_assoc(self, hidden, backend, coder):
        backend.unary_ops["pa_inc"] = lambda x: x + 1.0
        eq_inc = Equation("pa_inc_eq", None, hidden, hidden, nonlinearity="pa_inc")
        native_fns = _native_fns(backend, eq_inc)
        chain = seq(seq(eq("pa_inc_eq"), eq("pa_inc_eq")), eq("pa_inc_eq"))
        fn = compile_cell(chain, native_fns, coder, backend)
        np.testing.assert_allclose(fn(np.array([0.0])), np.array([3.0]))


class TestPar:
    """f ⊗ g — monoidal product. Pair input (a, b) -> (f(a), g(b))."""

    def test_par_on_pair(self, hidden, backend, coder):
        backend.unary_ops["pp_neg"]  = lambda x: -x
        backend.unary_ops["pp_sqr"]  = lambda x: x * x
        eq_neg = Equation("pp_neg_eq", None, hidden, hidden, nonlinearity="pp_neg")
        eq_sqr = Equation("pp_sqr_eq", None, hidden, hidden, nonlinearity="pp_sqr")
        native_fns = _native_fns(backend, eq_neg, eq_sqr)

        fn = compile_cell(par(eq("pp_neg_eq"), eq("pp_sqr_eq")), native_fns, coder, backend)
        a, b = fn((np.array([3.0]), np.array([4.0])))
        np.testing.assert_allclose(a, np.array([-3.0]))
        np.testing.assert_allclose(b, np.array([16.0]))


class TestCopy:
    """Δ_A : A → A × A — duplicates the input."""

    def test_duplicates(self, base_sort, backend, coder):
        fn = compile_cell(copy(base_sort), {}, coder, backend)
        a, b = fn(np.array([7.0]))
        np.testing.assert_allclose(a, np.array([7.0]))
        np.testing.assert_allclose(b, np.array([7.0]))


class TestDelete:
    """!_A : A → 1 — discards the input. Unit at runtime is None."""

    def test_returns_unit(self, base_sort, backend, coder):
        fn = compile_cell(delete(base_sort), {}, coder, backend)
        assert fn(np.array([1.0, 2.0])) is None


# ---------------------------------------------------------------------------
# Lens — Optic 1-cells (heights 1 and 2)
# ---------------------------------------------------------------------------

class TestLensConstruction:

    def test_height1_no_residual(self):
        c = lens(eq("get"), eq("put"))
        assert c.kind == "lens"
        assert c.forward.equation_name == "get"
        assert c.backward.equation_name == "put"
        assert c.residual_sort is None

    def test_height2_with_residual(self, base_sort):
        c = lens(eq("get"), eq("put"), residual=base_sort)
        assert c.kind == "lens"
        assert c.residual_sort.name == "base"

    def test_pretty_height1(self):
        c = lens(eq("g"), eq("p"))
        assert pretty(c) == "lens(g, p)"

    def test_pretty_height2(self, base_sort):
        c = lens(eq("g"), eq("p"), residual=base_sort)
        assert pretty(c) == "lens@base(g, p)"


class TestLensInterpreter:

    def test_atomic_compiles_to_compiled_lens(self, hidden, real_sr, backend, coder):
        backend.unary_ops["lns_id"]  = lambda x: x
        backend.unary_ops["lns_neg"] = lambda x: -x
        eq_g = Equation("lns_g", None, hidden, hidden, nonlinearity="lns_id")
        eq_p = Equation("lns_p", None, hidden, hidden, nonlinearity="lns_neg")
        native_fns = _native_fns(backend, eq_g, eq_p)

        compiled = compile_cell(lens(eq("lns_g"), eq("lns_p")), native_fns, coder, backend)
        assert isinstance(compiled, CompiledLens)
        assert compiled.residual_sort is None
        np.testing.assert_allclose(compiled.forward(np.array([3.0])),  np.array([3.0]))
        np.testing.assert_allclose(compiled.backward(np.array([3.0])), np.array([-3.0]))

    def test_height1_lens_seq(self, hidden, backend, coder):
        backend.unary_ops["la_double"]  = lambda x: 2.0 * x
        backend.unary_ops["la_halve"]   = lambda x: 0.5 * x
        backend.unary_ops["la_triple"]  = lambda x: 3.0 * x
        backend.unary_ops["la_third"]   = lambda x: x / 3.0
        eq_d = Equation("la_d",  None, hidden, hidden, nonlinearity="la_double")
        eq_h = Equation("la_h",  None, hidden, hidden, nonlinearity="la_halve")
        eq_t = Equation("la_t",  None, hidden, hidden, nonlinearity="la_triple")
        eq_th = Equation("la_th", None, hidden, hidden, nonlinearity="la_third")
        native_fns = _native_fns(backend, eq_d, eq_h, eq_t, eq_th)

        # L1 = (double, halve), L2 = (triple, third).
        # seq forward: triple ∘ double = ×6
        # seq backward: halve ∘ third = ÷6
        L1 = lens(eq("la_d"), eq("la_h"))
        L2 = lens(eq("la_t"), eq("la_th"))
        compiled = compile_cell(seq(L1, L2), native_fns, coder, backend)
        assert isinstance(compiled, CompiledLens)
        np.testing.assert_allclose(compiled.forward(np.array([1.0])),  np.array([6.0]))
        np.testing.assert_allclose(compiled.backward(np.array([6.0])), np.array([1.0]))

    def test_height1_lens_par(self, hidden, backend, coder):
        backend.unary_ops["lp_neg"]  = lambda x: -x
        backend.unary_ops["lp_abs"]  = lambda x: abs(x)
        eq_a = Equation("lp_a", None, hidden, hidden, nonlinearity="lp_neg")
        eq_b = Equation("lp_b", None, hidden, hidden, nonlinearity="lp_abs")
        native_fns = _native_fns(backend, eq_a, eq_b)

        L = lens(eq("lp_a"), eq("lp_a"))   # negate-negate
        M = lens(eq("lp_b"), eq("lp_a"))   # abs-negate
        compiled = compile_cell(par(L, M), native_fns, coder, backend)
        assert isinstance(compiled, CompiledLens)
        a_fwd, b_fwd = compiled.forward((np.array([3.0]), np.array([-4.0])))
        np.testing.assert_allclose(a_fwd, np.array([-3.0]))
        np.testing.assert_allclose(b_fwd, np.array([4.0]))

    def test_height2_lens_seq_threads_residual(self, hidden, base_sort, backend, coder):
        # h-2 forward: A -> R × B, backward: R × B' -> A'
        # We synthesize toy ops that match these signatures.
        backend.unary_ops["fwd1"] = lambda x: (x + 100, x * 2)        # A -> (R₁, B)
        backend.unary_ops["fwd2"] = lambda x: (x + 1000, x + 1)       # B -> (R₂, C)
        backend.binary_ops_dummy = None  # placeholder
        eq_f1 = Equation("h2_f1", None, hidden, hidden, nonlinearity="fwd1")
        eq_f2 = Equation("h2_f2", None, hidden, hidden, nonlinearity="fwd2")
        # Backward ops take a pair (residual, new_value):
        backend.unary_ops["bwd1"] = lambda p: p[0] - 100 + p[1]       # R₁ × B' -> A'
        backend.unary_ops["bwd2"] = lambda p: p[0] - 1000 + p[1]      # R₂ × C' -> B'
        eq_b1 = Equation("h2_b1", None, hidden, hidden, nonlinearity="bwd1")
        eq_b2 = Equation("h2_b2", None, hidden, hidden, nonlinearity="bwd2")
        native_fns = _native_fns(backend, eq_f1, eq_f2, eq_b1, eq_b2)

        L1 = lens(eq("h2_f1"), eq("h2_b1"), residual=base_sort)
        L2 = lens(eq("h2_f2"), eq("h2_b2"), residual=base_sort)
        compiled = compile_cell(seq(L1, L2), native_fns, coder, backend)
        assert isinstance(compiled, CompiledLens)
        assert compiled.residual_sort is not None  # composite has a residual

        # Forward of composite: a=5 → fwd1(5) = (105, 10); fwd2(10) = (1010, 11)
        # Result: ((105, 1010), 11)
        (r1, r2), c = compiled.forward(5)
        assert (r1, r2) == (105, 1010)
        assert c == 11

        # Backward: ((105, 1010), c'=11) → bwd2((1010, 11))=21 → bwd1((105, 21))=26
        out = compiled.backward(((105, 1010), 11))
        assert out == 26


class TestLensMixingRejected:

    def test_seq_para_with_lens_rejected(self, hidden, backend, coder):
        backend.unary_ops["mx"] = lambda x: x
        eq_m = Equation("mx_eq", None, hidden, hidden, nonlinearity="mx")
        native_fns = _native_fns(backend, eq_m)
        L = lens(eq("mx_eq"), eq("mx_eq"))
        with pytest.raises(ValueError, match="cannot mix Para and Optic"):
            compile_cell(seq(eq("mx_eq"), L), native_fns, coder, backend)

    def test_par_para_with_lens_rejected(self, hidden, backend, coder):
        backend.unary_ops["mx2"] = lambda x: x
        eq_m = Equation("mx2_eq", None, hidden, hidden, nonlinearity="mx2")
        native_fns = _native_fns(backend, eq_m)
        L = lens(eq("mx2_eq"), eq("mx2_eq"))
        with pytest.raises(ValueError, match="cannot mix Para and Optic"):
            compile_cell(par(L, eq("mx2_eq")), native_fns, coder, backend)


class TestComposedDecomposition:
    """Demonstrate the residual decomposition: Δ ; (f ⊗ id) ; ⊕

    Residual `f(x) ⊕ x` is structurally:
        copy(A) ; par(f, identity) ; binary-merge.
    All three pieces are ordinary Cell variants — no special "residual"
    primitive on Cell; it's just a composition.
    """

    def test_residual_via_copy_par_seq(self, hidden, real_sr, backend, coder):
        backend.unary_ops["d_id"]    = lambda x: x
        backend.unary_ops["d_double"] = lambda x: 2.0 * x
        eq_id    = Equation("d_id_eq",    None, hidden, hidden, nonlinearity="d_id")
        eq_dbl   = Equation("d_dbl_eq",   None, hidden, hidden, nonlinearity="d_double")
        # Binary merge: takes a pair, returns elementwise multiply via real semiring
        # (real.times = multiply, einsum 'i,i->i' contracts to elementwise multiply).
        # We build a wrapper that takes a Python pair argument.
        eq_mul = Equation("d_mul", "i,i->i", hidden, hidden, real_sr)
        prim_mul, mul_fn, *_ = resolve_equation(eq_mul, backend)
        # Wrap as pair-consumer for use in Cell composition.
        native_fns = _native_fns(backend, eq_id, eq_dbl)
        native_fns[Name("ua.equation.d_mul_pair")] = lambda p: mul_fn(p[0], p[1])

        # Cell expression: copy ; par(double, id) ; mul_pair
        # On input x: copy → (x, x); par → (2x, x); mul_pair → 2x * x = 2x²
        cell = seq(seq(copy(hidden), par(eq("d_dbl_eq"), eq("d_id_eq"))),
                   eq("d_mul_pair"))
        fn = compile_cell(cell, native_fns, coder, backend)
        assert fn is not None
        out = fn(np.array([3.0]))
        np.testing.assert_allclose(out, np.array([18.0]))  # 2 * 3 * 3
