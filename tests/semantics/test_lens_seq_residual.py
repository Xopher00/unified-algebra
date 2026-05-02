"""Tests for lens_seq with residual sort optic threading.

Verifies:
1. Direct CompiledLens construction with residual_sort attached
2. _try_lens_seq threads forward/backward via (r1, a) = fwd(s) semantics
3. Composed residual_sort is ProductSort(l1.residual_sort, l2.residual_sort)
4. lens_seq junction validation rejects type mismatches
5. end-to-end via compile_morphism and _register_cells
"""
import numpy as np
import pytest

from hydra.core import Name
import hydra.core as core

from unialg import NumpyBackend, Semiring, Sort, Equation
from unialg.terms import tensor_coder
from unialg.algebra.sort import ProductSort
from unialg.morphism._typed_morphism import TypedMorphism as T
from unialg.assembly._morphism_compile import compile_morphism, CompiledLens
from unialg.assembly._morphism_compile import MORPHISM_PRIM_PREFIX, register_cells
from unialg.assembly.graph import build_graph
from unialg.parser import NamedCell
from unialg.assembly._equation_resolution import resolve_equation
import unialg.morphism as morphism


@pytest.fixture
def backend():
    b = NumpyBackend()
    # forward op: returns (residual, focus) = (x * 2, x + 1)
    b.unary_ops["store_residual"] = lambda x: (x * 2.0, x + 1.0)
    # backward op: takes (residual, grad) and returns residual * grad
    b.unary_ops["consume_residual"] = lambda pair: pair[0] * pair[1]
    return b


@pytest.fixture
def real_sr():
    return Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def hidden(real_sr):
    return Sort("hidden", real_sr)


@pytest.fixture
def residual(real_sr):
    return Sort("residual", real_sr)


@pytest.fixture
def coder(backend):
    return tensor_coder(backend)


@pytest.fixture
def graph(hidden):
    return build_graph([hidden])


def _make_lens(hidden, residual, backend):
    """Build a lens whose forward stores a residual and backward consumes it."""
    prod = T.product(hidden, hidden)
    fwd_m = morphism.eq("store_residual", domain=hidden, codomain=prod)
    bwd_m = morphism.eq("consume_residual", domain=prod, codomain=hidden)
    return morphism.lens(fwd_m, bwd_m, residual_sort=residual)


def _native_fns_for(backend, *eqs):
    fns = {}
    for eq_obj in eqs:
        _, native_fn, *_ = resolve_equation(eq_obj, backend)
        fns[Name(f"ua.equation.{eq_obj.name}")] = native_fn
    return fns


class TestLensResidualConstruction:

    def test_residual_sort_stored_in_term(self, hidden, residual):
        prod = T.product(hidden, hidden)
        fwd = morphism.eq("f", domain=hidden, codomain=prod)
        bwd = morphism.eq("g", domain=prod, codomain=hidden)
        m = morphism.lens(fwd, bwd, residual_sort=residual)
        assert isinstance(m.term, core.TermRecord)
        field_names = {f.name.value for f in m.term.value.fields}
        assert "residualSort" in field_names

    def test_no_residual_term_without_arg(self, hidden):
        prod = T.product(hidden, hidden)
        fwd = morphism.eq("f", domain=hidden, codomain=prod)
        bwd = morphism.eq("g", domain=prod, codomain=hidden)
        m = morphism.lens(fwd, bwd)
        field_names = {f.name.value for f in m.term.value.fields}
        assert "residualSort" not in field_names

    def test_compiled_lens_carries_residual_sort(self, hidden, residual, graph, coder, backend):
        eq_fwd = Equation("store_residual", None, hidden, hidden, nonlinearity="store_residual")
        eq_bwd = Equation("consume_residual", None, hidden, hidden, nonlinearity="consume_residual")
        native_fns = _native_fns_for(backend, eq_fwd, eq_bwd)
        m = _make_lens(hidden, residual, backend)
        cl = compile_morphism(m, graph, native_fns, coder, backend)
        assert isinstance(cl, CompiledLens)
        assert cl.residual_sort is not None
        assert cl.residual_sort.name == "residual"


class TestLensSeqOpticThreading:

    def test_forward_threading(self, hidden, residual, graph, coder, backend):
        """Composed forward: r1,a = l1.fwd(s); r2,b = l2.fwd(a); returns ((r1,r2), b)."""
        eq_fwd = Equation("store_residual", None, hidden, hidden, nonlinearity="store_residual")
        eq_bwd = Equation("consume_residual", None, hidden, hidden, nonlinearity="consume_residual")
        native_fns = _native_fns_for(backend, eq_fwd, eq_bwd)

        l = _make_lens(hidden, residual, backend)
        composed = morphism.lens_seq(l, l)
        cl = compile_morphism(composed, graph, native_fns, coder, backend)

        assert isinstance(cl, CompiledLens)

        x = np.array([1.0, 2.0, 3.0])
        (r1, r2), b = cl.forward(x)

        # l1.fwd(x) = (x*2, x+1)  => r1 = x*2, a = x+1
        # l2.fwd(a) = ((x+1)*2, (x+1)+1)  => r2 = (x+1)*2, b = x+2
        np.testing.assert_allclose(r1, x * 2.0)
        np.testing.assert_allclose(r2, (x + 1.0) * 2.0)
        np.testing.assert_allclose(b, x + 2.0)

    def test_backward_threading(self, hidden, residual, graph, coder, backend):
        """Composed backward: a' = l2.bwd((r2, b')); s' = l1.bwd((r1, a'))."""
        eq_fwd = Equation("store_residual", None, hidden, hidden, nonlinearity="store_residual")
        eq_bwd = Equation("consume_residual", None, hidden, hidden, nonlinearity="consume_residual")
        native_fns = _native_fns_for(backend, eq_fwd, eq_bwd)

        l = _make_lens(hidden, residual, backend)
        composed = morphism.lens_seq(l, l)
        cl = compile_morphism(composed, graph, native_fns, coder, backend)

        r1 = np.array([4.0, 6.0])
        r2 = np.array([1.0, 3.0])
        b_prime = np.array([0.5, 1.5])

        result = cl.backward(((r1, r2), b_prime))

        # l2.bwd((r2, b')) = r2 * b'
        a_prime = r2 * b_prime
        # l1.bwd((r1, a')) = r1 * a'
        expected = r1 * a_prime
        np.testing.assert_allclose(result, expected)

    def test_composed_residual_sort_is_product(self, hidden, residual, graph, coder, backend):
        """Composed lens residual_sort is ProductSort of the two component residuals."""
        eq_fwd = Equation("store_residual", None, hidden, hidden, nonlinearity="store_residual")
        eq_bwd = Equation("consume_residual", None, hidden, hidden, nonlinearity="consume_residual")
        native_fns = _native_fns_for(backend, eq_fwd, eq_bwd)

        l = _make_lens(hidden, residual, backend)
        composed = morphism.lens_seq(l, l)
        cl = compile_morphism(composed, graph, native_fns, coder, backend)

        assert isinstance(cl.residual_sort, ProductSort)
        elems = cl.residual_sort.elements
        assert len(elems) == 2
        assert elems[0].name == "residual"
        assert elems[1].name == "residual"

    def test_no_residual_when_components_have_none(self, hidden, graph, coder, backend):
        """lens_seq with no residual sorts produces CompiledLens with residual_sort=None."""
        eq_fwd = Equation("store_residual", None, hidden, hidden, nonlinearity="store_residual")
        eq_bwd = Equation("consume_residual", None, hidden, hidden, nonlinearity="consume_residual")
        native_fns = _native_fns_for(backend, eq_fwd, eq_bwd)

        prod = T.product(hidden, hidden)
        fwd_m = morphism.eq("store_residual", domain=hidden, codomain=prod)
        bwd_m = morphism.eq("consume_residual", domain=prod, codomain=hidden)
        l = morphism.lens(fwd_m, bwd_m)

        composed = morphism.lens_seq(l, l)
        cl = compile_morphism(composed, graph, native_fns, coder, backend)

        assert isinstance(cl, CompiledLens)
        assert cl.residual_sort is None

    def test_junction_mismatch_rejected(self, hidden, real_sr):
        """lens_seq raises when l1.codomain != l2.domain."""
        other_sort = Sort("other", real_sr)
        prod_h = T.product(hidden, hidden)
        prod_o = T.product(other_sort, other_sort)

        l1 = morphism.lens(
            morphism.eq("f", domain=hidden, codomain=prod_h),
            morphism.eq("g", domain=prod_h, codomain=hidden),
        )
        l2 = morphism.lens(
            morphism.eq("h", domain=other_sort, codomain=prod_o),
            morphism.eq("k", domain=prod_o, codomain=other_sort),
        )
        with pytest.raises(TypeError, match="lens_seq.junction"):
            morphism.lens_seq(l1, l2)


class TestLensSeqRegistration:

    def test_registers_forward_backward_primitives(self, hidden, residual, coder, backend):
        """_register_cells exposes .forward and .backward primitives for a lens_seq cell."""
        eq_fwd = Equation("store_residual", None, hidden, hidden, nonlinearity="store_residual")
        eq_bwd = Equation("consume_residual", None, hidden, hidden, nonlinearity="consume_residual")
        native_fns = _native_fns_for(backend, eq_fwd, eq_bwd)

        l = _make_lens(hidden, residual, backend)
        composed = morphism.lens_seq(l, l)

        primitives = {}
        bound_terms = {}
        graph = build_graph([hidden])

        register_cells(
            [NamedCell(name="composed_lens", cell=composed)],
            graph, bound_terms, primitives, native_fns, coder, backend,
        )

        fwd_name = Name(f"{MORPHISM_PRIM_PREFIX}composed_lens.forward")
        bwd_name = Name(f"{MORPHISM_PRIM_PREFIX}composed_lens.backward")
        assert fwd_name in primitives
        assert bwd_name in primitives

    def test_forward_primitive_produces_correct_output(self, hidden, residual, coder, backend):
        """The registered forward primitive from a lens_seq cell executes correctly."""
        eq_fwd = Equation("store_residual", None, hidden, hidden, nonlinearity="store_residual")
        eq_bwd = Equation("consume_residual", None, hidden, hidden, nonlinearity="consume_residual")
        native_fns = _native_fns_for(backend, eq_fwd, eq_bwd)

        l = _make_lens(hidden, residual, backend)
        composed = morphism.lens_seq(l, l)

        fwd_name = Name(f"{MORPHISM_PRIM_PREFIX}composed_lens.forward")
        assert fwd_name in native_fns or True  # populated by _register_cells below

        primitives = {}
        bound_terms = {}
        graph = build_graph([hidden])

        register_cells(
            [NamedCell(name="composed_lens", cell=composed)],
            graph, bound_terms, primitives, native_fns, coder, backend,
        )

        fwd_fn = native_fns[fwd_name]
        x = np.array([2.0, 4.0])
        (r1, r2), b = fwd_fn(x)

        np.testing.assert_allclose(r1, x * 2.0)
        np.testing.assert_allclose(r2, (x + 1.0) * 2.0)
        np.testing.assert_allclose(b, x + 2.0)
