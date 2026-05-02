"""Test compile_morphism: term shapes compile to correct runtime callables."""
import numpy as np
import pytest

from hydra.core import Name
import hydra.core as core

from unialg import NumpyBackend, Semiring, Sort, Equation
from unialg.terms import tensor_coder
from unialg.morphism._typed_morphism import TypedMorphism as T
from unialg.assembly._morphism_compile import compile_morphism, CompiledLens
from unialg.assembly.graph import build_graph
from unialg.assembly._equation_resolution import resolve_equation
import unialg.morphism as morphism


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
def coder(backend):
    return tensor_coder(backend)

@pytest.fixture
def graph(hidden):
    return build_graph([hidden])


def _native_fns(backend, *eqs):
    fns = {}
    for eq_obj in eqs:
        _, native_fn, *_ = resolve_equation(eq_obj, backend)
        fns[Name(f"ua.equation.{eq_obj.name}")] = native_fn
    return fns


class TestCompileIden:
    def test_returns_callable(self, hidden, graph, coder, backend):
        fn = compile_morphism(morphism.iden(hidden), graph, {}, coder, backend)
        assert callable(fn)

    def test_identity_behavior(self, hidden, graph, coder, backend):
        fn = compile_morphism(morphism.iden(hidden), graph, {}, coder, backend)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(fn(x), x)


class TestCompileCopy:
    def test_returns_pair(self, hidden, graph, coder, backend):
        fn = compile_morphism(morphism.copy(hidden), graph, {}, coder, backend)
        x = np.array([1.0, 2.0])
        a, b = fn(x)
        np.testing.assert_array_equal(a, x)
        np.testing.assert_array_equal(b, x)


class TestCompileDelete:
    def test_returns_none(self, hidden, graph, coder, backend):
        fn = compile_morphism(morphism.delete(hidden), graph, {}, coder, backend)
        assert fn(np.array([1.0])) is None


class TestCompileLit:
    def test_returns_constant(self, hidden, graph, coder, backend):
        v = core.TermLiteral(core.LiteralFloat(core.FloatValueFloat32(42.0)))
        fn = compile_morphism(morphism.lit(v, hidden), graph, {}, coder, backend)
        assert fn("anything").value == pytest.approx(42.0)


class TestCompileEq:
    def test_equation_callable(self, hidden, real_sr, graph, coder, backend):
        eq_mul = Equation("mul", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_mul)
        m = morphism.eq("mul", domain=hidden, codomain=hidden)
        fn = compile_morphism(m, graph, native_fns, coder, backend)
        x, y = np.array([2.0]), np.array([3.0])
        np.testing.assert_allclose(fn(x, y), np.array([6.0]))


class TestCompileSeq:
    def test_seq_composes(self, hidden, graph, coder, backend):
        backend.unary_ops["halve"] = lambda x: 0.5 * x
        backend.unary_ops["double"] = lambda x: 2.0 * x
        eq_h = Equation("h", None, hidden, hidden, nonlinearity="halve")
        eq_d = Equation("d", None, hidden, hidden, nonlinearity="double")
        native_fns = _native_fns(backend, eq_h, eq_d)
        m = morphism.seq(
            morphism.eq("h", domain=hidden, codomain=hidden),
            morphism.eq("d", domain=hidden, codomain=hidden),
        )
        fn = compile_morphism(m, graph, native_fns, coder, backend)
        x = np.array([4.0])
        np.testing.assert_allclose(fn(x), np.array([4.0]))


class TestCompilePar:
    @pytest.mark.xfail(reason="par falls through to Hydra reduce_term — fast-path matcher pending")
    def test_par_applies_to_pair(self, hidden, graph, coder, backend):
        backend.unary_ops["halve"] = lambda x: 0.5 * x
        backend.unary_ops["double"] = lambda x: 2.0 * x
        eq_h = Equation("h", None, hidden, hidden, nonlinearity="halve")
        eq_d = Equation("d", None, hidden, hidden, nonlinearity="double")
        native_fns = _native_fns(backend, eq_h, eq_d)
        m = morphism.par(
            morphism.eq("h", domain=hidden, codomain=hidden),
            morphism.eq("d", domain=hidden, codomain=hidden),
        )
        fn = compile_morphism(m, graph, native_fns, coder, backend)
        x, y = np.array([4.0]), np.array([3.0])
        a, b = fn((x, y))
        np.testing.assert_allclose(a, np.array([2.0]))
        np.testing.assert_allclose(b, np.array([6.0]))


class TestCompileLens:
    def test_lens_produces_compiled_lens(self, hidden, graph, coder, backend):
        backend.unary_ops["id_op"] = lambda x: x
        eq_id = Equation("id_op", None, hidden, hidden, nonlinearity="id_op")
        native_fns = _native_fns(backend, eq_id)
        prod_sort = T.product(hidden, hidden)
        fwd = morphism.eq("id_op", domain=hidden, codomain=prod_sort)
        bwd = morphism.eq("id_op", domain=prod_sort, codomain=hidden)
        m = morphism.lens(fwd, bwd)
        result = compile_morphism(m, graph, native_fns, coder, backend)
        assert isinstance(result, CompiledLens)
        assert callable(result.forward)
        assert callable(result.backward)
