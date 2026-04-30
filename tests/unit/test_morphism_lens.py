"""Test lens smart constructor and compilation."""
import numpy as np
import pytest

from hydra.core import Name
import hydra.core as core

from unialg import NumpyBackend, Semiring, Sort, Equation
from unialg.terms import tensor_coder
from unialg.assembly._typed_morphism import TypedMorphism as T
from unialg.assembly._morphism_compile import compile_morphism, CompiledLens
from unialg.assembly.graph import (
    NamedCell, MORPHISM_PRIM_PREFIX, _register_cells, build_graph,
)
from unialg.assembly._equation_resolution import resolve_equation
import unialg.assembly.morphism as morphism


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


class TestLensConstruction:

    def test_domain_is_forward_domain(self, hidden):
        prod = T.product(hidden, hidden)
        fwd = morphism.eq("f", domain=hidden, codomain=prod)
        bwd = morphism.eq("g", domain=prod, codomain=hidden)
        m = morphism.lens(fwd, bwd)
        assert m.domain is hidden
        assert m.codomain is hidden

    def test_term_is_record(self, hidden):
        prod = T.product(hidden, hidden)
        fwd = morphism.eq("f", domain=hidden, codomain=prod)
        bwd = morphism.eq("g", domain=prod, codomain=hidden)
        m = morphism.lens(fwd, bwd)
        assert isinstance(m.term, core.TermRecord)


class TestLensValidation:

    def test_non_product_forward_codomain_rejected(self, hidden):
        fwd = morphism.eq("f", domain=hidden, codomain=hidden)
        bwd = morphism.eq("g", domain=hidden, codomain=hidden)
        with pytest.raises(TypeError, match="TypePair"):
            morphism.lens(fwd, bwd)

    def test_mismatched_residual_rejected(self, hidden, real_sr):
        base = Sort("base", real_sr)
        prod_hh = T.product(hidden, hidden)
        prod_bh = T.product(base, hidden)
        fwd = morphism.eq("f", domain=hidden, codomain=prod_hh)
        bwd = morphism.eq("g", domain=prod_bh, codomain=hidden)
        with pytest.raises(TypeError, match="lens.residual"):
            morphism.lens(fwd, bwd)

    def test_non_morphism_rejected(self, hidden):
        with pytest.raises(TypeError):
            morphism.lens("not a morphism", morphism.iden(hidden))


class TestLensCompilation:

    def test_compiles_to_compiled_lens(self, hidden, graph, coder, backend):
        backend.unary_ops["id_op"] = lambda x: x
        eq_id = Equation("id_op", None, hidden, hidden, nonlinearity="id_op")
        native_fns = _native_fns(backend, eq_id)
        prod = T.product(hidden, hidden)
        fwd = morphism.eq("id_op", domain=hidden, codomain=prod)
        bwd = morphism.eq("id_op", domain=prod, codomain=hidden)
        m = morphism.lens(fwd, bwd)
        result = compile_morphism(m, graph, native_fns, coder, backend)
        assert isinstance(result, CompiledLens)
        assert callable(result.forward)
        assert callable(result.backward)

    def test_lens_registers_two_primitives(self, hidden, coder, backend):
        backend.unary_ops["id_op"] = lambda x: x
        eq_id = Equation("id_op", None, hidden, hidden, nonlinearity="id_op")
        native_fns = _native_fns(backend, eq_id)
        primitives, compiled_fns = {}, {}
        graph = build_graph([hidden])
        prod = T.product(hidden, hidden)
        fwd = morphism.eq("id_op", domain=hidden, codomain=prod)
        bwd = morphism.eq("id_op", domain=prod, codomain=hidden)

        _register_cells(
            [NamedCell(name="my_lens", cell=morphism.lens(fwd, bwd))],
            graph, primitives, native_fns, compiled_fns, coder, backend,
        )

        assert Name(f"{MORPHISM_PRIM_PREFIX}my_lens.forward") in primitives
        assert Name(f"{MORPHISM_PRIM_PREFIX}my_lens.backward") in primitives
        assert isinstance(compiled_fns["my_lens"], CompiledLens)
