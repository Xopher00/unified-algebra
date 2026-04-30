"""Morphism registration and graph assembly tests.

Tests that typed morphisms compile and register as Hydra primitives
in the assembly graph. Lenses produce two primitives (forward + backward).
"""
import numpy as np
import pytest

from hydra.core import Name

from unialg import Sort, Equation
import unialg.assembly.morphism as morphism
from unialg.assembly._typed_morphism import TypedMorphism as T
from unialg.assembly.graph import (
    NamedCell, MORPHISM_PRIM_PREFIX, _register_cells, build_graph,
)
from unialg.assembly._morphism_compile import CompiledLens
from unialg.assembly._equation_resolution import resolve_equation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_sort(real_sr):
    return Sort("base", real_sr)


def _native_fns(backend, *eqs):
    fns = {}
    for eq_obj in eqs:
        _, native_fn, *_ = resolve_equation(eq_obj, backend)
        fns[Name(f"ua.equation.{eq_obj.name}")] = native_fn
    return fns


# ---------------------------------------------------------------------------
# _register_cells
# ---------------------------------------------------------------------------

class TestRegisterCells:

    def test_registers_morphism_into_dicts(self, hidden, real_sr, backend, coder):
        eq_step = Equation("rg_step", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_step)
        primitives, compiled_fns = {}, {}
        graph = build_graph([hidden])

        _register_cells(
            [NamedCell(name="my_cell",
                       cell=morphism.eq("rg_step", domain=hidden, codomain=hidden))],
            graph, primitives, native_fns, compiled_fns, coder, backend,
        )

        assert Name(f"{MORPHISM_PRIM_PREFIX}my_cell") in primitives
        assert "my_cell" in compiled_fns
        assert Name("ua.equation.my_cell") in primitives

    def test_missing_equation_raises(self, hidden, backend, coder):
        primitives, compiled_fns = {}, {}
        graph = build_graph([])
        with pytest.raises(ValueError):
            _register_cells(
                [NamedCell(name="dead",
                           cell=morphism.eq("ghost", domain=hidden, codomain=hidden))],
                graph, primitives, {}, compiled_fns, coder, backend,
            )

    def test_lens_registers_forward_and_backward(self, hidden, backend, coder):
        backend.unary_ops["bg_id"] = lambda x: x
        eq_id = Equation("bg_id_eq", None, hidden, hidden, nonlinearity="bg_id")
        native_fns = _native_fns(backend, eq_id)
        primitives, compiled_fns = {}, {}
        graph = build_graph([hidden])

        prod_sort = T.product(hidden, hidden)
        fwd = morphism.eq("bg_id_eq", domain=hidden, codomain=prod_sort)
        bwd = morphism.eq("bg_id_eq", domain=prod_sort, codomain=hidden)

        _register_cells(
            [NamedCell(name="some_lens", cell=morphism.lens(fwd, bwd))],
            graph, primitives, native_fns, compiled_fns, coder, backend,
        )

        assert Name(f"{MORPHISM_PRIM_PREFIX}some_lens.forward") in primitives
        assert Name(f"{MORPHISM_PRIM_PREFIX}some_lens.backward") in primitives
        assert "some_lens" in compiled_fns
        assert isinstance(compiled_fns["some_lens"], CompiledLens)
        assert Name("ua.equation.some_lens") in primitives


# ---------------------------------------------------------------------------
# End-to-end via assemble_graph
# ---------------------------------------------------------------------------

class TestAssembleGraphWithCells:

    def test_cell_registered_and_runnable(self, hidden, real_sr, backend, cx, coder):
        from unialg.assembly.graph import assemble_graph
        from hydra.dsl.terms import apply, var
        from conftest import encode_array, decode_term, assert_reduce_ok

        backend.unary_ops["ag_halve"]  = lambda x: 0.5 * x
        backend.unary_ops["ag_double"] = lambda x: 2.0 * x
        eq_h = Equation("ag_h", None, hidden, hidden, nonlinearity="ag_halve")
        eq_d = Equation("ag_d", None, hidden, hidden, nonlinearity="ag_double")

        named = NamedCell(
            name="ident_via_cells",
            cell=morphism.seq(
                morphism.eq("ag_h", domain=hidden, codomain=hidden),
                morphism.eq("ag_d", domain=hidden, codomain=hidden),
            ),
        )

        graph, native_fns, compiled_fns = assemble_graph(
            [eq_h, eq_d], backend, cells=[named],
        )

        assert "ident_via_cells" in compiled_fns
        out = compiled_fns["ident_via_cells"](np.array([4.0]))
        np.testing.assert_allclose(out, np.array([4.0]))

        x_enc = encode_array(coder, np.array([4.0]))
        out_term = assert_reduce_ok(
            cx, graph, apply(var("ua.equation.ident_via_cells"), x_enc),
        )
        np.testing.assert_allclose(decode_term(coder, out_term), np.array([4.0]))

    def test_validation_rejects_unknown_eq(self, hidden, real_sr, backend):
        from unialg.assembly.graph import assemble_graph

        eq_step = Equation("ag_v_step", "i,i->i", hidden, hidden, real_sr)
        named = NamedCell(
            name="bad",
            cell=morphism.eq("ghost", domain=hidden, codomain=hidden),
        )

        with pytest.raises(ValueError):
            assemble_graph([eq_step], backend, cells=[named])
