"""Cell ↔ Graph bridge: validate_cell + compile_cell_to_primitive.

Covers the two pieces of step 1: structural validation of Cell expressions
against an equation/sort scope, and compilation of a NamedCell into a Hydra
Primitive ready for graph registration.
"""
import numpy as np
import pytest

from hydra.core import Name
import hydra.core as core

from unialg import NumpyBackend, Sort, Equation
from unialg.assembly.functor import (
    Functor, sum_, prod, one, id_, const,
)
from unialg.assembly._para import (
    Cell, eq, lit, seq, par, copy, delete, lens, algebra_hom,
    validate_cell, CompiledLens,
)
from unialg.assembly._para_graph import (
    NamedCell, CELL_PRIM_PREFIX,
    compile_cell_to_primitive, register_named_cells,
)
from unialg.assembly._equation_resolution import resolve_equation


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
# validate_cell
# ---------------------------------------------------------------------------

class TestValidateCellEqRefs:

    def test_known_eq_passes(self):
        validate_cell(eq("foo"), eq_by_name={"foo": object()})

    def test_unknown_eq_rejected(self):
        with pytest.raises(ValueError, match="unknown equation 'ghost'"):
            validate_cell(eq("ghost"), eq_by_name={})

    def test_nested_eq_rejected_with_path(self):
        with pytest.raises(ValueError, match=r"seq\.right: eq references unknown"):
            validate_cell(seq(eq("a"), eq("b")), eq_by_name={"a": object()})


class TestValidateCellSortRefs:

    def test_copy_with_known_sort_passes(self, base_sort):
        validate_cell(copy(base_sort),
                     eq_by_name={}, sorts_by_name={"base": base_sort})

    def test_copy_with_unknown_sort_rejected(self, base_sort):
        with pytest.raises(ValueError, match="copy references unknown sort 'base'"):
            validate_cell(copy(base_sort),
                         eq_by_name={}, sorts_by_name={"other": base_sort})

    def test_delete_with_unknown_sort_rejected(self, base_sort):
        with pytest.raises(ValueError, match="delete references unknown sort"):
            validate_cell(delete(base_sort),
                         eq_by_name={}, sorts_by_name={})

    def test_no_sorts_table_skips_sort_check(self, base_sort):
        # When sorts_by_name is None, sort references aren't validated.
        validate_cell(copy(base_sort), eq_by_name={}, sorts_by_name=None)


class TestValidateCellLens:

    def test_lens_height1_passes(self):
        validate_cell(lens(eq("g"), eq("p")),
                     eq_by_name={"g": 1, "p": 2})

    def test_lens_with_residual_known_passes(self, base_sort):
        validate_cell(
            lens(eq("g"), eq("p"), residual=base_sort),
            eq_by_name={"g": 1, "p": 2},
            sorts_by_name={"base": base_sort},
        )

    def test_lens_residual_unknown_rejected(self, base_sort):
        with pytest.raises(ValueError, match="lens residual references unknown sort"):
            validate_cell(
                lens(eq("g"), eq("p"), residual=base_sort),
                eq_by_name={"g": 1, "p": 2},
                sorts_by_name={"other": base_sort},
            )

    def test_lens_inner_eq_unknown_rejected(self):
        with pytest.raises(ValueError, match=r"lens\.fwd: eq references unknown"):
            validate_cell(lens(eq("ghost"), eq("p")),
                         eq_by_name={"p": 1})


class TestValidateCellAlgebraHom:

    def test_algebra_hom_passes(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        c = algebra_hom(f, "algebra", [
            lit(core.TermLiteral(value=core.LiteralFloat(value=1.0))),
            eq("step"),
        ])
        validate_cell(c,
                     eq_by_name={"step": object()},
                     sorts_by_name={"base": base_sort})

    def test_algebra_hom_unknown_functor_sort_rejected(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        c = algebra_hom(f, "algebra", [
            lit(core.TermLiteral(value=core.LiteralFloat(value=0.0))),
            eq("step"),
        ])
        with pytest.raises(ValueError, match="functor 'F_list' references unknown sorts"):
            validate_cell(c,
                         eq_by_name={"step": object()},
                         sorts_by_name={"only_other": base_sort})

    def test_algebra_hom_unknown_inner_eq_rejected(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        c = algebra_hom(f, "algebra", [
            lit(core.TermLiteral(value=core.LiteralFloat(value=0.0))),
            eq("ghost"),
        ])
        with pytest.raises(ValueError, match=r"cell\[1\]: eq references unknown"):
            validate_cell(c,
                         eq_by_name={},
                         sorts_by_name={"base": base_sort})


# ---------------------------------------------------------------------------
# compile_cell_to_primitive
# ---------------------------------------------------------------------------

class TestCompileCellToPrimitive:

    def test_eq_named_cell_compiles_to_primitive(self, hidden, real_sr, backend, coder):
        eq_step = Equation("g_step", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_step)

        named = NamedCell(name="step_cell", cell=eq("g_step"))
        prim, fn = compile_cell_to_primitive(named, native_fns, coder, backend)

        assert prim is not None
        assert prim.name == Name(f"{CELL_PRIM_PREFIX}step_cell")
        assert fn is not None
        out = fn(np.array([2.0]), np.array([3.0]))
        np.testing.assert_allclose(out, np.array([6.0]))

    def test_missing_equation_returns_none(self, backend, coder):
        named = NamedCell(name="dead", cell=eq("nope"))
        prim, fn = compile_cell_to_primitive(named, {}, coder, backend)
        assert prim is None
        assert fn is None

    def test_seq_compiles(self, hidden, backend, coder):
        backend.unary_ops["bg_halve"]  = lambda x: 0.5 * x
        backend.unary_ops["bg_double"] = lambda x: 2.0 * x
        eq_h = Equation("bg_h", None, hidden, hidden, nonlinearity="bg_halve")
        eq_d = Equation("bg_d", None, hidden, hidden, nonlinearity="bg_double")
        native_fns = _native_fns(backend, eq_h, eq_d)

        named = NamedCell(name="round_trip", cell=seq(eq("bg_h"), eq("bg_d")))
        prim, fn = compile_cell_to_primitive(named, native_fns, coder, backend)
        assert prim is not None
        np.testing.assert_allclose(fn(np.array([4.0])), np.array([4.0]))

    def test_optic_cell_raises_not_implemented(self, hidden, backend, coder):
        backend.unary_ops["bg_id"] = lambda x: x
        eq_id = Equation("bg_id_eq", None, hidden, hidden, nonlinearity="bg_id")
        native_fns = _native_fns(backend, eq_id)

        named = NamedCell(name="some_lens",
                          cell=lens(eq("bg_id_eq"), eq("bg_id_eq")))
        with pytest.raises(NotImplementedError, match="Optic Cell"):
            compile_cell_to_primitive(named, native_fns, coder, backend)


# ---------------------------------------------------------------------------
# register_named_cells
# ---------------------------------------------------------------------------

class TestRegisterNamedCells:

    def test_registers_into_dicts(self, hidden, real_sr, backend, coder):
        eq_step = Equation("rg_step", "i,i->i", hidden, hidden, real_sr)
        native_fns = _native_fns(backend, eq_step)
        primitives, compiled_fns = {}, {}

        register_named_cells(
            [NamedCell(name="my_cell", cell=eq("rg_step"))],
            primitives=primitives,
            native_fns=native_fns,
            compiled_fns=compiled_fns,
            coder=coder,
            backend=backend,
        )

        assert Name(f"{CELL_PRIM_PREFIX}my_cell") in primitives
        assert "my_cell" in compiled_fns
        # Also exposed under ua.equation.{name} so downstream code can use it
        # uniformly with directly-resolved equations.
        assert Name("ua.equation.my_cell") in primitives

    def test_failed_compile_skipped(self, backend, coder):
        primitives, compiled_fns = {}, {}
        # Cell references nothing that exists; compile returns None and we skip.
        register_named_cells(
            [NamedCell(name="dead", cell=eq("ghost"))],
            primitives=primitives,
            native_fns={},
            compiled_fns=compiled_fns,
            coder=coder,
            backend=backend,
        )
        assert primitives == {}
        assert compiled_fns == {}


# ---------------------------------------------------------------------------
# End-to-end via assemble_graph
# ---------------------------------------------------------------------------

class TestAssembleGraphWithCells:
    """Pass NamedCell entries through assemble_graph alongside equations."""

    def test_cell_registered_and_runnable(self, hidden, real_sr, backend, cx, coder):
        from unialg.assembly.graph import assemble_graph
        from hydra.dsl.terms import apply, var
        from conftest import encode_array, decode_term, assert_reduce_ok

        # Two equations: halve, double.
        backend.unary_ops["ag_halve"]  = lambda x: 0.5 * x
        backend.unary_ops["ag_double"] = lambda x: 2.0 * x
        eq_h = Equation("ag_h", None, hidden, hidden, nonlinearity="ag_halve")
        eq_d = Equation("ag_d", None, hidden, hidden, nonlinearity="ag_double")

        # Cell composing halve ; double = identity.
        named = NamedCell(name="ident_via_cells",
                          cell=seq(eq("ag_h"), eq("ag_d")))

        graph, native_fns, compiled_fns = assemble_graph(
            [eq_h, eq_d], backend, cells=[named],
        )

        # Compiled fast-path is exposed by name.
        assert "ident_via_cells" in compiled_fns
        out = compiled_fns["ident_via_cells"](np.array([4.0]))
        np.testing.assert_allclose(out, np.array([4.0]))

        # The cell is registered as a Hydra primitive — reduce_term works.
        x_enc = encode_array(coder, np.array([4.0]))
        out_term = assert_reduce_ok(
            cx, graph, apply(var("ua.equation.ident_via_cells"), x_enc),
        )
        np.testing.assert_allclose(decode_term(coder, out_term), np.array([4.0]))

    def test_cell_validation_rejects_unknown_eq(self, hidden, real_sr, backend):
        from unialg.assembly.graph import assemble_graph

        eq_step = Equation("ag_v_step", "i,i->i", hidden, hidden, real_sr)
        named = NamedCell(name="bad", cell=eq("ghost"))

        with pytest.raises(ValueError, match="unknown equation 'ghost'"):
            assemble_graph([eq_step], backend, cells=[named])

    def test_algebra_hom_cell_via_graph(self, hidden, real_sr, backend, coder):
        """End-to-end: declare F + cells, register via assemble_graph, run."""
        from conftest import encode_array
        from unialg.assembly.graph import assemble_graph

        def list_matcher(seq_value):
            if not seq_value:
                return (0, [], [])
            head, *tail = seq_value
            return (1, [tail], [head])

        eq_step = Equation("ag_mul", "i,i->i", hidden, hidden, real_sr)

        f = Functor("F_list", sum_(one(), prod(const(hidden), id_())))
        init_term = encode_array(coder, np.array([1.0]))
        cell = algebra_hom(f, "algebra", [lit(init_term), eq("ag_mul")])

        named = NamedCell(name="list_prod", cell=cell,
                          matchers={"F_list": list_matcher})

        _, _, compiled_fns = assemble_graph(
            [eq_step], backend, cells=[named], extra_sorts=[hidden],
        )

        out = compiled_fns["list_prod"]([np.array([2.0]),
                                          np.array([3.0]),
                                          np.array([4.0])])
        np.testing.assert_allclose(out, np.array([24.0]))
