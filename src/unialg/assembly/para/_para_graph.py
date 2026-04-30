"""Morphism ↔ Hydra Graph bridge.

Wraps a compiled Cell or migrated morphism term into a Hydra Primitive
suitable for registration in the assembly graph. Used by ``assemble_graph``
during the migration from Cell-based compositions to Hydra-term morphisms.

Morphism expressions produce a single primitive with the compiled callable as
its compute function. Optic Cells (CompiledLens) currently raise —
bidirectional graph registration is a follow-on once the lens migration
begins.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import hydra.core as core
from hydra.dsl.prims import prim1
from hydra.lexical import empty_graph

from unialg.assembly._morphism_compile import compile_morphism


CELL_PRIM_PREFIX = "ua.cell."
CompiledMorphism: TypeAlias = Callable | object
Matcher: TypeAlias = Callable[[object], tuple[int, list, list]]


@dataclass(frozen=True)
class NamedCell:
    """A named Cell or morphism expression destined for graph registration."""
    name: str
    cell: object
    matchers: dict[str, Matcher] | None = None


def _missing_legacy_equation(term, native_fns: dict) -> str | None:
    """Return the first missing Equation reference in a legacy Cell tree."""
    from unialg.assembly.para._para import Cell
    from unialg.assembly._typed_morphism import TypedMorphism

    if isinstance(term, TypedMorphism):
        term = term.term
    if not isinstance(term, Cell):
        return None
    if term.kind == "eq":
        name = term.equation_name
        return None if core.Name(f"ua.equation.{name}") in native_fns else name
    if term.kind in {"seq", "par"}:
        return (
            _missing_legacy_equation(term.left, native_fns)
            or _missing_legacy_equation(term.right, native_fns)
        )
    if term.kind == "lens":
        return (
            _missing_legacy_equation(term.forward, native_fns)
            or _missing_legacy_equation(term.backward, native_fns)
        )
    if term.kind == "algebraHom":
        for child in term.cells:
            missing = _missing_legacy_equation(child, native_fns)
            if missing is not None:
                return missing
    return None


def compile_cell_to_primitive(
    named: NamedCell,
    graph,
    native_fns: dict | None = None,
    coder=None,
    backend=None,
) -> tuple[object | None, CompiledMorphism | None]:
    """Compile a NamedCell / morphism into ``(Primitive, compiled_fn)``.

    ``graph`` is a preliminary ``hydra.graph.Graph`` built before cell
    registration. It is threaded to ``compile_morphism`` so that Hydra
    graph-aware extractors can dereference and strip annotated terms.

    Raises ``ValueError`` if a referenced equation is missing from
    ``native_fns`` (propagated from ``compile_morphism``).

    Lens-flavoured Cells return a ``CompiledLens`` from the interpreter; this
    function does not yet register them as graph primitives. Callers should
    treat Optic Cells as a separate registration concern until the lens
    migration lands.
    """
    if backend is None:
        # Backwards-compatible direct-call form:
        # compile_cell_to_primitive(named, native_fns, coder, backend)
        backend = coder
        coder = native_fns
        native_fns = graph
        graph = empty_graph()
    fn = compile_morphism(named.cell, graph, native_fns, coder, backend, matchers=named.matchers)
    if fn is None:
        missing = _missing_legacy_equation(named.cell, native_fns)
        if missing is not None:
            raise ValueError(f"unknown equation {missing!r}")
        return None, None
    if not callable(fn):
        raise NotImplementedError(
            f"compile_cell_to_primitive: {named.name!r} produced non-callable "
            f"{type(fn).__name__}; bidirectional graph registration is not yet wired."
        )
    prim_name = core.Name(f"{CELL_PRIM_PREFIX}{named.name}")
    return prim1(prim_name, fn, [], coder, coder), fn


def register_named_cells(
    named_cells: list[NamedCell],
    graph=None,
    primitives: dict | None = None,
    native_fns: dict | None = None,
    compiled_fns: dict | None = None,
    coder=None,
    backend=None,
) -> None:
    """Compile each NamedCell and write its primitive into ``primitives``.

    ``graph`` is a preliminary ``hydra.graph.Graph`` passed through to
    ``compile_cell_to_primitive`` and onward to ``compile_morphism`` so that
    graph-aware extractors are available during compilation.

    Raises ``ValueError`` if a NamedCell references an equation that is absent
    from ``native_fns`` — this is a hard error, not a silent skip. The old
    silent-skip behaviour was masking misconfiguration; callers that want
    partial assembly should catch ``ValueError`` explicitly.

    ``compiled_fns`` is updated with the compiled callable for each registered
    cell, keyed by the user-facing name (no ``ua.cell.`` prefix), matching how
    ``_build_compositions`` records spec-level compiled functions.
    """
    if graph is None:
        graph = empty_graph()
    if primitives is None:
        primitives = {}
    if native_fns is None:
        native_fns = {}
    if compiled_fns is None:
        compiled_fns = {}
    for named in named_cells:
        prim, fn = compile_cell_to_primitive(named, graph, native_fns, coder, backend)
        if prim is None:
            continue
        primitives[prim.name] = prim
        compiled_fns[named.name] = fn
        native_fns[prim.name] = fn
        eq_alias = core.Name(f"ua.equation.{named.name}")
        primitives.setdefault(eq_alias, prim1(eq_alias, fn, [], coder, coder))
        native_fns.setdefault(eq_alias, fn)
