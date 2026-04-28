"""Cell ↔ Hydra Graph bridge.

Wraps a compiled Cell into a Hydra Primitive suitable for registration in
the assembly graph. Used by ``assemble_graph`` during the migration from
Spec-based compositions to Cell-based ones.

Para Cells produce a single primitive with the compiled callable as its
compute function. Optic Cells (CompiledLens) currently raise — bidirectional
graph registration is a follow-on once the lens migration begins.
"""
from __future__ import annotations

from dataclasses import dataclass

import hydra.core as core
from hydra.dsl.prims import prim1

from unialg.assembly._para import Cell
from unialg.assembly._para_runtime import (
    CompiledLens, CompiledMorphism, Matcher, compile_cell,
)


CELL_PRIM_PREFIX = "ua.cell."


@dataclass(frozen=True)
class NamedCell:
    """A named Cell expression destined for graph registration."""
    name: str
    cell: Cell
    matchers: dict[str, Matcher] | None = None


def compile_cell_to_primitive(
    named: NamedCell,
    native_fns: dict,
    coder,
    backend,
) -> tuple[object | None, CompiledMorphism | None]:
    """Compile a NamedCell into ``(Primitive, compiled_fn)``.

    Mirrors the ``Composition.resolve`` protocol: returns ``(None, None)`` if
    any referenced Equation is missing from ``native_fns``.

    Lens-flavoured Cells return a ``CompiledLens`` from the interpreter; this
    function does not yet register them as graph primitives. Callers should
    treat Optic Cells as a separate registration concern until the lens
    migration lands.
    """
    fn = compile_cell(named.cell, native_fns, coder, backend, matchers=named.matchers)
    if fn is None:
        return None, None
    if isinstance(fn, CompiledLens):
        raise NotImplementedError(
            f"compile_cell_to_primitive: Optic Cell {named.name!r} produced a "
            f"CompiledLens; bidirectional graph registration is not yet wired."
        )
    prim_name = core.Name(f"{CELL_PRIM_PREFIX}{named.name}")
    return prim1(prim_name, fn, [], coder, coder), fn


def register_named_cells(
    named_cells: list[NamedCell],
    primitives: dict,
    native_fns: dict,
    compiled_fns: dict,
    coder,
    backend,
) -> None:
    """Compile each NamedCell and write its primitive into ``primitives``.

    Names that fail to compile (missing equations) are skipped silently —
    the caller is responsible for any fall-back behaviour. ``compiled_fns``
    is updated with the compiled callable for each registered cell, keyed by
    the user-facing name (no ``ua.cell.`` prefix), matching how
    ``_build_compositions`` records spec-level compiled functions.
    """
    for named in named_cells:
        prim, fn = compile_cell_to_primitive(named, native_fns, coder, backend)
        if prim is None:
            continue
        primitives[prim.name] = prim
        compiled_fns[named.name] = fn
        native_fns[prim.name] = fn
        eq_alias = core.Name(f"ua.equation.{named.name}")
        primitives.setdefault(eq_alias, prim1(eq_alias, fn, [], coder, coder))
        native_fns.setdefault(eq_alias, fn)
