"""Tests for load_program: parse + backend primitive resolution.

Verifies that `load <backend>` directives bind real Prim nodes (not Ref
placeholders) so parsed morphisms are fully resolved and ready to lower.
"""
from unialg.main import load_program
from unialg.syntax.expressions import Compose, Prim, Ref


def test_load_numpy_resolves_add():
    prog, backends = load_program("load numpy\nroute plus = add")
    assert "numpy" in backends
    assert isinstance(prog.morphisms["plus"], Prim), (
        "Expected Prim (real backend node), got Ref — add not resolved"
    )


def test_load_numpy_compose_resolves():
    prog, _ = load_program("load numpy\nroute f = add >> multiply")
    f = prog.morphisms["f"]
    assert isinstance(f, Compose)
    assert isinstance(f.f, Prim)
    assert isinstance(f.g, Prim)


def test_load_program_returns_backends():
    _, backends = load_program("load numpy")
    assert "numpy" in backends
    assert "add" in backends["numpy"]
    assert "multiply" in backends["numpy"]


def test_without_load_aliases_are_refs():
    prog, _ = load_program("route f = add")
    assert isinstance(prog.morphisms["f"], Ref)


def test_load_records_in_program():
    prog, _ = load_program("load numpy\nroute f = add")
    assert prog.loads == ("numpy",)
