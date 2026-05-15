"""Tests for load_program: parse + backend primitive resolution.

Verifies that `load <backend>` directives bind real Prim nodes (not Ref
placeholders) so parsed morphisms are fully resolved and ready to lower.
"""
import pytest
import numpy as np

from unialg.main import compile_program, load_program
from unialg.syntax.expressions import Compose, Prim


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


def test_without_load_unresolved_name_is_parse_error():
    from unialg.syntax.parse import ParseError
    with pytest.raises(ParseError, match="unresolved references"):
        load_program("route f = add")


def test_load_records_in_program():
    prog, _ = load_program("load numpy\nroute f = add")
    assert prog.loads == ("numpy",)



def test_compile_program_runs_structural_unit_program():
    src = """
    map Nat = 1 | x
    route zero = ! >> |0
    route successor = |1
    route one = zero >> successor
    route two = one >> Nat{successor}
    route three = two >> Nat{Nat{successor}}
    route count = three
    """
    assert compile_program(src).run() == ("right", ("right", ("right", ("left", None))))


def test_loaded_binary_backend_does_not_decode_structural_output():
    src = """
    load numpy
    map Nat = 1 | x
    route zero = ! >> |0
    route successor = |1
    route one = zero >> successor
    route two = one >> Nat{successor}
    route pred = zero | id
    route result = two >> pred
    """
    assert compile_program(src).run() == ("right", ("left", None))


def test_loaded_binary_backend_decodes_binary_output():
    dot = compile_program("""
    load numpy
    route dot = multiply >> reduce.add
    """)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert np.isclose(dot.run(a, b), np.dot(a, b))
