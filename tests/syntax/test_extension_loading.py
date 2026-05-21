import pytest

from unialg.extensions import enable, is_enabled, registered_keywords


def test_enable_registers_keywords():
    enable("tensors")
    assert "algebra" in registered_keywords()
    assert is_enabled("tensors")


def test_enable_is_idempotent():
    enable("tensors")
    enable("tensors")


def test_load_extension_dsl():
    from unialg.syntax.parse import parse_program
    prog = parse_program(
        "load extension tensors\nalgebra A(plus=f, times=g, zero=z, one=o)"
    )
    assert len(prog.extensions.get("tensors", [])) == 1


def test_unknown_extension_raises():
    with pytest.raises((ImportError, AttributeError, ModuleNotFoundError)):
        enable("nonexistent_xyz")
