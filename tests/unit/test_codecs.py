"""Roundtrip tests for the type-directed codec layer.

Tests every supported type_from_spec shape. Each test:
  1. Parses the spec string/dict → Hydra Type
  2. Derives a TermCoder via coder_for_type
  3. Encodes a Python value → Term (coder.decode)
  4. Decodes back → Python (coder.encode)
  5. Asserts equality

Also tests strict rejection: wrong Python types must raise TypeError, not coerce.
"""

import pytest

from hydra.dsl.python import Left, Right

from unialg.emitters.codecs import (
    expect_right,
    coder_for_type,
    type_from_spec,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def roundtrip(spec, value):
    """Encode a Python value to a Term and back; return the recovered Python value."""
    typ = type_from_spec(spec)
    coder = coder_for_type(typ)
    term = expect_right(coder.decode(None, value), f"encode {spec}")
    return expect_right(coder.encode(None, None, term), f"decode {spec}")


# ---------------------------------------------------------------------------
# Scalar literals
# ---------------------------------------------------------------------------

def test_float_roundtrip():
    assert roundtrip("FLOAT", 3.14) == pytest.approx(3.14)


def test_float_roundtrip_int_input():
    # int is accepted for FLOAT (numeric widening); result is a float
    result = roundtrip("FLOAT", 2)
    assert result == pytest.approx(2.0)


def test_int_roundtrip():
    assert roundtrip("INT", 42) == 42


def test_string_roundtrip():
    assert roundtrip("STRING", "hello world") == "hello world"


def test_string_empty():
    assert roundtrip("STRING", "") == ""


def test_bool_true_roundtrip():
    result = roundtrip("BOOL", True)
    assert result is True


def test_bool_false_roundtrip():
    result = roundtrip("BOOL", False)
    assert result is False


def test_binary_roundtrip():
    data = b"\x00\xff\xab\xcd"
    assert roundtrip("BINARY", data) == data


def test_unit_roundtrip():
    assert roundtrip("UNIT", None) is None


# ---------------------------------------------------------------------------
# Compound types
# ---------------------------------------------------------------------------

def test_list_of_float_roundtrip():
    values = [1.0, 2.5, -3.0]
    assert roundtrip({"list": "FLOAT"}, values) == pytest.approx(values)


def test_list_of_list_of_float_roundtrip():
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    result = roundtrip({"list": {"list": "FLOAT"}}, matrix)
    assert len(result) == len(matrix)
    for got, want in zip(result, matrix):
        assert got == pytest.approx(want)


def test_list_empty_roundtrip():
    assert roundtrip({"list": "INT"}, []) == []


def test_pair_float_int_roundtrip():
    pair = (1.5, 7)
    result = roundtrip({"pair": ["FLOAT", "INT"]}, pair)
    assert result == (pytest.approx(1.5), 7)


def test_maybe_present_roundtrip():
    result = roundtrip({"maybe": "FLOAT"}, 2.71)
    assert result == pytest.approx(2.71)


def test_maybe_absent_roundtrip():
    assert roundtrip({"maybe": "FLOAT"}, None) is None


def test_either_left_roundtrip():
    value = Left(3.0)
    result = roundtrip({"either": ["FLOAT", "INT"]}, value)
    assert isinstance(result, Left)
    assert result.value == pytest.approx(3.0)


def test_either_right_roundtrip():
    value = Right(99)
    result = roundtrip({"either": ["FLOAT", "INT"]}, value)
    assert isinstance(result, Right)
    assert result.value == 99


# ---------------------------------------------------------------------------
# Strict rejection
# ---------------------------------------------------------------------------

def test_float_rejects_bool():
    coder = coder_for_type(type_from_spec("FLOAT"))
    with pytest.raises(TypeError):
        expect_right(coder.decode(None, True), "encode FLOAT bool")


def test_float_rejects_string():
    coder = coder_for_type(type_from_spec("FLOAT"))
    with pytest.raises(TypeError):
        expect_right(coder.decode(None, "3.14"), "encode FLOAT str")


def test_int_rejects_bool():
    coder = coder_for_type(type_from_spec("INT"))
    with pytest.raises(TypeError):
        expect_right(coder.decode(None, True), "encode INT bool")


def test_int_rejects_float():
    coder = coder_for_type(type_from_spec("INT"))
    with pytest.raises(TypeError):
        expect_right(coder.decode(None, 1.0), "encode INT float")


def test_bool_rejects_int():
    coder = coder_for_type(type_from_spec("BOOL"))
    with pytest.raises(TypeError):
        expect_right(coder.decode(None, 1), "encode BOOL int")


def test_string_rejects_int():
    coder = coder_for_type(type_from_spec("STRING"))
    with pytest.raises(TypeError):
        expect_right(coder.decode(None, 42), "encode STRING int")


def test_binary_rejects_string():
    coder = coder_for_type(type_from_spec("BINARY"))
    with pytest.raises(TypeError):
        expect_right(coder.decode(None, "not bytes"), "encode BINARY str")


def test_unit_rejects_non_none():
    coder = coder_for_type(type_from_spec("UNIT"))
    with pytest.raises(TypeError):
        expect_right(coder.decode(None, 0), "encode UNIT int")


# ---------------------------------------------------------------------------
# type_from_spec parse checks
# ---------------------------------------------------------------------------

def test_type_from_spec_unknown_string():
    with pytest.raises(ValueError, match="unknown shorthand"):
        type_from_spec("COMPLEX")


def test_type_from_spec_unknown_dict():
    with pytest.raises(ValueError, match="unknown spec"):
        type_from_spec({"tuple": ["FLOAT", "INT"]})
