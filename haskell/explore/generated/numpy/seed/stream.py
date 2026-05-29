# Note: this is an automatically generated file. Do not edit.
r"""Recursive definition."""
from __future__ import annotations
from functools import lru_cache
from typing import TypeVar, cast
import hydra.core
import hydra.lib.pairs

_a0 = TypeVar("_a0")
_a1 = TypeVar("_a1")


def unfold_stream(x: _):
    return (hydra.lib.pairs.first(x), unfold_stream(hydra.lib.pairs.second(x)))
