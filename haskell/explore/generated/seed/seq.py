# Note: this is an automatically generated file. Do not edit.
r"""Recursive definition."""
from __future__ import annotations
from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar, cast
import hydra.core
import hydra.lib.eithers
import hydra.lib.pairs
import numpy

_a0 = TypeVar("_a0")
_a1 = TypeVar("_a1")
_a2 = TypeVar("_a2")
_a3 = TypeVar("_a3")


def fold_seq(w: _, s0: _, x: _):
    return hydra.lib.eithers.either(
        (lambda l: s0),
        (
            lambda r: numpy.add(
                numpy.multiply(w, hydra.lib.pairs.first(r)),
                fold_seq(w, s0, hydra.lib.pairs.second(r)),
            )
        ),
        x,
    )
