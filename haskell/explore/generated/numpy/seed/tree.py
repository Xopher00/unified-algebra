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


def fold_tree(w: _, x: _):
    return hydra.lib.eithers.either(
        (
            lambda l: numpy.sum(
                numpy.multiply(
                    numpy.transpose(w, (0, 1)),
                    numpy.transpose(numpy.expand_dims(l, 1), (1, 0)),
                ),
                1,
            )
        ),
        (
            lambda r: numpy.add(
                fold_tree(w, hydra.lib.pairs.first(r)),
                fold_tree(w, hydra.lib.pairs.second(r)),
            )
        ),
        x,
    )
