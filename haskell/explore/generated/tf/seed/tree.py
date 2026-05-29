# Note: this is an automatically generated file. Do not edit.
r"""Recursive definition."""
from __future__ import annotations
from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar, cast
import hydra.core
import hydra.lib.eithers
import hydra.lib.pairs
import tensorflow
import tensorflow.math

_a0 = TypeVar("_a0")
_a1 = TypeVar("_a1")
_a2 = TypeVar("_a2")


def fold_tree(w: _, x: _):
    return hydra.lib.eithers.either(
        (
            lambda l: tensorflow.math.reduce_sum(
                tensorflow.math.multiply(
                    tensorflow.transpose(w, (0, 1)),
                    tensorflow.transpose(tensorflow.expand_dims(l, 1), (1, 0)),
                ),
                1,
            )
        ),
        (
            lambda r: tensorflow.math.add(
                fold_tree(w, hydra.lib.pairs.first(r)),
                fold_tree(w, hydra.lib.pairs.second(r)),
            )
        ),
        x,
    )
