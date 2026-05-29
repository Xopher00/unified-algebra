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
_a4 = TypeVar("_a4")
_a5 = TypeVar("_a5")


def fold_seq(w_in: _, w_rec: _, b: _, s0: _, x: _):
    return hydra.lib.eithers.either(
        (lambda l: s0),
        (
            lambda r: numpy.add(
                numpy.add(
                    numpy.sum(
                        numpy.multiply(
                            numpy.transpose(w_in, (0, 1)),
                            numpy.transpose(
                                numpy.expand_dims(hydra.lib.pairs.first(r), 1), (1, 0)
                            ),
                        ),
                        1,
                    ),
                    numpy.sum(
                        numpy.multiply(
                            numpy.transpose(w_rec, (0, 1)),
                            numpy.transpose(
                                numpy.expand_dims(
                                    fold_seq(
                                        w_in, w_rec, b, s0, hydra.lib.pairs.second(r)
                                    ),
                                    1,
                                ),
                                (1, 0),
                            ),
                        ),
                        1,
                    ),
                ),
                b,
            )
        ),
        x,
    )
