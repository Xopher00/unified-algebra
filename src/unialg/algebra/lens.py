"""Lens declarations: bidirectional morphisms pairing forward and backward equations."""

from __future__ import annotations

import hydra.core as core

from unialg.terms import _RecordView
from unialg.algebra.sort import sort_wrap


class Lens(_RecordView):
    """A bidirectional morphism pairing forward and backward equations.

    Construct:
        l = Lens("enc", "encoder", "decoder")
        l = Lens("optic", "fwd", "bwd", residual_sort=hidden_sort)

    Wrap an existing term:
        l = Lens.from_term(term)
    """

    _type_name = core.Name("ua.lens.Lens")

    name          = _RecordView.Scalar(str)
    forward       = _RecordView.Scalar(str)
    backward      = _RecordView.Scalar(str)
    residual_sort = _RecordView.Term(key="residualSort", optional=True, coerce=sort_wrap)

