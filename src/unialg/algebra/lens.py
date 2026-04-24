"""Lens declarations: bidirectional morphisms pairing forward and backward equations."""

from __future__ import annotations

import hydra.core as core
from hydra.dsl.meta.phantoms import record, string, unit, TTerm

from unialg.terms import _RecordView, _ScalarField, record_fields


class Lens(_RecordView):
    """A bidirectional morphism pairing forward and backward equations.

    Construct:
        l = Lens("enc", "encoder", "decoder")
        l = Lens("optic", "fwd", "bwd", residual_sort=hidden_sort)

    Wrap an existing term:
        l = Lens.from_term(term)
    """

    _type_name = core.Name("ua.lens.Lens")

    name     = _ScalarField("name", str)
    forward  = _ScalarField("forward", str)
    backward = _ScalarField("backward", str)

    def __init__(self, name: str, forward: str, backward: str, residual_sort=None):
        super().__init__(record(self._type_name, [
            core.Name("name") >> string(name),
            core.Name("forward") >> string(forward),
            core.Name("backward") >> string(backward),
            core.Name("residualSort") >> (TTerm(self._unwrap(residual_sort)) if residual_sort is not None else unit()),
        ]).value)

    @property
    def residual_sort(self):
        rs = record_fields(self._term).get("residualSort")
        if rs is None or isinstance(rs, core.TermUnit):
            return None
        return rs
