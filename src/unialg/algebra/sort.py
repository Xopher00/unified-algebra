"""Sorts and lenses: named tensor types and bidirectional morphisms.

A sort is a named space of tensors (e.g. "hidden", "output") bound to a
semiring. Two sorts are compatible iff they share the same semiring.

Sorts map to Hydra TypeVariables for type checking, and sort declarations
are registered in the Graph's schema_types.

A lens pairs forward and backward equations into a bidirectional morphism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra.core as core
import hydra.graph
from hydra.dsl.python import Nothing

from unialg.terms import _RecordView, tensor_coder
from .semiring import Semiring

if TYPE_CHECKING:
    import hydra.errors


class Sort(_RecordView):
    """A named tensor sort bound to a semiring.

    Construct:
        s = Sort("hidden", real_sr)
        s = Sort("batch_hidden", real_sr, batched=True)

    Wrap an existing term:
        s = Sort.from_term(term)
    """

    _type_name = core.Name("ua.sort.Sort")
    name     = _RecordView.Scalar(str)
    semiring = _RecordView.Term(coerce=Semiring.from_term)
    batched  = _RecordView.Scalar(bool, default=False)
    axes     = _RecordView.ScalarList(default=())

    @staticmethod
    def _parse_axis(s: str) -> tuple[str, int | None]:
        if ':' in s:
            name, size = s.rsplit(':', 1)
            return name, int(size)
        return s, None

    @property
    def rank(self) -> int | None:
        ax = self.axes
        return len(ax) if ax else None

    @property
    def axis_names(self) -> list[str]:
        return [self._parse_axis(a)[0] for a in self.axes]

    @property
    def axis_dims(self) -> list[int | None]:
        return [self._parse_axis(a)[1] for a in self.axes]

    @property
    def semiring_name(self) -> str:
        return self.semiring.name

    @property
    def type_(self) -> core.Type:
        base = core.TypeApplication(core.ApplicationType(
            core.TypeVariable(core.Name(f"ua.sort.{self.name}")),
            core.TypeVariable(core.Name(f"ua.semiring.{self.semiring_name}"))))
        if self.batched:
            return core.TypeApplication(core.ApplicationType(
                core.TypeVariable(core.Name("ua.batched")), base))
        return base

    def coder(self, backend) -> hydra.graph.TermCoder:
        return tensor_coder(backend, type_=self.type_)

    def register_schema(self, schema: dict) -> None:
        for name_str in (f"ua.sort.{self.name}", f"ua.semiring.{self.semiring_name}", "ua.batched"):
            n = core.Name(name_str)
            schema[n] = core.TypeScheme((), core.TypeVariable(n), Nothing())


# ---------------------------------------------------------------------------
# ProductSort class
# ---------------------------------------------------------------------------

class ProductSort(_RecordView):
    """A typed monoidal product of sorts.

    ProductSort represents the typed product domain for n-ary einsums and
    Cell ``par`` codomains. It is **monoidal-only**, not cartesian:

    - No projections (``fst``/``snd``/``π_i``).
    - No destructuring or pattern matching.
    - No pairing laws.
    - No copy/delete semantics (those live on Cell variants ``copy`` / ``delete``).

    The ``type_`` property emits a right-nested ``core.TypePair`` chain;
    ``coder`` builds a right-nested ``hydra.dsl.prims.pair`` coder. Runtime
    values are plain Python tuples.

    If projection is needed, lower to ``hydra.lib.pairs.first`` /
    ``hydra.lib.pairs.second`` (registered Hydra primitives) at the Cell-leaf
    level. Do not invent Cell-level projection variants.

    See ``ARCHITECTURE.md`` § "Hydra ↔ unified-algebra boundary" for context.

    Construct:
        ps = ProductSort([hidden_sort, output_sort])

    Wrap an existing term:
        ps = ProductSort.from_term(term)
    """

    _type_name = core.Name("ua.sort.Product")
    elements = _RecordView.TermList(key="sorts", coerce=lambda t: sort_wrap(t))

    def __init__(self, elements):
        if len(elements) < 2:
            raise ValueError("Product sort requires at least 2 component sorts")
        super().__init__(elements=elements)

    @property
    def type_(self) -> core.Type:
        types = [e.type_ for e in self.elements]
        result = types[-1]
        for t in reversed(types[:-1]):
            result = core.TypePair(core.PairType(first=t, second=result))
        return result

    def coder(self, backend):
        from hydra.dsl.prims import pair as pair_coder
        coders = [e.coder(backend) for e in self.elements]
        result = coders[-1]
        for c in reversed(coders[:-1]):
            result = pair_coder(c, result)
        return result

    def register_schema(self, schema: dict) -> None:
        for elem in self.elements:
            elem.register_schema(schema)


_sort_cache: dict[int, "Sort | ProductSort"] = {}

def sort_wrap(term) -> "Sort | ProductSort":
    """Return Sort or ProductSort wrapper for a raw sort term."""
    if isinstance(term, _RecordView):
        return term
    key = id(term)
    cached = _sort_cache.get(key)
    if cached is not None and cached._term is term:
        return cached
    if term.value.type_name == ProductSort._type_name:
        result = ProductSort.from_term(term)
    else:
        result = Sort.from_term(term)
    _sort_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Semiring compatibility
# ---------------------------------------------------------------------------


def check_sort_compatibility(sort_a, sort_b) -> bool:
    """Return True iff two sorts share the same semiring."""
    def _semiring(t):
        app = t.value
        if app.function == core.TypeVariable(core.Name("ua.batched")):
            app = app.argument.value
        return app.argument
    return _semiring(sort_a.type_) == _semiring(sort_b.type_)


# ---------------------------------------------------------------------------
# Lens
# ---------------------------------------------------------------------------

class Lens(_RecordView):
    """A bidirectional morphism pairing forward and backward equations."""

    _type_name = core.Name("ua.lens.Lens")

    name          = _RecordView.Scalar(str)
    forward       = _RecordView.Scalar(str)
    backward      = _RecordView.Scalar(str)
    residual_sort = _RecordView.Term(key="residualSort", optional=True, coerce=sort_wrap)


