"""Type constraint validation helper for assembly."""

from hydra.context import Context
from hydra.dsl.python import FrozenDict, Left
from hydra.unification import unify_type_constraints

_EMPTY_CX = Context(trace=(), messages=(), other=FrozenDict({}))


def unify_or_raise(constraints, schema):
    if constraints:
        st = schema if isinstance(schema, FrozenDict) else FrozenDict(schema)
        result = unify_type_constraints(_EMPTY_CX, st, tuple(constraints))
        if isinstance(result, Left):
            raise TypeError(result.value.message)
