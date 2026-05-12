from unialg.semantics.morphisms import _identity, pair, compose, Morphism
from unialg.syntax import expressions as expr


def const_float(value, typ):
    raw = P.constant(P.float64(float(value))).value
    return Morphism(expr.Prim(raw, typ, typ))

def section(op, const, value, *, side="right"):
    types = [op.dom().value.first, op.dom().value.second]
    fixed = {"left": 0, "right": 1}[side]
    open_ = 1 - fixed

    parts = [None, None]
    parts[fixed] = const(value, types[fixed])
    parts[open_] = _identity(types[open_])

    return compose(pair(*parts), op)

add1 = section(ops["add"], const_float, 1.0, side="right")   # x ↦ x + 1
mul2 = section(ops["multiply"], const_float, 2.0, side="left")  # x ↦ 2 * x

add_then_double = compose(add1, mul2)
