from hypothesis import strategies as st

from hydra.core import (
    EitherType,
    IntegerType,
    LiteralTypeInteger,
    PairType,
    TypeEither,
    TypeList,
    TypeLiteral,
    TypeMaybe,
    TypePair,
    TypeUnit,
    TypeVoid,
)

from unialg.syntax import expressions as expr
from unialg.semantics import morphisms as ops
from unialg.semantics.functors import Functor, apply_poly
from unialg.semantics.optics import Optic
from unialg.objects import LIST, MAYBE


INT = TypeLiteral(LiteralTypeInteger(IntegerType.INT32))
UNIT = TypeUnit()
VOID = TypeVoid()


def type_values(max_leaves: int = 5):
    """Generate small Hydra Type trees used by the current semantic core."""
    leaves = st.sampled_from([INT, UNIT, VOID])
    return st.recursive(
        leaves,
        lambda children: st.one_of(
            st.builds(lambda a, b: TypePair(PairType(a, b)), children, children),
            st.builds(lambda a, b: TypeEither(EitherType(a, b)), children, children),
            st.builds(TypeMaybe, children),
            st.builds(TypeList, children),
        ),
        max_leaves=max_leaves,
    )


def product_type_values():
    return st.builds(lambda a, b: TypePair(PairType(a, b)), type_values(), type_values())


def sum_type_values():
    return st.builds(lambda a, b: TypeEither(EitherType(a, b)), type_values(), type_values())


def poly_values(max_leaves: int = 6):
    """Generate small polynomial functor expression trees."""
    leaves = st.one_of(
        st.just(expr.Zero()),
        st.just(expr.One()),
        st.just(expr.Id()),
        type_values().map(expr.Const),
    )
    return st.recursive(
        leaves,
        lambda children: st.one_of(
            st.builds(expr.Sum, children, children),
            st.builds(expr.Prod, children, children),
            st.builds(expr.Exp, type_values(), children),
        ),
        max_leaves=max_leaves,
    )


def primitive_morphism_nodes():
    return st.one_of(
        type_values().map(expr.Identity),
        type_values().map(expr.Copy),
        type_values().map(expr.Delete),
        product_type_values().map(expr.First),
        product_type_values().map(expr.Second),
        sum_type_values().map(expr.Left),
        sum_type_values().map(expr.Right),
        type_values().map(expr.Absurd),
        st.builds(expr.Prim, st.just(object()), type_values(), type_values()),
    )


def plain_morphisms():
    return st.builds(lambda dom, cod: ops.Morphism(expr.Prim(object(), dom, cod)), type_values(), type_values())


def parametric_plain_morphisms():
    return st.builds(
        lambda param, dom, cod: ops.Morphism(
            expr.Prim(object(), ops.ProductType(param, dom), cod),
            param=param,
        ),
        type_values(),
        type_values(),
        type_values(),
    )


def maybe_lax_morphisms():
    return st.builds(
        lambda dom, cod: ops.Morphism(
            expr.Prim(object(), dom, MAYBE.wrap(cod)),
            monad=MAYBE,
        ),
        type_values(),
        type_values(),
    )


def list_lax_morphisms():
    return st.builds(
        lambda dom, cod: ops.Morphism(
            expr.Prim(object(), dom, LIST.wrap(cod)),
            monad=LIST,
        ),
        type_values(),
        type_values(),
    )


def composable_plain_pairs():
    return st.builds(
        lambda a, b, c: (
            ops.Morphism(expr.Prim(object(), a, b)),
            ops.Morphism(expr.Prim(object(), b, c)),
        ),
        type_values(),
        type_values(),
        type_values(),
    )


def same_domain_plain_pairs():
    return st.builds(
        lambda a, b, c: (
            ops.Morphism(expr.Prim(object(), a, b)),
            ops.Morphism(expr.Prim(object(), a, c)),
        ),
        type_values(),
        type_values(),
        type_values(),
    )


def same_codomain_plain_pairs():
    return st.builds(
        lambda a, b, c: (
            ops.Morphism(expr.Prim(object(), a, c)),
            ops.Morphism(expr.Prim(object(), b, c)),
        ),
        type_values(),
        type_values(),
        type_values(),
    )


def optic_values():
    """Generate valid Optic objects with arbitrary Id-bearing polynomial functors."""
    id_bearing = poly_values().filter(lambda b: count_id(b) > 0)
    return st.builds(
        lambda body, s, a, b, t: Optic(
            functor=Functor("_", body),
            forward=ops.Morphism(expr.Prim(object(), s, apply_poly(body, a))),
            backward=ops.Morphism(expr.Prim(object(), apply_poly(body, b), t)),
        ),
        id_bearing, type_values(), type_values(), type_values(), type_values(),
    )


def flatten_sum(node: expr.PolyExpr) -> tuple[expr.PolyExpr, ...]:
    if isinstance(node, expr.Sum):
        return flatten_sum(node.left) + flatten_sum(node.right)
    return (node,)


def count_id(node: expr.PolyExpr) -> int:
    if isinstance(node, expr.Id):
        return 1
    if isinstance(node, (expr.Sum, expr.Prod)):
        return count_id(node.left) + count_id(node.right)
    if isinstance(node, expr.Exp):
        return count_id(node.body)
    return 0


def collect_consts(node: expr.PolyExpr):
    if isinstance(node, expr.Const):
        return [node.space]
    if isinstance(node, (expr.Sum, expr.Prod)):
        return collect_consts(node.left) + collect_consts(node.right)
    if isinstance(node, expr.Exp):
        return [node.base] + collect_consts(node.body)
    return []
