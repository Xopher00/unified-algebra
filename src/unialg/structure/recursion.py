from __future__ import annotations

from collections.abc import Callable

import hydra.dsl.meta.phantoms as P
from hydra.dsl.python import Right
import hydra.dsl.terms as Terms
import hydra.reduction as R
from hydra.graph import Primitive

import hydra.rewriting as RW

from unialg.objects import ExpType, Name, ProductType, SumType, Type, TypeList, TypeScheme, TypeUnit
from unialg.syntax import expressions as expr
from . import terms as H
from . import actions
from unialg.semantics.functors import Functor
from unialg.semantics.morphisms import Morphism, MorphismError, compose, raw_signature
from unialg.semantics.optics import Optic, fixed_point_optic, identity_optic
from . import realize

# IDENTITY = "hydra.lib.equality.identity"

# def _is_identity_primitive(term) -> bool:
#     # Adapt field names to Hydra-Python's generated Term/Function wrappers.
#     # Intended shape: TermFunction(FunctionPrimitive(Name("hydra.lib.equality.identity")))
#     return (
#         getattr(term, "name", None) == IDENTITY
#         or str(getattr(term, "name", "")) == IDENTITY
#         or IDENTITY in repr(term)
#     )

# def _rewrite_identity_apps(recurse, term):
#     term = recurse(term)

#     # Intended Hydra AST shape:
#     # apply(identity, x) -> x
#     fun = getattr(term, "function", None)
#     arg = getattr(term, "argument", None)

#     if fun is not None and arg is not None and _is_identity_primitive(fun):
#         return arg

#     return term

# def normalize_recursive_body(term):
#     return RW.rewrite_term(_rewrite_identity_apps, term)


def act_forward(t: Optic, h: Morphism) -> Morphism:
    """Decompose through an optic, then lift ``h`` through the optic functor."""
    return compose(t.forward, actions.poly_fmap(t.functor, h))


def act_backward(t: Optic, h: Morphism) -> Morphism:
    """Lift ``h`` through the optic functor, then reconstruct through the optic."""
    return compose(actions.poly_fmap(t.functor, h), t.backward)


def act(t: Optic, h: Morphism) -> Morphism:
    """Apply an optic action to ``h``.

    Composition: ``S --forward--> F(A) --actions.poly_fmap(F,h)--> F(B) --backward--> T``.
    If ``h`` is lax, plain optic boundaries are lifted into the same monad by the
    morphism composition rules.
    """
    return compose(act_forward(t, h), t.backward)


def compose_optic(outer: Optic, inner: Optic) -> Optic:
    """Compose two optics: focus through ``outer`` then ``inner``."""
    composed_functor = outer.functor.compose(inner.functor)
    fwd = act_forward(outer, inner.forward)
    bwd = act_backward(outer, inner.backward)
    return Optic(functor=composed_functor, forward=fwd, backward=bwd)


def recursive_carrier(*, functor: Functor, carrier: Type, unroll, roll) -> Optic:
    return fixed_point_optic(
        functor=functor,
        carrier=carrier,
        unroll=H.term_lambda("x", unroll).value,
        roll=H.term_lambda("layer", roll).value,
    )


def _require_carrier(fp: Optic) -> Type:
    if fp.carrier is None:
        raise MorphismError("recursive optic must define carrier")
    return fp.carrier


def _recursive_morphism(name: str, dom: Type, cod: Type, context: Morphism,
    build: Callable[[Morphism], Morphism]) -> Morphism:
    prim_name = Name(name)
    raw_dom, raw_cod = raw_signature(context.param, context.monad, dom, cod)
    self_ref = Morphism(
        expr.Prim(P.primitive(prim_name).value, raw_dom, raw_cod),
        param=context.param,
        monad=context.monad,
    )
    body = build(self_ref)
    MorphismError.check(body.param, context.param, "recursive body param mismatch")
    if body.monad != context.monad:
        raise MorphismError(f"recursive body monad mismatch: {body.monad!r} != {context.monad!r}")

    def impl(ctx, graph, args):
        term = Terms.apply(raw_body.value, args[0])
        result = R.reduce_term(ctx, graph, True, term)
        if isinstance(result, Right):
            return result
        raise RuntimeError(f"{name} reduction failed: {result!r}")

    prim = Primitive(
        prim_name,
        TypeScheme(variables=(), body=ExpType(raw_dom, raw_cod), constraints=None),
        impl,
    )
    raw_body, morphism = actions.primitive_from_raw(realize.realize_term(body.node), raw_dom, raw_cod, body, (prim,))
    return morphism


def cata(fp: Optic, alg: Morphism) -> Morphism:
    carrier = _require_carrier(fp)
    expected_dom = fp.functor.apply(alg.cod())
    MorphismError.check(alg.dom(), expected_dom, "cata algebra has wrong shape")
    return _recursive_morphism(
        name=f"unialg.cata.{id(fp):x}.{id(alg):x}",
        dom=carrier,
        cod=alg.cod(),
        context=alg,
        build=lambda self_ref: compose(
            act_forward(fp, self_ref),
            alg,
            shared_context=True,
        ),
    )


def ana(fp: Optic, coalg: Morphism) -> Morphism:
    carrier = _require_carrier(fp)
    expected_cod = fp.functor.apply(coalg.dom())
    MorphismError.check(coalg.cod(), expected_cod, "ana coalgebra has wrong shape")
    return _recursive_morphism(
        name=f"unialg.ana.{id(fp):x}.{id(coalg):x}",
        dom=coalg.dom(),
        cod=carrier,
        context=coalg,
        build=lambda self_ref: compose(
            coalg,
            act_backward(fp, self_ref),
            shared_context=True,
        ),
    )


def hylo(fp: Optic, coalg: Morphism, alg: Morphism) -> Morphism:
    """Hylomorphism: unfold with coalg then fold with alg."""
    return compose(ana(fp, coalg), cata(fp, alg), shared_context=True)
