from __future__ import annotations

from collections.abc import Callable

import hydra.dsl.meta.phantoms as P
from hydra.dsl.python import Right
import hydra.dsl.terms as Terms
import hydra.reduction as R
from hydra.graph import Primitive

from unialg.objects import ExpType, Name, Type, TypeScheme
from unialg.syntax import expressions as expr
from . import terms as T
from unialg.semantics.functors import Functor
from unialg.semantics.morphisms import Morphism, MorphismError, compose, raw_signature
from unialg.semantics.optics import Optic, fixed_point_optic
from . import realize


# def recursive_carrier(*, functor: Functor, carrier: Type, unroll, roll) -> Optic:
#     return fixed_point_optic(
#         functor=functor,
#         carrier=carrier,
#         unroll=T.term_lambda("x", unroll).value,
#         roll=T.term_lambda("layer", roll).value,
#     )


# def _require_carrier(fp: Optic) -> Type:
#     if fp.carrier is None:
#         raise MorphismError("recursive optic must define carrier")
#     return fp.carrier


# def _recursive_morphism(name: str, dom: Type, cod: Type, context: Morphism,
#     build: Callable[[Morphism], Morphism]) -> Morphism:
#     prim_name = Name(name)
#     raw_dom, raw_cod = raw_signature(context.param, context.monad, dom, cod)
#     self_ref = Morphism(
#         expr.Prim(P.primitive(prim_name).value, raw_dom, raw_cod),
#         param=context.param,
#         monad=context.monad,
#     )
#     body = build(self_ref)
#     MorphismError.check(body.param, context.param, "recursive body param mismatch")
#     if body.monad != context.monad:
#         raise MorphismError(f"recursive body monad mismatch: {body.monad!r} != {context.monad!r}")

#     def impl(ctx, graph, args):
#         term = Terms.apply(raw_body.value, args[0])
#         result = R.reduce_term(ctx, graph, True, term)
#         if isinstance(result, Right):
#             return result
#         raise RuntimeError(f"{name} reduction failed: {result!r}")

#     prim = Primitive(
#         prim_name,
#         TypeScheme(variables=(), body=ExpType(raw_dom, raw_cod), constraints=None),
#         impl,
#     )
#     raw_body, morphism = realize.primitive_from_raw(realize.realize_term(body.node), raw_dom, raw_cod, body, (prim,))
#     return morphism


# def cata(fp: Optic, alg: Morphism) -> Morphism:
#     carrier = _require_carrier(fp)
#     expected_dom = fp.functor.apply(alg.cod())
#     MorphismError.check(alg.dom(), expected_dom, "cata algebra has wrong shape")
#     return _recursive_morphism(
#         name=f"unialg.cata.{id(fp):x}.{id(alg):x}",
#         dom=carrier,
#         cod=alg.cod(),
#         context=alg,
#         build=lambda self_ref: compose(
#             fp.act_forward(self_ref),
#             alg,
#             shared_context=True,
#         ),
#     )


# def ana(fp: Optic, coalg: Morphism) -> Morphism:
#     carrier = _require_carrier(fp)
#     expected_cod = fp.functor.apply(coalg.dom())
#     MorphismError.check(coalg.cod(), expected_cod, "ana coalgebra has wrong shape")
#     return _recursive_morphism(
#         name=f"unialg.ana.{id(fp):x}.{id(coalg):x}",
#         dom=coalg.dom(),
#         cod=carrier,
#         context=coalg,
#         build=lambda self_ref: compose(
#             coalg,
#             fp.act_backward(self_ref),
#             shared_context=True,
#         ),
#     )


# def hylo(fp: Optic, coalg: Morphism, alg: Morphism) -> Morphism:
#     """Hylomorphism: unfold with coalg then fold with alg."""
#     return compose(ana(fp, coalg), cata(fp, alg), shared_context=True)



def fixed_point_optic(*, functor: Functor, carrier: Type, unroll, roll) -> Optic:
    """Shim: structural fixed-point optic constructor. Retained for review."""
    layer = functor.apply(carrier)
    return Optic(
        functor=functor,
        forward=Morphism(expr.Prim(unroll, carrier, layer)),
        backward=Morphism(expr.Prim(roll, layer, carrier)),
        carrier=carrier,
    )


def recursive_carrier(*, functor, carrier, unroll, roll) -> Optic:
    return fixed_point_optic(
        functor=functor, carrier=carrier,
        unroll=T.term_lambda("x", unroll).value,
        roll=T.term_lambda("layer", roll).value,
    )
