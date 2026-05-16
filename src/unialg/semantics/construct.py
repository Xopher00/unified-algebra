"""Semantic construction: parsed expression tree → typed Morphism.

Walks a MorphismExpr tree (placeholder-typed, with Ref nodes) and
reconstructs it through the semantic combinators, producing a fully
typed Morphism ready for compile_program.
"""
from __future__ import annotations

from unialg.syntax import expressions as expr
from unialg.objects import Type, TypeUnit
from . import morphisms as ops
from .morphisms import Morphism, MorphismError
from .functors import Functor, poly_fmap


def construct(
    node: expr.MorphismExpr,
    env: dict[str, Morphism],
    functor_env: dict[str, expr.PolyExpr] | None = None,
) -> Morphism:
    """Resolve a parsed expression tree into a typed Morphism.

    Walks the tree recursively, resolving Ref nodes from ``env`` and
    calling semantic combinators at each binary node.
    """
    match node:
        case expr.Ref(name=name):
            if name not in env:
                raise MorphismError(f"construct: unresolved reference {name!r}")
            return env[name]

        case expr.Prim(raw, dom, cod):
            return Morphism(node=node)

        case expr.Identity(space):
            return ops.identity(space)

        case expr.First(ab):
            return ops._fst(ab)

        case expr.Second(ab):
            return ops._snd(ab)

        case expr.Left(ab):
            return ops._inl(ab)

        case expr.Right(ab):
            return ops._inr(ab)

        case expr.Copy(space):
            return ops._copy(space)

        case expr.Delete(space):
            return ops._delete(space)

        case expr.Absurd(cod):
            return ops.absurd(cod)

        case expr.Assoc(dom=dom):
            return ops._assoc(dom)

        case expr.Symmetry(dom=dom):
            return ops._symmetry(dom)

        case expr.Compose(f=f, g=g):
            return ops.compose(
                construct(f, env, functor_env),
                construct(g, env, functor_env),
                allow_unification=True,
            )

        case expr.Pair(f=f, g=g):
            return ops.pair(
                construct(f, env, functor_env),
                construct(g, env, functor_env),
                allow_unification=True,
            )

        case expr.Parallel(f=f, g=g):
            return ops.par(
                construct(f, env, functor_env),
                construct(g, env, functor_env),
            )

        case expr.Case(f=f, g=g):
            return ops.case(
                construct(f, env, functor_env),
                construct(g, env, functor_env),
                allow_unification=True,
            )

        case expr.PolyFmap(body=body, f=f):
            fenv = functor_env or {}
            if isinstance(body, expr.PolyRef):
                if body.name not in fenv:
                    raise MorphismError(f"construct: unresolved functor {body.name!r}")
                resolved_body = fenv[body.name]
            else:
                resolved_body = body
            functor = Functor(name="anonymous", body=resolved_body)
            return poly_fmap(functor, construct(f, env, functor_env))

        case _:
            raise TypeError(f"construct: unhandled node {type(node).__name__!r}")
