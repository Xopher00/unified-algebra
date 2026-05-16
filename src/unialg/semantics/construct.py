"""Semantic construction: parsed expression tree → typed Morphism.

Walks a MorphismExpr tree (placeholder-typed, with Ref nodes) and
reconstructs it through the semantic combinators, producing a fully
typed Morphism ready for compile_program.
"""
from __future__ import annotations

from dataclasses import dataclass

from unialg.syntax import expressions as expr
from unialg.syntax.parse import Program
from . import morphisms as ops
from .morphisms import Morphism, MorphismError
from .functors import Functor, poly_fmap


@dataclass(frozen=True)
class ConstructedProgram:
    """A parsed program after semantic route construction."""

    routes: dict[str, Morphism]
    functors: dict[str, expr.PolyExpr]


def _route_refs(node: expr.MorphismExpr) -> set[str]:
    """Collect route references from a parsed morphism expression."""
    match node:
        case expr.Ref(name=name):
            return {name}
        case expr.MonadicEmbed(f=f):
            return _route_refs(f)
        case expr.ContextualBinary(f=f, g=g):
            return _route_refs(f) | _route_refs(g)
        case expr.PolyFmap(f=f):
            return _route_refs(f)
        case expr.MorphismApp(fun=fun, args=args):
            refs = _route_refs(fun)
            for arg in args:
                refs |= _route_refs(arg)
            return refs
        case expr.AlgExpr(body=body):
            return _route_refs(body)
        case _:
            return set()


def construct_program(
    program: Program,
    env: dict[str, Morphism] | None = None,
) -> ConstructedProgram:
    """Resolve every parsed route in ``program`` into a typed ``Morphism``.

    ``parse_program`` deliberately leaves names as ``Ref`` nodes. This pass
    gives those names semantic meaning by resolving them against backend/builtin
    morphisms in ``env`` and against other routes in the same program.
    """
    if program.route_params:
        names = ", ".join(sorted(program.route_params))
        raise MorphismError(
            f"construct_program: parametric routes are not wired yet: {names}"
        )

    base_env = dict(env or {})
    routes: dict[str, Morphism] = {}
    constructing: list[str] = []

    def resolve_route(name: str) -> Morphism:
        if name in routes:
            return routes[name]
        if name not in program.routes:
            if name in base_env:
                return base_env[name]
            raise MorphismError(f"construct_program: unresolved route {name!r}")
        if name in constructing:
            cycle = " -> ".join((*constructing, name))
            raise MorphismError(f"construct_program: cyclic route reference: {cycle}")

        constructing.append(name)
        route_env = dict(base_env)
        for ref in sorted(_route_refs(program.routes[name])):
            if ref in program.routes:
                route_env[ref] = resolve_route(ref)
        route_env.update(routes)
        route = construct(program.routes[name], route_env, program.functors)
        routes[name] = route
        constructing.pop()
        return route

    for name in program.routes:
        resolve_route(name)

    return ConstructedProgram(routes=routes, functors=dict(program.functors))


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
