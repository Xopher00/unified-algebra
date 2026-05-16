"""Semantic construction: parsed expression tree → typed Morphism.

Walks a MorphismExpr tree (placeholder-typed, with Ref nodes) and
reconstructs it through the semantic combinators, producing a fully
typed Morphism ready for compile_program.
"""
from __future__ import annotations

from dataclasses import dataclass

from unialg.syntax import expressions as expr
from unialg.syntax.parse import Program
import hydra.lexical as L

from . import morphisms as ops
from . import typeops as Ty
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

    base_env = dict(env or {})
    routes: dict[str, Morphism] = {}
    constructing: list[str] = []
    _cx = [L.empty_context()]

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
            if ref in program.routes and ref not in program.route_params:
                route_env[ref] = resolve_route(ref)
        route_env.update(routes)
        route = construct(program.routes[name], route_env, program.functors, program.routes, program.route_params, _cx)
        routes[name] = route
        constructing.pop()
        return route

    for name in program.routes:
        if name not in program.route_params:
            resolve_route(name)

    return ConstructedProgram(routes=routes, functors=dict(program.functors))


def construct(
    node: expr.MorphismExpr,
    env: dict[str, Morphism],
    functor_env: dict[str, expr.PolyExpr] | None = None,
    route_bodies: dict[str, expr.MorphismExpr] | None = None,
    route_params: dict[str, tuple[str, ...]] | None = None,
    _cx: list | None = None,
) -> Morphism:
    """Resolve a parsed expression tree into a typed Morphism.

    Walks the tree recursively, resolving Ref nodes from ``env`` and
    calling semantic combinators at each binary node.
    """
    if _cx is None:
        _cx = [L.empty_context()]

    def _recurse(n):
        return construct(n, env, functor_env, route_bodies, route_params, _cx)

    from unialg.objects import TypeUnit, ProductType, SumType

    def _fresh():
        var, new_cx = Ty.fresh_variable_type(_cx[0])
        _cx[0] = new_cx
        return var

    def _fresh_pair():
        return ProductType(_fresh(), _fresh())

    def _fresh_sum():
        return SumType(_fresh(), _fresh())

    match node:
        case expr.Ref(name=name):
            if name not in env:
                raise MorphismError(f"construct: unresolved reference {name!r}")
            return env[name]

        case expr.Prim(raw, dom, cod):
            return Morphism(node=node)

        case expr.Identity(space):
            return ops.identity(space if space != TypeUnit() else _fresh())

        case expr.First(ab):
            return ops._fst(ab if ab != ProductType(TypeUnit(), TypeUnit()) else _fresh_pair())

        case expr.Second(ab):
            return ops._snd(ab if ab != ProductType(TypeUnit(), TypeUnit()) else _fresh_pair())

        case expr.Left(ab):
            return ops._inl(ab if ab != SumType(TypeUnit(), TypeUnit()) else _fresh_sum())

        case expr.Right(ab):
            return ops._inr(ab if ab != SumType(TypeUnit(), TypeUnit()) else _fresh_sum())

        case expr.Copy(space):
            return ops._copy(space if space != TypeUnit() else _fresh())

        case expr.Delete(space):
            return ops._delete(space if space != TypeUnit() else _fresh())

        case expr.Absurd(cod):
            return ops.absurd(cod if cod != TypeUnit() else _fresh())

        case expr.Assoc(dom=dom):
            return ops._assoc(dom)

        case expr.Symmetry(dom=dom):
            return ops._symmetry(dom)

        case expr.SharedCompose(f=f, g=g):
            return ops.compose(
                _recurse(f), _recurse(g),
                shared_context=True, allow_unification=True,
            )

        case expr.Compose(f=f, g=g):
            return ops.compose(
                _recurse(f), _recurse(g),
                allow_unification=True,
            )

        case expr.Pair(f=f, g=g):
            return ops.pair(
                _recurse(f), _recurse(g),
                allow_unification=True,
            )

        case expr.Parallel(f=f, g=g):
            return ops.par(_recurse(f), _recurse(g))

        case expr.Case(f=f, g=g):
            return ops.case(
                _recurse(f), _recurse(g),
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
            return poly_fmap(functor, _recurse(f))

        case expr.MorphismApp(fun=fun, args=args):
            bodies = route_bodies or {}
            params = route_params or {}
            # Parametric route: substitute args into body
            if isinstance(fun, expr.Ref) and fun.name in params:
                body = bodies[fun.name]
                declared_params = params[fun.name]
                if len(args) != len(declared_params):
                    raise MorphismError(
                        f"construct: {fun.name} expects {len(declared_params)} "
                        f"params, got {len(args)}"
                    )
                local_env = dict(env)
                for pname, arg_node in zip(declared_params, args):
                    local_env[pname] = _recurse(arg_node)
                return construct(body, local_env, functor_env, route_bodies, route_params, _cx)
            # Backend primitive with arity > 1: populate args
            resolved_fun = _recurse(fun)
            if isinstance(resolved_fun.node, expr.BackendPrim) and resolved_fun.node.arity > 1:
                bp = resolved_fun.node
                if len(args) != bp.arity:
                    raise MorphismError(
                        f"construct: {fun!r} expects {bp.arity} args, got {len(args)}"
                    )
                resolved_args = [_recurse(a) for a in args]
                all_aux = resolved_fun.aux_primitives
                for ra in resolved_args:
                    all_aux = all_aux + ra.aux_primitives
                applied_dom = resolved_args[0].dom()
                return Morphism(
                    node=expr.BackendPrim(
                        bp.primitive, bp.arity, applied_dom, bp.cod,
                        args=tuple(ra.node for ra in resolved_args),
                    ),
                    aux_primitives=all_aux,
                )
            raise MorphismError(
                f"construct: cannot apply {fun!r} (not parametric, not arity > 1)"
            )

        case _:
            raise TypeError(f"construct: unhandled node {type(node).__name__!r}")
