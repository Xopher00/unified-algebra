"""Semantic construction: parsed expression tree → typed Morphism.

Walks a MorphismExpr tree (placeholder-typed, with Ref nodes) and
reconstructs it through the semantic combinators, producing a fully
typed Morphism ready for compile_program.
"""
from __future__ import annotations

from dataclasses import dataclass

from unialg.syntax import expressions as expr
from unialg.syntax.parse import Program, poly_to_focus_expr
import hydra.lexical as L

from . import morphisms as ops
from . import typeops as Ty
from .morphisms import Morphism, MorphismError
from .functors import Functor, poly_fmap
from .optics import Optic, RecursiveCarrier, ana, cata, hylo, recursive_carrier
from unialg.objects import monad_by_name


@dataclass(frozen=True)
class ConstructedProgram:
    """A parsed program after semantic morphism construction."""

    morphisms: dict[str, Morphism]
    functors: dict[str, expr.PolyExpr]
    carriers: dict[str, RecursiveCarrier]
    focuses: dict[str, Optic]


def _morphism_refs(node: expr.MorphismExpr) -> set[str]:
    """Collect morphism references from a parsed morphism expression."""
    match node:
        case expr.Ref(name=name):
            return {name}
        case (
            expr.Identity()
            | expr.Copy()
            | expr.Delete()
            | expr.First()
            | expr.Second()
            | expr.Left()
            | expr.Right()
            | expr.Absurd()
            | expr.Assoc()
            | expr.Symmetry()
            | expr.Prim()
            | expr.SelfRef()
            | expr.CarrierBoundary()
        ):
            return set()
        case expr.BackendPrim(args=args):
            refs = set()
            for arg in args:
                refs |= _morphism_refs(arg)
            return refs
        case expr.MonadicEmbed(f=f):
            return _morphism_refs(f)
        case expr.ContextualBinary(f=f, g=g):
            return _morphism_refs(f) | _morphism_refs(g)
        case expr.PolyFmap(f=f):
            return _morphism_refs(f)
        case expr.MorphismApp(fun=fun, args=args):
            refs = _morphism_refs(fun)
            for arg in args:
                refs |= _morphism_refs(arg)
            return refs
        case expr.RecursionApp(args=args):
            refs = set()
            for arg in args:
                refs |= _morphism_refs(arg)
            return refs
        case expr.MonadicLift(body=body):
            return _morphism_refs(body)
        case expr.AlgExpr(body=body):
            return _morphism_refs(body)
        case _:
            raise TypeError(f"_morphism_refs: unknown MorphismExpr {type(node).__name__!r}")


def _resolve_poly_refs(
    node: expr.PolyExpr,
    functors: dict[str, expr.PolyExpr],
    stack: tuple[str, ...] = (),
) -> expr.PolyExpr:
    """Inline named functor references inside a polynomial expression."""
    match node:
        case expr.PolyRef(name=name):
            if name not in functors:
                raise MorphismError(f"construct_program: unresolved functor {name!r}")
            if name in stack:
                cycle = " -> ".join((*stack, name))
                raise MorphismError(f"construct_program: cyclic functor reference: {cycle}")
            return _resolve_poly_refs(functors[name], functors, (*stack, name))
        case expr.Prod(left=left, right=right):
            return expr.Prod(
                _resolve_poly_refs(left, functors, stack),
                _resolve_poly_refs(right, functors, stack),
            )
        case expr.Sum(left=left, right=right):
            return expr.Sum(
                _resolve_poly_refs(left, functors, stack),
                _resolve_poly_refs(right, functors, stack),
            )
        case expr.PolyCompose(left=left, right=right):
            return expr.PolyCompose(
                _resolve_poly_refs(left, functors, stack),
                _resolve_poly_refs(right, functors, stack),
            )
        case expr.Exp(base=base, body=body):
            return expr.Exp(base, _resolve_poly_refs(body, functors, stack))
        case expr.List(body=body):
            return expr.List(_resolve_poly_refs(body, functors, stack))
        case expr.Maybe(body=body):
            return expr.Maybe(_resolve_poly_refs(body, functors, stack))
        case expr.Zero() | expr.One() | expr.Id() | expr.Const():
            return node
        case _:
            raise TypeError(f"construct_program: unknown PolyExpr {type(node).__name__!r}")


def _focus_alias_candidate(node: expr.PolyExpr) -> bool:
    """Return True when ``node`` could be an optic alias expression."""
    match node:
        case expr.PolyRef():
            return True
        case expr.PolyCompose(left=left, right=right):
            return _focus_alias_candidate(left) and _focus_alias_candidate(right)
        case _:
            return False


def construct_program(
    program: Program,
    env: dict[str, Morphism] | None = None,
) -> ConstructedProgram:
    """Resolve every parsed morphism in ``program`` into a typed ``Morphism``.

    ``parse_program`` deliberately leaves names as ``Ref`` nodes. This pass
    gives those names semantic meaning by resolving them against backend/builtin
    morphisms in ``env`` and against other morphisms in the same program.
    """

    base_env = dict(env or {})
    raw_functors = dict(program.functors)
    functors: dict[str, expr.PolyExpr] = {}
    morphisms: dict[str, Morphism] = {}
    carriers: dict[str, RecursiveCarrier] = {}
    focuses: dict[str, Optic] = {}
    constructing: list[str] = []
    constructing_focus: list[str] = []
    _cx = [L.empty_context()]

    def resolve_functor(name: str) -> expr.PolyExpr:
        if name in functors:
            return functors[name]
        if name not in raw_functors:
            raise MorphismError(f"construct_program: unresolved functor {name!r}")
        body = raw_functors[name]
        resolved = _resolve_poly_refs(body, raw_functors, (name,))
        functors[name] = resolved
        return resolved

    def resolve_carrier(name: str) -> RecursiveCarrier:
        if name in carriers:
            return carriers[name]
        if name not in program.carriers:
            raise MorphismError(f"construct_program: unresolved carrier {name!r}")

        decl = program.carriers[name]
        functor_body = _resolve_poly_refs(decl.functor, raw_functors, (name,))
        functor = Functor(name=f"{name}F", body=functor_body)
        carrier = recursive_carrier(name, functor)
        carriers[name] = carrier
        focuses.setdefault(name, carrier.optic())
        return carrier

    def resolve_focus_expr(node: expr.FocusExpr) -> Optic:
        match node:
            case expr.FocusRef(name=name):
                return resolve_focus(name)
            case expr.FocusCompose(left=left, right=right):
                return resolve_focus_expr(left).compose(resolve_focus_expr(right))
            case _:
                raise TypeError(f"construct_program: unknown FocusExpr {type(node).__name__!r}")

    def morphism_env_for(node: expr.MorphismExpr) -> dict[str, Morphism]:
        morphism_env = dict(base_env)
        for ref in sorted(_morphism_refs(node)):
            if ref in program.morphisms and ref not in program.morphism_params:
                morphism_env[ref] = resolve_morphism(ref)
        morphism_env.update(morphisms)
        return morphism_env

    def resolve_focus_alias(name: str, decl: expr.FocusDecl) -> Optic:
        if decl.expr is None:
            raise MorphismError(f"construct_program: incomplete focus alias {name!r}")
        return resolve_focus_expr(decl.expr)

    def resolve_focus_from_carrier(name: str, decl: expr.FocusDecl) -> Optic:
        if decl.carrier is None:
            raise MorphismError(f"construct_program: incomplete carrier focus {name!r}")
        return resolve_carrier(decl.carrier).optic()

    def resolve_explicit_focus(name: str, decl: expr.FocusDecl) -> Optic:
        if decl.functor is None or decl.forward is None or decl.backward is None:
            raise MorphismError(f"construct_program: incomplete focus {name!r}")
        if decl.functor not in functors:
            resolve_functor(decl.functor)

        morphism_env = morphism_env_for(decl.forward)
        morphism_env.update(morphism_env_for(decl.backward))
        forward = construct(
            decl.forward, morphism_env, functors, program.morphisms, program.morphism_params, focuses, _cx,
        )
        backward = construct(
            decl.backward, morphism_env, functors, program.morphisms, program.morphism_params, focuses, _cx,
        )
        functor = Functor(name=decl.functor, body=functors[decl.functor])
        carrier = forward.dom()
        layer = functor.apply(carrier)
        try:
            Ty.require_equal(None, forward.cod(), layer, f"focus {name}.forward")
            Ty.require_equal(None, backward.dom(), layer, f"focus {name}.backward")
            Ty.require_equal(None, backward.cod(), carrier, f"focus {name}.carrier")
        except TypeError as e:
            raise MorphismError(str(e)) from e

        return Optic(functor=functor, forward=forward, backward=backward, carrier=carrier)

    def resolve_focus(name: str) -> Optic:
        if name in focuses:
            return focuses[name]
        if name in carriers:
            focus = carriers[name].optic()
            focuses[name] = focus
            return focus
        if name in program.carriers:
            focus = resolve_carrier(name).optic()
            focuses[name] = focus
            return focus
        if name not in program.focuses:
            if name in raw_functors and _focus_alias_candidate(raw_functors[name]):
                decl = expr.FocusDecl(expr=poly_to_focus_expr(raw_functors[name]))
            else:
                raise MorphismError(f"construct_program: unresolved focus {name!r}")
        else:
            decl = program.focuses[name]
        if name in constructing_focus:
            cycle = " -> ".join((*constructing_focus, name))
            raise MorphismError(f"construct_program: cyclic focus reference: {cycle}")

        if decl.carrier is not None:
            focus = resolve_focus_from_carrier(name, decl)
            focuses[name] = focus
            return focus
        constructing_focus.append(name)
        try:
            if decl.expr is not None:
                focus = resolve_focus_alias(name, decl)
            else:
                focus = resolve_explicit_focus(name, decl)
        finally:
            constructing_focus.pop()
        focuses[name] = focus
        return focus

    def resolve_morphism(name: str) -> Morphism:
        if name in morphisms:
            return morphisms[name]
        if name not in program.morphisms:
            if name in base_env:
                return base_env[name]
            raise MorphismError(f"construct_program: unresolved morphism {name!r}")
        if name in constructing:
            cycle = " -> ".join((*constructing, name))
            raise MorphismError(f"construct_program: cyclic morphism reference: {cycle}")

        constructing.append(name)
        try:
            morphism_env = morphism_env_for(program.morphisms[name])
            morphism = construct(
                program.morphisms[name],
                morphism_env,
                functors,
                program.morphisms,
                program.morphism_params,
                focuses,
                _cx,
            )
            morphisms[name] = morphism
            return morphism
        finally:
            constructing.pop()

    for name in program.carriers:
        resolve_carrier(name)

    for name in program.focuses:
        resolve_focus(name)

    for name, body in raw_functors.items():
        try:
            resolve_functor(name)
        except MorphismError:
            if not _focus_alias_candidate(body):
                raise
            resolve_focus(name)

    for name in program.morphisms:
        if name not in program.morphism_params:
            resolve_morphism(name)

    return ConstructedProgram(
        morphisms=morphisms,
        functors=functors,
        carriers=carriers,
        focuses=focuses,
    )


def construct(
    node: expr.MorphismExpr,
    env: dict[str, Morphism],
    functor_env: dict[str, expr.PolyExpr] | None = None,
    morphism_bodies: dict[str, expr.MorphismExpr] | None = None,
    morphism_params: dict[str, tuple[str, ...]] | None = None,
    focus_env: dict[str, Optic] | None = None,
    _cx: list | None = None,
) -> Morphism:
    """Resolve a parsed expression tree into a typed Morphism.

    Walks the tree recursively, resolving Ref nodes from ``env`` and
    calling semantic combinators at each binary node. ``morphism_bodies`` and
    ``morphism_params`` make parameterized ``let`` declarations available for
    substitution; ``focus_env`` makes constructed optics available to recursive
    schemes and carrier boundaries.
    """
    if _cx is None:
        _cx = [L.empty_context()]

    def _recurse(n):
        return construct(n, env, functor_env, morphism_bodies, morphism_params, focus_env, _cx)

    from unialg.objects import TypeUnit, ProductType, SumType

    def _fresh():
        var, new_cx = Ty.fresh_variable_type(_cx[0])
        _cx[0] = new_cx
        return var

    def _fresh_pair():
        return ProductType(_fresh(), _fresh())

    def _fresh_sum():
        return SumType(_fresh(), _fresh())

    def _apply_parameterized_morphism(fun: expr.Ref, args: tuple[expr.MorphismExpr, ...]) -> Morphism:
        bodies = morphism_bodies or {}
        params = morphism_params or {}
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
        return construct(body, local_env, functor_env, morphism_bodies, morphism_params, focus_env, _cx)

    def _apply_backend_primitive(resolved_fun: Morphism, fun, args: tuple[expr.MorphismExpr, ...]) -> Morphism:
        bp = resolved_fun.node
        if not isinstance(bp, expr.BackendPrim) or bp.arity <= 1:
            raise MorphismError(
                f"construct: cannot apply {fun!r} (not parameterized, not arity > 1)"
            )
        if len(args) != bp.arity:
            raise MorphismError(
                f"construct: {fun!r} expects {bp.arity} args, got {len(args)}"
            )
        resolved_args = [_recurse(a) for a in args]
        all_aux = resolved_fun.aux_primitives
        for ra in resolved_args:
            all_aux = all_aux + ra.aux_primitives
        return Morphism(
            node=expr.BackendPrim(
                bp.primitive, bp.arity, resolved_args[0].dom(), bp.cod,
                args=tuple(ra.node for ra in resolved_args),
            ),
            aux_primitives=all_aux,
        )

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
            params = morphism_params or {}
            if isinstance(fun, expr.Ref) and fun.name in params:
                return _apply_parameterized_morphism(fun, args)
            return _apply_backend_primitive(_recurse(fun), fun, args)

        case expr.RecursionApp(kind=kind, focus=focus_name, args=args):
            foci = focus_env or {}
            if focus_name not in foci:
                raise MorphismError(f"construct: unresolved focus {focus_name!r}")
            fp = foci[focus_name]
            resolved_args = [_recurse(a) for a in args]
            match kind:
                case "cata":
                    if len(resolved_args) != 1:
                        raise MorphismError(f"construct: cata[{focus_name}] expects 1 arg, got {len(resolved_args)}")
                    return cata(fp, resolved_args[0])
                case "ana":
                    if len(resolved_args) != 1:
                        raise MorphismError(f"construct: ana[{focus_name}] expects 1 arg, got {len(resolved_args)}")
                    return ana(fp, resolved_args[0])
                case "hylo":
                    if len(resolved_args) != 2:
                        raise MorphismError(f"construct: hylo[{focus_name}] expects 2 args, got {len(resolved_args)}")
                    return hylo(fp, resolved_args[0], resolved_args[1])
                case _:
                    raise MorphismError(f"construct: unknown recursion scheme {kind!r}")

        case expr.CarrierBoundary(kind=kind, focus=focus_name):
            foci = focus_env or {}
            if focus_name not in foci:
                raise MorphismError(f"construct: unresolved focus {focus_name!r}")
            fp = foci[focus_name]
            match kind:
                case "roll":
                    return fp.backward
                case "unroll":
                    return fp.forward
                case _:
                    raise MorphismError(f"construct: unknown carrier boundary {kind!r}")

        case expr.MonadicLift(monad=monad_name, body=body):
            try:
                monad = monad_by_name(monad_name)
            except ValueError as e:
                raise MorphismError(str(e)) from e
            return _recurse(body).to_lax(monad)

        case _:
            raise TypeError(f"construct: unhandled node {type(node).__name__!r}")
