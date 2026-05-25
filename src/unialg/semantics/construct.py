"""Semantic construction: parsed expression tree → typed Morphism.

Walks a MorphismExpr tree (placeholder-typed, with Ref nodes) and
reconstructs it through the semantic combinators, producing a fully
typed Morphism ready for compile_program.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import re

from hydra.core import LiteralType, TypeLiteral
from hydra.sorting import topological_sort
from hydra.dsl.python import Left

from unialg.syntax import expressions as expr
from unialg.syntax.parse import Program, poly_to_focus_expr
import hydra.lexical as L

from . import morphisms as ops
from . import typeops as Ty
from ._construct_helpers import (
    construct_carrier_boundary,
    construct_domain_expr, construct_domain_extensions,
    construct_monadic_lift, construct_poly_fmap,
    construct_recursion_app, finalize_domain_morphisms, 
    focus_alias_candidate, focus_expr_refs, 
    morphism_refs, poly_refs, resolve_poly_refs,
)
from .morphisms import Morphism, MorphismError
from .functors import Functor
from .optics import (
    Optic, RecursiveCarrier, recursive_carrier,
    lens_optic, prism_optic, traversal_optic,
)
from unialg.objects import TypePair, TypeUnit, ProductType, SumType

_FOCUS_CONSTRUCTORS: dict = {
    "lens":      lambda name, fwd, bwd, _ftr: lens_optic(name, fwd, bwd),
    "prism":     lambda name, fwd, bwd, _ftr: prism_optic(name, fwd, bwd),
    "traversal": lambda name, fwd, bwd, ftr:  traversal_optic(name, ftr, fwd, bwd),
    "optic":     lambda name, fwd, bwd, ftr:  traversal_optic(name, ftr, fwd, bwd),
}


def _parse_bool(text: str) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise MorphismError(f"construct: invalid BOOL literal {text!r}")


def _parse_int(text: str) -> int:
    if not re.fullmatch(r"[+-]?[0-9]+", text):
        raise MorphismError(f"construct: invalid INT literal {text!r}")
    return int(text, 10)


def _parse_float(text: str) -> float:
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text):
        raise MorphismError(f"construct: invalid FLOAT literal {text!r}")
    return float(text)


_LITERAL_PARSERS: dict = {
    LiteralType.STRING:  lambda text: text,
    LiteralType.BOOLEAN: _parse_bool,
    LiteralType.INTEGER: _parse_int,
    LiteralType.FLOAT:   _parse_float,
}


def _parse_literal_value(text: str, expected) -> object:
    """Convert a quoted literal string to its Python value given the expected type."""
    if expected == TypeUnit():
        raise MorphismError("construct: quoted literal cannot inhabit UNIT; use delete")
    if not isinstance(expected, TypeLiteral):
        raise MorphismError("construct: quoted literal requires a scalar receiving type")
    kind = expected.value
    if kind == LiteralType.BINARY:
        raise MorphismError("construct: quoted literal cannot be used for BINARY input")
    parser = _LITERAL_PARSERS.get(kind)
    if parser is None:
        raise MorphismError("construct: quoted literal requires INT, FLOAT, BOOL, or STRING input")
    return parser(text)


def _argument_types(dom, arity: int) -> tuple:
    """Flatten a nested TypePair domain into a tuple of arity individual types."""
    if arity == 1:
        return (dom,)
    if not isinstance(dom, TypePair):
        raise MorphismError("construct: primitive argument domain does not match arity")
    return _argument_types(dom.value.first, arity - 1) + (dom.value.second,)


@dataclass(frozen=True)
class ConstructedProgram:
    """A parsed program after semantic morphism construction."""

    morphisms: dict[str, Morphism]
    functors: dict[str, expr.PolyExpr]
    carriers: dict[str, RecursiveCarrier]
    focuses: dict[str, Optic]
    domain_data: dict[str, object] = dataclass_field(default_factory=dict)


def _topo_order(names, deps, kind):
    names = tuple(names)
    pairs = []
    for name in names:
        refs = deps(name)
        if name in refs:
            raise MorphismError(f"construct_program: cyclic {kind} reference: {name} -> {name}")
        pairs.append((name, tuple(r for r in refs if r in names)))
    result = topological_sort(tuple(pairs))
    if isinstance(result, Left):
        cycles = " | ".join(" -> ".join((*scc, scc[0])) for scc in result.value)
        raise MorphismError(f"construct_program: cyclic {kind} reference: {cycles}")
    return result.value


def construct_program(program: Program, env: dict[str, Morphism] | None = None,
            domain_context: object | None = None) -> ConstructedProgram:
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
    domain_data: dict[str, object] = {}
    focuses: dict[str, Optic] = {}
    _cx = [L.empty_context()]

    def resolve_functor(name: str) -> expr.PolyExpr:
        if name in functors:
            return functors[name]
        if name not in raw_functors:
            raise MorphismError(f"construct_program: unresolved functor {name!r}")
        body = raw_functors[name]
        resolved = resolve_poly_refs(body, raw_functors)
        functors[name] = resolved
        return resolved

    def resolve_carrier(name: str) -> RecursiveCarrier:
        if name in carriers:
            return carriers[name]
        if name not in program.carriers:
            raise MorphismError(f"construct_program: unresolved carrier {name!r}")

        decl = program.carriers[name]
        functor_body = resolve_poly_refs(decl.functor, raw_functors)
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
        for ref in sorted(morphism_refs(node)):
            if ref not in program.morphisms or ref in program.morphism_params:
                continue
            morphism_env[ref] = resolve_morphism(ref)
        morphism_env.update(morphisms)
        return morphism_env

    def focus_decl_for(name: str) -> expr.FocusDecl:
        if name in program.focuses:
            return program.focuses[name]
        if name in raw_functors and focus_alias_candidate(raw_functors[name]):
            return expr.FocusDecl(expr=poly_to_focus_expr(raw_functors[name]))
        raise MorphismError(f"construct_program: unresolved focus {name!r}")

    def resolve_focus_alias(name: str, decl: expr.FocusDecl) -> Optic:
        if decl.expr is None:
            raise MorphismError(f"construct_program: incomplete focus alias {name!r}")
        return resolve_focus_expr(decl.expr)

    def resolve_focus_from_carrier(name: str, decl: expr.FocusDecl) -> Optic:
        if decl.carrier is None:
            raise MorphismError(f"construct_program: incomplete carrier focus {name!r}")
        return resolve_carrier(decl.carrier).optic()

    def resolve_explicit_focus(name: str, decl: expr.FocusDecl) -> Optic:
        if decl.forward is None or decl.backward is None:
            raise MorphismError(f"construct_program: incomplete focus {name!r}")
        ctor = _FOCUS_CONSTRUCTORS.get(decl.kind)
        if ctor is None:
            raise MorphismError(f"construct_program: unknown focus kind {decl.kind!r}")

        morphism_env = morphism_env_for(decl.forward)
        morphism_env.update(morphism_env_for(decl.backward))
        forward = construct(
            decl.forward, morphism_env, functors, program.morphisms, program.morphism_params, focuses, _cx, domain_data, domain_context,
        )
        backward = construct(
            decl.backward, morphism_env, functors, program.morphisms, program.morphism_params, focuses, _cx, domain_data, domain_context,
        )

        ftr = None
        if decl.kind in ("traversal", "optic"):
            if decl.functor is None:
                raise MorphismError(f"construct_program: {decl.kind} focus {name!r} requires a functor")
            if decl.functor not in functors:
                resolve_functor(decl.functor)
            ftr = Functor(name=decl.functor, body=functors[decl.functor])
        return ctor(name, forward, backward, ftr)

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
        decl = focus_decl_for(name)
        if decl.carrier is not None:
            focus = resolve_focus_from_carrier(name, decl)
            focuses[name] = focus
            return focus
        if decl.expr is not None:
            focus = resolve_focus_alias(name, decl)
        else:
            focus = resolve_explicit_focus(name, decl)
        focuses[name] = focus
        return focus

    def resolve_morphism(name: str) -> Morphism:
        if name in morphisms:
            return morphisms[name]
        if name not in program.morphisms:
            if name in base_env:
                return base_env[name]
            raise MorphismError(f"construct_program: unresolved morphism {name!r}")
        morphism_env = morphism_env_for(program.morphisms[name])
        morphism = construct(
            program.morphisms[name],
            morphism_env,
            functors,
            program.morphisms,
            program.morphism_params,
            focuses,
            _cx,
            domain_data,
            domain_context,
        )
        morphisms[name] = morphism
        return morphism

    morphism_names = [n for n in program.morphisms if n not in program.morphism_params]
    focus_order = _topo_order(program.focuses, lambda n: focus_expr_refs(program.focuses[n].expr), "focus")
    functor_order = _topo_order(raw_functors, lambda n: poly_refs(raw_functors[n]), "functor")
    morphism_order = _topo_order(morphism_names, lambda n: morphism_refs(program.morphisms[n]), "morphism")

    for name in program.carriers:
        resolve_carrier(name)

    for name in focus_order:
        resolve_focus(name)

    for name in functor_order:
        body = raw_functors[name]
        try:
            resolve_functor(name)
        except MorphismError:
            if not focus_alias_candidate(body):
                raise
            resolve_focus(name)

    domain_data = construct_domain_extensions(program, base_env, domain_context)

    for name in morphism_order:
        resolve_morphism(name)

    finalize_domain_morphisms(morphisms, base_env, domain_context, domain_data)

    return ConstructedProgram(
        morphisms=morphisms,
        functors=functors,
        carriers=carriers,
        focuses=focuses,
        domain_data=domain_data,
    )


def construct(node: expr.MorphismExpr, env: dict[str, Morphism],
    functor_env: dict[str, expr.PolyExpr] | None = None,
    morphism_bodies: dict[str, expr.MorphismExpr] | None = None,
    morphism_params: dict[str, tuple[str, ...]] | None = None,
    focus_env: dict[str, Optic] | None = None,
    _cx: list | None = None,
    _domain_data: dict[str, object] | None = None,
    _domain_context: object | None = None,
    _expected_cod=None,
    _bound_exprs: dict[str, expr.MorphismExpr] | None = None,
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
    bound_exprs = dict(_bound_exprs or {})

    def _recurse(n, expected_cod=None):
        return construct(
            n, env, functor_env, morphism_bodies, morphism_params, focus_env,
            _cx, _domain_data, _domain_context, expected_cod, bound_exprs,
        )

    def _fresh():
        var, new_cx = Ty.fresh_variable_type(_cx[0])
        _cx[0] = new_cx
        return var

    def _fresh_pair():
        return ProductType(_fresh(), _fresh())

    def _fresh_sum():
        return SumType(_fresh(), _fresh())

    def _expand_bound(n: expr.MorphismExpr) -> expr.MorphismExpr:
        seen: set[str] = set()
        while isinstance(n, expr.Ref) and n.name in bound_exprs:
            if n.name in seen:
                raise MorphismError(f"construct: cyclic parameter binding {n.name!r}")
            seen.add(n.name)
            n = bound_exprs[n.name]
        return n

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
        local_bound = dict(bound_exprs)
        for pname, arg_node in zip(declared_params, args):
            local_bound[pname] = arg_node
        return construct(
            body, env, functor_env, morphism_bodies, morphism_params, focus_env,
            _cx, _domain_data, _domain_context, None, local_bound,
        )

    def _apply_backend_primitive(resolved_fun: Morphism, fun, args: tuple[expr.MorphismExpr, ...]) -> Morphism:
        bp = resolved_fun.node
        if not isinstance(bp, expr.BackendPrim):
            raise MorphismError(
                f"construct: cannot apply {fun!r} (not parameterized or backend primitive)"
            )
        if len(args) != bp.arity:
            raise MorphismError(
                f"construct: {fun!r} expects {bp.arity} args, got {len(args)}"
            )
        expected_types = _argument_types(bp.dom, bp.arity)
        resolved_args: list[tuple[Morphism, bool]] = []
        for arg, expected in zip(args, expected_types):
            expanded = _expand_bound(arg)
            is_literal = isinstance(expanded, expr.Literal)
            resolved = _recurse(expanded, expected)
            try:
                Ty.require_equal(None, resolved.cod(), expected, "Backend primitive argument")
            except TypeError as e:
                raise MorphismError(str(e)) from e
            resolved_args.append((resolved, is_literal))

        visible_dom = TypeUnit()
        for resolved, is_literal in resolved_args:
            if is_literal:
                continue
            if visible_dom == TypeUnit():
                visible_dom = resolved.dom()
            else:
                try:
                    Ty.require_equal(None, visible_dom, resolved.dom(), "Backend primitive input context")
                except TypeError as e:
                    raise MorphismError(str(e)) from e

        lifted_args: list[Morphism] = []
        for resolved, is_literal in resolved_args:
            if is_literal:
                resolved = ops.compose(
                    ops._delete(visible_dom), resolved, allow_unification=True,
                )
            lifted_args.append(resolved)

        all_aux = resolved_fun.aux_primitives
        for ra in lifted_args:
            all_aux = all_aux + ra.aux_primitives
        return Morphism(
            node=expr.BackendPrim(
                bp.primitive, bp.arity, visible_dom, bp.cod,
                args=tuple(ra.node for ra in lifted_args),
            ),
            aux_primitives=all_aux,
        )

    match node:
        case expr.Ref(name=name):
            if name in bound_exprs:
                return _recurse(bound_exprs[name], _expected_cod)
            if name not in env:
                raise MorphismError(f"construct: unresolved reference {name!r}")
            return env[name]

        case expr.Prim(raw, dom, cod):
            return Morphism(node=node)

        case expr.Identity(space):
            return ops.identity(
                space if space != TypeUnit() else (_expected_cod or _fresh())
            )

        case expr.Literal(text=text, value=value, cod=cod):
            if value is not None and cod != TypeUnit():
                return ops.lit(value, cod, text)
            if _expected_cod is None:
                raise MorphismError("construct: quoted literal requires a typed argument context")
            return ops.lit(_parse_literal_value(text, _expected_cod), _expected_cod, text)

        case expr.First(ab):
            return ops._first(ab if ab != ProductType(TypeUnit(), TypeUnit()) else _fresh_pair())

        case expr.Second(ab):
            return ops._second(ab if ab != ProductType(TypeUnit(), TypeUnit()) else _fresh_pair())

        case expr.Left(ab):
            return ops._inject_left(ab if ab != SumType(TypeUnit(), TypeUnit()) else _fresh_sum())

        case expr.Right(ab):
            return ops._inject_right(ab if ab != SumType(TypeUnit(), TypeUnit()) else _fresh_sum())

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

        case expr.DistributeLeft(dom=dom):
            a, bc = ops._lr(dom)
            b, c = ops._lr(bc)
            return ops.distribute_left(a, b, c)

        case expr.DistributeRight(dom=dom):
            ab, c = ops._lr(dom)
            a, b = ops._lr(ab)
            return ops.distribute_right(a, b, c)

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

        case expr.Coparallel(f=f, g=g):
            return ops.copar(_recurse(f), _recurse(g))

        case expr.Case(f=f, g=g):
            return ops.case(
                _recurse(f), _recurse(g),
                allow_unification=True,
            )

        case expr.PolyFmap(body=body, f=f):
            return construct_poly_fmap(body, f, functor_env, _recurse)

        case expr.MorphismApp(fun=fun, args=args):
            params = morphism_params or {}
            if isinstance(fun, expr.Ref) and fun.name in params:
                return _apply_parameterized_morphism(fun, args)
            return _apply_backend_primitive(_recurse(fun), fun, args)

        case expr.RecursionApp(kind=kind, focus=focus_name, args=args):
            return construct_recursion_app(kind, focus_name, args, focus_env, _recurse)

        case expr.CarrierBoundary(kind=kind, focus=focus_name):
            return construct_carrier_boundary(kind, focus_name, focus_env)

        case expr.MonadicLift(monad=monad_name, body=body):
            return construct_monadic_lift(monad_name, body, _recurse)

        case _ if hasattr(node, '_domain_tag'):
            return construct_domain_expr(node, env, _domain_data, _domain_context)

        case _:
            raise TypeError(f"construct: unhandled node {type(node).__name__!r}")
