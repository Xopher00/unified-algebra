from __future__ import annotations

from unialg.objects import monad_by_name
from unialg.syntax import expressions as expr
from unialg.syntax.parse import Program

from .functors import Functor, poly_fmap
from .morphisms import Morphism, MorphismError
from .optics import Optic, ana, cata, hylo


_MORPHISM_REF_EMPTY: frozenset[type] = frozenset({
    expr.Identity, expr.Copy, expr.Delete, expr.Literal, expr.First, expr.Second,
    expr.Left, expr.Right, expr.Absurd, expr.Assoc, expr.Symmetry,
    expr.Prim, expr.SelfRef, expr.CarrierBoundary, expr.Coerce,
})

_MORPHISM_REF_SINGLE = (expr.MonadicEmbed, expr.PolyFmap, expr.MonadicLift, expr.AlgExpr)


def _many_morphism_refs(nodes) -> set[str]:
    refs = set()
    for node in nodes:
        refs |= morphism_refs(node)
    return refs


def _morphism_refs_single(node: expr.MorphismExpr) -> set[str]:
    match node:
        case expr.MonadicEmbed(f=f) | expr.PolyFmap(f=f):
            return morphism_refs(f)
        case expr.MonadicLift(body=body) | expr.AlgExpr(body=body):
            return morphism_refs(body)
        case _:
            raise TypeError(f"morphism_refs: unexpected single-child node {type(node).__name__!r}")


def _morphism_refs_multi(node: expr.MorphismExpr) -> set[str]:
    match node:
        case expr.BackendPrim(args=args) | expr.RecursionApp(args=args):
            return _many_morphism_refs(args)
        case expr.ContextualBinary(f=f, g=g):
            return morphism_refs(f) | morphism_refs(g)
        case expr.MorphismApp(fun=fun, args=args):
            return morphism_refs(fun) | _many_morphism_refs(args)
        case _:
            raise TypeError(f"morphism_refs: unknown MorphismExpr {type(node).__name__!r}")


def morphism_refs(node: expr.MorphismExpr) -> set[str]:
    """Collect morphism references from a parsed morphism expression."""
    if isinstance(node, expr.Ref):
        return {node.name}
    if type(node) in _MORPHISM_REF_EMPTY or hasattr(node, '_domain_tag'):
        return set()
    if isinstance(node, _MORPHISM_REF_SINGLE):
        return _morphism_refs_single(node)
    return _morphism_refs_multi(node)


def expanded_morphism_refs(
    node: expr.MorphismExpr,
    bodies: dict[str, expr.MorphismExpr],
    params: dict[str, tuple[str, ...]],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    """Like morphism_refs, but expands through parameterized callee bodies.

    For MorphismApp(Ref(name), args) where name is parameterized, also
    collects refs from the callee body (minus its declared param names),
    recursively. This gives the full transitive dependency set needed for
    topological ordering and environment preparation.
    """
    if isinstance(node, expr.Ref):
        return {node.name}
    if type(node) in _MORPHISM_REF_EMPTY or hasattr(node, '_domain_tag'):
        return set()
    if isinstance(node, _MORPHISM_REF_SINGLE):
        return _morphism_refs_single(node)
    if not isinstance(node, expr.MorphismApp):
        return _morphism_refs_multi(node)

    # MorphismApp: collect from args, then expand callee body if parameterized
    fun, args = node.fun, node.args
    refs = _many_expanded_morphism_refs(args, bodies, params, seen)
    if not isinstance(fun, expr.Ref):
        refs |= expanded_morphism_refs(fun, bodies, params, seen)
        return refs
    callee = fun.name
    refs.add(callee)
    if callee in params and callee in bodies and callee not in seen:
        formal_params = set(params[callee])
        body_refs = expanded_morphism_refs(bodies[callee], bodies, params, seen | {callee})
        refs |= body_refs - formal_params
    return refs


def _many_expanded_morphism_refs(nodes, bodies, params, seen) -> set[str]:
    refs = set()
    for node in nodes:
        refs |= expanded_morphism_refs(node, bodies, params, seen)
    return refs


def resolve_poly_refs(
    node: expr.PolyExpr,
    functors: dict[str, expr.PolyExpr],
) -> expr.PolyExpr:
    """Inline named functor references inside a polynomial expression."""
    match node:
        case expr.PolyRef(name=name):
            if name not in functors:
                raise MorphismError(f"construct_program: unresolved functor {name!r}")
            return resolve_poly_refs(functors[name], functors)
        case expr.Prod(left=l, right=r):
            return expr.Prod(resolve_poly_refs(l, functors), resolve_poly_refs(r, functors))
        case expr.Sum(left=l, right=r):
            return expr.Sum(resolve_poly_refs(l, functors), resolve_poly_refs(r, functors))
        case expr.PolyCompose(left=l, right=r):
            return expr.PolyCompose(resolve_poly_refs(l, functors), resolve_poly_refs(r, functors))
        case expr.Exp(base=base, body=body):
            return expr.Exp(resolve_poly_refs(base, functors), resolve_poly_refs(body, functors))
        case expr.List(body=body):
            return expr.List(resolve_poly_refs(body, functors))
        case expr.Maybe(body=body):
            return expr.Maybe(resolve_poly_refs(body, functors))
        case expr.Zero() | expr.One() | expr.Id() | expr.Const():
            return node
        case _:
            raise TypeError(f"construct_program: unknown PolyExpr {type(node).__name__!r}")


def focus_alias_candidate(node: expr.PolyExpr) -> bool:
    """Return True when ``node`` could be an optic alias expression."""
    match node:
        case expr.PolyRef():
            return True
        case expr.PolyCompose(left=left, right=right):
            return focus_alias_candidate(left) and focus_alias_candidate(right)
        case _:
            return False


def focus_expr_refs(node) -> set[str]:
    match node:
        case None:
            return set()
        case expr.FocusRef(name=name):
            return {name}
        case expr.FocusCompose(left=l, right=r):
            return focus_expr_refs(l) | focus_expr_refs(r)
        case _:
            raise TypeError(f"focus_expr_refs: unknown FocusExpr {type(node).__name__!r}")


def poly_refs(node: expr.PolyExpr) -> set[str]:
    match node:
        case expr.PolyRef(name=name):
            return {name}
        case expr.Prod(left=l, right=r) | expr.Sum(left=l, right=r) | expr.PolyCompose(left=l, right=r):
            return poly_refs(l) | poly_refs(r)
        case expr.Exp(body=b) | expr.List(body=b) | expr.Maybe(body=b):
            return poly_refs(b)
        case expr.Zero() | expr.One() | expr.Id() | expr.Const():
            return set()
        case _:
            raise TypeError(f"poly_refs: unknown PolyExpr {type(node).__name__!r}")


def construct_domain_extensions(
    program: Program,
    base_env: dict[str, Morphism],
    domain_context: object | None,
) -> dict[str, object]:
    from unialg.extensions import get_domain_protocol

    domain_data: dict[str, object] = {}
    for tag, decls in program.extensions.items():
        protocol = get_domain_protocol(tag)
        if protocol is None:
            continue
        domain_data[tag] = protocol.construct(
            decls,
            {**base_env, "_domain_context": domain_context},
        )
    return domain_data


def finalize_domain_morphisms(
    morphisms: dict[str, Morphism],
    base_env: dict[str, Morphism],
    domain_context: object | None,
    domain_data: dict[str, object],
) -> None:
    from unialg.extensions import get_domain_protocol, registered_domains

    if not registered_domains():
        return

    fin_env = {**base_env, "_domain_context": domain_context, "_domain_data": domain_data}
    for tag in registered_domains():
        protocol = get_domain_protocol(tag)
        if protocol is None or protocol.finalize is None:
            continue
        for name in list(morphisms.keys()):
            morphisms[name] = protocol.finalize(morphisms[name], fin_env)


def construct_poly_fmap(body, f, functor_env, recurse) -> Morphism:
    fenv = functor_env or {}
    if isinstance(body, expr.PolyRef):
        if body.name not in fenv:
            raise MorphismError(f"construct: unresolved functor {body.name!r}")
        body = fenv[body.name]
    functor = Functor(name="anonymous", body=body)
    return poly_fmap(functor, recurse(f))


def focus_for_construct(focus_name: str, focus_env: dict[str, Optic] | None) -> Optic:
    foci = focus_env or {}
    if focus_name not in foci:
        raise MorphismError(f"construct: unresolved focus {focus_name!r}")
    return foci[focus_name]


def require_recursion_arity(
    kind: str,
    focus_name: str,
    resolved_args: list[Morphism],
    expected: int,
) -> None:
    if len(resolved_args) == expected:
        return
    arg_word = "arg" if expected == 1 else "args"
    raise MorphismError(
        f"construct: {kind}[{focus_name}] expects {expected} {arg_word}, got {len(resolved_args)}"
    )


def construct_recursion_app(kind, focus_name, args, focus_env, recurse) -> Morphism:
    fp = focus_for_construct(focus_name, focus_env)
    resolved_args = [recurse(a) for a in args]

    if kind == "cata":
        require_recursion_arity(kind, focus_name, resolved_args, 1)
        return cata(fp, resolved_args[0])
    if kind == "ana":
        require_recursion_arity(kind, focus_name, resolved_args, 1)
        return ana(fp, resolved_args[0])
    if kind == "hylo":
        require_recursion_arity(kind, focus_name, resolved_args, 2)
        return hylo(fp, resolved_args[0], resolved_args[1])
    raise MorphismError(f"construct: unknown recursion scheme {kind!r}")


def construct_carrier_boundary(kind, focus_name, focus_env) -> Morphism:
    fp = focus_for_construct(focus_name, focus_env)
    if kind == "roll":
        return fp.backward
    if kind == "unroll":
        return fp.forward
    raise MorphismError(f"construct: unknown carrier boundary {kind!r}")


def construct_monadic_lift(monad_name: str, body, recurse) -> Morphism:
    try:
        monad = monad_by_name(monad_name)
    except ValueError as e:
        raise MorphismError(str(e)) from e
    return recurse(body).to_lax(monad)


def construct_domain_expr(
    node,
    env: dict[str, Morphism],
    domain_data: dict[str, object] | None,
    domain_context: object | None,
) -> Morphism:
    from unialg.extensions import get_domain_protocol

    protocol = get_domain_protocol(node._domain_tag)
    if protocol is None:
        raise TypeError(f"construct: no registered domain for tag {node._domain_tag!r}")

    ext_env = dict(env)
    ext_env["_domain_data"] = domain_data or {}
    ext_env["_domain_context"] = domain_context
    return protocol.construct_expr(node, ext_env)
