"""Helper routines for semantic construction.

Kept separate from construct.py so the main constructor reads as orchestration
rather than a pile of branch-specific helpers.
"""
from __future__ import annotations

from unialg.objects import monad_by_name
from unialg.syntax import expressions as expr
from unialg.syntax.parse import Program

from .functors import Functor, poly_fmap
from .morphisms import Morphism, MorphismError
from .optics import Optic, ana, cata, hylo


def morphism_refs(node: expr.MorphismExpr) -> set[str]:
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
                refs |= morphism_refs(arg)
            return refs
        case expr.MonadicEmbed(f=f):
            return morphism_refs(f)
        case expr.ContextualBinary(f=f, g=g):
            return morphism_refs(f) | morphism_refs(g)
        case expr.PolyFmap(f=f):
            return morphism_refs(f)
        case expr.MorphismApp(fun=fun, args=args):
            refs = morphism_refs(fun)
            for arg in args:
                refs |= morphism_refs(arg)
            return refs
        case expr.RecursionApp(args=args):
            refs = set()
            for arg in args:
                refs |= morphism_refs(arg)
            return refs
        case expr.MonadicLift(body=body):
            return morphism_refs(body)
        case expr.AlgExpr(body=body):
            return morphism_refs(body)
        case _ if hasattr(node, '_domain_tag'):
            return set()
        case _:
            raise TypeError(f"morphism_refs: unknown MorphismExpr {type(node).__name__!r}")


def resolve_poly_refs(
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
            return resolve_poly_refs(functors[name], functors, (*stack, name))
        case expr.Prod(left=left, right=right):
            return expr.Prod(
                resolve_poly_refs(left, functors, stack),
                resolve_poly_refs(right, functors, stack),
            )
        case expr.Sum(left=left, right=right):
            return expr.Sum(
                resolve_poly_refs(left, functors, stack),
                resolve_poly_refs(right, functors, stack),
            )
        case expr.PolyCompose(left=left, right=right):
            return expr.PolyCompose(
                resolve_poly_refs(left, functors, stack),
                resolve_poly_refs(right, functors, stack),
            )
        case expr.Exp(base=base, body=body):
            return expr.Exp(base, resolve_poly_refs(body, functors, stack))
        case expr.List(body=body):
            return expr.List(resolve_poly_refs(body, functors, stack))
        case expr.Maybe(body=body):
            return expr.Maybe(resolve_poly_refs(body, functors, stack))
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


def domain_construct_env(
    base_env: dict[str, Morphism],
    domain_context: object | None,
) -> dict:
    ext_env = dict(base_env)
    ext_env["_domain_context"] = domain_context
    return ext_env


def domain_finalize_env(
    base_env: dict[str, Morphism],
    domain_context: object | None,
    domain_data: dict[str, object],
) -> dict:
    fin_env = domain_construct_env(base_env, domain_context)
    fin_env["_domain_data"] = domain_data
    return fin_env


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
            domain_construct_env(base_env, domain_context),
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

    fin_env = domain_finalize_env(base_env, domain_context, domain_data)
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
