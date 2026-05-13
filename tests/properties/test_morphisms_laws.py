import pytest
from hypothesis import assume, given, settings

from hydra.core import (
    EitherType,
    FunctionType,
    PairType,
    TypeEither,
    TypeFunction,
    TypeList,
    TypeMaybe,
    TypePair,
)

from unialg.syntax import expressions as expr
from unialg.semantics import morphisms as ops
from unialg.semantics import functors as sem
from unialg.semantics import typeops as Ty
from unialg.objects import LIST, MAYBE
from support.strategies import (
    UNIT,
    VOID,
    collect_consts,
    composable_plain_pairs,
    count_id,
    flatten_sum,
    list_lax_morphisms,
    maybe_lax_morphisms,
    parametric_plain_morphisms,
    plain_morphisms,
    poly_values,
    product_type_values,
    same_codomain_plain_pairs,
    same_domain_plain_pairs,
    sum_type_values,
    type_values,
)


pytestmark = [pytest.mark.semantics, pytest.mark.property]


@settings(max_examples=60)
@given(type_values(), type_values())
def test_type_constructors_are_structural_and_order_sensitive(left, right):
    assert ops.ProductType(left, right) == TypePair(PairType(left, right))
    assert ops.SumType(left, right) == TypeEither(EitherType(left, right))

    if left != right:
        assert ops.ProductType(left, right) != ops.ProductType(right, left)
        assert ops.SumType(left, right) != ops.SumType(right, left)


@settings(max_examples=80)
@given(poly_values(), type_values())
def test_apply_poly_matches_polynomial_structure(body, space):
    result = sem.apply_poly(body, space)

    if isinstance(body, expr.Id):
        assert result == space
    elif isinstance(body, expr.One):
        assert result == UNIT
    elif isinstance(body, expr.Zero):
        assert result == VOID
    elif isinstance(body, expr.Const):
        assert result == body.space
    elif isinstance(body, expr.Prod):
        assert result == ops.ProductType(
            sem.apply_poly(body.left, space),
            sem.apply_poly(body.right, space),
        )
    elif isinstance(body, expr.Sum):
        assert result == ops.SumType(
            sem.apply_poly(body.left, space),
            sem.apply_poly(body.right, space),
        )
    elif isinstance(body, expr.Exp):
        assert result == TypeFunction(
            FunctionType(body.base, sem.apply_poly(body.body, space))
        )
    else:
        pytest.fail(f"unhandled body: {body!r}")


@settings(max_examples=80)
@given(poly_values())
def test_functor_introspection_matches_tree_walks(body):
    functor = sem.Functor("F", body)

    assert functor.summands() == flatten_sum(body)
    assert functor.x_arity() == count_id(body)
    assert functor.consts() == collect_consts(body)


@settings(max_examples=80)
@given(poly_values(), poly_values(), type_values())
def test_functor_compose_apply_law(outer_body, inner_body, space):
    outer = sem.Functor("F", outer_body)
    inner = sem.Functor("G", inner_body)

    assert outer.compose(inner).apply(space) == outer.apply(inner.apply(space))


@settings(max_examples=60)
@given(type_values())
def test_basic_morphism_constructors_have_expected_types(space):
    assert ops.identity(space).dom() == space
    assert ops.identity(space).cod() == space
    assert ops._copy(space).dom() == space
    assert ops._copy(space).cod() == ops.ProductType(space, space)
    assert ops._delete(space).dom() == space
    assert ops._delete(space).cod() == UNIT


@settings(max_examples=60)
@given(product_type_values())
def test_product_projection_constructors_have_expected_types(product):
    assert ops._fst(product).dom() == product
    assert ops._fst(product).cod() == product.value.first
    assert ops._snd(product).dom() == product
    assert ops._snd(product).cod() == product.value.second


@settings(max_examples=60)
@given(sum_type_values())
def test_sum_injection_constructors_have_expected_types(sum_type):
    assert ops._inl(sum_type).dom() == sum_type.value.left
    assert ops._inl(sum_type).cod() == sum_type
    assert ops._inr(sum_type).dom() == sum_type.value.right
    assert ops._inr(sum_type).cod() == sum_type


@settings(max_examples=60)
@given(type_values())
def test_absurd_has_void_domain_and_requested_codomain(codomain):
    assert ops.absurd(codomain).dom() == VOID
    assert ops.absurd(codomain).cod() == codomain


@settings(max_examples=60)
@given(plain_morphisms())
def test_signature_dom_of_and_cod_of_agree(morphism):
    assert ops.signature(morphism.node) == (ops.dom_of(morphism.node), ops.cod_of(morphism.node))


@settings(max_examples=60)
@given(type_values(), type_values())
def test_prim_type_law(domain, codomain):
    raw = object()
    node = expr.Prim(raw, domain, codomain)

    assert ops.signature(node) == (domain, codomain)
    assert node.raw is raw


@settings(max_examples=60)
@given(type_values(), type_values(), type_values())
def test_contextual_nodes_report_stored_domain_and_codomain(domain, codomain, param):
    f = expr.Identity(domain)
    g = expr.Identity(domain)
    node = expr.Pair(f, g, UNIT, UNIT, param, None, domain, codomain)

    assert ops.signature(node) == (domain, codomain)


@settings(max_examples=60)
@given(plain_morphisms())
def test_monadic_embed_type_law(morphism):
    embedded = expr.MonadicEmbed(morphism.node, MAYBE)

    assert ops.dom_of(embedded) == ops.dom_of(morphism.node)
    assert ops.cod_of(embedded) == TypeMaybe(ops.cod_of(morphism.node))


def test_unknown_type_and_poly_nodes_are_rejected():
    with pytest.raises(TypeError):
        ops.signature(expr.MorphismExpr())

    with pytest.raises(TypeError):
        sem.apply_poly(expr.PolyExpr(), UNIT)


@settings(max_examples=60)
@given(plain_morphisms())
def test_plain_morphism_wrapper_returns_raw_domain_and_codomain(morphism):
    assert morphism.dom() == ops.dom_of(morphism.node)
    assert morphism.cod() == ops.cod_of(morphism.node)


@settings(max_examples=60)
@given(parametric_plain_morphisms())
def test_parametric_morphism_wrapper_strips_param_prefix(morphism):
    assume(morphism.param != UNIT)
    raw_domain = ops.dom_of(morphism.node)

    assert morphism.dom() == raw_domain.value.second


@settings(max_examples=60)
@given(type_values(), type_values(), type_values())
def test_parametric_morphism_rejects_wrong_raw_domain(param, raw_domain, codomain):
    assume(param != UNIT)
    morphism = ops.Morphism(expr.Prim(object(), raw_domain, codomain), param=param)

    if not isinstance(raw_domain, TypePair) or raw_domain.value.first != param:
        with pytest.raises(ops.MorphismError):
            morphism.dom()


@settings(max_examples=60)
@given(maybe_lax_morphisms())
def test_lax_morphism_wrapper_strips_monad_wrapper(morphism):
    assert morphism.cod() == ops.cod_of(morphism.node).value


@settings(max_examples=60)
@given(type_values(), type_values())
def test_lax_morphism_rejects_wrong_raw_codomain(domain, codomain):
    morphism = ops.Morphism(expr.Prim(object(), domain, codomain), monad=MAYBE)

    if not isinstance(codomain, TypeMaybe):
        with pytest.raises(ops.MorphismError):
            morphism.cod()


@settings(max_examples=60)
@given(plain_morphisms())
def test_node_in_and_to_lax_laws_for_plain_morphisms(morphism):
    assert morphism.node_in(None) is morphism.node

    embedded_node = morphism.node_in(MAYBE)
    assert isinstance(embedded_node, expr.MonadicEmbed)
    assert embedded_node.f is morphism.node
    assert embedded_node.monad is MAYBE

    lax = morphism.to_lax(MAYBE)
    assert lax.param == morphism.param
    assert lax.monad is MAYBE
    assert lax.aux_primitives == morphism.aux_primitives


@settings(max_examples=60)
@given(maybe_lax_morphisms())
def test_node_in_rejects_different_existing_monad(morphism):
    with pytest.raises(ops.MorphismError):
        morphism.node_in(LIST)


@settings(max_examples=60)
@given(composable_plain_pairs())
def test_compose_derives_type_and_preserves_aux(pair):
    f, g = pair
    f = ops.Morphism(f.node, aux_primitives=("f",))
    g = ops.Morphism(g.node, aux_primitives=("g",))
    composed = ops.compose(f, g)

    assert composed.dom() == f.dom()
    assert composed.cod() == g.cod()
    assert composed.aux_primitives == ("f", "g")


@settings(max_examples=60)
@given(type_values(), type_values())
def test_identity_is_neutral_for_composition_type_structure(domain, codomain):
    f = ops.Morphism(expr.Prim(object(), domain, codomain))

    left = ops.compose(ops.identity(domain), f)
    right = ops.compose(f, ops.identity(codomain))

    assert (left.dom(), left.cod()) == (f.dom(), f.cod())
    assert (right.dom(), right.cod()) == (f.dom(), f.cod())


@settings(max_examples=60)
@given(type_values(), type_values(), type_values(), type_values())
def test_compose_is_associative_for_type_structure(a, b, c, d):
    f = ops.Morphism(expr.Prim(object(), a, b))
    g = ops.Morphism(expr.Prim(object(), b, c))
    h = ops.Morphism(expr.Prim(object(), c, d))

    left = ops.compose(ops.compose(f, g), h)
    right = ops.compose(f, ops.compose(g, h))

    assert (left.dom(), left.cod()) == (right.dom(), right.cod()) == (a, d)


@settings(max_examples=60)
@given(type_values(), type_values(), type_values(), type_values())
def test_plain_combinators_derive_expected_types(a, b, c, d):
    f = ops.Morphism(expr.Prim(object(), a, b))
    g = ops.Morphism(expr.Prim(object(), c, d))
    h = ops.Morphism(expr.Prim(object(), b, c))

    composed = ops.compose(f, h)
    assert composed.dom() == a
    assert composed.cod() == c

    parallel = ops.par(f, g)
    assert parallel.dom() == ops.ProductType(a, c)
    assert parallel.cod() == ops.ProductType(b, d)

    paired = ops.pair(f, ops.Morphism(expr.Prim(object(), a, d)))
    assert paired.dom() == a
    assert paired.cod() == ops.ProductType(b, d)

    cased = ops.case(f, ops.Morphism(expr.Prim(object(), c, b)))
    assert cased.dom() == ops.SumType(a, c)
    assert cased.cod() == b


@settings(max_examples=60)
@given(type_values(), type_values(), type_values(), type_values())
def test_rejects_incompatible_compose_pair_and_case(a, b, c, d):
    f = ops.Morphism(expr.Prim(object(), a, b))
    g = ops.Morphism(expr.Prim(object(), c, d))

    if b != c:
        with pytest.raises(ops.MorphismError):
            ops.compose(f, g)

    if a != c:
        with pytest.raises(ops.MorphismError):
            ops.pair(f, g)

    if b != d:
        with pytest.raises(ops.MorphismError):
            ops.case(f, g)


@settings(max_examples=60)
@given(same_domain_plain_pairs())
def test_pair_validity_and_type_law(pair):
    f, g = pair
    paired = ops.pair(f, g)

    assert paired.dom() == f.dom()
    assert paired.cod() == ops.ProductType(f.cod(), g.cod())


@settings(max_examples=60)
@given(same_codomain_plain_pairs())
def test_case_validity_and_type_law(pair):
    f, g = pair
    cased = ops.case(f, g)

    assert cased.dom() == ops.SumType(f.dom(), g.dom())
    assert cased.cod() == f.cod()


@settings(max_examples=60)
@given(type_values(), type_values())
def test_param_combination_law(p, q):
    assert Ty.combine_params(UNIT, p) == p
    assert Ty.combine_params(p, UNIT) == p
    assert Ty.combine_params(p, q) == (q if p == UNIT else p if q == UNIT else ops.ProductType(q, p))


@settings(max_examples=60)
@given(type_values(), type_values(), type_values())
def test_shared_context_compose_plain_type_law(a, b, c):
    f = ops.Morphism(expr.Prim(object(), a, b))
    g = ops.Morphism(expr.Prim(object(), b, c))
    composed = ops.compose(f, g, shared_context=True)

    assert composed.param == UNIT
    assert composed.monad is None
    assert composed.dom() == a
    assert composed.cod() == c


@settings(max_examples=60)
@given(type_values(), type_values(), type_values(), type_values())
def test_shared_context_compose_shares_matching_param(p, a, b, c):
    assume(p != UNIT)
    f = ops.Morphism(expr.Prim(object(), ops.ProductType(p, a), b), param=p)
    g = ops.Morphism(expr.Prim(object(), ops.ProductType(p, b), c), param=p)
    composed = ops.compose(f, g, shared_context=True)

    assert composed.param == p
    assert composed.monad is None
    assert composed.dom() == a
    assert composed.cod() == c


@settings(max_examples=60)
@given(type_values(), type_values(), type_values(), type_values(), type_values())
def test_shared_context_compose_rejects_distinct_params(p, q, a, b, c):
    assume(p != q)
    assume(p != UNIT)
    assume(q != UNIT)
    f = ops.Morphism(expr.Prim(object(), ops.ProductType(p, a), b), param=p)
    g = ops.Morphism(expr.Prim(object(), ops.ProductType(q, b), c), param=q)

    with pytest.raises(ops.MorphismError):
        ops.compose(f, g, shared_context=True)


@settings(max_examples=60)
@given(type_values(), type_values(), type_values())
def test_shared_context_compose_preserves_lax_context(a, b, c):
    f = ops.Morphism(expr.Prim(object(), a, MAYBE.wrap(b)), monad=MAYBE)
    g = ops.Morphism(expr.Prim(object(), b, MAYBE.wrap(c)), monad=MAYBE)
    composed = ops.compose(f, g, shared_context=True)

    assert composed.param == UNIT
    assert composed.monad is MAYBE
    assert composed.dom() == a
    assert composed.cod() == c


@settings(max_examples=60)
@given(type_values(), type_values(), type_values(), type_values())
def test_shared_context_compose_preserves_lax_para_context(p, a, b, c):
    assume(p != UNIT)
    f = ops.Morphism(expr.Prim(object(), ops.ProductType(p, a), MAYBE.wrap(b)), param=p, monad=MAYBE)
    g = ops.Morphism(expr.Prim(object(), ops.ProductType(p, b), MAYBE.wrap(c)), param=p, monad=MAYBE)
    composed = ops.compose(f, g, shared_context=True)

    assert composed.param == p
    assert composed.monad is MAYBE
    assert composed.dom() == a
    assert composed.cod() == c


@settings(max_examples=60)
@given(type_values(), type_values(), type_values(), type_values(), type_values())
def test_shared_context_applies_to_all_contextual_combinators(p, a, b, c, d):
    assume(p != UNIT)
    f = ops.Morphism(expr.Prim(object(), ops.ProductType(p, a), b), param=p)
    g = ops.Morphism(expr.Prim(object(), ops.ProductType(p, c), d), param=p)

    parallel = ops.par(f, g, shared_context=True)
    assert parallel.param == p
    assert parallel.dom() == ops.ProductType(a, c)
    assert parallel.cod() == ops.ProductType(b, d)

    left = ops.Morphism(expr.Prim(object(), ops.ProductType(p, a), b), param=p)
    right = ops.Morphism(expr.Prim(object(), ops.ProductType(p, a), c), param=p)
    paired = ops.pair(left, right, shared_context=True)
    assert paired.param == p
    assert paired.dom() == a
    assert paired.cod() == ops.ProductType(b, c)

    first = ops.Morphism(expr.Prim(object(), ops.ProductType(p, a), b), param=p)
    second = ops.Morphism(expr.Prim(object(), ops.ProductType(p, c), b), param=p)
    cased = ops.case(first, second, shared_context=True)
    assert cased.param == p
    assert cased.dom() == ops.SumType(a, c)
    assert cased.cod() == b


@settings(max_examples=60)
@given(plain_morphisms(), maybe_lax_morphisms())
def test_plain_morphism_auto_embeds_into_lax_context(plain, lax):
    bridge = ops.Morphism(expr.Prim(object(), plain.cod(), lax.dom()))
    composed = ops.compose(ops.compose(plain, bridge).to_lax(MAYBE), lax)

    assert composed.monad is MAYBE
    assert composed.dom() == plain.dom()
    assert composed.cod() == lax.cod()


@settings(max_examples=60)
@given(maybe_lax_morphisms(), list_lax_morphisms())
def test_conflicting_monads_are_rejected(maybe_morphism, list_morphism):
    with pytest.raises(ops.MorphismError):
        ops.par(maybe_morphism, list_morphism)


@settings(max_examples=60)
@given(type_values(), poly_values(), poly_values())
def test_polyexpr_constructor_functions_return_matching_nodes(space, left, right):
    assert isinstance(sem.zero(), expr.Zero)
    assert isinstance(sem.one(), expr.One)
    assert isinstance(sem.id_(), expr.Id)
    assert sem.const(space) == expr.Const(space)
    assert sem.sum_(left, right) == expr.Sum(left, right)
    assert sem.prod(left, right) == expr.Prod(left, right)
    assert sem.exp(space, left) == expr.Exp(space, left)


@settings(max_examples=60)
@given(poly_values())
def test_functor_category_validation(body):
    assert sem.Functor("F", body, category="set").body == body

    if isinstance(body, expr.Id):
        assert sem.Functor("F", body, category="poset").body == body
    else:
        with pytest.raises(ValueError):
            sem.Functor("F", body, category="poset")
