"""Polynomial endofunctor data layer.

Functor and PolyExpr are Hydra-backed: PolyExpr wraps a TermInject of union
type ua.functor.PolyExpr; Functor is a record view of ua.functor.Functor.
Sort references in const / exp variants carry typed Sort terms, matching
the pattern used by Equation.domain_sort and Lens.residual_sort.
"""
import pytest

from unialg import Sort, Semiring
from unialg.morphism.functor import (
    Functor,
    PolyExpr,
    POLY_TYPE_NAME,
    zero, one, id_, const, sum_, prod, exp,
    pretty,
)


# ---------------------------------------------------------------------------
# Local fixtures — sort references for polynomial constructors
# ---------------------------------------------------------------------------

@pytest.fixture
def base_sort(real_sr):
    return Sort("base", real_sr)


@pytest.fixture
def output_sort(real_sr):
    return Sort("output", real_sr)


@pytest.fixture
def input_sort(real_sr):
    return Sort("input", real_sr)


# ---------------------------------------------------------------------------
# Construction & variant kinds
# ---------------------------------------------------------------------------

class TestPolyExprKinds:
    """Each constructor produces the expected variant tag."""

    def test_zero(self):
        assert zero().kind == "zero"

    def test_one(self):
        assert one().kind == "one"

    def test_id(self):
        assert id_().kind == "id"

    def test_const(self, base_sort):
        e = const(base_sort)
        assert e.kind == "const"
        assert e.sort.name == "base"

    def test_sum(self):
        e = sum_(one(), id_())
        assert e.kind == "sum"
        assert e.left.kind == "one"
        assert e.right.kind == "id"

    def test_prod(self, base_sort):
        e = prod(const(base_sort), id_())
        assert e.kind == "prod"
        assert e.left.sort.name == "base"
        assert e.right.kind == "id"

    def test_exp(self, input_sort, output_sort):
        e = exp(input_sort, prod(const(output_sort), id_()))
        assert e.kind == "exp"
        assert e.base_sort.name == "input"
        assert e.body.kind == "prod"


class TestPolyExprAccessorErrors:
    """Wrong-kind accessors raise AttributeError with a clear message."""

    def test_sort_only_for_const(self):
        with pytest.raises(AttributeError, match="kind='one'"):
            _ = one().sort

    def test_left_only_for_sum_prod(self):
        with pytest.raises(AttributeError, match="kind='id'"):
            _ = id_().left

    def test_base_sort_only_for_exp(self, base_sort):
        with pytest.raises(AttributeError, match="kind='const'"):
            _ = const(base_sort).base_sort

    def test_body_only_for_exp(self):
        with pytest.raises(AttributeError, match="kind='zero'"):
            _ = zero().body


# ---------------------------------------------------------------------------
# Hydra union encoding
# ---------------------------------------------------------------------------

class TestHydraEncoding:
    """The wrapped term is genuinely a TermInject of the polynomial union."""

    def test_term_is_term_inject(self):
        import hydra.core as core
        e = sum_(one(), id_())
        assert isinstance(e.term, core.TermInject)
        injection = e.term.value
        assert injection.type_name == POLY_TYPE_NAME
        assert injection.field.name.value == "sum"

    def test_unit_payload_for_nullary(self):
        import hydra.core as core
        for ctor, kind in [(zero, "zero"), (one, "one"), (id_, "id")]:
            term = ctor().term
            assert term.value.field.name.value == kind
            assert isinstance(term.value.field.term, core.TermUnit)

    def test_pair_payload_for_sum_prod(self):
        import hydra.core as core
        for ctor in (sum_, prod):
            e = ctor(one(), id_())
            assert isinstance(e._payload, core.TermPair)

    def test_pair_payload_for_exp(self, base_sort):
        import hydra.core as core
        e = exp(base_sort, id_())
        assert isinstance(e._payload, core.TermPair)


# ---------------------------------------------------------------------------
# Equality and hashing — content-based via underlying term
# ---------------------------------------------------------------------------

class TestEquality:

    def test_eq_by_content(self, base_sort):
        a = sum_(one(), prod(const(base_sort), id_()))
        b = sum_(one(), prod(const(base_sort), id_()))
        assert a == b
        assert hash(a) == hash(b)

    def test_neq_by_content(self):
        a = sum_(one(), id_())
        b = sum_(id_(), one())
        assert a != b

    def test_const_neq_different_sorts(self, base_sort, output_sort):
        assert const(base_sort) != const(output_sort)

    def test_eq_rejects_non_polyexpr(self):
        assert one() != "one"
        assert one() != 1


# ---------------------------------------------------------------------------
# Summand extraction
# ---------------------------------------------------------------------------

class TestSummands:
    """summands() flattens nested sum variants."""

    def test_list_endo(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        ss = f.summands()
        assert len(ss) == 2
        assert ss[0].kind == "one"
        assert ss[1].kind == "prod"

    def test_tree_endo(self, base_sort):
        f = Functor("F_tree", sum_(const(base_sort), prod(id_(), id_())))
        ss = f.summands()
        assert len(ss) == 2
        assert ss[0].kind == "const"
        assert ss[1].kind == "prod"

    def test_single_summand_no_sum(self, output_sort):
        f = Functor("F_stream", prod(const(output_sort), id_()))
        assert len(f.summands()) == 1
        assert f.summands()[0].kind == "prod"

    def test_three_way_sum_flattens(self, real_sr):
        a, b, c = (Sort(n, real_sr) for n in ("a", "b", "c"))
        # A + B + C parses left-associated as Sum(Sum(A, B), C)
        f = Functor("F_three", sum_(sum_(const(a), const(b)), const(c)))
        ss = f.summands()
        assert len(ss) == 3
        assert [s.sort.name for s in ss] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

class TestStructuralProperties:

    def test_x_arity_list(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        assert f.x_arity() == 1

    def test_x_arity_tree(self, base_sort):
        f = Functor("F_tree", sum_(const(base_sort), prod(id_(), id_())))
        assert f.x_arity() == 2

    def test_x_arity_constant_functor(self, base_sort):
        f = Functor("F_const", const(base_sort))
        assert f.x_arity() == 0
        assert not f.is_recursive()

    def test_is_recursive(self, base_sort):
        assert Functor("F_id", id_()).is_recursive()
        assert Functor("F_list",
                       sum_(one(), prod(const(base_sort), id_()))).is_recursive()
        assert not Functor("F_const", const(base_sort)).is_recursive()

    def test_consts_returns_typed_sorts(self, input_sort, output_sort):
        f = Functor("F_mealy", exp(input_sort, prod(const(output_sort), id_())))
        sorts = f.consts()
        assert len(sorts) == 2
        names = [s.name for s in sorts]
        assert names == ["input", "output"]
        # And they are real Sort objects
        from unialg import Sort as _Sort
        assert all(isinstance(s, _Sort) for s in sorts)

    def test_no_consts_when_only_id(self):
        assert Functor("F_id", id_()).consts() == ()

    def test_no_consts_when_only_one(self):
        assert Functor("F_one", one()).consts() == ()


# ---------------------------------------------------------------------------
# Functor record-view roundtrip
# ---------------------------------------------------------------------------

class TestFunctorRoundtrip:
    """Functor wraps a Hydra record term; from_term reconstructs it."""

    def test_field_access(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        assert f.name == "F_list"
        assert f.category == "set"
        assert f.body.kind == "sum"

    def test_explicit_category(self):
        f = Functor("F_poset", id_(), category="poset")
        assert f.category == "poset"

    def test_from_term_idempotent(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        f2 = Functor.from_term(f.term)
        assert f2.name == f.name
        assert f2.category == f.category
        assert f2.body == f.body

    def test_from_term_returns_existing(self):
        f = Functor("F_list", id_())
        assert Functor.from_term(f) is f

    def test_const_roundtrips_a_sort(self, base_sort):
        f = Functor("F_const", const(base_sort))
        roundtripped = Functor.from_term(f.term).body.sort
        # Same sort name + semiring after roundtrip
        assert roundtripped.name == "base"
        assert roundtripped.semiring_name == base_sort.semiring_name


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_set_id_ok(self):
        Functor("F_id", id_()).validate()

    def test_set_complex_ok(self, base_sort):
        Functor("F_list", sum_(one(), prod(const(base_sort), id_()))).validate()

    def test_poset_id_ok(self):
        Functor("F_poset", id_(), category="poset").validate()

    def test_poset_with_non_id_rejected(self):
        f = Functor("F_bad", sum_(one(), id_()), category="poset")
        with pytest.raises(ValueError, match="poset requires body=X"):
            f.validate()

    def test_unknown_category_rejected(self):
        f = Functor("F_weird", id_(), category="profunctor")
        with pytest.raises(ValueError, match="category must be 'set' or 'poset'"):
            f.validate()


# ---------------------------------------------------------------------------
# Pretty-printing — the polynomial surface form
# ---------------------------------------------------------------------------

class TestPretty:

    def test_list_endo(self, base_sort):
        f = Functor("F_list", sum_(one(), prod(const(base_sort), id_())))
        assert repr(f) == "functor F_list : 1 + base * X"

    def test_tree_endo(self, base_sort):
        f = Functor("F_tree", sum_(const(base_sort), prod(id_(), id_())))
        assert repr(f) == "functor F_tree : base + X * X"

    def test_stream_endo(self, output_sort):
        f = Functor("F_stream", prod(const(output_sort), id_()))
        assert repr(f) == "functor F_stream : output * X"

    def test_id(self):
        assert repr(Functor("F_id", id_())) == "functor F_id : X"

    def test_zero(self):
        assert repr(Functor("F_zero", zero())) == "functor F_zero : 0"

    def test_poset(self):
        assert repr(Functor("F_poset", id_(), category="poset")) \
            == "functor F_poset : X [category=poset]"

    def test_sum_inside_prod_parenthesises(self, base_sort, output_sort):
        # (A + B) * X — sum on the left of a prod, must be parenthesised
        f = Functor("F_par", prod(sum_(const(base_sort), const(output_sort)), id_()))
        assert repr(f) == "functor F_par : (base + output) * X"

    def test_exp_with_compound_body_parenthesises(self, input_sort, output_sort):
        f = Functor("F_mealy", exp(input_sort, prod(const(output_sort), id_())))
        assert repr(f) == "functor F_mealy : input -> (output * X)"
