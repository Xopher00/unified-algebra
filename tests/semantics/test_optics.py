"""Tests for Optic combinators, focusing on the Optic.par (product of optics)."""
import pytest

from unialg.semantics.optics import Optic
from unialg.semantics.functors import Functor
from unialg.semantics import morphisms as ops
from unialg.syntax import expressions as expr
from unialg.objects import BINARY


def _id_functor(name: str) -> Functor:
    return Functor(name=name, body=expr.Id())


def _prod_functor(name: str) -> Functor:
    return Functor(name=name, body=expr.Prod(expr.Id(), expr.Id()))


def _carrier_optic(functor: Functor) -> Optic:
    """Build a trivial optic with forward/backward = identity at F(BINARY)."""
    fa = functor.apply(BINARY)
    fwd = ops.identity(fa)
    bwd = ops.identity(fa)
    return Optic(functor=functor, forward=fwd, backward=bwd, carrier=BINARY)


class TestOpticPar:
    def test_par_combines_functor_bodies(self):
        F = _id_functor("F")
        G = _id_functor("G")
        O_L = _carrier_optic(F)
        O_R = _carrier_optic(G)

        O = O_L.par(O_R)
        assert O.functor.body == expr.Prod(expr.Id(), expr.Id())

    def test_par_forward_backward_are_par_of_components(self):
        F = _id_functor("F")
        G = _id_functor("G")
        O_L = _carrier_optic(F)
        O_R = _carrier_optic(G)

        O = O_L.par(O_R)
        expected_fwd = ops.par(O_L.forward, O_R.forward)
        expected_bwd = ops.par(O_L.backward, O_R.backward)

        assert O.forward.dom() == expected_fwd.dom()
        assert O.forward.cod() == expected_fwd.cod()
        assert O.backward.dom() == expected_bwd.dom()
        assert O.backward.cod() == expected_bwd.cod()

    def test_par_carrier_is_shared_carrier(self):
        # par requires both optics to share the same carrier type X.
        # The combined carrier is X, not Prod(X, X): the functor Prod(L,R) still
        # maps X → apply_poly(L,X) × apply_poly(R,X), so X (not X×X) is the
        # common element type for both positions.
        F = _id_functor("F")
        G = _id_functor("G")
        O_L = _carrier_optic(F)
        O_R = _carrier_optic(G)

        O = O_L.par(O_R)
        assert O.carrier == BINARY

    def test_par_passes_optic_validation(self):
        F = _prod_functor("F")
        G = _id_functor("G")
        O_L = _carrier_optic(F)
        O_R = _carrier_optic(G)

        # Optic.__post_init__ validates via functor.unapply — must not raise.
        O = O_L.par(O_R)
        assert O.functor.body == expr.Prod(expr.Prod(expr.Id(), expr.Id()), expr.Id())

    def test_par_source_and_target(self):
        from unialg.objects import ProductType
        F = _id_functor("F")
        G = _id_functor("G")
        O_L = _carrier_optic(F)
        O_R = _carrier_optic(G)

        O = O_L.par(O_R)
        assert O.source == ProductType(BINARY, BINARY)
        assert O.target == ProductType(BINARY, BINARY)

    def test_par_without_carrier_yields_none(self):
        F = _id_functor("F")
        fwd = ops.identity(BINARY)
        bwd = ops.identity(BINARY)
        O_no_carrier = Optic(functor=F, forward=fwd, backward=bwd, carrier=None)

        O = O_no_carrier.par(O_no_carrier)
        assert O.carrier is None

    def test_par_compose_associativity_of_functor_bodies(self):
        """par of two composed optics has a Prod of composed bodies."""
        F = _id_functor("F")
        G = _id_functor("G")
        O_base = _carrier_optic(_prod_functor("H"))
        O_L = _carrier_optic(F)
        O_R = _carrier_optic(G)

        # (O_base ∘ O_L) | (O_base ∘ O_R)
        composed_L = O_base.compose(O_L)
        composed_R = O_base.compose(O_R)
        O = composed_L.par(composed_R)

        # The functor bodies should be Prod of composed bodies
        assert isinstance(O.functor.body, expr.Prod)
        assert O.functor.body.left == composed_L.functor.body
        assert O.functor.body.right == composed_R.functor.body
