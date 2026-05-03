from __future__ import annotations

import hydra.dsl.terms as Terms

from unialg.algebra.sort import ProductSort
from ._typed_morphism import TypedMorphism, Name

_LENS_TYPE = Name("ua.morphism.Lens")
_LENS_SEQ_TYPE = Name("ua.morphism.LensSeq")

T = TypedMorphism

def _lens_term(forward, backward, residual_sort=None):
    fields = [
        Terms.field("forward", forward.term),
        Terms.field("backward", backward.term),
    ]
    if residual_sort is not None:
        from unialg.terms import _RecordView
        fields.append(Terms.field("residualSort", _RecordView._unwrap(residual_sort)))
    return Terms.record(_LENS_TYPE, fields)

def lens(forward, backward, residual_sort=None) -> TypedMorphism:
    """Lens morphism ``Lens(S, T; A, B, R) : S → T``.

    Sorts are derived from the typed forward and backward morphisms:

    - Forward must be ``S → R × A`` — codomain is a 2-element ``ProductSort``
      whose first element is the residual ``R`` and second is the
      forward focus ``A``.
    - Backward must be ``R × B → T`` — domain is a 2-element ``ProductSort``
      whose first element is the residual ``R`` and second is the
      backward focus ``B``.
    - The residual ``R`` must be the same in both products.

    The resulting lens has type ``S → T`` (forward.domain → backward.codomain).

    Assembly contract (enforced in ``assembly._morphism_compile._try_lens``):
    At runtime the forward callable must return a Python 2-tuple ``(residual, focus)``
    and the backward callable must accept a 2-tuple ``(residual, focus_prime)`` and
    return the updated input.  These conventions are the glue that lets
    ``lens_seq`` pair residuals across multiple lenses in sequence.
    """
    forward = T.require(forward, "lens.forward")
    backward = T.require(backward, "lens.backward")

    fwd_residual, _focus_in = T.split_product2(
        forward.codomain, "lens.forward.codomain",
    )
    bwd_residual, _focus_out = T.split_product2(
        backward.domain, "lens.backward.domain",
    )
    T.same_sort(fwd_residual, bwd_residual, "lens.residual")

    return T(
        _lens_term(forward, backward, residual_sort),
        forward.domain, backward.codomain,
    )


def lens_seq(l1: TypedMorphism, l2: TypedMorphism) -> TypedMorphism:
    """Sequential composition of two lenses.

    Forward: ``(r1, a) = l1.fwd(s); (r2, b) = l2.fwd(a); return ((r1, r2), b)``
    Backward: ``a' = l2.bwd((r2, b')); s' = l1.bwd((r1, a')); return s'``

    The ``residual_sort`` of the result is ``ProductSort([l1.residual, l2.residual])``
    when both lenses carry a residual sort, else ``None``.

    Assembly contract (enforced in ``assembly._morphism_compile._try_lens_seq``):
    The forward pass pairs residuals left-to-right as ``(r1, r2)`` and returns
    them alongside the final focus ``b``.  The backward pass must consume the
    paired residuals in reverse order: l2.bwd receives ``(r2, b')`` first,
    then l1.bwd receives ``(r1, a')``.  The outer residual sort of the compiled
    lens is ``ProductSort([l1.residual_sort, l2.residual_sort])``.
    """
    l1 = T.require(l1, "lens_seq.l1")
    l2 = T.require(l2, "lens_seq.l2")
    T.same_sort(l1.codomain, l2.domain, "lens_seq.junction")
    term = Terms.record(_LENS_SEQ_TYPE, [
        Terms.field("first", l1.term),
        Terms.field("second", l2.term),
    ])
    return T(term, l1.domain, l2.codomain)