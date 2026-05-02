from __future__ import annotations

from ._typed_morphism import TypedMorphism, Terms, Name

_LENS_TYPE = Name("ua.morphism.Lens")

T = TypedMorphism

def _lens_term(forward, backward):
    """Lens record term: ``{forward: <term>, backward: <term>}``.

    Sort metadata (source/target/focus/residual) lives on the wrapping
    ``TypedMorphism``, not in the runtime record. Embedding sorts in the term
    would double-encode information already carried at the type level.
    """
    return Terms.record(_LENS_TYPE, [
        Terms.field("forward", forward.term),
        Terms.field("backward", backward.term),
    ])

def lens(forward, backward) -> TypedMorphism:
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
        _lens_term(forward, backward),
        forward.domain, backward.codomain,
    )