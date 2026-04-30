from __future__ import annotations

from hydra.core import Name
import hydra.dsl.terms as Terms

from unialg.algebra.sort import ProductSort
from ._typed_morphism import TypedMorphism

_LENS_TYPE = Name("ua.morphism.Lens")

T = TypedMorphism


def _split_residual_focus(sort, label: str):
    """Destructure a 2-element ``ProductSort`` into ``(residual, focus)``."""
    if not isinstance(sort, ProductSort):
        raise TypeError(
            f"{label}: expected ProductSort of (residual, focus), got {sort!r}"
        )
    if len(sort.elements) != 2:
        raise TypeError(
            f"{label}: expected 2-element ProductSort, got "
            f"{len(sort.elements)} elements"
        )
    return sort.elements[0], sort.elements[1]


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

    fwd_residual, _focus_in = _split_residual_focus(
        forward.codomain, "lens.forward.codomain")
    bwd_residual, _focus_out = _split_residual_focus(
        backward.domain, "lens.backward.domain")
    T.same_sort(fwd_residual, bwd_residual, "lens.residual")

    return T(_lens_term(forward, backward), forward.domain, backward.codomain)
