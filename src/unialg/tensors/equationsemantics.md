# Einsum semantics

## intent

`Einsum` is a morphism expression.

It is not meant to introduce a separate semantic system.
Its meaning should be accounted for within the existing morphism semantics.

## semantic shape

An `Einsum` expression should eventually determine:

- a visible domain
- a visible codomain
- any contextual parameter behavior, if applicable
- any monadic/lax behavior, if applicable

These should align with the existing `Morphism` wrapper semantics.

## interpretation strategy

`Einsum` is not explained by special-case composition rules.

Instead, its semantics should be understood by relating it to the existing
morphism machinery:

- `signature`
- `dom_of`
- `cod_of`
- `compose`
- `par`
- `pair`
- contextual parameter handling
- monad resolution

## placeholder law

The semantics of `Einsum` must be expressible in terms of the existing morphism
framework.

Exact elaboration is intentionally left unspecified here.

## constraints

Any eventual semantics for `Einsum` must preserve:

- object-level domain/codomain discipline
- compatibility with sequential composition
- compatibility with parallel composition
- compatibility with visible/raw signature conventions
- compatibility with parameterized morphisms
- compatibility with lax/effectful morphisms

## deferred

This placeholder does not yet specify:

- the exact elaboration of einsum structure
- index validation rules
- tensor object/type representation
- backend realization
- normalization or optimization rules
