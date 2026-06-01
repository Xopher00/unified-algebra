# Haskell Branch Overview

## Purpose

A mathematician writes a program in ordinary Haskell.
GHC type-checks it.
The Haskell runtime evaluates the expressions, which build Hydra IR.
Hydra renders that IR to executable backend code.
Generated examples are checked with differential tests under `explore/archs/`.

The pipeline is domain-agnostic. ML backends are one choice, not the only
choice; a user can target graph algorithms, signal processing, tropical
combinatorics, or any other domain by supplying a custom backend JSON file.
The aim is to make the classification of structure explicit and transferable
across domains. An architecture is a coordinate on independent axes such as
functor shape, semiring, and direction, regardless of whether the application is
ML or not.

## Pipeline

The Haskell surface is the DSL. Types, functor structure, recursion direction,
and tensor equations are expressed directly in Haskell, then evaluated into
Hydra IR for backend lowering and code generation.

```haskell
surface expression → typed interpretation → algebraic construction → executable assembly → backend realization
```

JSON backend specs map logical op names to backend-specific paths. Lowering
rewrites those names before codegen, so the same architecture can target
different libraries or domains by changing the backend spec.
