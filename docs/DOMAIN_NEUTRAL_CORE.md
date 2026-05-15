# Domain-Neutral Core and Future Domain Extensions

## Overview

This project is best understood as a typed compositional DSL implemented in Python. It provides
a pipeline for describing, composing, lowering, and executing typed transformations through
interchangeable backends. Machine learning — specifically tensor contraction and semiring-based
matrix products — is one important use case, but the deeper architectural goal is more general:
any domain that can be expressed as typed objects, primitive morphisms, and their compositions
can be hosted here without modifying the core.

---

## Current Layer Map

```
surface syntax (DSL source)
    ↓
syntax/         — tokenize, parse → MorphismExpr / PolyExpr trees
    ↓
semantics/      — type-check, derive dom/cod, build Morphism handles
    ↓
structure/      — lower to raw Hydra terms; realize recursive schemes
    ↓
emitters/       — bind backend JSON specs → Hydra primitives + Morphisms
    ↓
Hydra runtime   — evaluate / reduce terms
```

| Package | Key files | Responsibility |
|---|---|---|
| `unialg/` | `main.py`, `objects.py` | Entry point, object-space type constructors, monad descriptors |
| `unialg/syntax/` | `expressions.py`, `parse.py` | ADT nodes for morphisms and polynomial functors; Pratt parser |
| `unialg/semantics/` | `morphisms.py`, `functors.py`, `optics.py`, `typeops.py` | Type derivation, composition checking, functor object action, optic algebra |
| `unialg/structure/` | `realize.py`, `terms.py`, `recursion.py` | Translation of validated expression trees to raw Hydra lambda terms |
| `unialg/emitters/` | `backend.py`, `codecs.py`, `backends/*.json` | Load JSON specs, register backend primitives, wrap as typed Morphisms |
| `unialg/tensors/` | `semirings.py`, `contractions.py` | Domain-adjacent: semiring algebra and tensor contraction over backends |

---

## Layer Responsibilities

### `syntax/` — Surface representation

The syntax layer owns two expression families:

- **`MorphismExpr`** — arrow nodes: identity, copy, delete, product/sum structure, composition,
  parallel, pair, case, polynomial functor actions (`PolyFmap`), recursive schemes (`Cata`, `Ana`),
  raw backend escapes (`Prim`), and name references (`Ref`).
- **`PolyExpr`** — polynomial functor nodes: `Zero`, `One`, `Id`, `Const`, `Sum`, `Prod`, `Exp`,
  `List`, `Maybe`, and name references (`PolyRef`).

The parser produces `dom`/`cod` placeholders (`TypeUnit()`) on binary nodes — it does not resolve
types. Unresolved name references (`Ref`, `PolyRef`) survive parsing and are resolved by the
semantic layer. The syntax layer does not import from `semantics/`.

The surface language accepts three declaration kinds:

```
load  BACKEND           — load backend primitives into parser env
route NAME = <expr>     — define a named morphism
map   NAME = <functor>  — define a named polynomial functor
```

### `semantics/` — Type derivation and algebraic meaning

The semantics layer gives expression nodes their types. It does not emit Hydra terms.

- **`morphisms.py`** — resolves `dom`/`cod` for every `MorphismExpr`; enforces compatibility
  at composition sites; constructs `Morphism` handles that carry the node plus its parametric
  context (`param`, `monad`, `aux_primitives`).
- **`functors.py`** — `Functor` descriptor wrapping a `PolyExpr`; computes object action
  `F(A)`, functor composition, summand decomposition, and recursion variable arity.
- **`optics.py`** — `Optic` descriptor: `(functor, forward, backward)` triples. Defines lens,
  prism, and traversal as polynomial functor optics; does not produce Hydra terms.
- **`typeops.py`** — type-unification utilities over Hydra's `Type` objects; provides the
  empty base `Graph` (`_EMPTY_GRAPH`) used during lowering setup.

### `structure/` — Lowering to Hydra terms

The structure layer translates validated expression trees into raw Hydra lambda terms.

- **`realize.py`** — the main dispatch: every `MorphismExpr` variant maps to a Hydra term
  combinator. Contextual nodes (composition, parallel, pair, case) are realized with
  parameter-splitting so that parametric morphisms propagate context correctly. Recursive
  schemes (`Cata`, `Ana`) are realized as self-referential Hydra `Primitive` objects registered
  on-the-fly.
- **`terms.py`** — Hydra term combinators (lambda helpers, pair/either constructors, monadic
  bind/pure wrappers, structural normalization).
- **`recursion.py`** — optic action: applying an optic to a morphism via functor action.

### `emitters/` — Backend primitive binding

The emitter layer connects the DSL to executable Python implementations.

- **`backend.py`** — loads a JSON spec, resolves dotted import paths, wraps callables as Hydra
  `Primitive` objects, and exposes them as `Morphism`s via `BackendOps`. The `BackendOps` class
  is the canonical handle a program interacts with.
- **`codecs.py`** — type-directed codec layer. `type_from_spec(spec)` parses a JSON type
  declaration into a Hydra `Type`; `coder_for_type(typ)` derives a `TermCoder` recursively.
  Knows Hydra types and plain Python values only — no numpy, torch, or framework details.
- **`backends/*.json`** — NumPy, JAX, PyTorch, CuPy specs; same logical op names across all
  backends (e.g. `unialg.backend.add`) so DSL programs are backend-agnostic.

A backend spec looks like:

```json
{
  "backend": "numpy",
  "operations": {
    "add": { "kind": "elementwise binary", "path": "numpy.add",
             "arity": 2, "arg_type": "FLOAT", "result_type": "FLOAT" }
  }
}
```

### `objects.py` — Object-space type constructors

Provides thin wrappers over Hydra's `Type` hierarchy: `ProductType`, `SumType`, `ExpType`,
`ListType`, `MaybeType`, `VoidType`. Also defines the `Monad` descriptor (`MAYBE`, `LIST`)
used by lax morphisms to wrap codomains and sequence effects.

### `main.py` — Orchestration entry point

`load_program` / `compile_program` stitch the layers together: parse source, resolve `load`
directives into backend environments, lower each route's expression through `structure/`,
augment the Hydra graph with aux primitives, and wrap the result in `CompiledProgram`.

---

## Domain-Neutral Kernel

The stable centre of the project is a small set of concepts that should remain usable across
domains without needing to know which domain is active:

| Concept | Where it lives |
|---|---|
| Objects / types | `objects.py`, Hydra `Type` |
| Morphisms | `semantics/morphisms.py` — `Morphism`, `signature`, `compose` |
| Polynomial functors | `syntax/expressions.py` (`PolyExpr`), `semantics/functors.py` (`Functor`) |
| Product / sum structure | `syntax/expressions.py` — `Pair`, `Case`, `Parallel`, `Copy`, `Delete` |
| Composition | `morphisms.py` — `compose`, `Compose` node; Kleisli lifting via `Monad` |
| Optics | `semantics/optics.py` — `Optic`, `Lens`, `Prism`, `Traversal` |
| Recursion schemes | `syntax/expressions.py` — `Cata`, `Ana`, `AlgExpr`; `structure/realize.py` |
| Primitive declarations | `emitters/backend.py` — `BackendPrimitive`, `register_backend_primitive` |
| Lowering | `structure/realize.py` — `realize`, `realize_normalized` |
| Backend binding | `emitters/backend.py` — `BackendOps` |

This kernel should not need to import from any domain extension (tensors, ML models, planning,
etc.) to define what morphism composition or functor action means. Future domains should extend
the kernel, not modify it.

---

## Domain Extensions

A domain extension is a package or module that contributes:

- **Type / object declarations** — new carrier types relevant to the domain, expressed as
  Hydra `Type` constructors.
- **Primitive morphism declarations** — domain operations registered via `register_backend_primitive`
  or `BackendOps`, with canonical names under `unialg.backend.*`.
- **Algebraic structure** — higher-level compositions of primitives (e.g. `Semiring`, optic
  families) built entirely from `Morphism` objects and `compose`.
- **Convenience syntax and aliases** — shorthand DSL expressions or helper factories.
- **Optional backend binding hints** — JSON spec files mapping domain ops to concrete libraries.

The domain extension should ideally normalize into ordinary core expressions rather than
introducing a parallel semantic universe. For example, a planning domain's `proposal_compose`
should be a `Morphism` built via `compose`, not a separate call path that bypasses the
expression layer.

### Current domain extension: `tensors/`

`tensors/semirings.py` defines `Semiring` — a carrier type with four typed `Morphism`
components (plus, times, zero, one) and an optional adjoint. This is domain-adjacent structure:
it is built entirely from `Morphism` objects, lives on top of the core without modifying it,
and can be instantiated for any backend that provides the required operations.

`tensors/contractions.py` implements tensor contraction using a `Semiring`; it reduces to
ordinary `Morphism` composition and backend primitives.

### Hypothetical future domains

These are examples of what a domain extension might look like — they are not proposed
implementation targets:

- **ML domain** — `Tensor`, `Layer`, `Activation`, `Loss`, `Attention` as typed carrier
  objects; forward/backward pass as `Morphism` composition chains; parameter sharing as
  `Copy` / `Delete` structure.
- **Planning domain** — `Proposal`, `Budget`, `Constraint`, `Impact` as carrier types;
  feasibility checking as a morphism from proposals to `Maybe[Plan]`.
- **Security domain** — `Asset`, `Control`, `Finding`, `Risk` as carrier types; evidence
  aggregation as a `Semiring`-like structure over risk scores.
- **Workflow domain** — process steps as morphisms; branching as `Case`; loops as `Cata`
  over a list functor.

None of these require modifying the semantic layer. Each can be expressed as typed objects,
primitive morphisms (potentially backed by simple Python functions), and compositions.

---

## Optional Acquisition / Extraction Layer

The architecture leaves room for a future layer that extracts candidate DSL expressions from
unstructured domain material (documents, tables, schemas, diagrams).

Its role would be to produce ordinary DSL-level material:

- candidate objects / types
- candidate primitive morphisms with estimated dom / cod
- constants or holes where values are known or missing
- evidence and provenance metadata

The key constraint is that this layer is an alternate *front door* into the same expression
layer used by the normal parser — not a separate compiler. Extracted material should produce
`MorphismExpr` / `PolyExpr` trees that can be type-checked, composed, lowered, and executed
through the same path as hand-written programs.

```
manual DSL syntax               domain material (docs, tables, schemas)
    ↓                                   ↓
syntax/parse.py             (future) extraction layer
    ↓                                   ↓
     ──────── MorphismExpr / PolyExpr ───────
                      ↓
               semantics/
                      ↓
               structure/
                      ↓
              emitters / backend
```

A separate *composer* or *search* process might later assemble extracted expressions into
larger programs. That is distinct from extraction: the extraction layer populates the available
expression graph; the composer selects and connects from it. The existing DSL compiler path
remains the authority for what can be parsed, type-checked, lowered, and executed.

---

## What Belongs in the Core vs. a Domain Extension

| Core (stable, domain-neutral) | Domain extension (domain-specific) |
|---|---|
| `Morphism`, `compose`, `pair`, `case`, `parallel` | Carrier types for a specific domain |
| `Functor`, `apply_poly`, `PolyFmap` | Domain-specific functor shapes |
| `Optic`, `Lens`, `Prism`, `Traversal` | Domain-specific optic families |
| `Cata`, `Ana`, recursion schemes | Domain-specific algebras / coalgebras |
| `BackendPrimitive`, `register_backend_primitive` | Backend specs for domain operations |
| `Monad`, `MAYBE`, `LIST` | Domain-specific effect descriptors (if any) |
| `realize`, `lower` | — (not domain-specific) |
| Type constructors: `ProductType`, `SumType`, … | Domain carrier types using these constructors |

A useful design pressure: if adding support for a new domain requires changing `morphisms.py`,
`realize.py`, or `objects.py`, that is a signal that domain-specific concerns are leaking into
the kernel. Prefer making the new domain declare its vocabulary and register its primitives
rather than widening the semantic layer.

---

## Dependency Direction

The intended import direction is strictly downward:

```
main.py
  → syntax/, semantics/, structure/, emitters/
      → objects.py
          → Hydra
```

Domain extensions sit *beside* or *above* the core, not beneath it:

```
tensors/  →  semantics/morphisms.py, objects.py
          →  emitters/backend.py
```

The semantic layer should not import from `tensors/`. The emitter layer should not import from
`semantics/functors.py` for anything other than resolving morphism types. Future domain
packages should follow the same constraint: depend on the core downward, do not require the
core to depend on the domain.

`import-linter` contracts in `.importlinter` enforce the key boundaries. Check them before
large refactors (`lint-imports`), and update them when a new domain package is added.

---

## Architectural Intent

The architecture leaves room for other domains when those domains can be represented as typed
objects, primitive transformations, and compositions. The core machinery — morphism composition,
functor action, optic algebra, recursion schemes, backend primitive binding — is intended to be
domain-neutral. A new domain extension can be thought of as a vocabulary layer: it names its
types, declares its primitive operations, and expresses its patterns using the same structural
combinators as the ML/tensor work already present.

Future contributors should look at `tensors/` as a model: it introduces domain-relevant
structure (`Semiring`, `contractions`) as `Morphism`-level algebra without touching the parsing,
type-checking, or lowering layers.
