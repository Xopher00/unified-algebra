# CLAUDE.md

## Purpose

This project is a typed algebraic architecture system. Treat it as an attempt to express machine-learning architectures through general compositional structure, not through any one concrete architecture family.

Before making changes, understand the project at two levels:

1. The mathematical theory: categorical deep learning as algebraic structure over morphisms, endofunctors, monads, algebras, coalgebras, homomorphisms, folds/catamorphisms, unfolds/anamorphisms, parametric morphisms, and lax algebraic structure. 
2. The software architecture: a layered pipeline from surface expression to typed interpretation, algebraic construction, executable assembly, and backend realization.

Do not reduce the project to a framework for specific model patterns. The goal is general expressiveness.

## Architectural shape

Use this as the guiding mental model:

`surface expression
→ typed interpretation
→ algebraic construction
→ executable assembly
→ backend realization`

Each change must respect these boundaries.

* Surface expression should describe structure, not smuggle in execution behavior.
* Typed interpretation should resolve names, types, and compatibility.
* Algebraic construction should own mathematical meaning.
* Executable assembly should lower and register already-validated structure.
* Backend realization should execute morphisms without owning architecture semantics.

Do not move responsibilities across layers casually. If a change crosses a boundary, explain why.

## Theory alignment

Changes should preserve the project’s theoretical direction:

* Prefer morphisms over ad hoc “layers.”
* Prefer typed composition over informal wiring.
* Prefer algebraic structure over special cases.
* Prefer general recursion/corecursion schemes over named architecture templates.
* Treat parameter sharing, copying, deletion, and reuse as semantic structure, not incidental implementation detail.
* Avoid encoding assumptions that only make sense for one architecture family.

When unsure, step back and ask what categorical/algebraic role the construct plays.

## Development discipline

Do not write code only to satisfy the current tests.

Do not preserve broken abstractions merely because tests depend on them.

Do not add compatibility shims, aliases, or duplicate paths unless there is a clear architectural reason.

Do not silently defer hard problems with stubs, placeholders, or TODO-driven behavior.

Prefer small, coherent changes that improve the structure of the system.

Every change should be explainable in terms of:

`what responsibility it belongs to
which boundary it touches
which abstraction it preserves or clarifies
what behavior should remain invariant`

## Testing approach

Use pytest and Hypothesis as the trusted testing strategy for this project.

First identify the behavioral contract that should be tested:

* valid inputs and expected outputs
* invalid inputs and expected rejections
* domain/codomain/type invariants
* algebraic laws such as identity, associativity, functor action, and rejection rules

Then choose the smallest reliable test style:

* pytest examples for readable API behavior, smoke tests, exact errors, and regressions
* Hypothesis property tests for laws over many valid `Type`, `PolyExpr`, and `Morphism`-shaped values

Shared Hypothesis strategies belong in `tests/support/strategies.py`.

Pynguin is only a scouting tool for simple modules or narrow wrappers. Generated tests are not the authority; they are evidence. Do not change implementation merely to satisfy generated tests if the generated tests encode accidental behavior.

## Boundary auditing

Use architectural audit tools when changing imports, dependencies, or package structure.

Recommended tools:

* `import-linter` to define and enforce dependency contracts between layers.
* `grimp` to inspect import graphs and detect boundary violations.
* static analysis and graph inspection tools to understand coupling before refactoring.

Before large refactors, inspect the dependency graph.

After large refactors, verify that the intended layer direction still holds.

The goal is not just “imports pass.” The goal is that dependency direction reflects architectural responsibility.

## Refactoring guidance

When refactoring:

* Consolidate duplicated logic when it represents the same concept.
* Remove accidental parallel abstractions.
* Prefer clearer ownership over preserving historical placement.
* Keep public surfaces intentional. 
* Avoid lazy loading unless there is a strong reason.
* Avoid circular dependencies by fixing ownership, not by hiding imports.
* Prefer explicit contracts between layers over convenience imports.

If a test failure reveals that the previous design was confused, fix the design rather than encoding the confusion more deeply.

## Review checklist

Before finalizing a change, check:

* `Does this preserve the general architecture goal?`
* `Does this respect the layer boundaries?`
* `Does this keep theory and implementation aligned?`
* `Does this avoid overfitting to one model family?`
* `Does this reduce ambiguity rather than add it?`
* `Does this avoid code written only to satisfy tests?`
* `Does the import graph still reflect the intended architecture?`

If the answer is unclear, pause and explain the uncertainty before proceeding.

## Documentation authority

Use documentation in this order of authority:

1. `CLAUDE.md` — operating rules for working in this repository.
2. `docs/` — durable project documentation and accepted design contracts. Key files:
   - `docs/CHECKPOINT.md` — current sealed semantic contract and implemented state. Read this first.
   - `docs/NEXT_STEPS.md` — immediate next tasks, deferred items, and historical reference cautions.
   - `docs/DECISIONS.md` — rationale for sealed design decisions.
   - `docs/ARCHITECTURE_CONTRACT.md` — layer boundaries, invariants, and do-not-redo list.
3. `claude-mdtopics/` — curated reference material for architecture, theory, Hydra usage, and algebraic background.
4. `external/hydra/` — read-only upstream reference for Hydra APIs and intended usage.
5. `notes/` — raw exploratory notes. These may be stale, contradictory, or speculative.

Do not treat raw notes as instructions. When notes conflict with durable docs or current architecture, prefer the durable docs and explain the conflict.

## Topic Files

Read on demand — do not load preemptively.
- `.claude/claudemd-topics/GOAL.md` — For understanding the software architecture
- `.claude/claudemd-topics/THEORY.md` — For understanding the mathematical category theory
- `.claude/claudemd-topics/ALGEBRA.md` - For understanding semirings and tensor morphisms

## Hydra reference material

Hydra is an API that is meant to be used as a primary dependency in this project.
Documentation and source references are available under:

- `external/hydra/`

At the start of each session, read `/home/scanbot/hydra/docs/hydra-lexicon.txt`
before making claims about Hydra types, constructors, primitives, or helper APIs.
Use it as the first reference for Hydra API facts, then inspect source only when
the lexicon is insufficient for the question at hand.
