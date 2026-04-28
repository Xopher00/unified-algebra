# Architecture

unified-algebra is a typed DSL for wiring algebraic tensor programs.

## Design contract

1. **Hydra is the semantic and type substrate.** Every DSL declaration compiles to a Hydra `Term`/`Type`. There is no separate AST — Hydra terms are the internal representation.

2. **Compositions may compile to backend-native callables.** Sequential paths, fans, folds, unfolds, fixpoints, and lenses can be compiled once at program-creation time into closures that call backend operations directly.

3. **Compiled callables are registered back as Hydra primitives.** The Hydra graph remains the single source of truth. Compiled closures are wrapped as primitives in the same graph, not a parallel execution path.

4. **The DSL is operation-agnostic.** Semantics come from declared semirings and backends. The user names operations (`plus=add`, `times=multiply`); the backend provides implementations. The framework never assumes real arithmetic.

5. **The framework owns wiring, typing, contraction, and composition** — not ML-specific operations. It does not know what a convolution is. It knows how to contract tensors over a semiring, compose typed morphisms, and validate sort junctions.

## What this is not

- **Not a neural network library.** It expresses architectures but does not provide layers, optimizers, or training loops.
- **Not a replacement for PyTorch or JAX.** Those are backends. This is the algebra above them.
- **Not hardcoded to any architecture.** The same composition constructs express MLPs, message-passing networks, attention patterns, Bellman-Ford, Viterbi, and anything else that decomposes into semiring-parameterized tensor equations.
- **Not a tensor compiler.** It does not lower to hardware instructions. It wires typed equations and delegates execution to the backend.

## Module map

```
src/unialg/
    algebra/
        semiring.py      Semiring declaration + law validation
        sort.py          Sort (named tensor type), ProductSort, Lens
        equation.py      Equation (typed tensor morphism) — pure declaration only
        contraction.py   CompiledEinsum, semiring contraction engine
        expr.py          Expression compiler for inline `define` ops

    assembly/
        graph.py                  DAG assembly, topological sort, sort/rank validation
        compositions.py           Path, Fan, Fold, Unfold, Fixpoint composition objects
        specs.py                  Spec dataclasses (PathSpec, FanSpec, etc.)
        _primitives.py            Hydra primitive registration
        _validation.py            Type unification helper (unify_or_raise)
        _equation_resolution.py   Equation lowering to Hydra Primitives

    parser/
        _grammar.py      PEG grammar for .ua surface syntax
        _resolver.py     Name resolution: raw tuples -> UASpec

    runtime/
        program.py       compile_program(), Program callable wrapper

    backend.py           Backend ABC + NumpyBackend, PytorchBackend, JaxBackend, CupyBackend
    terms.py             Hydra record-view helpers, tensor_coder, literal encoding
```

## Data flow

```
.ua source text
    |  parser/_grammar.py
    v
raw declaration tuples
    |  parser/_resolver.py
    v
UASpec (semirings, sorts, equations, compositions, defines)
    |  algebra/expr.py (register custom ops on backend)
    |  assembly/graph.py (resolve equations, validate DAG, build Hydra graph)
    v
hydra.graph.Graph + compiled entry points
    |  runtime/program.py
    v
Program(name, *arrays) -> arrays
```

## Core abstractions

| Concept | What it is | DSL keyword |
|---------|-----------|-------------|
| Semiring | Algebraic structure (plus, times, zero, one) | `algebra` |
| Sort | Named tensor type bound to a semiring | `spec` |
| Equation | Typed morphism: einsum contraction or pointwise op | `op` |
| Path | Sequential composition (left to right) | `seq` |
| Fan | Parallel branches merged by a stack-machine chain | `branch` |
| Fold | Catamorphism (scan / left-recursive accumulation) | `scan` |
| Unfold | Anamorphism (iterated state transition) | `unroll` |
| Fixpoint | Convergence iteration (epsilon-bounded) | `fixpoint` |
| Lens | Bidirectional morphism (forward + backward) | `lens` |
| Define | Inline custom operation from expression | `define` |

## Semiring parameterization

The same equation `"ij,j->i"` means different things under different semirings:

- **Real** (plus=add, times=multiply): standard matrix-vector product
- **Tropical** (plus=minimum, times=add): shortest-path relaxation (Bellman-Ford)
- **Log-space** (plus=logaddexp, times=add): numerically stable probability

Swapping the semiring changes semantics without changing the architecture.

## Backend contract

A backend maps operation names to implementations. The framework calls:

- `backend.elementwise(name)` — binary elementwise (add, multiply, maximum, ...)
- `backend.reduce(name)` — reduction along axis (sum, min, ...)
- `backend.unary(name)` — pointwise transform (relu, tanh, exp, ...)
- `backend.compile(fn)` — optional JIT wrapper
- `backend.while_loop(cond, body, state)` — optional loop fusion (JAX lax.while_loop)

Users extend backends by adding to `backend.unary_ops` / `backend.binary_ops`, or by using `define` in the DSL.
