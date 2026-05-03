# Architecture

See [CURRENT_CONTRACT.md](CURRENT_CONTRACT.md) for the layer-by-layer boundary contract.

unified-algebra is a typed DSL for wiring algebraic tensor programs.

## Design contract

1. **Hydra is the semantic and type substrate.** Every DSL declaration compiles to a Hydra `Term`/`Type`. There is no separate AST — Hydra terms are the internal representation.

2. **Compositions may compile to backend-native callables.** Sequential paths, fans, folds, unfolds, fixpoints, and lenses can be compiled once at program-creation time into closures that call backend operations directly.

3. **Compiled callables are registered back as Hydra primitives.** The Hydra graph remains the single source of truth. Compiled closures are wrapped as primitives in the same graph, not a parallel execution path.

4. **The DSL is operation-agnostic.** Semantics come from declared semirings and backends. The user names operations (`plus=add`, `times=multiply`); the backend provides implementations. The framework never assumes real arithmetic.

5. **The framework owns wiring, typing, contraction, and composition** — not ML-specific operations. It does not know what a convolution is. It knows how to contract tensors over a semiring, compose typed morphisms, and validate sort junctions.

## Hydra ↔ unified-algebra boundary

unified-algebra is layered on top of [Hydra](https://github.com/CategoricalData/hydra). The boundary is intentional and load-bearing; its purpose is to keep generic typed-functional plumbing in Hydra and concentrate algebraic/tensor/semiring semantics in unified-algebra.

### Hydra owns

- **Parser combinators.** `hydra.parsers` — used by `parser/_grammar.py`. unified-algebra does not maintain its own parser library.
- **Typed term IR.** `core.Term*`, `core.Type*`, including `TermLambda`, `TermApplication`, `TermVariable`, `TermLet`, `TermPair`, `TermRecord`, `TermInject`, `TermCases`. All DSL declarations compile to terms in this IR.
- **Term reduction.** `hydra.reduction.reduce_term` is the fallback interpreter for entry points that cannot be statically compiled. unified-algebra does not replace or wrap it.
- **Type checking.** `hydra.checking.type_of_term` validates assembled terms against their declared types. System F + HM polymorphism is available but currently unused (every unified-algebra primitive is monomorphic).
- **Term construction helpers.** `hydra.dsl.meta.phantoms` (`record`, `lam`, `var`, `apply`, etc.) — the canonical, phantom-typed way to build terms. Direct `core.TermLambda` / `core.TermLet` construction would mix DSL levels (Hydra pitfall #5) and is forbidden.
- **Primitive registration.** `hydra.dsl.prims.prim1` / `prim2` / `prim3`, with arity ceiling 3. Higher-arity primitives must pack into list-coders.
- **Standard library.** `hydra.sources.libraries.standard_library()` provides `hydra.lib.pairs.{first,second,bimap}`, `hydra.lib.equality.identity`, `hydra.lib.lists.{foldl,...}`, `hydra.lib.maybes.*`, `hydra.lib.eithers.*`, etc. Use these instead of reimplementing.

### unified-algebra owns

- **Sorts as tensor/domain spaces.** `algebra/sort.py` — `Sort` (named tensor type bound to a Semiring), `ProductSort` (typed monoidal product), `UnitSort` (terminal object).
- **Semiring algebra and law checking.** `algebra/semiring.py` — declarative spec + backend-resolved runtime. No Hydra overlap; semirings do not exist in `standard_library()`.
- **Equation / einsum / contraction.** `algebra/equation.py`, `algebra/contraction.py` — declarative tensor morphism specs and the contraction engine. No Hydra overlap.
- **Functor (polynomial F-algebra).** `morphism/functor.py` — body kinds `zero`/`one`/`id`/`const`/`sum`/`prod`/`exp`. Hydra has type extraction but no polynomial-functor algebra.
- **Typed morphism IR.** `morphism/_typed_morphism.py` — `TypedMorphism` carries a Hydra term plus boundary sorts (input/output). Constructors in `morphism/morphism.py`: `eq`, `lit`, `iden`, `copy`, `delete`, `seq`, `par`, `lens`.
- **Lenses / optics.** `assembly/_morphism_compile.py` — `CompiledLens` dataclass with forward/backward boundary sorts. Lenses are registered as Hydra primitives (forward + backward pass) in the same graph as all other primitives. `reduce_term` is the single execution path.
- **algebraHom / fold / unfold / fixpoint.** `morphism/algebra_hom.py` — catamorphism and anamorphism construction. Recursion schemes compile to Hydra primitives via `assemble_graph`; Hydra has no fixpoint/μ term.
- **Backend lowering and execution.** `backend.py` is intentionally Hydra-free. JIT compilation, while_loop, tensor ops, and contraction execution are all backend-native and do not become Hydra primitives.

### `terms.py` is the narrow adapter

`src/unialg/terms.py` is the **only** module permitted to construct Hydra records / unwrap literal values for general use. It contains:

- `tensor_coder(backend, type_)` — `hydra.graph.TermCoder` factory bridging arrays ↔ Hydra terms.
- `_RecordView` — Python descriptor protocol over Hydra records (`Term`, `Scalar`, `TermList`, `ScalarList` field types). All unified-algebra record-shaped declarations (Sort, Semiring, Equation, Functor, Cell payloads) inherit from `_RecordView`.
- `_literal_value(term)` — unwraps `core.LiteralInteger`/`Float`/`String`/`Boolean`/`Binary` to Python scalars.

`terms.py` must stay thin. **Do not** put Cell semantics, Equation semantics, Semiring law checks, or backend dispatch in `terms.py`. New record-view subclasses are fine; new categorical or tensor logic is not. A behavioral regression test pins `terms.py` line count under a soft budget (~250 lines).

### ProductSort is monoidal-only

`ProductSort` represents a typed monoidal product (or equivalently a tensor/domain product for n-ary einsums). It does **not** implement:

- Projections (no `fst`/`snd`/`π_i`).
- Destructuring or pattern matching.
- Pairing laws.
- Copy/delete laws (those live on `Cell` variants).
- Cartesian-product structure.

`ProductSort` exists at the type level (right-nested `core.TypePair` chain) and at the wire format (right-nested `hydra.dsl.prims.pair` coder). Runtime values are plain Python tuples.

If projections become necessary, lower to `hydra.lib.pairs.first` / `hydra.lib.pairs.second` (registered Hydra primitives) at the Cell-leaf level. Do not invent Cell-level projection variants.

### Forbidden moves

- Collapsing TypedMorphism categorical structure (monoidal product, copy/delete, optic residual, recursion schemes) into unstructured Hydra terms.
- Putting Equation, TypedMorphism, Semiring, or backend semantics inside `terms.py`.
- Constructing `core.TermLambda` / `core.TermLet` directly (use `hydra.dsl.meta.phantoms.lam` / `let`).
- Treating `ProductSort` as cartesian.
- Reimplementing primitives that exist in `hydra.lib.*` (audit `standard_library()` first).
- Adding Hydra imports to `backend.py`.

## What this is not

- **Not a neural network library.** It expresses architectures but does not provide layers, optimizers, or training loops.
- **Not a replacement for PyTorch or JAX.** Those are backends. This is the algebra above them.
- **Not hardcoded to any architecture.** The same composition constructs express MLPs, message-passing networks, attention patterns, Bellman-Ford, Viterbi, and anything else that decomposes into semiring-parameterized tensor equations.
- **Not a tensor compiler.** It does not lower to hardware instructions. It wires typed equations and delegates execution to the backend.

## Module map

```
src/unialg/
    algebra/
        semiring.py          Semiring declaration + law validation
        sort.py              Sort, ProductSort, UnitSort
        equation.py          Equation (typed tensor morphism) — pure declaration
        contraction.py       CompiledEinsum, semiring contraction engine
        expr.py              Expression compiler for inline define ops

    assembly/
        graph.py                  build_graph, assemble_graph, rebind_params
        program.py                compile_program, Program callable wrapper
        _equation_resolution.py   Equation → Hydra primitives + validation
        _morphism_compile.py      Cell → Hydra primitives/bound_terms, CompiledLens
        _validation.py            Pipeline sort/rank/axis validation

    morphism/
        _typed_morphism.py   TypedMorphism — term + boundary sorts
        morphism.py          Constructors: eq, lit, iden, copy, delete, seq, par
        functor.py           PolynomialFunctor, PolyExpr
        algebra_hom.py       Catamorphism/anamorphism construction
        lens.py              Lens, lens_seq construction

    parser/
        _grammar.py          Pratt parser for .ua surface syntax
        _resolver.py         Name resolution orchestration → UASpec
        _resolve_cells.py    Cell expression → TypedMorphism
        _cell_ast.py         CellExpr frozen dataclasses
        _types.py            NamedCell, UASpec dataclasses
        __init__.py          parse_ua, parse_ua_spec public API

    backend.py       Backend ABC + NumpyBackend, PytorchBackend, JaxBackend, CupyBackend
    terms.py         Hydra record-view helpers, tensor_coder, literal encoding
```

## Data flow

```
.ua source text
    |  parser/_grammar.py  (Pratt parser)
    v
raw declaration tuples
    |  parser/_resolver.py  (name resolution → UASpec)
    v
UASpec (semirings, sorts, equations, cells, defines)
    |  algebra/expr.py  (register custom ops on backend)
    |  assembly/graph.py  (resolve equations, validate DAG, build Hydra graph)
    v
hydra.graph.Graph + Hydra primitives
    |  assembly/program.py  (compile_program → Program)
    v
Program(name, *arrays) -> arrays   [executes via reduce_term]
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
