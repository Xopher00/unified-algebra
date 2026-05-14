# Next Steps

## Done: Revise tests

Tests revised. 94 passing. No references to old API remain in live tests; legacy tests quarantined under `tests/regression/stale_old_api/`.

## Done: Functor cleanup

- `Functor.apply(space)` added as instance method (object action)
- `Functor.compose(inner)` added as instance method (functor composition)
- Three `functor_*` compatibility wrappers removed
- `poly_fmap` signature changed to `(functor: Functor, h: Morphism)`, uses `functor.apply()` internally
- `Functor.map` not added — `poly_fmap` stays as a free function in `actions.py`; this is the correct placement since `poly_fmap` takes two arguments (functor + morphism) and requires Hydra-backed realization

## Done: Optics layer (unified polynomial optic)

- Single `Optic(functor, forward, backward)` dataclass replaces former Lens/Prism/Traversal subclasses
- `forward: S → F(A)` decomposes source; `backward: F(B) → T` reconstructs target
- Uniform action: `compose(forward, poly_fmap(F, h), backward)` — implemented as `act()` in `actions.py`
- `focus`/`replacement` derived via strict `functor.unapply()` (Hydra unification plus validated inverse of `apply_poly`)
- `Functor.unapply(fa)` builds `F(A)` with a Hydra type variable, asks `hydra.unification.unify_types` to solve `F(A) = fa`, then performs round-trip validation
- Lens = `Prod(Id, Const(residue))`, Prism = `Sum(Id, Const(residue))`, Traversal = arbitrary polynomial
- For simple lenses/prisms where S = F(A), forward and backward are identity morphisms
- Convenience constructors: `fst_lens(a, b)`, `snd_lens(a, b)`, `left_prism(a, b)`, `right_prism(a, b)`
- Height-2 optics need no structural change — just deeper polynomial bodies
- 8 unified Hypothesis property tests (type laws + rejection laws), 103 tests total passing
- No Hydra term construction in `optics.py`; action in `actions.py` preserves layer discipline

## Done: surface syntax layer (2026-05-14)

`load`/`route`/`map` program syntax works end-to-end. `compile_program(src).run(...)` requires no Hydra imports from the caller. See CHECKPOINT.md for full change list.

## Immediate: compile-time type checking for parsed compositions

**Problem:** `compile_program("load numpy\nroute bad = tanh >> add")` silently compiles. Type mismatch (`FLOAT → FLOAT×FLOAT`) only surfaces at `run()` time as a Hydra reduction error.

**Investigation so far:**
- `morphisms.py` imports `typeops` directly — the type checking infrastructure is already there
- `Compose`/`Pair`/`Case` nodes carry `dom=TypeUnit()`, `cod=TypeUnit()` from the parser (correct: semantics owns type derivation per `parse.py` docstring)
- The fix belongs inside `morphisms.py` — it must handle the case where these nodes arrive with placeholder types and need to derive real types from their `f`/`g` children
- Grimp shows: importers of `typeops` are `main`, `semantics.functors`, `semantics.morphisms`, `semantics.optics` — no new import edges are needed
- **Resume:** read `morphisms.py` to understand how `dom_of`/`cod_of` (or `signature`) handles `ContextualBinary` with placeholder fields, and where the existing check for `Morphism`-built compositions lives — that is the pattern to follow for parsed nodes

## Next: tensor operations (revised 2026-05-13)

### What was tried and why it failed

Three distinct attempts all broke down in the same ways:

1. **Parallel expression language** — each attempt invented `TensorExpr`, `TensorVar`, `ContractExpr`, or `TensorSemiring` as a new syntax tree running beside `MorphismExpr`. This duplicates the semantic layer.
2. **Python-native evaluator** — each attempt ended with a `_RUNTIME` dict (op name → numpy function) that bypasses Hydra entirely. This defeats the purpose of the backend/Hydra machinery.
3. **`ContractSpec` embedded in `Prim`** — the most mature attempt (notebook cells 66–89) correctly used `Morphism` as the carrier but then stuffed semantic metadata into `Prim(raw, dom, cod)`, which takes a raw Hydra `Term`, not a dataclass.

### What the Hydra API survey ruled out (2026-05-13)

- Hydra has **no built-in einsum/tensor operations**. `hydra.lib.math` is scalar-only.
- `hydra.reduction.contract_term` is beta-reduction cleanup — unrelated to tensor contraction despite the name.
- `hydra.parsers` is a Hydra-internal parser combinator library — not a surface syntax parser for user expressions.
- Creating new `MorphismExpr` subclasses needs stronger justification: Hydra's `TermPrimitive` + correctly named `BackendPrimitive` registration may be sufficient representation.

### Hard constraints for any implementation

- **No new expression language under `tensors/`** — new semantic files belong in `semantics/`, new structure files in `structure/`, following the existing layer split.
- **No Python-native evaluator** — execution must go through `run()` → `structure/realize.py` → Hydra reduction.
- **No new `MorphismExpr` subclass without clear justification** — exhaust `Prim` + named primitive first.
- **Explore Hydra before adding** — use `pkgutils`/`importlib`/`inspect` to verify a capability does not already exist.
- **`tensors/semirings.py`** is correctly placed (semantics layer); extend it in place rather than moving it.
- **`tensors/tensorexpressions.py` and `tensors/equations.py`** are stubs that contradict the above constraints — they should be removed, not extended.

### Correct layer mapping (to validate before implementing)

| Concern | Layer | Correct file | Status |
|---------|-------|--------------|--------|
| Subscript parsing (`"ij,jk->ik"`) | Syntax | `syntax/expressions.py` or keep as `str` | Unclear — may not need a node |
| Semiring dataclass | Semantics | `tensors/semirings.py` | Exists ✓ |
| `from_backend` factory + `op_env` | Semantics | `tensors/semirings.py` | Missing |
| Tensor type `ExpType(I, A)` | Semantics | `semantics/morphisms.py` (helper) | Missing |
| `contract_morphism(sr, eq) → Morphism` | Semantics | `semantics/` | Missing |
| Contract fusion rewrite | Structure | `structure/` | Missing (notebook only) |
| Contraction kernel registration | Structure | `structure/` | Missing |

### Before writing any code

1. Delete `tensors/tensorexpressions.py` and `tensors/equations.py` — they contradict the design.
2. Write skeleton files (docstring + comments only, no logic) for each new file, getting sign-off on placement before filling in.
3. For each piece, ask: does Hydra already provide this? Check with `importlib`/`inspect` before implementing.


## Next: Clarify the core API flow

The current code has a natural but partly scattered flow:

```text
Expr node -> typed semantic wrapper -> type action -> backend realization -> lowering/run
```

For morphisms, that flow is:

- `MorphismExpr` in `expressions.py` — pure syntax tree
- `Morphism` in `morphisms.py` — typed arrow handle
- `signature` / `dom_of` / `cod_of` — type derivation
- `realize` in `realize.py` — Hydra interpretation
- `lower` / `run` — execution boundary

For polynomial functors, the parallel flow is currently less explicit:

- `PolyExpr` in `expressions.py` — pure syntax tree
- `Functor` in `functors.py` — named descriptor with `apply(space)` object action method
- `functors.apply_poly` — internal recursive implementation of F(A)
- `Functor.unapply(fa)` — inverse object action, using Hydra unification to solve `F(A) = fa`
- `poly_fmap(functor, h)` in `actions.py` — arrow action `h -> F(h)`, takes a `Functor`

Refactor direction:

- Do **not** collapse `Morphism` and `Functor` into one universal class; morphisms are arrows, functors are object-and-arrow transformers.
- Keep `MorphismExpr` and `PolyExpr` as pure ADT syntax.
- Keep `Morphism` as the typed semantic handle for arrows.
- `Functor` upgraded: `apply(space)` is now an instance method; `summands()`, `x_arity()`, `consts()` are methods.
- `Functor.unapply(fa)` now uses Hydra type unification instead of a hand-written structural inverse walker.
- `poly_fmap(functor, h)` in `actions.py` takes a `Functor`, uses `functor.apply()` internally. Stays as a free function — it takes two arguments (functor + morphism) and requires Hydra-backed realization.
- Import-order-sensitive monkey patches remain removed; no layer boundary violations.

Current reader-facing shape:

```python
Maybe = Functor("Maybe", Sum(One(), Id()))

Maybe.apply(INT)              # object action: F(A)
poly_fmap(Maybe, add1)        # arrow action: F(f)
lower(poly_fmap(Maybe, add1), "maybe_add1")
```

## Next: Optics — remaining work

The unified `Optic` handles Lens, Prism, Traversal, and height-2 cases structurally. Remaining:

- **Runtime behavioral tests** — lens get/set, set/get, set/set laws; prism review/preview roundtrips (these require `realize`/`lower`/`run`)

## Done: Recursion schemes over carrier optics

The old `recursion.py` (with `rec`, `Inductive`, `LIST_IND`, `MAYBE_IND`, `AlgebraError`)
moved to `.bak`. The current API is generic over any carrier optic:

```python
Optic(functor, forward, backward, carrier=mu)
```

The fixed-point boundary is supplied by the optic:

- `fp.forward = unroll : μF → F(μF)` (destructor — peel one layer)
- `fp.backward = roll : F(μF) → μF` (constructor — wrap one layer)
- `fp.carrier = μF`
- No `FixedPoint` subclass — plain `Optic` is sufficient

Built-in carrier helpers, if added, are adapters which produce this optic boundary. They are not core semantics.

`cata`, `ana`, and `hylo` are implemented for plain, para, lax, and lax-para `Morphism` algebras/coalgebras. Parameter context is shared by calling contextual combinators with `shared_context=True`; effects are sequenced with the existing monad `bind`.

**`cata(fp, alg)`** — catamorphism / fold

```
cata(fp, alg) : P × μF → T(A)
             = compose(act_forward(fp, self), alg, shared_context=True)
             = compose(compose(fp.forward, poly_fmap(F, self)), alg, shared_context=True)
```

`alg` may be plain (`F(A) → A`), para (`P × F(A) → A`), lax (`F(A) → T(A)`), or lax-para (`P × F(A) → T(A)`). `act_forward` performs `forward ∘ poly_fmap`; the recursive self-reference has the same `param` and `monad` as `alg`.

For the list functor `F(X) = 1 + (E × X)`, `alg : F(A) → A` decomposes into:
- nil branch: `alg(Left(())) : A` — the base value
- cons branch: `(e: E, acc: A) → alg(Right(e, acc)) : A` — the step function

**`ana(fp, coalg)`** — anamorphism / unfold

```
ana(fp, coalg) : P × A → T(μF)
              = compose(coalg, act_backward(fp, self), shared_context=True)
              = compose(coalg, compose(poly_fmap(F, self), fp.backward), shared_context=True)
```

`coalg` may be plain (`A → F(A)`), para (`P × A → F(A)`), lax (`A → T(F(A))`), or lax-para (`P × A → T(F(A))`). `act_backward` performs `poly_fmap ∘ backward`; the recursive self-reference has the same `param` and `monad` as `coalg`.

**`hylo(fp, coalg, alg)`** — hylomorphism

```
hylo(fp, coalg, alg) = compose(ana(fp, coalg), cata(fp, alg), shared_context=True)
```

The shared-context composition is what keeps a lax-para hylo at `P × A → T(B)` instead of expanding into `P × P × A → T(B)`.

## Done: Runtime recursion smoke tests

Runtime tests now prove that the recursive primitive wiring reduces.

The smoke carrier uses `F(X) = Unit + X` and takes the terminating `Unit` branch. This keeps the functor valid for `Optic` validation while avoiding an infinite self-call:

- `cata(fp, alg)(value) == expected`
- `ana(fp, coalg)(seed) == expected`
- `hylo(fp, coalg, alg)(seed) == expected`

The lax-para runtime cases verify that the shared parameter is supplied once:

```text
P × A -> T(B)
```

not:

```text
P × P × A -> T(B)
```

This also caught and fixed a runtime bug: recursive primitives must register their actual raw function type as their Hydra `TypeScheme`; a dummy `Unit` scheme makes Hydra treat plain recursive primitives as nullary.

## Done: List carrier adapter

`list_carrier(element)` is implemented as a convenience constructor:

```python
Optic(functor=F, forward=unroll, backward=roll, carrier=mu)
```

It represents Hydra lists as:

```text
μX. 1 + (A × X)
```

Runtime coverage verifies:

- carrier boundary type laws
- `cata(list_carrier(INT), sum_alg)` sums a concrete Hydra list
- `ana(list_carrier(INT), countdown_coalg)` builds a concrete Hydra list
- `hylo(list_carrier(INT), countdown_coalg, sum_alg)` unfolds and folds in one pass

This adapter did not change the core recursion semantics.

## Next: Non-list carrier adapters

Maybe/tree adapters should also be convenience constructors, but they need a little design care:

- `Maybe(A)` is structurally `1 + A`, a constant polynomial with no `Id`; current `Optic` validation expects a functor position it can `unapply`.
- Tree carriers need an agreed recursive shape and carrier encoding before writing roll/unroll.

Do not weaken the core recursion semantics for these. Add only the adapter support needed to produce the same carrier optic boundary cleanly.

## Next: Algebra structure above recursion

`Morphism` remains the algebra/coalgebra representation; no separate `ParaAlgebra` wrapper is needed for this layer. The next algebraic layer is about named relationships between algebras.

Typed maps between algebras that commute with the algebra structure. Needed for relating model components (encoder/decoder adjointness, residual connections as natural transformations). `AlgebraHom(f, src, tgt)` where `f: Morphism` is the carrier map.

Design questions:
- Does `AlgebraHom` live in `recursion.py` or a new `algebra.py`?
- How are coherence cells `ε_A`, `δ_A` represented for lax cases?
- Does the optic structure make algebra/coalgebra typing constraints more explicit?

## Next: Semiring tensor structure

For multi-headed and branching architectures. Tensor products of morphisms, bilinearity. See `docs/ALGEBRA.md` / `claude-mdtopics/ALGEBRA.md` for context.

## Deferred: Python operator syntax via Hydra Expr

The Pratt parser (string path) is implemented in `syntax/`. A Python operator
path — where users compose `Morphism` objects with `>>`, `&`, `|` directly — is
the natural next step. This section records the Hydra mechanisms discovered
during exploration (2026-05-14).

### Hydra infrastructure available

- `hydra.ast.Op(symbol, padding, precedence, associativity)` — operator metadata.
  Shared `Op` objects are already defined in `syntax/_ops.py` for `>>`, `&`, `||`, `|`.
- `hydra.ast.Expr` — generic expression ADT: `ExprConst(Symbol)`, `ExprOp(OpExpr)`,
  `ExprBrackets(BracketExpr)`, `ExprSeq(SeqExpr)`.
- `hydra.serialization.sym(name) -> Symbol` — creates a named symbol.
- `hydra.serialization.print_expr(expr) -> str` — renders an `Expr` tree to surface
  syntax with operator symbols and whitespace.
- `hydra.serialization.parenthesize(expr) -> Expr` — precedence-aware parenthesization.
- `hydra.dsl.ast.op_expr(op, lhs, rhs) -> TTerm` — builds an `OpExpr` term
  (phantom-typed, returns Hydra `Term`, not `MorphismExpr`).

These together give: build an `Expr` tree with `ExprConst`/`ExprOp`, call
`print_expr(parenthesize(expr))`, and get readable surface syntax like
`(f & g) >> h` back. Verified working.

### Design direction

Operator dunders (`__rshift__`, `__and__`, `__or__`) live in `syntax/`, not on
`Morphism` in `semantics/`. They wrap **expressions** (`MorphismExpr`), not
semantic objects (`Morphism`). The syntax layer owns the surface representation;
semantics owns typed composition. The dunders build `MorphismExpr` trees and
parallel `hydra.ast.Expr` trees — they do not call semantic combinators directly.

Each operator call:
1. Constructs a `MorphismExpr` node (e.g., `Compose`, `Pair`) via shared
   constructors in `_ops.py`.
2. Builds a parallel `hydra.ast.Expr` tree for rendering.
3. `__repr__` calls `print_expr(parenthesize(expr))` for readable display.

`||` cannot be overloaded in Python. Alternative: `//` (`__floordiv__`) for `par`,
or a `.par()` method.

### Exploratory: automatic Python-to-Hydra Expr parsing

Python's `ast` module can inspect operator expressions at the source level. An
alternative to manual dunder overloads: intercept a Python function or lambda via
`inspect.getsource` or `ast.parse`, walk the AST, and translate `BinOp` nodes
(`>>`, `&`, `|`) directly into `hydra.ast.Expr` trees (`ExprOp` with the
corresponding `Op` objects from `_ops.py`). Name nodes become `ExprConst(sym(name))`.

This would allow writing plain Python:

```python
def transformer(x, attention, add, layer_norm, token_ffn):
    return ((x & attention) >> add >> layer_norm) >> \
           ((x & x_star(token_ffn)) >> add >> layer_norm)
```

and extracting the `Expr` tree automatically without executing the function or
requiring operator overloads at all. The `Expr` tree feeds into the same
`print_expr` rendering and could lower to `MorphismExpr` via the shared
lowering path.

This approach has tradeoffs (source inspection is fragile, closures lose source,
notebooks may need cell-level AST hooks) but is worth prototyping as a
zero-boilerplate alternative to the dunder approach.

### Shared infrastructure (already in place)

`syntax/_ops.py` defines shared `Op` objects, BP derivation (`morphism_bp()`,
`functor_bp()`), binary `MorphismExpr` constructors (`make_compose`, `make_pair`,
`make_par`, `make_case`), and `Expr` builder + render helpers (`atom_expr`,
`binary_expr`, `render`). Both the string parser grammar and the future Python
operator interface use these.

### Constraint

All operator syntax code lives in `syntax/`. The `syntax/` layer NEVER imports
from `semantics/`. The operator interface must call semantic combinators through
a boundary that respects this (e.g., the user passes `Morphism` objects in, and
the syntax layer wraps the result).

## Next: Einsum specs and backend-to-expression mapping

### Einsum spec files

Einsum specs register the einsum methods that every backend provides — `einsum`
is a backend primitive like `add`, `softmax`, `gelu`, and `layer_norm`. The spec
file declares that the backend HAS an einsum operation; the equation strings
are supplied at use-time by architecture code.

The spec does not name or own specific contraction patterns. Names like
`linear_seq` or `attention_score` are architecture-level choices — the user
passes the equation string when instantiating a parametric morphism from the
einsum atom.

### Backend loading as expressions

A backend spec (e.g., `backends/numpy.json`) already registers primitives via
`BackendOps.from_spec()`. The next step: automatically map each registered
backend primitive to a `MorphismExpr` node (or a `Morphism` with its `Prim` node)
that the surface syntax can reference directly.

The flow:

1. Load a backend spec: `ops = BackendOps.from_spec("backends/numpy.json")`
2. Auto-generate an expression environment: `env = backend_to_env(ops)` — each
   op name maps to a `MorphismExpr` (or `Ref` resolved to its `Prim`-backed node).
3. The parser's `env` parameter accepts this: `parse_morphism("f >> add >> gelu", env=env)`
   resolves `add` and `gelu` to their concrete backend-backed expressions.
4. For the Python operator path: the env entries are directly usable as expression
   operands — no manual atom construction needed.

This closes the gap between "user loads a backend" and "user writes expressions
that reference backend primitives by name." The einsum specs and backend env
together give the testing scaffold described in `notes/syntax.md` deliverable #4.

### Constraint

Einsum specs define reusable contraction equations only. They do not encode
q/k/v, attention, FFN, GNN, transformer, or multimodal blocks. Backend loading
produces primitive-backed expressions — it does not define architecture layers.

## Watch: Strength and distributivity

Lax para composition handles parameter threading with shared context plus `bind`/lambda capture. No explicit strength morphism is part of the current semantics.

## Tensor contraction: open design questions (2026-05-13)

These gaps must be resolved before any tensor contraction code is written.
Ordered by how much they block forward progress.

### 1. Index type encoding — DECISION REQUIRED (blocks everything)

`tensor_type(index, element)` in `semantics/morphisms.py` and the dom/cod of
`contract_morphism` in `semantics/contractions.py` both depend on what Hydra
`Type` represents an index set.

Options:
- **Nominal** — `Name("I")`, `Name("J")` as opaque phantom types.  Sufficient
  for checking that `contract_morphism` output type matches a downstream
  morphism's input type.  Does not encode size.
- **TypeVariable** — polymorphic; `contract_morphism` returns a scheme not a
  monotype.  Clean but harder to instantiate in practice.
- **Nat-indexed** — encodes concrete sizes statically.  Precise but requires
  size arithmetic in the type system; more than is needed now.

Nominal is the likely starting point.  Decide before touching `tensor_type()`
or any `contract_morphism` dom/cod.

### 2. Structural tensor ops not in backend specs — needs placement decision

`structure/tensor_lowering.py` needs `expand_dims` and `transpose` for the
alignment step (unsqueeze + permute input tensors to `output_vars ++ reduced_vars`
dim order).  These are **not** in `tensors/backends/*.json`, which only carries
elementwise binary, unary, and reduce ops.

Options:
- Add them to the existing JSON specs under a `"structural"` kind.
- Register them separately in `structure/tensor_lowering.py` outside the JSON
  spec mechanism (direct `register_backend_primitive` calls).
- Handle alignment at the Hydra term level using existing pair/list primitives
  rather than delegating to the backend.

Decide before implementing `structure/tensor_lowering.py`.

### 3. Batching via poly_fmap needs grounding

The claim is that `bij,bjk->bik` = `poly_fmap(BatchFunctor, contract_morphism(sr, "ij,jk->ik"))`.
The existing functor machinery handles **polynomial** endofunctors (Prod, Sum,
Const, Id, Comp).  A function-space functor `(B → -)` is exponential, not
polynomial.

For finite batch dimensions it collapses to `Prod(Id, Id, ..., Id)` — a
product functor — which `poly_fmap` can handle.  But this requires:
- The batch index type to be finite and explicitly encoded as a product.
- A clear statement of what the `BatchFunctor` PolyExpr looks like.

Decide whether batching is handled by poly_fmap over a product functor, or
by a separate mechanism (e.g. explicit loop / vmap primitive), before
implementing `semantics/contractions.py`.

### 4. CompiledEquation has no layer home

The equation parser (`compile_einsum` from `unified-algebra/algebra/contraction.py`)
is pure data — no backend coupling.  It parses a string like `"ij,jk->ik"` into
`input_vars`, `output_vars`, `reduced_vars` index structures.

Candidate placements:
- `syntax/` — it is string parsing, which is syntax-layer work.
- Private helper inside `semantics/contractions.py` — keeps it close to its
  only consumer.

Decide before implementing `contract_morphism`.

### 5. Post-contraction nonlinearity — hook needed

The old `Equation` had an optional nonlinearity applied after contraction
(`contract_and_apply`).  This is common in practice: sigmoid/relu/softmax after
a semiring contraction.  In the new system, this is just morphism composition
(`compose(contract_morphism(sr, eq), nl)`), but `contract_morphism` should note
this as the intended pattern so users know not to bake it into the semiring.

No implementation needed now — note the pattern in `semantics/contractions.py`.

### 6. No Semiring law-checking equivalent

The old `Semiring.check_laws` verified associativity, commutativity, identity,
annihilation, and distributivity on scalar samples before the semiring was used.
The new `Semiring` has no equivalent.  Useful for custom and research semirings.

Add as a future method on `Semiring` in `semantics/semirings.py`, gated behind
a flag (default off for performance, on for development/research use).

### 7. Import linter boundary rules not updated

The new files (`semantics/semirings.py`, `semantics/contractions.py`,
`structure/tensor_lowering.py`, `structure/semiring_factory.py`) are not
reflected in `.importlinter`.  Update after skeletons are finalised and before
any implementation is merged.

## Deferred: Surface grammar — remaining gaps

`parse_program` now handles `route NAME = <morphism-expr>` and
`map NAME = <functor-expr>` as a multi-definition program. Remaining gaps:

- **Other declaration forms not yet supported**: `data`, `type`, imports,
  inline comments, multi-line bodies, forward references across definitions.
- **Test files**: `tests/syntax/test_pratt.py` covers the new program API.
  Check all other test files (e.g. regression tests, strategy tests) to confirm
  no remaining calls to `parse_morphism`/`parse_functor` that should now
  flow through `parse_program` instead.
- **`tests/support/strategies.py`**: The Hypothesis strategies currently generate
  isolated expression strings. They should be reviewed and extended with a
  `program_strategy` that generates full `route`/`map` programs for property
  testing the whole pipeline. Confirm `strategies.py` is still aligned with the
  current expression ADT after the `Ref`/`PolyRef` additions.

## Deferred

- Surface syntax / grammar (no timeline)
- Backend expansion beyond current Hydra primitives
- `CarrierExpr` — a syntax-layer expression type for declaratively describing carrier roll/unroll structure, so recursive carriers (List, Maybe, RoseTree, user-defined) do not require hand-written Hydra plumbing. Design is unsettled: the expression ADT must be general enough to not enumerate per-carrier primitives, and the derivation of `unroll`/`roll` from that description must stay clean. Do not implement until the design is clear.

## Historical Reference

`/home/scanbot/unified-algebra/src/unialg` is a prior, ad-hoc version of this project. It may be useful for understanding old experiments, but it is a **reference only**:

- Do not copy code or port abstractions wholesale.
- Do not resurrect `_RecordView`; it created invisible structural coupling by backing domain objects with Hydra record terms.
- Do not reintroduce a manual `TypedMorphism.kind`-style tag; the current ADT structure is clearer.
- Treat the old `algebra_hom` bridge as a cautionary example: it exposed a broad functor surface while only executing a narrow subset. Avoid shipping similarly incomplete abstractions as stable API.
