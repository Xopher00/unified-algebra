# Next Steps

Strategic order: cheap vocabulary and safety work first so expensive architectural items land cleanly the first time rather than requiring rework. Items within each tier are roughly independent and can be tackled in any sub-order.

---

## Tier 1 — Cheap foundations

Low-complexity items that remove assumptions from everything above. Do these before anything in later tiers.

### Lexer comment robustness
**Impact: Low** | **Complexity: Low**

`RecursionError` on long comments. Isolated to `syntax/_lex.py`. Fix comment tokenization to use iteration instead of recursion.

### Semiring operation validation
**Impact: High** | **Complexity: Low**

Fail early when `plus`, `times`, `reduce`, or `adjoint` are missing, ill-typed, or carrier-incompatible. Hard validation gate at semiring declaration time, distinct from the optional law checking below. Without this, every subsequent tensor test runs on potentially invalid semirings.

### Custom semiring end-to-end
**Impact: High** | **Complexity: Low**

Implement the smooth-tropical semiring. Exercises the full pipeline — notation, semantics, primitives, fusion — with a non-standard semiring. Validates that no part of the pipeline is implicitly real-semiring-specific. Required as a precondition for `AlgebraHom` (Tier 4 deferred) and as the counterfactual case for backend fast-path detection.

### Semiring law checking
**Impact: Medium** | **Complexity: Low**

Add a `check_laws(samples)` method to `Semiring` in `tensors/semantics.py`. Verifies associativity, commutativity, identity, annihilation, and distributivity on scalar samples. Gate behind a flag (default off; on for development/research use). Natural pairing with semiring validation — same file, same object.

### Finalizer contract documentation
**Impact: Medium** | **Complexity: Low**

Document `finalize` as an official whole-morphism rewrite hook, not only a domain elaboration hook. Tensor fusion already uses this power; making it explicit prevents future extension conflicts and forces precision before `TensorPlan` (Tier 4) formalizes the pipeline stages.

### `merge` / codiagonal
**Impact: Medium** | **Complexity: Low**

```
merge : A + A → A
```

Derivable as `case(id, id)` but must exist as a named combinator. It is listed in the cross-layer combinator vocabulary (Tier 2). Adding it before the vocabulary audit means the audit becomes a verification pass rather than a discovery exercise.

### Canonical distributivity combinators
**Impact: High** | **Complexity: Low**

Add named structural rewrites:
```
A × (B + C) ↔ (A × B) + (A × C)
(A + B) × C ↔ (A × C) + (B × C)
```

Same rationale as `merge`: listed in the contract vocabulary; cheap to add; makes the cross-layer audit a check not a search. Should have morphism, functor, and optic interpretations where applicable.

### Combinator laws documentation
**Impact: High** | **Complexity: Low**

Document expected laws for each combinator: identity, associativity, product/sum coherence, copy/delete laws, merge/case laws, map functor laws, optic composition laws, distributivity. Write this before implementing the structural normalization layer (Tier 4) — stating the laws first forces precision and gives future extensions a clear semantic target.

### Unified arrow notation
**Impact: High** | **Complexity: Low** (design decision; implementation follows in Tier 2)

`>>`, `|`, and `*` already appear at both the morphism and functor levels with the same meaning. This is not accidental — it reflects a deeper principle: the same symbols should encode the same wiring rule at every layer.

The missing pieces are `><` and `<>`. Treating `>` and `<` as directed half-arrows, the rule is:

```
>      identity / pass-through (1 in, 1 out)
>>     sequential composition  (chain)
><     converge — two inputs consumed into one (monoidal tensor, times)
<>     diverge — one input fanned to two outputs (copy, Δ)
>><    binary consume to one  (multiplication / contraction)
<>>    one to binary produce   (comultiplication)
```

The net count of `>` minus `<` determines arity. By the Frobenius laws, arrangement within a symbol is free — only the counts matter. This means the symbol itself is a complete description of the wiring rule, in the same spirit as Hehner's unified algebra where notation exposes rather than hides the algebra.

The practical consequence: tensor contractions, structural combinators, functor actions, and user-defined semiring operations all become writable with the same primitive vocabulary. A user expressing a contraction writes `><`; a user expressing copy writes `<>`; the language does not require them to know which layer they are operating at.

Design decision to make now: adopt this as the intended notation direction. The cross-layer combinator contract audit (Tier 2) should verify that the existing `>>`, `|`, `*` usage is consistent with it, and identify where `><` and `<>` need to be added. Implementation of `><` and `<>` in the grammar and semantics follows as part of that audit.

---

## Tier 2 — Safety nets and vocabulary settlement

Establish test coverage and settle the shared combinator vocabulary before structural work modifies the layers they cover.

### Optic runtime behavioral tests
**Impact: Medium** | **Complexity: Low**

Lens get/set, set/get, set/set laws; prism review/preview roundtrips. These require `realize` → `lower` → `run` and are blocked on nothing — just not yet written. Live in `tests/semantics/`. Write before combinator/optic work so regressions are caught immediately.

### Cross-layer structural combinator contract
**Impact: High** | **Complexity: Medium**

Define a shared combinator vocabulary that applies consistently across morphisms, optics, and functors:
```
identity, compose, parallel, pair, case, map, associate, swap, distribute, copy, delete, merge, focus, traverse
```

With `merge` and `distributivity` already added (Tier 1), this becomes an audit rather than a design exercise: confirm every item is consistently present and named across `morphisms.py`, `optics.py`, `functors.py`. Prevents each layer from inventing its own wiring semantics. Directly informs what the `construct` elaboration phase (Tier 4) should produce.

### `Exp.base: Type → PolyExpr`
**Impact: High** | **Complexity: Medium**

`Exp.base` is currently a Hydra `Type`, which cannot be produced from source text. Changing it to `PolyExpr` unblocks user-facing `Exp[base, body]` syntax and makes the base consistent with the rest of the `PolyExpr` tree. Five targeted sites:

| Site | File | Change |
|------|------|--------|
| `apply_poly` | `semantics/functors.py:216` | `ExpType(base, ...)` → `ExpType(apply_poly(base, TypeUnit()), apply_poly(b, space))` |
| `_collect_consts` | `semantics/functors.py:253` | recurse into `node.base` instead of treating it as a `Type` |
| alpha-rename | `tensors/fusion.py:98` | replace `substitute_type_variables(subst, s.base)` with a PolyExpr walk substituting in `Const` nodes |
| `_labels_from_base` | `tensors/fusion.py:51, 283` | decode `Const`/`PolyRef` instead of `TypeVariable`/`TypePair` — simpler |
| pretty printer | `syntax/expressions.py:497` | `show_type(expr.base)` → `pretty(expr.base)` |

After these five sites: add `Exp[base, body]` case to `_functor_nud` in `syntax/_grammar.py`, matching the `List[body]`/`Maybe[body]` pattern. The tensor domain construction in `tensors/semantics.py` (`_index_product`) would wrap its base in `Const(...)`.

This change also makes `_labels_from_base` in `fusion.py` structurally simpler and removes the reason the Hydra-type decoder exists there at all — directly enabling the tensor boundary cleanup in Tier 3.

Note: a pending plan (plan file `compiled-petting-acorn.md`) covers a related `ContractSpec.dom/cod` change to emit `ExpType`-wrapped types. Treat that plan as part of this work block.

### Extension activation API
**Impact: High** | **Complexity: Medium**

Tensor support currently activates on import. Add an explicit activation path — `load extension tensors` in the DSL or `enable("tensors")` in the Python API — so syntax availability is intentional and testable. Aligns with the existing extension registry model in `extensions.py`. Do this before writing further integration tests against the tensor extension so those tests exercise correct activation semantics.

---

## Tier 3 — Core structural work

Medium-complexity items that build on the clean vocabulary and Exp.base changes from Tier 2.

### Tensor semantic boundary cleanup
**Impact: Medium** | **Complexity: Medium**

Remove direct Hydra-type encoding from tensor semantic objects where possible. Introduce a tensor-owned label/base abstraction; encode into Hydra-facing form only during lowering. Substantially easier after `Exp.base` (Tier 2): the main coupling point (`TypeVariable`/`TypePair` in Exp bases) becomes `Const`/`PolyRef` and the decoder moves to the right layer.

### Label/axis invariant tests
**Impact: Medium** | **Complexity: Medium**

Focused tests for label preservation through rewriting and decomposition. Prioritize: repeated labels, output label ordering, diagonal axis adjustment, alpha-renamed reduced labels, invalid label reuse. Write after boundary cleanup so tests target the stable label representation.

### `eval` / `curry` for closed structure
**Impact: High** | **Complexity: Medium**

Add explicit combinators for function-space semantics:
```
eval  : (A → B) × A → B
curry : (C × A → B) → (C → A → B)
```

Makes exponential structure operational rather than type-level only.

### Cross-layer `map` semantics
**Impact: High** | **Complexity: Medium**

Formalize what `map` means across: Morphism, Functor, Optic, Recursive carrier, Monad/lax context. Bridge between structural computation and domain-specific extensions. Informs the traversal combinator design (Tier 4).

### Non-list carrier adapters (Maybe, Tree)
**Impact: Medium** | **Complexity: Medium**

`maybe_carrier` and `tree_carrier` convenience constructors analogous to `list_carrier`. Design notes:
- `Maybe(A) = 1 + A` is a constant polynomial with no `Id`; current `Optic` validation expects a functor position it can `unapply`. The adapter may need a `Just` wrapper that carries the `Id` position explicitly.
- Tree carrier needs an agreed recursive shape (`1 + X × X`) and Hydra encoding of `roll`/`unroll` before implementing. `Tree[x]` is now parseable as `Sum(One(), Prod(Id(), Id()))`, which gives the functor shape for free.
- Do not weaken core recursion semantics. Add only adapter support that produces the existing carrier optic boundary cleanly.

---

## Tier 4 — Architectural refactors

High-complexity items that benefit from settled vocabulary (Tier 1–2) and clean objects (Tier 3).

### `construct` — phase-based split
**Impact: High** | **Complexity: High**

`semantics/construct.py:243`. Currently acts as resolver + typechecker + elaborator + finalizer coordinator in one monolithic function. Split by **compiler phase**, not node kind:

```
resolve    → name lookup, ref resolution, scope
check      → type and shape compatibility
elaborate  → AST → semantic core forms
finalize   → whole-program rewrites, extension hooks
```

`construct.py` becomes the public facade (`construct_program`, `construct_morphism`, `construct_carrier`); internal implementation moves to private modules (`_construct_context.py`, `_construct_expr.py`, `_construct_decl.py`, `_finalize.py`). Public import surface stays stable.

Do after the cross-layer combinator contract audit (Tier 2) — that audit settles what the elaboration phase should produce, preventing a second pass on `_construct_expr.py`. Resolves `construct_program` ruff C901(40) automatically as closures become top-level functions.

### TensorPlan
**Impact: High** | **Complexity: Medium**

Separate tensor planning from finalization. Fusion should be one optimizer pass over a `TensorPlan` object, with room for optional backend-specific passes.

Target pipeline:
```
contract syntax
→ TensorPlan
→ generic validation
→ generic fusion
→ optional backend optimizer
→ substrate Morphism
```

Do after tensor boundary cleanup (Tier 3) so the planning API operates on clean objects from the start rather than Hydra-typed internals.

### Structural backend capability interface
**Impact: High** | **Complexity: Medium**

Formalize the sanctioned structural operations used by tensor lowering: transpose, expand dims, diagonal, reduce, elementwise combine, backend fast paths. Makes the current structural cheat explicit rather than relying on callable discovery. Do before backend fast-paths and attention ops so those features validate the interface rather than extend the cheat.

### Semantic annotation system
**Impact: High** | **Complexity: High**

Hydra annotations as a semantic side-channel: type evidence, error provenance, optimizer hints, proof obligations, domain claims. Allows the DSL to accumulate obligations lazily (`obligation native_roundtrip ≅ id`) without immediate verification.

The Tier 1 combinator laws and semiring work produce two concrete payloads that drive the syntax design: combinator law obligations and semiring obligations. Position here so the structural normalization layer (below) has a proper channel for optimizer hints rather than encoding them implicitly. AlgebraHom's coherence cells (`ε_A`, `δ_A`) are a third concrete use case when that work unblocks.

### Structural normalization layer
**Impact: High** | **Complexity: High**

Normalization/rewrite API for structural morphisms: associate, swap, copy/delete, pair/case, projection/injection, distribute, merge, map. Needed for optimizer soundness and for comparing equivalent programs across layers. Requires the complete, tested combinator vocabulary from Tiers 1–2, the documented laws from Tier 1, and the annotation system (above) as the hint/obligation channel.

### Fusion equivalence property tests
**Impact: High** | **Complexity: Medium**

Randomized tests comparing unfused and fused tensor contractions against reference NumPy/einsum behavior. Cover: repeated labels, diagonals, scalar/vector/matrix cases, opaque leaves, nested `Pair`, adjoints, custom semirings. Write after tensor boundary cleanup and TensorPlan so tests target the stable, cleaned-up pipeline rather than a messy intermediate state.

### Backend fast-paths — native einsum
**Impact: Medium** | **Complexity: Medium**

Route real-semiring fused contraction to native `numpy.einsum` (or backend equivalent) instead of the align/fold/reduce decomposition. Requires detecting the special case in `compile_contract_spec`. Keeps the generic decomposition for non-standard semirings. The custom semiring end-to-end (Tier 1) provides the required counterfactual test.

### Monadic tensor contractions
**Impact: High** | **Complexity: High**

`ContractSpec` currently has no monad field; the fusion and decompose pipeline assumes pure computation. Extend tensor contractions to thread effects: effectful semiring operations (`plus`/`times` returning `M(A)`) and effectful contraction output (`M(output_type)`).

Requires clean semiring semantics (Tier 1), `TensorPlan` (above) to provide a clean stage for monad threading separated from finalization, and cross-layer `map` semantics (Tier 3) to be settled — monadic map is traverse. Closely related to the traversal combinator below; the two may converge in design.

### Traversal combinator
**Impact: Medium** | **Complexity: High**

First-class structural traversal for optics, functors, lists, maybe-types, and recursive carriers. Supports applying the same morphism uniformly across structured positions. Requires `map` semantics (Tier 3) and carrier adapter work (Tier 3).

---

## Deferred

Items that are explicitly blocked or premature. Revisit when blockers resolve.

### Contraction order + backend optimizer
**Impact: Low** | **Complexity: High** | **Blocked on TensorPlan**

Cost-model-driven contraction order selection, plus optional backend-registered optimizer modules. Both depend on `TensorPlan` existing as a stable internal API.

### Transformer / attention ops
**Impact: Medium** | **Complexity: Medium**

Ops not yet in any backend JSON spec: axis-specialized softmax, ReLU/GELU, layer normalization, attention masking. Blocked on resolving the design question about routing carried parameters through larger composed blocks. Do not add until that design is clear.

### Runtime allocation/performance benchmarks
**Impact: Medium** | **Complexity: Low**

Benchmark fused vs. unfused tensor lowering and direct NumPy/einsum. Track primitive count, intermediate allocation count, and wall-clock runtime. Meaningful only after backend fast-paths exist to make the comparison interesting.

### AlgebraHom — typed maps between algebras
**Impact: High** | **Complexity: High** | **Blocked on carrier adapters + custom semiring**

Typed maps between algebras that commute with the algebra structure. `AlgebraHom(f: Morphism, src, tgt)` where `f` is the carrier map. Needed for encoder/decoder adjointness, residual connections as natural transformations.

Open design questions:
- Does `AlgebraHom` live in `semantics/optics.py` or a new `semantics/algebra.py`?
- How are coherence cells `ε_A`, `δ_A` represented for lax cases?

### `.ualg` files and CLI runner
**Impact: High** | **Complexity: High**

Promote the DSL to first-class files: `parse_program` reads `.ualg` files directly; `unialg compile program.ualg --backend torch`; `unialg run program.ualg --backend torch --input input.json`. Independent of most other items but high complexity — defer until the core system is settled.

---

## Historical reference — do not revisit

`/home/scanbot/unified-algebra/src/unialg` is a prior, ad-hoc version. Reference only:

- Do not copy code or port abstractions wholesale.
- Do not resurrect `_RecordView` — it coupled domain objects to Hydra record terms invisibly.
- Do not reintroduce `TypedMorphism.kind`-style tags — the current ADT is clearer.
- The old `algebra_hom` bridge exposed a broad functor surface while executing a narrow subset. Avoid similarly incomplete abstractions as stable API.
