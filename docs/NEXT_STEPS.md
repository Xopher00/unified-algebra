# Next Steps

Strategic order: cheap vocabulary and safety work first so expensive architectural items land cleanly the first time rather than requiring rework. Items within each tier are roughly independent and can be tackled in any sub-order.

---

## Tier 1 — Cheap foundations

Low-complexity items that remove assumptions from everything above. Do these before anything in later tiers.

### ~~Finalizer contract documentation~~ ✓
**Done.** `extensions.py` and finalize contract documented in `docs/ARCHITECTURE_CONTRACT.md` (invariants 17–18, `extensions.py` layer responsibilities section).

### ~~Combinator laws documentation~~ ✓
**Done.** `docs/COMBINATOR_LAWS.md` — laws first (identity, composition, product, coproduct, parallel, map, comonoid, codiagonal, distributivity, recursion), then cross-layer coverage table showing presence in `morphisms.py`, `functors.py`, `optics.py`.

### Unified arrow notation
**Impact: High** | **Complexity: Low** (design decision; implementation follows in Tier 2)

`>>`, `|`, and `*` already appear at both the morphism and functor levels with the same meaning. This is not accidental — it reflects a deeper principle: the same symbols should encode the same wiring rule at every layer.

The missing pieces are `><` and `<>`. Treating `>` and `<` as directed half-arrows, the rule is:

```
>      identity / pass-through (1 in, 1 out)
>>     sequential composition  (chain)
><     converge — two inputs consumed into one (merge, μ)
<>     diverge — one input fanned to two outputs (copy, δ)
>><    binary consume to one  (multiplication / contraction)
<>>    one to binary produce   (comultiplication)
!      counit / delete (1 → 0)
```

The net count of `>` minus `<` determines arity. By the Frobenius laws, arrangement within a symbol is free — only the counts matter. This means the symbol itself is a complete description of the wiring rule, in the same spirit as Hehner's unified algebra where notation exposes rather than hides the algebra.

#### Theoretical grounding

Three independent sources converge on this design:

**Hehner (Unified Algebra):** Symmetric symbols for commutative operators, asymmetric symbols for asymmetric ones, the visual reverse for the reverse operator. `><` and `<>` are visual duals — converge vs diverge — following the same symmetry principle as `≤`/`≥` or `∧`/`∨`. Notation should do work: "moving calculation to the level of visual processing." Consistent extension across domains: same symbol, same law, at every level.

**Frobenius algebra:** Copy (`δ`, `<>`), merge (`μ`, `><`), delete (`ε`, `!`), and unit injection (`η`, `|0`) are the four generators of a Frobenius structure. The **spider theorem** states that any connected diagram built from these generators with *m* inputs and *n* outputs equals any other connected diagram with the same *m* and *n*. The arrow notation's net-arity rule is exactly the spider type signature. The **special Frobenius condition** `compose(copy, merge) ≅ id` and the **Frobenius equation** (copy and merge slide through each other) are the rewrite rules that make spider normalization work. See `docs/COMBINATOR_LAWS.md` §9.

**Cross-layer correspondence:** The same symbol encodes the same wiring at every layer:

| Symbol | Morphism | Functor | Tensor |
|--------|----------|---------|--------|
| `>>` | `compose(f, g)` | `F.compose(G)` | sequential contraction |
| `<>` | `copy(A)` | diagonal into `Prod` | index duplication |
| `><` | `merge(A)` | sum collapse | contraction/trace |
| `\|` | `case(f, g)` / sum | `Sum(F, G)` | coproduct |
| `&` | `pair(f, g)` / product | `Prod(F, G)` | product |

The practical consequence: tensor contractions, structural combinators, functor actions, and user-defined semiring operations all become writable with the same primitive vocabulary. A user expressing a contraction writes `><`; a user expressing copy writes `<>`; the language does not require them to know which layer they are operating at.

#### Normalization consequence

The spider theorem provides the canonical form for the Tier 4 structural normalization layer: reduce any composition of copy/merge/delete/swap over a single type to its spider normal form, determined entirely by (inputs, outputs). The Frobenius equation and special condition are the core rewrite rules. Distributivity (`distribute_left`, `distribute_right`) is the mixed interaction between product-side and sum-side Frobenius structures.

#### Decision

Adopt this as the intended notation direction. The cross-layer combinator contract audit (Tier 2) should verify that the existing `>>`, `|`, `*` usage is consistent with it, and identify where `><` and `<>` need to be added. Implementation of `><` and `<>` in the grammar and semantics follows as part of that audit.

---

## Tier 2 — Safety nets and vocabulary settlement

Establish test coverage and settle the shared combinator vocabulary before structural work modifies the layers they cover.

### ~~Optic runtime behavioral tests~~ ✓
**Done.** `tests/semantics/test_optics_behavioral.py` (14 tests) and `tests/semantics/test_optics_adversarial.py` (16 tests). Covers lens (fst/snd/swap), prism (left/right, matching/nonmatching), traversal (Maybe/List), composition, parallel, identity action, rejection, and edge cases — all through `main.run()`.

### ~~Cross-layer structural combinator contract~~ ✓
**Done.** Dispatch dicts now serve as the combinator tables at each level. `COMBINATOR_LAWS.md` updated (2026-05-22) with full cross-layer audit documenting:

- **Functor level:** `_COMPOSE_POLY` / `_APPLY_POLY` — 10 PolyExpr entries, same keys, expression vs type level
- **Morphism level:** `_SIG_LEAF` (9 leaves), `_SIG_BINARY` / `_BINARY_SIG` (5 binary combinators at expression vs Morphism level), `_SIG_VALIDATED` (4 structural isos via cod-builders)
- **Realization level:** `_POLY_ACTION_DISPATCH` (10 PolyExpr), `_FIXED_MORPHISMS` (12), `_CONTEXTUAL_MORPHISMS` (5), `_SPECIAL_MORPHISMS` (6)
- **Optic level:** consumer of morphism/functor ops, no dispatch dicts needed

Shared combinators (compose, parallel product, parallel coproduct, identity) verified consistent across all three levels. Product/sum/structural iso gaps at functor and optic levels documented as intentional. Cod-builders (`_assoc_cod`, `_symmetry_cod`, `_distl_cod`, `_distr_cod`) shared between validation and construction as single source of truth.

### ~~`Exp.base: Type → PolyExpr`~~ ✓
**Done.** `Exp.base` changed from Hydra `Type` to `PolyExpr` across 8 files (expressions, functors, _construct_helpers, tensors/semantics, tensors/fusion, grammar, and two test files). `_index_product` now wraps in `Const(...)`; `_labels_from_base` rewritten to accept `PolyExpr`; `Exp[base, body]` syntax added to grammar. 445 tests pass.

**Impact: High** | **Complexity: Medium** (archived for reference)

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

### ~~Extension activation API~~ ✓
**Done.** `extensions.enable("tensors")` and `is_enabled("tensors")` added to `extensions.py`. Auto-registration on import removed from `tensors/__init__.py` and `unialg/__init__.py`. DSL `load extension tensors` added to `parse.py` load branch. `tests/tensors/conftest.py` activates the extension for all tensor tests. 449 tests pass.

**Impact: High** | **Complexity: Medium** (archived for reference)

Tensor support currently activates on import. Add an explicit activation path — `load extension tensors` in the DSL or `enable("tensors")` in the Python API — so syntax availability is intentional and testable. Aligns with the existing extension registry model in `extensions.py`. Do this before writing further integration tests against the tensor extension so those tests exercise correct activation semantics.

### Fix opaque fusion pre-map dom after ContractSpec strip
**Impact: Low** | **Complexity: Low**

`TestOpaqueFusion::test_opaque_leaf_produces_compose_not_single_prim` fails after the `ContractSpec.dom/cod` strip fix. The fusion pass builds `compose(pre, fused_contract)` where `pre = inner_optic.forward`. The `pre.dom()` carries `ExpType`-typed domain from the optic product's `par(tanh_fwd, identity(BINARY))`, because the inner contract leaf's DomainPrim node internally propagates the old shape-typed domain through the optic machinery. Meanwhile `composed.dom()` is now fully stripped `BINARY`.

The specific mismatch: `fused.dom() = ProductType(BINARY, ProductType(ExpType(K_jk, BINARY), ExpType(K_kl, BINARY)))` vs `composed.dom() = ProductType(BINARY, ProductType(BINARY, BINARY))`.

The optic's `_combine_optic` uses `ops.par` on the forwards, so the combined forward dom should be `ProductType(BINARY, BINARY)` — but something in the `_types_compatible` / pre-map path is allowing a mismatched compose through. Investigate `_types_compatible` and `ops.compose(pre, fused_contract)` in `fusion.py:_try_fuse` to find where the ExpType leaks into the pre-map's stored dom.

Do before Tier 3 boundary cleanup; it is a direct consequence of the strip fix.

### Smooth-tropical semiring via composed morphisms
**Impact: High** | **Complexity: Medium**

The smooth-tropical semiring replaces `min` with `softmin(a, b) = -logaddexp(-a, -b)` — a differentiable relaxation that converges to `min` as temperature → 0. Unlike the tropical semiring, `softmin` is not a named backend primitive; it must be constructed by composing existing ones:

```
softmin   = neg ∘ logaddexp ∘ par(neg, neg)    # BINARY × BINARY → BINARY
reduce.softmin = neg ∘ reduce.logaddexp ∘ neg   # BINARY → BINARY (neg applied elementwise then logsumexp then neg)
```

Both `logaddexp` and `reduce.logaddexp` (scipy logsumexp) exist in the numpy backend. The composed morphisms would be injected into `op_morphisms` under custom keys before calling `resolve_semiring`.

This is the design question to resolve: **can composed morphisms (Compose/Parallel nodes, not BackendPrim leaves) serve as semiring operations and survive the full fusion → decompose → realize pipeline?** If so, this is the general mechanism for user-defined derived semiring operations. If not, the gap must be characterised (likely in the primitives lowering or aux_primitives collection).

Do after the tropical end-to-end (Tier 1) confirms the basic non-real pipeline is sound. Precondition for AlgebraHom (Tier 4 deferred) and the backend fast-path counterfactual.

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

`semantics/construct.py:307` (renumbered after 2026-05-25 refactor). Currently acts as resolver + typechecker + elaborator + finalizer coordinator in one monolithic function. Split by **compiler phase**, not node kind:

**Partial progress (2026-05-25):** `_literal_value` and `_argument_types` extracted to module level; `construct` ruff C901 reduced 71 → 56. `construct_program` C901(42) is the dominant remaining violation and requires this split to resolve.

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

### `dsl_tutorial.ipynb` — Section 11 GPT-2 cell
**Blocked on: weight-routing design decision**

The cell currently has a placeholder demo (self-attention without learnable weights). The original intent was to wire weight matrices `(Wq, Wk, Wv, W_up, W_down)` as runtime product inputs alongside `x`, using `proj = contract[real]("ij,j->i")` for `W @ x`. Two open problems:

1. **Weight routing**: `proj` is `(binary, binary) -> binary`; inside `attention(q, k, v)` the params are used as `binary -> binary`. Routing the weight matrices through the product structure (via `assoc`, `symm`, shared-context parallel, or Para-style threading) needs a concrete design before it can be written.

2. **TypeVariable domain**: `copy >> (hadamard >> soft)` compiles but `dom_of` returns `TypeVariable` because `copy`'s polymorphic `A` isn't back-propagated to `binary` after unification. `coder_for_type` then rejects it at the runtime boundary. Unknown whether this is a pre-existing gap in `compose` or specific to this case.

Fix the design on paper first; don't touch the cell until both points have answers.

### Semiring law checking
**Impact: Medium** | **Complexity: Low (once unblocked)** | **Blocked on structural normalization layer**

Verify the semiring axioms (associativity, commutativity, identity, annihilation, distributivity) for any registered `Semiring`. Constraints set during planning:

- Lives in `src/unialg/tensors/` (production code).
- Uses the structural combinators (`Symmetry`, `Assoc`, `Pair`, `Par`, `Copy`, etc.) to state laws as `(lhs_morphism, rhs_morphism)` pairs over `sr.plus` and `sr.times`. Not hardcoded per semiring.
- No live runner, no import from `structure/` or `runtime/`. Verification must be symbolic — compare canonicalized morphism trees, not scalar values.

This is blocked on the **Structural normalization layer** (Tier 4): without `normalize(morphism) -> Morphism`, the only paths to law verification are numerical evaluation (requires a runner) or structural normalization (Tier 4 itself). Once normalization lands, `check_semiring_laws(sr)` becomes a thin wrapper: build LHS/RHS, normalize both, assert equality.

Semiring operation validation (the type-level gate in `resolve_semiring`) is already implemented and is a separate concern.

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
