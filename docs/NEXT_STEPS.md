# Next Steps

Tasks ordered by module. Each entry carries an **impact** (what it unblocks or improves) and **complexity** (files, design decisions, risk) rating.

---

## syntax/

### `Exp` surface syntax — `Exp.base: Type → PolyExpr`
**Impact: High** | **Complexity: Medium**

`Exp.base` is currently a Hydra `Type`, which cannot be produced from source text. Changing it to `PolyExpr` unblocks user-facing `Exp[base, body]` syntax and makes the base consistent with the rest of the `PolyExpr` tree.

Five sites across three files:

| Site | File | Change |
|------|------|--------|
| `apply_poly` | `semantics/functors.py:216` | `ExpType(base, ...)` → `ExpType(apply_poly(base, TypeUnit()), apply_poly(b, space))` |
| `_collect_consts` | `semantics/functors.py:253` | recurse into `node.base` instead of treating it as a `Type` |
| alpha-rename | `tensors/fusion.py:98` | replace `substitute_type_variables(subst, s.base)` with a PolyExpr walk substituting in `Const` nodes |
| `_labels_from_base` | `tensors/fusion.py:51, 283` | decode `Const`/`PolyRef` instead of `TypeVariable`/`TypePair` — simpler |
| pretty printer | `syntax/expressions.py:497` | `show_type(expr.base)` → `pretty(expr.base)` |

After these five sites: add `Exp[base, body]` case to `_functor_nud` in `syntax/_grammar.py`, matching the `List[body]`/`Maybe[body]` pattern. The tensor domain construction in `tensors/semantics.py` (`_index_product`) would wrap its base in `Const(...)`.

### Complexity: `_morphism_nud` — E (32)
**Impact: Medium** | **Complexity: Medium**

`syntax/_grammar.py:83`. Large match dispatch over morphism token kinds (identity, copy, delete, fst, snd, inl, inr, absurd, assoc, swap, ref, app, recursion, carrier boundary, monadic lift, parenthesized). Same extraction pattern as the `signature` refactor: group into `_nud_structural` (identity/copy/delete/fst/snd/inl/inr/absurd/assoc/swap), `_nud_special` (ref/app/recursion/carrier/lift), keep the main function as a dispatcher. Target: all sub-functions ≤ B.

### Complexity: `_pretty_morphism` — D (28)
**Impact: Low** | **Complexity: Low**

`syntax/expressions.py:282`. Match dispatch over all MorphismExpr variants for pretty-printing. Extract `_pretty_leaf` for structural primitives (Identity through Absurd) and `_pretty_contextual` for binary nodes. Mechanical — no semantic changes.

### Complexity: `_pretty_poly` — C (14)
**Impact: Low** | **Complexity: Low**

`syntax/expressions.py:472`. Same pattern — match dispatch over PolyExpr variants. Can be addressed together with `_pretty_morphism`.

### Complexity: `parse_program` — C (17)
**Impact: Medium** | **Complexity: Medium**

`syntax/parse.py:66`. Main parser loop that classifies declaration kinds (morphism, functor, carrier, focus, domain extensions) and dispatches. Extract `_parse_declaration` helper for the per-kind logic. The loop body would become: classify → dispatch → accumulate.

---

## tensors/

### Phase 7 — Custom semiring end-to-end
**Impact: High** | **Complexity: Low**

Smooth-tropical semiring (from legacy example 05). Exercises the full pipeline — notation, semantics, primitives, fusion — with a non-standard semiring. Validates that no part of the pipeline is implicitly real-semiring-specific.

### Semiring law checking
**Impact: Medium** | **Complexity: Low**

Add a `check_laws(samples)` method to `Semiring` in `tensors/semantics.py` (or the semiring dataclass). Verifies associativity, commutativity, identity, annihilation, and distributivity on scalar samples. Gate behind a flag (default off; on for development/research use). Useful for custom and research semirings.

### Fusion — pair nesting end-to-end test
**Impact: Medium** | **Complexity: Low**

`Pair(Pair(c1, id), c2)` — multi-level Pair trees. The `_par_to_optic` walk handles recursive Pair structurally (each Pair factors through Copy, inner Pairs recurse). Needs an end-to-end test to confirm label alignment and shape correctness for deeply nested cases.

### Fusion — opaque leaf optimization metric
**Impact: Low** | **Complexity: Low**

Add a test verifying that `compose(pre_map, fused_contract)` has fewer `BackendPrim` leaves than the unfused chain when the opaque leaf introduces a saving.

### Backend fast-paths — native einsum
**Impact: Medium** | **Complexity: Medium**

Route real-semiring fused contraction to native `numpy.einsum` (or backend equivalent) instead of the align/fold/reduce decomposition. Requires detecting the special case in `compile_contract_spec`. Keeps the generic decomposition for non-standard semirings.

### Contraction order optimization
**Impact: Low** | **Complexity: High** | **Deferred**

Cost-model-driven choice of which contractions to fuse and in what order. Current fusion is greedy (fixpoint iteration). Not needed until the pipeline is exercised with larger expressions.

### Transformer / attention ops
**Impact: Medium** | **Complexity: Medium**

Ops not yet in any backend JSON spec:
- Axis-specialized softmax (axis=-1 baked into primitive)
- ReLU / GELU
- Layer normalization (axis-aware mean / variance / sqrt)
- Attention masking (mask-aware softmax or pre-softmax addition)

Each requires adding the op to the backend JSON specs and wiring up the primitive registration. Routing carried parameters through larger composed blocks is a design question that should be resolved first.

### Complexity: `_par_to_optic` — D (25)
**Impact: Medium** | **Complexity: Medium**

`tensors/fusion.py:230`. Shape-driven recursive descent with three concerns mixed: binary node handling (Parallel/Pair), leaf dispatch (Identity/contract/opaque), and Pair factoring (Copy insertion). Extract `_optic_leaf` for the three leaf cases and `_optic_pair_wrap` for the Copy/validation logic. The binary recursion stays in `_par_to_optic`. Target: main function ≤ B.

### Complexity: `_try_fuse` — C (11)
**Impact: Low** | **Complexity: Low**

`tensors/fusion.py:319`. Borderline — one guard clause extraction or early-return restructuring brings it to B. Low priority.

### Complexity: `Equation.parse` — C (16), `parse_algebra` — C (14)
**Impact: Medium** | **Complexity: Low**

`tensors/notation.py:30` and `notation.py:204`. `Equation.parse` validates and splits einsum notation strings; `parse_algebra` parses semiring declaration blocks. Both are string-processing functions with many validation branches. Extract validation into `_validate_equation` and `_validate_algebra_fields` helpers.

### Complexity: `_adjust_diagonal_axes` — C (11)
**Impact: Low** | **Complexity: Low**

`tensors/primitives.py:101`. Borderline — iterative axis adjustment for diagonal extraction. Low priority.


---

## semantics/ + structure/

### Optic runtime behavioral tests
**Impact: Medium** | **Complexity: Low**

Lens get/set, set/get, set/set laws; prism review/preview roundtrips. These require `realize` → `lower` → `run` and are blocked on nothing — just not yet written. Live in `tests/semantics/`.

### Non-list carrier adapters (Maybe, Tree)
**Impact: Medium** | **Complexity: Medium**

`maybe_carrier` and `tree_carrier` convenience constructors analogous to `list_carrier`. Design notes:
- `Maybe(A) = 1 + A` is a constant polynomial with no `Id`; current `Optic` validation expects a functor position it can `unapply`. The adapter may need a `Just` wrapper that carries the `Id` position explicitly.
- Tree carrier needs an agreed recursive shape (`1 + X × X`) and Hydra encoding of `roll`/`unroll` before implementing. `Tree[x]` is now parseable as `Sum(One(), Prod(Id(), Id()))`, which gives the functor shape for free.
- Do not weaken core recursion semantics. Add only adapter support that produces the existing carrier optic boundary cleanly.
- A full `CarrierExpr` DSL was considered and dropped — the adapter pattern (`list_carrier`, `maybe_carrier`, `tree_carrier`) is sufficient. Revisit only if a concrete case emerges that adapters cannot handle.

### AlgebraHom — typed maps between algebras
**Impact: High** | **Complexity: High** | **Deferred — blocked on carrier adapters + Phase 7**

Typed maps between algebras that commute with the algebra structure. Needed for relating model components (encoder/decoder adjointness, residual connections as natural transformations). `AlgebraHom(f: Morphism, src, tgt)` where `f` is the carrier map.

Blocked on: non-list carrier adapters (Maybe, Tree) and tensor Phase 7 end-to-end — the design questions need concrete use cases to resolve, and those come from exercising the pipeline with real compositions.

Open design questions:
- Does `AlgebraHom` live in `semantics/optics.py` or a new `semantics/algebra.py`?
- How are coherence cells `ε_A`, `δ_A` represented for lax cases?
- Does the optic structure make algebra/coalgebra typing constraints more explicit?

### Complexity: `construct` — E (37)
**Impact: High** | **Complexity: High**

`semantics/construct.py:243`. The largest function in the codebase — a single closure-heavy function that resolves refs, applies parameterized morphisms, dispatches backend primitives, handles domain expressions, poly_fmap, recursion apps, carrier boundaries, and monadic lifts. Most of the inner functions are already extracted into `_construct_helpers.py` but the main match dispatch remains monolithic. Extract by node kind: `_construct_ref`, `_construct_app`, `_construct_domain`, `_construct_recursion` — each a standalone function taking `(node, env, ...)` instead of closing over the environment. This is the highest-value refactoring target in the project.

### Complexity: `construct_program` — B (10) / ruff C901 (40)
**Impact: Medium** | **Complexity: Medium**

`semantics/construct.py:59`. Radon scores B(10) but ruff McCabe flags it at 40 — the discrepancy comes from nested closures that radon counts separately but ruff counts against the enclosing function. Addressing `construct` (above) will resolve this automatically since the closures will become top-level functions.

### Complexity: `realize` — E (31)
**Impact: High** | **Complexity: Medium**

`structure/realize.py:190`. Match dispatch over all MorphismExpr variants to produce Hydra terms. Same extraction pattern as `signature`: split into `_realize_structural` (Identity/Copy/Delete/First/Second/Left/Right/Absurd), `_realize_recursive` (Compose/Parallel/Pair/Case/MonadicEmbed), and leave Prim/DomainPrim/BackendPrim/PolyFmap/SelfRef in the main function. Target: all sub-functions ≤ B.

### Complexity: `poly_action_term` — C (12)
**Impact: Low** | **Complexity: Low**

`structure/realize.py:91`. Match dispatch over PolyExpr variants for functor action on Hydra terms. Borderline — can be addressed alongside the `realize` refactor.

### Complexity: `morphism_refs` — C (12)
**Impact: Low** | **Complexity: Low**

`semantics/_construct_helpers.py:19`. Recursive walk collecting Ref names from MorphismExpr trees. The nesting comes from the many isinstance checks. Low priority — works correctly, rarely modified.

### Complexity: `apply_poly` — C (11)
**Impact: Low** | **Complexity: Low**

`semantics/functors.py:198`. Match dispatch over PolyExpr variants for F(A) type computation. Borderline — one point above threshold. Will gain one more case when `Const` functor is added for `Exp.base` refactor. Worth splitting then, not now.


---

## runtime/

### Lexer comment robustness
**Impact: Low** | **Complexity: Low**

`RecursionError` on long comments. Isolated to `syntax/_lex.py`. Fix the comment tokenization to use iteration instead of recursion.

### Complexity: `load_spec` — C (11)
**Impact: Low** | **Complexity: Low**

`runtime/backend.py:227`. JSON backend spec loader with validation branches for each op kind (elementwise, reduce, structural). Borderline.

### Complexity: `type_from_spec` — C (12), `term_value` — C (11)
**Impact: Low** | **Complexity: Low**

`runtime/codecs.py:116` and `codecs.py:66`. Match dispatches over Hydra type/term variants for encoding/decoding. Straightforward extraction if needed — low priority since these files are rarely modified.

---

## Historical reference — do not revisit

`/home/scanbot/unified-algebra/src/unialg` is a prior, ad-hoc version. Reference only:

- Do not copy code or port abstractions wholesale.
- Do not resurrect `_RecordView` — it coupled domain objects to Hydra record terms invisibly.
- Do not reintroduce `TypedMorphism.kind`-style tags — the current ADT is clearer.
- The old `algebra_hom` bridge exposed a broad functor surface while executing a narrow subset. Avoid similarly incomplete abstractions as stable API.
