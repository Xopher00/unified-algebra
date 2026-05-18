# Tensor Extension Checkpoint — 2026-05-18

## Status: Phases 1–5 complete; Phase 6+ deferred

### What's built and tested (342 tests passing, 6 skipped)

**Phase 1 — Extension framework (core)**
- `src/unialg/extensions.py` — generic registration API (DomainProtocol, register_keyword, register_expr_form, register_domain)
- Parser hook in `syntax/parse.py` — unknown keywords delegate to registered domains, `Program.extensions` field, `_is_decl_start()` for RHS slicing
- Semantic hook in `semantics/construct.py` — domain declarations resolved before morphisms, `_domain_data` and opaque `_domain_context` threaded through standalone `construct()`, domain-tagged expression dispatch
- `main.py` remains orchestration only: it loads backends, passes `BackendOps` opaquely as domain context, and does not walk or lower tensor/domain expression trees

**Phase 2 — Lexer additions**
- `syntax/_lex.py` — STRING (`"..."`), FLOAT (`0.0`), MINUS (`-`) tokens
- `_raw_number()` handles INT/FLOAT with shared-prefix (no backtrack needed)
- MINUS after multi-char operators to preserve `<->` etc.

**Phase 3 — Tensor notation + syntax registration**
- `tensors/notation.py` — `Equation` (parse, target_vars, alignment_plan, replace_input), `AlignmentPlan`, `SemiringDecl`, `ContractExpr`
- `tensors/__init__.py` — `_parse_algebra` keyword parser, `_parse_contract` expression parser, self-registration at import
- `syntax/_grammar.py` — expression form dispatch via `get_expr_handler`
- Repeated labels within single operand rejected (no diagonal/trace semantics yet)

**Phase 4 — Tensor semantics**
- `tensors/semantics.py` — `resolve_semiring`, `contract_morphism`, `ContractSpec`, domain protocol (`construct`, `construct_expr`, `refs`)
- `Semiring.zero` and `Semiring.one` are `float` values (identity elements for future law checking, not Morphism objects, not part of contraction pipeline)
- `ContractSpec` is internal tensor compile data only; it must not be returned upward to core semantics
- All semiring ops (plus, times, reduce, adjoint) are Morphism objects wrapping BackendPrim nodes

**Phase 5 — Tensor contraction compilation**
- `tensors/primitives.py` builds ordinary substrate `Morphism` trees for contractions
- `contract_morphism(sr, eq, context=backend)` now returns a composed Morphism, not `Prim(ContractSpec)`
- Standard contraction shape: `par_all(align_i...) >> fold_product(sr.times, n_inputs) >> axis_reduce(sr.plus_reduce, axes)` when axes are reduced
- Adjoint contraction shape: `par_all(align_i...) >> fold_product(sr.adjoint, n_inputs) >> axis_reduce(sr.times_reduce, axes)`, rejected if required adjoint ops are missing
- Generated tensor primitive leaves are only baked `BINARY -> BINARY` operations:
  - alignment: expand dims + transpose using backend structural metadata
  - axis reduce: backend reduce callable closed over contraction axes
- Backend structural callables live in backend JSON `"structural"` sections and are exposed through generic `BackendOps.structural_op`; no tensor- or numpy-specific primitives were added to `runtime/backend.py`
- End-to-end tests cover real NumPy matmul and tropical/min-plus matvec

### What's NOT built

**Phase 6+ — Fusion and advanced tensor features**
- Contract fusion / semantic rewrite optimizer
- Backend fast paths such as native einsum
- Block-wise splitting for memory-safe broadcast
- Diagonal/trace semantics for repeated labels within one operand
- Semiring law checking using zero/one identity values

### Bugs fixed during Phases 1–4
- `_morphism_refs` handles domain-tagged nodes (returns empty set)
- `domain_data` initialized early in `construct_program` scope, populated in-place
- `_is_decl_start()` checks NAME values against registered extension keywords (fixes `let f = id\nalgebra ...`)

### Files touched (relative to pre-extension state)

**Created:**
- `src/unialg/extensions.py`
- `src/unialg/tensors/notation.py`
- `src/unialg/tensors/semantics.py`
- `src/unialg/tensors/primitives.py`
- `tests/syntax/test_lex_extensions.py` (19 tests)
- `tests/tensors/__init__.py`
- `tests/tensors/test_notation.py` (20 tests)
- `tests/tensors/test_syntax_integration.py` (12 tests)
- `tests/tensors/test_semantics.py` (20 tests)
- `tests/tensors/test_e2e.py` (2 tests)

**Modified:**
- `src/unialg/syntax/_lex.py` — STRING, FLOAT, MINUS tokens
- `src/unialg/syntax/_grammar.py` — extension expr form dispatch
- `src/unialg/syntax/parse.py` — Program.extensions, keyword dispatch, _is_decl_start
- `src/unialg/semantics/construct.py` — domain_data, _domain_context, _domain_tag dispatch, _morphism_refs
- `src/unialg/main.py` — passes loaded backend as opaque domain context during `compile_program`
- `src/unialg/runtime/backend.py` — generic backend callable metadata (`BackendPrimitive.fn`) and JSON-backed structural callables
- `src/unialg/runtime/backends/*.json` — structural sections for backend-native expand/transpose calls
- `src/unialg/tensors/__init__.py` — full rewrite (registration + parsers)
- `src/unialg/tensors/semirings.py` — zero/one changed from Morphism to float
- `src/unialg/__init__.py` — `import tensors` for registration trigger
