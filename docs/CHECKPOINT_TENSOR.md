# Tensor Extension Checkpoint — 2026-05-18

## Status: Phases 1–4 complete, Phase 5 blocked on lowering context design

### What's built and tested (339 tests passing)

**Phase 1 — Extension framework (core)**
- `src/unialg/extensions.py` — generic registration API (DomainProtocol, register_keyword, register_expr_form, register_domain)
- Parser hook in `syntax/parse.py` — unknown keywords delegate to registered domains, `Program.extensions` field, `_is_decl_start()` for RHS slicing
- Semantic hook in `semantics/construct.py` — domain declarations resolved before morphisms, `_domain_data` threaded through standalone `construct()`, domain-tagged expression dispatch
- Pre-realize hook in `main.py` — `_lower_domain_prims()` walks ContextualBinary children, delegates domain Prim nodes to `protocol.lower()`

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
- `contract_morphism(sr, eq)` returns `Morphism(Prim(ContractSpec, dom, cod))` — staging node
- All semiring ops (plus, times, reduce, adjoint) are Morphism objects wrapping BackendPrim nodes

### What's NOT built

**Phase 5 — Lowering (ContractSpec → executable substrate)**
- `tensors/primitives.py` does not exist
- The lowering stub raises `NotImplementedError`
- Design blocked on: how runtime context (RuntimeStore, structural callables, axis-aware reduce) flows from `compile_program` → lowering

**Phase 6–7 — End-to-end runtime, fusion**
- Not started

### Bugs fixed during Phases 1–4
- `_morphism_refs` handles domain-tagged nodes (returns empty set)
- `domain_data` initialized early in `construct_program` scope, populated in-place
- `_is_decl_start()` checks NAME values against registered extension keywords (fixes `let f = id\nalgebra ...`)

### Files touched (relative to pre-extension state)

**Created:**
- `src/unialg/extensions.py`
- `src/unialg/tensors/notation.py`
- `src/unialg/tensors/semantics.py`
- `tests/syntax/test_lex_extensions.py` (19 tests)
- `tests/tensors/__init__.py`
- `tests/tensors/test_notation.py` (20 tests)
- `tests/tensors/test_syntax_integration.py` (12 tests)
- `tests/tensors/test_semantics.py` (20 tests)

**Modified:**
- `src/unialg/syntax/_lex.py` — STRING, FLOAT, MINUS tokens
- `src/unialg/syntax/_grammar.py` — extension expr form dispatch
- `src/unialg/syntax/parse.py` — Program.extensions, keyword dispatch, _is_decl_start
- `src/unialg/semantics/construct.py` — domain_data, _domain_tag dispatch, _morphism_refs
- `src/unialg/main.py` — _lower_domain_prims, hooked into compile_morphism/lower/run
- `src/unialg/tensors/__init__.py` — full rewrite (registration + parsers)
- `src/unialg/tensors/semirings.py` — zero/one changed from Morphism to float
- `src/unialg/__init__.py` — `import tensors` for registration trigger
