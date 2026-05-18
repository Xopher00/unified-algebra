# Tensor Extension Checkpoint — 2026-05-18

## Status: Phases 1–6.5 complete (392 tests passing, 4 skipped)

### What's built and tested

**Phase 1 — Extension framework (core)**
- `src/unialg/extensions.py` — generic registration API (DomainProtocol, register_keyword, register_expr_form, register_domain)
- Parser hook in `syntax/parse.py` — unknown keywords delegate to registered domains, `Program.extensions` field, `_is_decl_start()` for RHS slicing
- Semantic hook in `semantics/construct.py` — domain declarations resolved before morphisms, `_domain_data` and opaque `_domain_context` threaded through standalone `construct()`, domain-tagged expression dispatch, finalize hooks invoked after morphism construction
- `main.py` remains orchestration only

**Phase 2 — Lexer additions**
- `syntax/_lex.py` — STRING, FLOAT, MINUS tokens

**Phase 3 — Tensor notation + syntax registration**
- `tensors/notation.py` — `Equation` (parse, target_vars, alignment_plan, replace_input, diagonal_axes, post_diagonal_labels), `AlignmentPlan`, `SemiringDecl`, `ContractExpr`
- `tensors/__init__.py` — self-registration with finalize hook

**Phase 4 — Tensor semantics**
- `tensors/semantics.py` — `resolve_semiring`, `contract_morphism` (lazy — returns DomainPrim), `ContractSpec` with `shape: PolyExpr` field, `_left_nested_shape`, `_count_id`
- `ContractSpec.dom` uses `apply_poly(shape, BINARY)` for exact nested ProductType

**Phase 5 — Tensor contraction compilation**
- `tensors/primitives.py` — shape-guided `_build_alignment_tree` and `_build_fold_tree` match polynomial nesting; `compile_contract_spec` uses these when `spec.shape` is present
- `diagonal_extract_morphism` with `_adjust_diagonal_axes` for iterative diagonal extraction
- `_call_diagonal` adapts for torch (dim1/dim2) vs numpy (axis1/axis2) API differences

**Phase 6 — Semantic fusion pass**
- `tensors/fusion.py::normalize_contracts` — domain finalize hook
- `DomainPrim` MorphismExpr subclass; `realize.py` guard raises if one escapes
- `_rewrite_bottom_up` — generic bottom-up Morphism tree traversal (catamorphism pattern)
- `_fuse_to_fixpoint` — iterates fusion to convergence

**Phase 6.5A — Polynomial-shape-correct fusion**
- `_absorb_par_tree` — recursive tree walk (no flattening), builds equation inputs + polynomial shape together
- `apply_poly(fused_shape, BINARY)` computes exact nested ProductType — fixes left-nesting mismatch
- Invariants: `count_id(shape) == len(equation.inputs)` and `apply_poly(shape, BINARY) == spec.dom`

**Phase 6.5B — Opaque-leaf preserving fusion**
- Par-tree leaves that are neither Identity nor DomainPrim are treated as residue
- Result: `compose(pre_map, fused_contract)` — the optic operational form
- `pre_map` runs opaque morphisms at their positions; passthrough inputs go directly to fused contraction
- Guard: opaque-only trees (no absorbed contracts) correctly blocked

**Phase 6.5C — Pair/shared-input fusion**
- `_try_fuse_pair` — detects `Compose(Pair(c1, id), outer)`, verifies preconditions
- Alpha-renaming via Hydra: `_labels_to_type` encodes equation labels as `TypeVariable(Name(...))` trees; `substitute_type_variables` performs renaming; `_type_to_labels` decodes
- `_rename_reduced_labels` finds fresh single-char labels for collisions with identity-branch labels
- Result: `compose(Copy(shared_dom), fused_contract)` with renamed equation
- Guards: plain-context (TypeUnit param, no monad), semiring/adjoint match, slot count, label correspondence

**Phase 6.5D — Diagonal/trace semantics**
- `Equation.parse` accepts repeated labels (rejection removed)
- `diagonal_axes(i)` returns axis pairs; `post_diagonal_labels(i)` simulates numpy.diagonal reordering
- `alignment_plan(i)` uses post-diagonal labels for correct unsqueeze/transpose
- `take_diagonal` structural op added to all 4 backend JSON specs
- `diagonal_extract_morphism` inserted before alignment for operands with repeated labels

### Files created (since Phase 5)
- `src/unialg/tensors/fusion.py` — fusion pass with Hydra-based alpha-renaming
- `tests/tensors/test_fusion.py` — 33 fusion tests (structural, numerical, optimization, opaque, pair)
- `tests/tensors/test_diagonal.py` — 11 diagonal/trace tests

### Files modified (since Phase 5)
- `src/unialg/syntax/expressions.py` — `DomainPrim` MorphismExpr subclass
- `src/unialg/semantics/morphisms.py` — signature dispatch for DomainPrim
- `src/unialg/semantics/construct.py` — finalize hook invocation
- `src/unialg/structure/realize.py` — DomainPrim guard
- `src/unialg/extensions.py` — `DomainProtocol.finalize`, `registered_domains()`
- `src/unialg/tensors/semantics.py` — `ContractSpec.shape`, `_left_nested_shape`, `_count_id`, lazy `contract_morphism`
- `src/unialg/tensors/primitives.py` — shape-guided alignment/fold trees, diagonal extraction, fail-closed shape checks
- `src/unialg/tensors/notation.py` — diagonal methods, post_diagonal_labels with axis reordering
- `src/unialg/tensors/__init__.py` — finalize hook registration
- `src/unialg/runtime/backends/*.json` — `take_diagonal` structural op
- `pyproject.toml` — `unialg.tensors.fusion` added to import-linter contract
- `tests/tensors/test_semantics.py` — updated for lazy contract_morphism
- `tests/tensors/test_notation.py` — diagonal parsing tests replace rejection test
