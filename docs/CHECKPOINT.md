# Checkpoint — Current State (2026-05-25)

**Tests:** 524 passing, 6 skipped. Typed literal points, configured axis arguments, and coproduct bimap syntax/realization are covered across semantic, runtime, and tensor composition tests.

---

## Recent changes (2026-05-25)

**Hydra dependency migrated to `hydra-kernel==0.15.0` (PyPI)**
- `hatch_build.py` and its copy-from-local-clone hook removed.
- `src/hydra/` removed from the source tree; `[tool.hatch.build.targets.wheel]` no longer lists it.
- `hydra-kernel==0.15.0` added to `[project].dependencies`.
- `hydra-python` must not be co-installed — it owns the same `hydra/` namespace directory and clobbers the kernel modules on install. Only `hydra-kernel` is needed.
- The local clone at `/home/scanbot/hydra` (121 commits ahead of v0.15.0) is retained as a reference but is no longer part of the build.

**`construct.py` literal parser refactor**
- `_literal_value`, `_argument_types` extracted from inside `construct()` to module level.
- `_parse_literal_value` rewritten as a dispatch via `_LITERAL_PARSERS` dict; per-type helpers `_parse_bool`, `_parse_int`, `_parse_float` extracted as named module-level functions.
- `from unialg.objects import TypePair, TypeUnit, ProductType, SumType` hoisted from inside `construct()` to module level.
- `Coparallel` match case added to the `construct()` dispatch (`ops.copar(_recurse(f), _recurse(g))`).
- `construct` ruff C901: 71 → 56. `construct_program` C901(42) unchanged — requires the Tier 4 phase-based split.

---

## Sealed semantic contract (Contract 8)

Do not revisit or redesign the following.

- **Types are native Hydra** — `SpaceT` eliminated; use `hydra.core.Type` variants directly
- Type vocabulary: `TypeVariable`, `TypePair(PairType(l,r))`, `TypeEither(EitherType(l,r))`, `TypeUnit()`, `TypeVoid()`, `TypeFunction(FunctionType(d,c))`, `TypeList(b)`, `TypeMaybe(b)`
- Convenience constructors in `objects.py`: `ProductType(l,r)` → `TypePair(...)`, `SumType(l,r)` → `TypeEither(...)`
- `Morphism(node: MorphismExpr, param: Type, monad: Monad|None, aux_primitives: tuple)` — three modes: plain, parametric, lax
- `Morphism.to_lax(monad)` — universal coercion; `_resolve_monad(*morphisms)` — derives shared monad or errors
- `compose` and `case` auto-embed plain morphisms into lax context
- dom/cod derived via `dom_of(node)` / `cod_of(node)` — NOT stored on Morphism
- `MorphismError(TypeError)` — single error class with `check(a, b, msg)` classmethod
- `MorphismExpr` ADT: Identity, Copy, Delete, Literal, First, Second, Left, Right, Absurd, Assoc, Symmetry, DistributeLeft, DistributeRight, MonadicEmbed, ContextualBinary (base), Compose, SharedCompose, Parallel, Coparallel, Pair, Case, Prim, DomainPrim, BackendPrim
- Typed literal points are restored as `lit(value, A) : Unit -> A`; single-quoted payloads resolve only at typed application sites and lift into context through `delete >> lit`
- Distributivity: `distribute_left(a,b,c) : A×(B+C)→(A×B)+(A×C)`, `distribute_right(a,b,c) : (A+B)×C→(A×C)+(B×C)`; DSL keywords `distl`, `distr`
- `merge(a) : A+A→A` — codiagonal; derivable as `case(id,id)` but named in the vocabulary; DSL keyword `merge`
- `ContextualBinary` subclasses carry `f, g, f_param, g_param, param, monad, dom, cod` — dom/cod are stored and authoritative
- `copar(f,g) : A+C -> B+D`; DSL notation `f && g` constructs the coproduct bimap
- `PolyExpr` ADT: Zero, One, Id, Const(space), Sum(l,r), Prod(l,r), Exp(base: PolyExpr, body), PolyCompose(l,r), List(body), Maybe(body)
- Extension activation: `extensions.enable("tensors")` / `is_enabled("tensors")`; DSL `load extension tensors`; auto-registration removed from `tensors/__init__.py`
- `Functor(name, body: PolyExpr)` — named polynomial endofunctor; `apply`, `unapply`, `compose`, `map` as instance methods
- `apply_poly(body, space)` — pure type substitution F(space)
- `realize(node: MorphismExpr)` → raw Hydra term; explicit call, no monkey-patch
- `poly_action_term(body, h, monad=None)` — plain map or monadic traversal
- `poly_fmap(functor, h)` — lifts Morphism through polynomial functor; deferred `PolyFmap` node
- Contextual combinators accept `shared_context=True` to share matching non-unit params
- `Optic(functor, forward, backward, carrier)` — unified polynomial optic; Lens/Prism/Traversal are functor choices
- `Optic.par(other)` — parallel/product combinator; shared carrier required
- `cata`, `ana`, `hylo` — plain/para/lax/lax-para algebras and coalgebras
- `main.lower(m, graph)`, `main.run(m, arg, ctx, graph)` — execution boundary
- `Monad(type_ctor, bind_name, pure_name)`: `MAYBE`, `LIST`

---

## Architecture layers

```
surface expression
  → typed interpretation
  → algebraic construction
  → executable assembly
  → backend realization
```

| Layer | Files | Hydra? |
|-------|-------|--------|
| Type ground | `objects.py` | types only |
| Surface syntax | `syntax/expressions.py`, `syntax/parse.py` | no; quoted literal payloads remain unresolved |
| Semantic construction | `semantics/construct.py` | types only |
| Algebraic construction | `semantics/morphisms.py` | types only |
| Polynomial functors | `semantics/functors.py` | type unification only |
| Optics | `semantics/optics.py` | no term construction |
| Term realization | `structure/realize.py` | yes; realizes typed literal points as Hydra constants |
| Recursion schemes | `structure/recursion.py` | yes |
| Hydra vocabulary | `structure/terms.py` | yes |
| Runtime boundary | `runtime/` | yes |
| Orchestration | `main.py` | yes |

---

## Package layout

```
src/unialg/
├── syntax/
│   ├── expressions.py        MorphismExpr + PolyExpr ADTs — no Hydra
│   ├── parse.py              Surface parser — Program/MorphismExpr nodes
│   ├── _grammar.py           Pratt grammar callbacks — morphism + functor
│   ├── _lex.py               Tokenizer
│   └── _ops.py               Operator binding powers and node constructors
├── semantics/
│   ├── construct.py          Parser AST → semantic Morphisms
│   ├── _construct_helpers.py PolyExpr ref resolution, focus-expr helpers
│   ├── morphisms.py          dom_of/cod_of, Morphism, combinators
│   ├── functors.py           Functor, PolyExpr helpers, apply_poly, poly_fmap
│   ├── optics.py             Optic, RecursiveCarrier, cata/ana/hylo
│   └── typeops.py            Type unification, require_equal, visible_domain
├── structure/
│   ├── realize.py            realize(MorphismExpr) → raw Hydra term
│   ├── recursion.py          recursive_carrier, carrier adapter shims
│   └── terms.py              Hydra primitive name catalog + pair/term helpers
├── runtime/
│   ├── backend.py            BackendPrimitive, BackendOps, load_spec
│   ├── boundary.py           RuntimeStore, encode/decode boundary
│   ├── codecs.py             Term coders, value boundary helpers
│   └── backends/             JSON backend specs (numpy, torch, jax, cupy)
├── tensors/
│   ├── notation.py           Equation, AlignmentPlan, SemiringDecl, ContractExpr
│   ├── semirings.py          Semiring dataclass (carrier, plus, times, zero, one, adjoint, reduce fields)
│   ├── semantics.py          resolve_semiring, contract_morphism, ContractSpec, _strip_exp
│   ├── primitives.py         compile_contract_spec, alignment/fold trees, diagonal extraction
│   ├── fusion.py             normalize_contracts, _par_to_optic, shape-based fusion
│   └── __init__.py           Self-registration with finalize hook
├── extensions.py             DomainProtocol, register_keyword/expr_form/domain
├── objects.py                Type aliases, ProductType, SumType, Monad, show_type
├── main.py                   compile_program/morphism, lower, run, CompiledProgram
└── __init__.py               Public surface
```

---

## Tensor extension — implemented (Phase 6.7)

- **Extension framework** — `extensions.py` generic registration API; parser and semantic hooks in `parse.py` and `construct.py`
- **Tensor notation** — `tensors/notation.py`: `Equation` (parse, alignment_plan, diagonal_axes, post_diagonal_labels), `SemiringDecl`, `ContractExpr`
- **Tensor semantics** — `tensors/semantics.py`: `resolve_semiring`, lazy `contract_morphism` returning `DomainPrim`, `ContractSpec` with `shape: PolyExpr` field using `Exp(index_product(labels), Id())` per slot
- **ContractSpec dom/cod** — `dom` and `cod` now return substrate-level `BINARY` types (stripped via `_strip_exp`). The `ExpType`-wrapped label information lives only in `ContractSpec.shape` (the `PolyExpr`). This allows tensor contractions to compose directly with `BINARY→BINARY` activation functions. `_strip_exp` is retained at the substrate boundary in `_decompose_leaf` as a no-op safety check.
- **Tensor compilation** — `tensors/primitives.py`: shape-guided `_build_alignment_tree` and `_build_fold_tree`; `diagonal_extract_morphism` with cross-backend `_call_diagonal`
- **Diagonal/trace** — repeated labels accepted; `diagonal_axes`, `post_diagonal_labels` with numpy reordering; `take_diagonal` in all backend JSON specs
- **Fusion pass** — `tensors/fusion.py`: `normalize_contracts` finalize hook; `_par_to_optic` catamorphism using `Optic.par`; handles Parallel-tree, opaque leaves, all Pair variants; alpha-renaming of reduced labels in `_rename_shape_labels`; shape-driven label extraction via `Exp` bases (no parallel label tracking)

---

## Semiring residuals — design note

`Semiring.adjoint` is the only residual hook. The underlying law (`a ⊗ b ≤ c iff b ≤ a ⇒ c`) relates to quantales and enriched adjunctions, but no first-class `Quantale`, `Poset`, or inequality DSL should be introduced until there is a concrete non-tensor use case. `Functor.category="poset"` is a placeholder guard only.
