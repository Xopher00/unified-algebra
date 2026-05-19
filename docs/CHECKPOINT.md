# Checkpoint — Current State (2026-05-19)

**Tests:** 402 passing, 4 skipped. All import-linter boundaries clean.

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
- `MorphismExpr` ADT: Identity, Copy, Delete, First, Second, Left, Right, Absurd, Assoc, Symmetry, MonadicEmbed, ContextualBinary (base), Compose, Parallel, Pair, Case, Prim, DomainPrim, BackendPrim
- `ContextualBinary` subclasses carry `f, g, f_param, g_param, param, monad, dom, cod` — dom/cod are stored and authoritative
- `PolyExpr` ADT: Zero, One, Id, Const(space), Sum(l,r), Prod(l,r), Exp(base: Type, body), List(body), Maybe(body)
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
| Surface syntax | `syntax/expressions.py`, `syntax/parse.py` | no |
| Semantic construction | `semantics/construct.py` | types only |
| Algebraic construction | `semantics/morphisms.py` | types only |
| Polynomial functors | `semantics/functors.py` | type unification only |
| Optics | `semantics/optics.py` | no term construction |
| Term realization | `structure/realize.py` | yes |
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
│   ├── semantics.py          resolve_semiring, contract_morphism, ContractSpec
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
- **ContractSpec dom/cod** — emit `ExpType`-wrapped types; `_strip_exp` at substrate boundary in `_decompose_leaf`
- **Tensor compilation** — `tensors/primitives.py`: shape-guided `_build_alignment_tree` and `_build_fold_tree`; `diagonal_extract_morphism` with cross-backend `_call_diagonal`
- **Diagonal/trace** — repeated labels accepted; `diagonal_axes`, `post_diagonal_labels` with numpy reordering; `take_diagonal` in all backend JSON specs
- **Fusion pass** — `tensors/fusion.py`: `normalize_contracts` finalize hook; `_par_to_optic` catamorphism using `Optic.par`; handles Parallel-tree, opaque leaves, all Pair variants; alpha-renaming of reduced labels in `_rename_shape_labels`; shape-driven label extraction via `Exp` bases (no parallel label tracking)

---

## Semiring residuals — design note

`Semiring.adjoint` is the only residual hook. The underlying law (`a ⊗ b ≤ c iff b ≤ a ⇒ c`) relates to quantales and enriched adjunctions, but no first-class `Quantale`, `Poset`, or inequality DSL should be introduced until there is a concrete non-tensor use case. `Functor.category="poset"` is a placeholder guard only.
