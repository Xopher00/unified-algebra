# Checkpoint — Semantic Contract 8 (sealed)

## Date
2026-05-17 (updated after runtime/main layering refresh)

## Sealed contract

The following constitute the sealed semantic contract. Do not revisit or redesign these.

- **No parallel type hierarchy** — `SpaceT` eliminated; types are native `hydra.core.Type` variants directly
- Type vocabulary: `TypeVariable`, `TypePair(PairType(l,r))`, `TypeEither(EitherType(l,r))`, `TypeUnit()`, `TypeVoid()`, `TypeFunction(FunctionType(d,c))`, `TypeList(b)`, `TypeMaybe(b)`
- Convenience constructors in `objects.py`: `ProductType(l, r)` → `TypePair(PairType(l,r))`, `SumType(l, r)` → `TypeEither(EitherType(l,r))`
- Typed morphisms: `Morphism(node: MorphismExpr, param: Type, monad: Monad|None, aux_primitives: tuple)`
- Three modes: plain (param=TypeUnit(), monad=None), para (param≠TypeUnit(), monad=None), lax (monad≠None)
- `Morphism.to_lax(monad)` — universal coercion: plain → lax via MonadicEmbed; same monad → self; conflict → MorphismError
- `_resolve_monad(*morphisms)` — derives target monad from a set of morphisms; errors on conflict
- `compose` and `case` auto-embed plain morphisms into lax context via `_resolve_monad` + `to_lax`
- `embed` removed — was a shim over `to_lax`; use `m.to_lax(monad)` directly
- dom/cod derived via `dom_of(node)` / `cod_of(node)` — NOT stored on Morphism
- `MorphismError(TypeError)` — single error class with `check(a, b, msg)` classmethod; replaces former `CompositionError` and `CaseError`
- `MorphismExpr` ADT: Identity, Copy, Delete, First, Second, Left, Right, Absurd, Assoc, MonadicEmbed, ContextualBinary (base), Compose, Parallel, Pair, Case, Prim(raw, dom, cod)
- `ContextualBinary` subclasses carry fields: `f, g, f_param, g_param, param, monad, dom, cod` — dom/cod are stored and authoritative, not recomputed
- `PolyExpr` ADT: Zero, One, Id, Const(space), Sum(l,r), Prod(l,r), Exp(base, body)
- `Functor(name, body: PolyExpr)` — named polynomial endofunctor, defined in `functors.py`
- `functors.apply_poly(body, space)` — pure type substitution F(space)
- `Functor.apply(space)`, `Functor.unapply(fa)`, `Functor.compose(inner)` — public object action, inverse object action, and functor composition methods
- `realize(node: MorphismExpr)` → raw Hydra term; call explicitly, no monkey patch on syntax nodes
- `structure.realize.poly_action_term(body, h, monad=None)` → raw TTerm-level polynomial action; handles plain map (monad=None) and monadic traversal (monad set) in one function
- `semantics.functors.poly_fmap(functor: Functor, h: Morphism)` — lifts Morphism through polynomial functor; uniform across plain/para/lax
- Contextual combinators accept `shared_context=True` to share matching non-unit params; used by recursion schemes
- Recursion optic actions: `act`, `act_forward`, `act_backward`, `compose_optic`, `list_carrier`, `cata`, `ana`, `hylo`
- `main.lower(m, graph)` — pure extraction: Morphism → Hydra term
- `main.run(m, arg, ctx, graph)` — apply and reduce; augments graph with aux_primitives
- `Monad(type_ctor, bind_name, pure_name)`: `MAYBE`, `LIST`

## Architectural note — DSL layers clarified by `explore.ipynb`

The reader-first explorer clarified the intended conceptual stack:

1. **Morphism layer** — typed arrows, products, sums, params, effects, functor lifting
2. **Optics layer** — lenses, prisms, traversals, and height-2/polynomial lenses built from morphisms
3. **Recursion layer** — fixed points, folds, unfolds, and recursive carriers
4. **Execution layer** — Hydra term generation and reduction boundary

Do not conflate these layers:

- `PolyExpr` / `poly_fmap` describes and maps over one layer of a shape.
- `recursion.py` is for repeated roll/unroll over recursive fixed points.
- Lenses and optics are focus/update abstractions built above morphisms; they do **not** belong inside `recursion.py`.
- Height-2 or polynomial lenses belong near the functor/shape layer, depending on `PolyExpr`, `apply_poly`, `poly_fmap`, and `Morphism`.

## Package layout (as of 2026-05-17)

The package now uses subdirectories matching the architectural layers.

```
src/unialg/
├── syntax/
│   ├── expressions.py     MorphismExpr + PolyExpr ADTs — pure Python, no Hydra
│   └── parse.py           Surface parser — produces Program/MorphismExpr nodes
├── semantics/
│   ├── construct.py        Program construction — parser AST to semantic Morphisms
│   ├── morphisms.py        dom_of/cod_of + combinators — Hydra-free
│   ├── functors.py         Functor + PolyExpr helpers — Hydra type unification only
│   ├── optics.py           Unified polynomial functor Optic — no Hydra terms
│   └── typeops.py          fresh_type_var, require_equal, type unification helpers
├── structure/
│   ├── realize.py          realize(MorphismExpr) → raw Hydra term
│   ├── recursion.py        act/act_forward/act_backward, cata/ana/hylo — optic actions + recursion schemes
│   └── terms.py            Hydra primitive name catalog + term helpers (includes product_arg)
├── runtime/
│   ├── backend.py          BackendPrimitive, register_backend_primitive, BackendOps, load_spec
│   ├── boundary.py         RuntimeStore, native boundary traversal, program I/O
│   ├── codecs.py           Term coders and structural Hydra-term decoding
│   └── backends/           JSON backend specs (numpy, torch, jax, cupy)
├── objects.py              Type aliases, TypeScheme, show_type, ProductType, SumType, Monad, etc.
├── main.py                 compile_program/compile_morphism/lower/run orchestration
├── tensors/
│   ├── semirings.py        Semiring dataclass and backend semiring helpers
│   ├── contractions.py     Tensor contraction helpers
│   ├── layer.md            Tensor layer notes
│   └── equationsemantics.md Tensor equation notes
└── __init__.py             Public surface
```

## Design note — residuals and quantale-enriched structure

Legacy contained an adjoint/residual mode for semiring tensor contraction:
standard contraction used `plus_reduce(times)`, while adjoint mode used
`times_reduce(residual)`. This corresponds to the order-theoretic residuation
law:

```text
a ⊗ b ≤ c  iff  b ≤ (a ⇒ c)
```

This is related to posets, quantales, Galois connections, and enriched
adjunctions, but the current project should not generalize it prematurely.

Current sealed interpretation:

- `tensors/semirings.py` owns the concrete residual hook through optional
  `Semiring.adjoint`.
- `Semiring.op_env(adjoint=True)` selects the tensor contraction role
  `product=adjoint`, `fold=times_reduce`, `seed=one`.
- A future `contract_morphism(sr, equation, adjoint=True)` should use that
  selection when tensor contraction is implemented.
- Do not add first-class `Quantale`, `Poset`, `EnrichedAdjunction`, or
  inequality syntax until residuals become useful outside tensor contraction.
- `Functor.category="poset"` remains a placeholder/guard, not a real enriched
  functor implementation.

### Layer responsibilities

| File | Layer | Hydra? |
|------|-------|--------|
| objects.py | type-level ground | types only |
| syntax/expressions.py | expression data | no Hydra terms |
| syntax/parse.py | surface parsing | no Hydra terms |
| semantics/construct.py | semantic program construction | type objects only |
| semantics/morphisms.py | algebraic construction | type objects only |
| semantics/functors.py | polynomial functor semantics | type unification only |
| semantics/optics.py | unified polynomial functor optic | no term construction |
| structure/terms.py | Hydra term vocabulary | yes |
| structure/realize.py | semantic expression realization | yes |
| structure/recursion.py | fixed-point carrier adapters | yes |
| runtime/ | backend/native value boundary | yes |
| main.py | orchestration and execution boundary | yes |

### expressions.py
- `MorphismExpr` sealed hierarchy (leaves + binary combinators + Prim)
- `PolyExpr` sealed hierarchy
- Both are frozen dataclasses, no Hydra imports

### objects.py
- `ProductType(l, r) -> TypePair`, `SumType(l, r) -> TypeEither` — thin constructors for object shapes
- `Monad(type_ctor, bind_name, pure_name)`, `MAYBE`, `LIST` — effect descriptors used by lax morphisms

### morphisms.py
- `dom_of(node: MorphismExpr) -> Type`, `cod_of(node: MorphismExpr) -> Type`
- `Morphism(node, param, monad, aux_primitives)` + `Morphism.to_lax(monad)` — universal lax coercion
- `_resolve_monad(*morphisms)` — determines target monad; errors on conflict
- `MorphismError(TypeError)` with `check(a, b, msg)` classmethod — replaces former `CompositionError` and `CaseError`
- Constructors: `_identity`, `_copy`, `_delete`, `_fst`, `_snd`, `_inl`, `_inr`, `absurd`
- Combinators: `compose`, `par`, `pair`, `case` — pass class directly to `_contextual_binary`; `compose`/`case` auto-embed via `_resolve_monad`+`to_lax`
- `shared_context=True` — available on all contextual combinators; shares a matching non-unit param instead of multiplying contexts; rejects distinct non-unit params
- `_contextual_binary(cls, f, g, dom, cod)` — constructs the appropriate `ContextualBinary` subclass; resolves monad and param
- `embed` removed — use `m.to_lax(monad)` directly
- PolyExpr helpers live in `functors.py`: `zero`, `one`, `id_`, `const`, `sum_`, `prod`, `exp`, `apply_poly`
- Functor introspection lives on `Functor`: `summands()`, `x_arity()`, `consts()`
- `signature` reads stored `dom`/`cod` from `ContextualBinary` nodes directly

### functors.py
- `Functor(name, body: PolyExpr)` — named polynomial endofunctor descriptor
- `Functor.apply(space)` — object action method, delegates to `apply_poly(self.body, space)`
- `Functor.unapply(fa)` — inverse object action, delegates type matching to Hydra unification by solving `F(A) = fa`
- `Functor.compose(inner)` — public functor composition, backed by `compose_poly(self.body, inner.body)`
- PolyExpr constructors: `zero`, `one`, `id_`, `const`, `sum_`, `prod`, `exp`
- `apply_poly(body, space)` — internal recursive implementation of F(A)
- Introspection methods: `summands()`, `x_arity()`, `consts()`
- Former `functor_summands`/`functor_x_arity`/`functor_consts` free functions removed — use methods on `Functor` directly

### realize.py
- `realize(node: MorphismExpr) -> raw Hydra term` — dispatches on all MorphismExpr cases; Compose/Parallel/Pair/Case use `_ctx_preamble`
- `_ctx_preamble(node: ContextualBinary) -> (value, call_f, call_g)` — splits input, builds param-aware call closures; `param_term` captured by closure, not returned
- `_pair_effects(monad, left, right)` — pairs two terms; sequences monadic effects when monad is set
- Uses `structure/terms.py` as the Hydra term vocabulary; no scattered ad hoc primitive-name construction

### functors.py / realize.py
- `semantics.functors.poly_fmap(functor, h: Morphism) -> Morphism` — produces a deferred `PolyFmap` node; uniform across plain/para/lax
- `structure.realize.poly_action_term(body, h, monad=None)` — realizes that deferred functor action; plain map when monad=None, monadic traversal when monad set

### recursion.py
- `recursive_carrier(*, functor, carrier, unroll, roll)` — wraps callable roll/unroll boundaries as `Prim` morphisms
- `fixed_point_optic` — lower-level shim for direct raw-term injection
- Optic action and recursion-scheme functions (`act`, `act_forward`, `act_backward`, `cata`, `ana`, `hylo`) live on/in `semantics/optics.py`; `structure/realize.py` materializes their deferred nodes
- Carrier-specific roll/unroll helpers are optional adapters that produce `forward : μF → F(μF)` and `backward : F(μF) → μF`; they are not required core semantics
- Old API (`rec`, `cata`, `ana`, `Inductive`, `LIST_IND`, `MAYBE_IND`, `AlgebraError`) in `recursion.py.bak` — do not reference

### optics.py
- Single `Optic(functor: Functor, forward: Morphism, backward: Morphism)` — unified polynomial functor optic
- `forward: S → F(A)` decomposes source into F-shaped container; `backward: F(B) → T` reconstructs target
- Uniform action for all optics: `compose(forward, poly_fmap(F, h), backward)` — implemented as methods on `Optic`
- `focus` and `replacement` derived via strict `functor.unapply()` on forward codomain / backward domain
- `source` and `target` derived from `forward.dom()` / `backward.cod()`
- `Functor.unapply(fa)` — strict public inverse: builds `F(A)` with a Hydra type variable, asks `hydra.unification.unify_types` to solve `F(A) = fa`, then checks round-trip `self.apply(A) == fa`
- Validation: `Optic.__post_init__` relies on strict `Functor.unapply()` for forward/backward compatibility
- Lens, Prism, Traversal are functor choices, not separate types:
  - Lens: `F = Prod(Id(), Const(residue))` — product focus
  - Prism: `F = Sum(Id(), Const(residue))` — sum focus
  - Traversal: arbitrary polynomial F — multi-element focus
- For simple cases (fst_lens, left_prism) where S = F(A), forward and backward are identity morphisms
- Convenience constructors: `fst_lens(a, b)`, `snd_lens(a, b)`, `left_prism(a, b)`, `right_prism(a, b)`
- No Hydra term construction; validation reuses `morphisms.MorphismError.check`
- Height-2 optics require no structural change — just use deeper polynomial bodies

### main.py
- `compile_program(src, ...)` → parse source, construct semantic program tree, realize selected final route/map, return `CompiledProgram`
- `compile_morphism(m, ...)` → realize an existing typed `Morphism`, collect auxiliary primitives, return `CompiledProgram`
- `lower(m, graph)` → realize a raw Hydra term without evaluating it
- `run(m, arg, ctx, graph)` → augment graph with `m.aux_primitives`, reduce a raw Hydra argument
- `CompiledProgram.run(*args)` → source-program execution path; uses `runtime/boundary.py` for Python value boundary handling

### Public surface (`__init__.py`)
- Spaces, Morphism, combinators, PolyExpr types, Functor, recursion exports, lower, run

## runtime/ layer (as of 2026-05-17)

### runtime/backend.py
- `BackendPrimitive(primitive, arity, arg_type, result_type, arg_coder, result_coder)` — resolved backend op
- `register_backend_primitive(canonical_name, path, arg_type, arity, ...)` → `BackendPrimitive`
- `load_spec(spec)` — loads JSON, returns `dict[str, BackendPrimitive]`
- `BackendOps.from_spec(path)` — top-level runtime entry point; primitives keyed by logical op name
- `library_to_graph(library, base=None)` — installs backend Library into a Hydra Graph; uses `objects.standard_graph()` as default base
- Canonical primitive names follow `unialg.backend.<op>` pattern
- `backend_coverage`, `compare_backend_coverage`, `backend_required_for_term` — utility queries

### runtime/codecs.py
- Pure value-boundary layer: no morphism or expression dependencies
- `expect_right`, `_literal_value`, `term_value`, `_mk_term_coder` — shared boundary helpers
- `type_from_spec`, `coder_for_type`, `encode_python` — type/coder entry points
- Legacy registry helpers remain only for backend-spec compatibility

### runtime/boundary.py
- `RuntimeStore` stores native values behind Hydra binary handles
- `BinaryAdapter`, `encode_boundary_input`, `decode_boundary_output` handle native BINARY leaves
- `pack_args(args)` preserves the single-argument vs multiple-argument program boundary
- `encode_input(backend, domain, ctx, value)` delegates to backend/native boundary code
- `decode_output(backend, codomain, ctx, graph, result_term)` gives `CompiledProgram.run()` one output path for both backend and non-backend programs

### structure/terms.py additions
- `product_arg(x: TTerm, n: int) → list[TTerm]` — destructures a left-nested Hydra pair into n components; lives here alongside other pair utilities (`pair_first`, `pair_second`, `pair_swap`, `pairs_bimap`)

## Hydra API survey (2026-05-13)

Surveyed with `pkgutils.walk_packages` + `importlib`/`inspect`. Key findings relevant to tensor contraction:

- **No built-in einsum/tensor operations** in Hydra's standard library (`hydra.lib.math` has scalar math only)
- `hydra.reduction.contract_term` — beta-reduction cleanup (NOT tensor contraction; name is misleading)
- `hydra.rewriting.rewrite_term` — bottom-up term rewriter; available and correct for contraction fusion
- `hydra.differentiation.differentiate_term` — symbolic differentiation; knows about binary op names via `primitive_derivative`; works at Hydra term level
- `hydra.parsers` — parser combinator library (for Hydra-internal parsing, not for user-facing einsum syntax)
- `hydra.ast` — code-generation pretty-printing AST; not relevant

Constraint derived: **Do not create new `MorphismExpr` subclasses for tensor operations without clear justification** that Hydra cannot represent the same concept natively. The existing `Prim` node (raw Hydra term escape) plus correctly named `BackendPrimitive` registration is likely sufficient.

## What has been verified
- `import unialg` clean
- `SpaceT` fully eliminated; all types are native `hydra.core.Type` variants
- `ProductType`/`SumType` constructors centralized in `objects.py`
- `Morphism.to_lax(monad)` — correct coercion; MonadicEmbed wraps node cleanly
- `_resolve_monad` — correctly derives monad from set of morphisms
- `compose` and `case` auto-embed plain into lax context
- `embed` removed; no remaining references in codebase
- `compose` type check correct for plain, lax, and para modes
- `dom_of`/`cod_of` return correct `Type` instances for all MorphismExpr cases
- `apply_poly` returns correct `Type` for all PolyExpr cases
- `hydra.lib.lists.bind` and `hydra.lib.lists.pure` confirmed to exist ✓
- `realize` produces valid Hydra terms (all MorphismExpr cases including new Compose/Parallel/Pair/Case subclasses)
- `_poly_action_term` covers all plain and monadic PolyExpr cases; `One()` ≠ `Const(B)` distinction verified; monadic traversal raises for `Exp` as expected
- Plain product functor mapping targets Hydra `pairs.bimap`; plain sum mapping targets Hydra `eithers.bimap`
- `_ctx_preamble` correctly closes over `param_term` in `call_f`/`call_g`; Hydra built-ins (P.compose, bimap) confirmed insufficient for para/lax
- `MorphismError.check` replaces all `CompositionError`/`CaseError` call sites

## Current status and remaining work
- **Tests updated** — live suite uses pytest + Hypothesis law tests; stale legacy tests remain quarantined under `tests/regression/stale_old_api/`; currently 240 passing, 6 skipped
- **Optics layer complete for current semantics** — unified `Optic(functor, forward, backward)` subsumes Lens, Prism, and Traversal; height-2 optics supported via deeper polynomial bodies
- **Recursion layer complete for current semantics** — `act`, `act_forward`, `act_backward`, `compose_optic`, `list_carrier`, `cata`, `ana`, `hylo`; plain/para/lax/lax-para algebras and coalgebras are represented as `Morphism` values
- Semiring tensor equations beyond the current tensor helper notes
- Surface syntax coverage beyond the current parser
- Algebra homomorphisms as first-class typed objects (`AlgebraHom(f, src, tgt)`)
- Named algebra/coherence objects above the current `Morphism` semantics
- Backend expansion beyond current Hydra primitives
