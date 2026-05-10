# Checkpoint — Semantic Contract 8 (sealed)

## Date
2026-05-09 (updated after unified polynomial optic generalization)

## Sealed contract

The following constitute the sealed semantic contract. Do not revisit or redesign these.

- **No parallel type hierarchy** — `SpaceT` eliminated; types are native `hydra.core.Type` variants directly
- Type vocabulary: `TypeVariable`, `TypePair(PairType(l,r))`, `TypeEither(EitherType(l,r))`, `TypeUnit()`, `TypeVoid()`, `TypeFunction(FunctionType(d,c))`, `TypeList(b)`, `TypeMaybe(b)`
- Convenience constructors in `space.py`: `ProductType(l, r)` → `TypePair(PairType(l,r))`, `SumType(l, r)` → `TypeEither(EitherType(l,r))`
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
- `actions._poly_action_term(body, h, monad=None)` → raw TTerm-level polynomial action; handles plain map (monad=None) and monadic traversal (monad set) in one function
- `actions.poly_fmap(functor: Functor, h: Morphism)` — lifts Morphism through polynomial functor; uniform across plain/para/lax
- Contextual combinators accept `shared_context=True` to share matching non-unit params; used by recursion schemes
- Recursion optic actions: `act`, `act_forward`, `act_backward`, `compose_optic`, `list_carrier`, `cata`, `ana`, `hylo`
- `lower(m)` — pure extraction: Morphism → Hydra term
- `run(m, arg, ctx, graph)` — apply and reduce; augments graph with aux_primitives
- `Monad(type_ctor, bind_name, pure_name)`: `MAYBE`, `LIST`

## Architectural note — DSL layers clarified by `explore.ipynb`

The reader-first explorer clarified the intended conceptual stack:

1. **Morphism layer** — typed arrows, products, sums, params, effects, functor lifting
2. **Optics layer** — lenses, prisms, traversals, and height-2/polynomial lenses built from morphisms
3. **Recursion layer** — fixed points, folds, unfolds, and recursive carriers
4. **Lowering layer** — Hydra term generation and reduction boundary

Do not conflate these layers:

- `PolyExpr` / `poly_fmap` describes and maps over one layer of a shape.
- `recursion.py` is for repeated roll/unroll over recursive fixed points.
- Lenses and optics are focus/update abstractions built above morphisms; they do **not** belong inside `recursion.py`.
- Height-2 or polynomial lenses belong near the functor/shape layer, depending on `PolyExpr`, `apply_poly`, `poly_fmap`, and `Morphism`.

## Package layout (flat, as of 2026-05-08)

```
src/unialg/
├── space.py        Space hierarchy + Monad descriptor — pure Python, no Hydra
├── expressions.py  MorphismExpr + PolyExpr ADTs — pure Python, no Hydra
├── morphisms.py   dom_of/cod_of + combinators (compose/par/pair/case/embed) — Hydra-free
├── functors.py     Functor + PolyExpr object/introspection helpers — Hydra type unification, no terms
├── hydra_primitives.py  Hydra primitive catalog and thin wrappers
├── realize.py       realize(MorphismExpr) — raw Hydra term construction
├── actions.py     poly_fmap — derived compiler actions
├── optics.py       Unified polynomial functor Optic; no Hydra term construction
├── recursion.py    act/act_forward/act_backward/compose_optic/list_carrier/cata/ana/hylo — optic actions + recursion schemes
├── lowering.py     lower/run — execution boundary
└── __init__.py     public surface
```

### Layer responsibilities

| File | Layer | Hydra? |
|------|-------|--------|
| space.py | typed interpretation | no |
| expressions.py | typed interpretation | no |
| morphisms.py | algebraic construction | no |
| functors.py | polynomial functor semantics | type unification only |
| hydra_primitives.py | backend primitive catalog | yes |
| realize.py | backend realization | yes |
| actions.py | executable actions | yes |
| optics.py | Unified polynomial functor optic | no term construction |
| recursion.py | algebraic construction + actions | yes (Primitive) |
| lowering.py | backend realization | yes |

### expressions.py
- `MorphismExpr` sealed hierarchy (leaves + binary combinators + Prim)
- `PolyExpr` sealed hierarchy
- Both are frozen dataclasses, no Hydra imports

### space.py
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
- Uses `hydra_primitives.py` for backend primitive names/wrappers; no scattered `Name("hydra.lib...")` strings

### actions.py
- `_poly_action_term(body, h, monad=None)` — unified polynomial functor action; plain map when monad=None (uses `pairs.bimap`, `eithers.bimap`), monadic traversal when monad set (uses `bind`/`pure` + sum/product reconstruction)
- `poly_fmap(functor, h: Morphism) -> Morphism` — uniform across plain/para/lax via `_action_section` + `_poly_action_term`

### recursion.py
- `act_forward(t, h)` — `compose(t.forward, poly_fmap(t.functor, h))`
- `act_backward(t, h)` — `compose(poly_fmap(t.functor, h), t.backward)`
- `act(t, h)` — `compose(act_forward(t, h), t.backward)` — full optic action
- `compose_optic(outer, inner)` — composes two optics via polynomial functor composition
- `list_carrier(element)` — convenience carrier optic for Hydra lists as `μX. 1 + (A × X)`
- `cata(fp, alg)` — catamorphism over `Optic(..., carrier=μF)`; supports plain, para, lax, and lax-para algebras
- `ana(fp, coalg)` — anamorphism over `Optic(..., carrier=μF)`; supports plain, para, lax, and lax-para coalgebras
- `hylo(fp, coalg, alg)` — composes `ana` and `cata` with shared parameter context
- Carrier-specific roll/unroll helpers are optional adapters that produce `forward : μF → F(μF)` and `backward : F(μF) → μF`; they are not required core semantics
- Old API (`rec`, `cata`, `ana`, `Inductive`, `LIST_IND`, `MAYBE_IND`, `AlgebraError`) in `recursion.py.bak` — do not reference

### optics.py
- Single `Optic(functor: Functor, forward: Morphism, backward: Morphism)` — unified polynomial functor optic
- `forward: S → F(A)` decomposes source into F-shaped container; `backward: F(B) → T` reconstructs target
- Uniform action for all optics: `compose(forward, poly_fmap(F, h), backward)` — implemented as `act()` in `actions.py`
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

### lowering.py
- `lower(m)` → `realize(m.node)`
- `run(m, arg, ctx, graph)` → augments graph with `m.aux_primitives`, reduces

### Public surface (`__init__.py`)
- Spaces, Morphism, combinators, PolyExpr types, Functor, recursion exports, lower, run

## What has been verified
- `import unialg` clean
- `SpaceT` fully eliminated; all types are native `hydra.core.Type` variants
- `ProductType`/`SumType` constructors centralized in `space.py`
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

## What is not yet implemented
- **Tests updated** — live suite uses pytest + Hypothesis law tests; stale legacy tests remain quarantined under `tests/regression/stale_old_api/`; 103 tests passing
- **Optics layer complete** — unified `Optic(functor, forward, backward)` subsumes Lens, Prism, and Traversal; height-2 optics supported via deeper polynomial bodies
- **Recursion layer complete for current semantics** — `act`, `act_forward`, `act_backward`, `compose_optic`, `list_carrier`, `cata`, `ana`, `hylo`; plain/para/lax/lax-para algebras and coalgebras are represented as `Morphism` values
- Semiring tensor equations
- Surface syntax / grammar
- Algebra homomorphisms as first-class typed objects (`AlgebraHom(f, src, tgt)`)
- Named algebra/coherence objects above the current `Morphism` semantics
- Backend expansion beyond current Hydra primitives
