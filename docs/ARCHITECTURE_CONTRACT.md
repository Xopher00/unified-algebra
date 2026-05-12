# Architecture Contract

## Layer model

```
objects.py                               ← type constructors, monad descriptors
syntax/expressions.py                    ← expression syntax ADTs
        ↓
semantics/morphisms.py                   ← typed morphism handles and combinators
semantics/functors.py                    ← polynomial functor object action
semantics/optics.py                      ← optics and recursion schemes (cata/ana/hylo)
        ↓
structure/terms.py                       ← Hydra term vocabulary
structure/realize.py                     ← DSL-to-Hydra translation
structure/recursion.py                   ← fixed-point optic bridge
        ↓
lowering.py                              ← execution boundary (morphism → reduction)
        ↓
Hydra                                    ← external graph reduction engine
```

Dependency flows downward only. No `semantics/` module imports from `structure/`.
No `structure/` module imports from `semantics/`. `objects.py` and `syntax/` are
imported by all layers.

## Layer responsibilities

### `objects.py` — type-level ground
- Thin constructors for object shapes: `ProductType(l, r)` → `TypePair`, `SumType(l, r)` → `TypeEither`, `ListType(a)` → `TypeList`, `MaybeType(a)` → `TypeMaybe`
- `Monad(type_ctor, bind_name, pure_name)` descriptors: `MAYBE`, `LIST`
- No Hydra term construction; no imports from other unialg modules

### `syntax/expressions.py` — expression syntax
- `MorphismExpr` sealed ADT: Identity, Copy, Delete, First, Second, Left, Right, Absurd, Assoc, MonadicEmbed, ContextualBinary (Compose, Parallel, Pair, Case), PolyFmap, SelfRef, AlgExpr, Cata, Ana, Prim
- `PolyExpr` sealed ADT: Zero, One, Id, Const, Sum, Prod, Exp, List, Maybe
- `pretty` display via singledispatch
- Frozen dataclasses; no Hydra term imports

### `semantics/morphisms.py` — algebraic construction, morphism level
- `Morphism(node, param, monad, aux_primitives)` — typed handle with plain/para/lax/lax-para modes
- `dom_of` / `cod_of` / `signature` — type derivation from `MorphismExpr`
- `MorphismError(TypeError)` with `check(a, b, msg)` — single error class
- Combinators: `compose`, `par`, `pair`, `case` — via `_contextual_binary`
- `shared_context=True` on contextual combinators — shares a matching non-unit param; recursion uses this to avoid `P × P` contexts
- `_resolve_monad` — derives target monad; errors on conflict
- `Morphism.to_lax(monad)` — universal coercion into lax context
- Imports from `objects.py` and `syntax/expressions.py` only; no Hydra terms

### `semantics/functors.py` — polynomial functor semantics
- `Functor(name, body: PolyExpr)` — named polynomial endofunctor descriptor
- `Functor.apply(space)` — object action F(A)
- `Functor.unapply(fa)` — inverse object action via Hydra type unification
- `Functor.compose(inner)` — functor composition F∘G
- `Functor.map(h)` — semantic arrow action, returns `poly_fmap(self, h)`
- `poly_fmap(functor, h)` — builds a `Morphism` wrapping `PolyFmap`; no Hydra term construction
- `apply_poly`, `compose_poly` — recursive object-action implementations
- Introspection: `summands()`, `x_arity()`, `consts()`
- Uses Hydra type unification only; no Hydra term construction

### `semantics/optics.py` — unified polynomial functor optics and recursion
- `Optic(functor, forward, backward, carrier=None)` — single class for Lens, Prism, Traversal
- `forward: S → F(A)`, `backward: F(B) → T`; focus/replacement derived via strict `functor.unapply()`
- `act(h)`, `act_forward(h)`, `act_backward(h)` — optic actions on morphisms
- `Optic.compose(inner)` — optic composition via functor composition
- `cata(fp, alg)` — returns `Morphism(Cata(...))` deferred node; no Hydra construction
- `ana(fp, coalg)` — returns `Morphism(Ana(...))` deferred node; no Hydra construction
- `hylo(fp, coalg, alg)` — `compose(ana(...), cata(...), shared_context=True)`
- `identity_optic(name, functor, focus)` — optic where S = T = F(focus)
- No Hydra term construction

### `structure/terms.py` — Hydra term vocabulary
- Single source of truth for how to build a Hydra term for a semantic concept
- Lambda and application builders: `prim2`, `term_lambda`, `lam2`
- Monad-polymorphic term builders: `bind`, `pure`, `apply_effect`, `map_effect`, `lift2_effect`
- Shape-specific builders: `pure_unit`, `pure_identity`, `product_action`, `pair_effects`, `case_effects`, `list_effects`, `maybe_effects`
- Projection/injection terms: `pair_first`, `pair_second`, `pair_swap`, `either_swap`, `pairs_bimap`, `eithers_bimap`, `eithers_either`, `left_injection`, `right_injection`, `absurd`
- List/maybe terms: `lists_cons`, `lists_empty`, `lists_foldr`, `lists_map`, `lists_uncons`, `maybes_maybe`, `maybes_nothing`, `maybes_just`
- `optimize_term` — peephole simplifier

### `structure/realize.py` — DSL-to-Hydra translation
- `realize(node: MorphismExpr, _prims: list | None = None)` → raw Hydra term
- `realize_term(node, _prims=None)` → typed `TTerm` handle
- `poly_action_term(body, h, monad)` — functor action realization
- Handles the deferred-node pattern: `SelfRef` resolves to a named primitive; `Cata`/`Ana` create and register a `Primitive` in `_prims`
- Does not evaluate terms — that is `lowering.py`'s job

### `structure/recursion.py` — fixed-point optic bridge
- `recursive_carrier(*, functor, carrier, unroll, roll)` — wraps Python callables as `Prim` morphisms and returns a carrier `Optic`
- `fixed_point_optic` — lower-level shim for direct raw-term injection; retained for performance review

### `lowering.py` — execution boundary
- `lower(m, extra_prims=None)` → `realize(m.node, extra_prims)` (pure extraction)
- `run(m, arg, ctx, graph)` — realize, collect `_prims`, augment graph, apply and reduce

### `__init__.py` — public surface
- Aggregates exports; no logic

## Invariants

1. `compose(f, g)` is defined iff `f.cod() == g.dom()`; mismatch raises `MorphismError` at construction time.
2. `identity(A).dom() == identity(A).cod() == A`
3. `fst(AB).dom() == AB`, `fst(AB).cod() == AB.value.first`
4. `snd(AB).dom() == AB`, `snd(AB).cod() == AB.value.second`
5. `par(f, g).dom() == ProductType(f.dom(), g.dom())`, `par(f, g).cod() == ProductType(f.cod(), g.cod())`
6. Types are native `hydra.core.Type` variants; equality is Hydra type equality.
7. Dependency direction flows downward through the layer model; never upward. No `semantics/` module imports from `structure/`.
8. `Optic.__post_init__` validates that `functor.unapply()` succeeds on both `forward.cod()` and `backward.dom()`.
9. `poly_fmap(F, h).dom() == F.apply(h.dom())`, `poly_fmap(F, h).cod() == F.apply(h.cod())`
10. For recursive optics, `fp.carrier` is present and marks `μF`.
11. `fp.forward.dom() == fp.carrier`, `fp.forward.cod() == fp.functor.apply(fp.carrier)`
12. `fp.backward.dom() == fp.functor.apply(fp.carrier)`, `fp.backward.cod() == fp.carrier`
13. `cata(fp, alg)` and `ana(fp, coalg)` preserve the `param` and `monad` of their algebra/coalgebra.
14. `hylo(fp, coalg, alg)` uses `compose(..., shared_context=True)`, so matching non-unit params are shared, not duplicated.
15. `Cata`/`Ana` nodes are created by `semantics/optics.py` with no Hydra imports; `structure/realize.py` materializes them.
16. The `_prims` list accumulator in `realize` is the only mechanism for registering recursion-created `Primitive` objects; `lowering.run()` always calls `lower(m, extra_prims=[])` and registers collected primitives before reduction.

## Do not redo
- Do not redo Hydra API exploration (already settled)
- Do not redo the typed morphism design (stable)
- Do not redo the product boundary design (fst/snd/par is the settled interface)
- Do not redo the unified optic design (Optic(functor, forward, backward) is sealed)
- Do not redo the deferred-node pattern (SelfRef/AlgExpr/Cata/Ana nodes are the settled separation between semantics and structure)
- Do not move cata/ana/hylo back to structure/; they are pure semantic functions that return deferred nodes
- Do not reintroduce SpaceT or parallel type hierarchies (eliminated)
- Do not reintroduce space.py, hydra_primitives.py, or actions.py (renamed to objects.py, structure/terms.py, and structure/realize.py respectively)
- Do not introduce tensor equations yet (not in scope)
- Do not introduce surface syntax yet (deferred)
- Do not expose copy, delete, or fanout yet (deliberate boundary)
