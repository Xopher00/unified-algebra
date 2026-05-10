# Architecture Contract

## Layer model

```
surface expression        (not yet built — syntax/grammar deferred)
→ typed interpretation    (not yet built — type resolution deferred)
→ algebraic construction  ← space.py, expressions.py, morphisms.py, functors.py, optics.py, recursion.py
→ executable assembly     ← realize.py, actions.py
→ backend realization     ← hydra_primitives.py, lowering.py, Hydra (external)
```

## Layer responsibilities

### `space.py` — typed interpretation, object level
- Thin constructors for object shapes: `ProductType(l, r)` → `TypePair`, `SumType(l, r)` → `TypeEither`
- `Monad(type_ctor, bind_name, pure_name)` descriptors: `MAYBE`, `LIST`
- No Hydra term construction; no imports from other unialg modules

### `expressions.py` — typed interpretation, syntax
- `MorphismExpr` sealed ADT: Identity, Copy, Delete, First, Second, Left, Right, Absurd, Assoc, MonadicEmbed, ContextualBinary (Compose, Parallel, Pair, Case), Prim
- `PolyExpr` sealed ADT: Zero, One, Id, Const, Sum, Prod, Exp
- `pretty` display via singledispatch
- Frozen dataclasses, no Hydra imports beyond core types

### `morphisms.py` — algebraic construction, morphism level
- `Morphism(node, param, monad, aux_primitives)` — typed handle with plain/para/lax/lax-para modes
- `dom_of` / `cod_of` / `signature` — type derivation from `MorphismExpr`
- `MorphismError(TypeError)` with `check(a, b, msg)` — single error class
- Combinators: `compose`, `par`, `pair`, `case` — via `_contextual_binary`
- `shared_context=True` on contextual combinators — shares a matching non-unit param; recursion uses this to avoid `P × P` contexts
- `_resolve_monad` — derives target monad; errors on conflict
- `Morphism.to_lax(monad)` — universal coercion into lax context
- Imports from `space.py` and `expressions.py` only; no Hydra terms

### `functors.py` — polynomial functor semantics
- `Functor(name, body: PolyExpr)` — named polynomial endofunctor descriptor
- `Functor.apply(space)` — object action F(A)
- `Functor.unapply(fa)` — inverse object action via Hydra type unification
- `Functor.compose(inner)` — functor composition
- `apply_poly`, `compose_poly` — internal recursive implementations
- Introspection: `summands()`, `x_arity()`, `consts()`
- Uses Hydra type unification only; no Hydra terms

### `optics.py` — unified polynomial functor optics
- `Optic(functor, forward, backward)` — single class for Lens, Prism, Traversal
- `forward: S → F(A)`, `backward: F(B) → T`
- Focus/replacement derived via strict `functor.unapply()`
- Convenience constructors: `fst_lens`, `snd_lens`, `left_prism`, `right_prism`
- No Hydra term construction; action lives in `actions.py`

### `hydra_primitives.py` — backend primitive catalog
- Single source of truth for Hydra primitive names and wrappers
- `bind`, `pure`, `pairs_bimap`, `eithers_bimap`, `left_injection`, `right_injection`
- No unialg imports

### `realize.py` — backend realization, term construction
- `realize(node: MorphismExpr)` → raw Hydra term
- `_ctx_preamble` — shared argument plumbing for contextual nodes
- Shared term-level helpers: `_t`, `_split_input`, `_pair_effects`
- Does not evaluate terms — that is `lowering.py`'s job

### `actions.py` — executable assembly, derived actions
- `poly_fmap(functor, h)` — polynomial functor arrow action
- `_poly_action_term(body, h, monad=None)` — plain and monadic functor action in one descent
- `primitive_from_raw(raw, raw_dom, raw_cod, template, aux_primitives=())` — wraps a raw Hydra term as a typed `Morphism`

### `recursion.py` — algebraic construction + recursive primitives
- `act_forward(t, h)`, `act_backward(t, h)`, `act(t, h)` — optic actions
- `compose_optic(outer, inner)` — optic composition via functor composition
- `list_carrier(element)` — convenience carrier optic for Hydra lists as `μX. 1 + (A × X)`
- `cata(fp, alg)` — catamorphism over `Optic(..., carrier=μF)`
- `ana(fp, coalg)` — anamorphism over `Optic(..., carrier=μF)`
- `hylo(fp, coalg, alg)` — hylomorphism with shared parameter context
- Recursive self-references inherit the algebra/coalgebra `param` and `monad`
- Carrier-specific roll/unroll helpers are adapters that provide `forward : μF → F(μF)` and `backward : F(μF) → μF`; not required core semantics

### `lowering.py` — backend realization, execution boundary
- `lower(m)` → `realize(m.node)` (pure extraction)
- `run(m, arg, ctx, graph)` — apply and reduce with aux primitive registration

### `__init__.py` — public surface
- Aggregates exports; no logic

## Invariants

1. `compose(f, g)` is defined iff `f.cod() == g.dom()`; mismatch raises `MorphismError` at construction time.
2. `identity(A).dom() == identity(A).cod() == A`
3. `fst(AB).dom() == AB`, `fst(AB).cod() == AB.value.first`
4. `snd(AB).dom() == AB`, `snd(AB).cod() == AB.value.second`
5. `par(f, g).dom() == ProductType(f.dom(), g.dom())`, `par(f, g).cod() == ProductType(f.cod(), g.cod())`
6. Types are native `hydra.core.Type` variants; equality is Hydra type equality.
7. Dependency direction flows downward through the layer model; never upward.
8. `Optic.__post_init__` validates that `functor.unapply()` succeeds on both `forward.cod()` and `backward.dom()`.
9. `poly_fmap(F, h).dom() == F.apply(h.dom())`, `poly_fmap(F, h).cod() == F.apply(h.cod())`
10. For recursive optics, `fp.carrier` is present and marks `μF`.
11. `fp.forward.dom() == fp.carrier`, `fp.forward.cod() == fp.functor.apply(fp.carrier)`
12. `fp.backward.dom() == fp.functor.apply(fp.carrier)`, `fp.backward.cod() == fp.carrier`
13. `cata(fp, alg)` and `ana(fp, coalg)` preserve the `param` and `monad` of their algebra/coalgebra.
14. `hylo(fp, coalg, alg)` uses `compose(..., shared_context=True)`, so matching non-unit params are shared, not duplicated.

## Do not redo
- Do not redo Hydra API exploration (already settled)
- Do not redo the typed morphism design (stable)
- Do not redo the product boundary design (fst/snd/par is the settled interface)
- Do not redo the unified optic design (Optic(functor, forward, backward) is sealed)
- Do not reintroduce SpaceT or parallel type hierarchies (eliminated)
- Do not introduce tensor equations yet (not in scope)
- Do not introduce surface syntax yet (deferred)
- Do not expose copy, delete, or fanout yet (deliberate boundary)
