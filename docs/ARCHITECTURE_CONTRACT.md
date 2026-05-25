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
runtime/                                 ← native backend boundary and Python codecs
        ↓
main.py                                  ← orchestration and execution boundary
        ↓
Hydra                                    ← external graph reduction engine
```

Dependency policy is ownership-based, not a simple "downward only" import rule.
`objects.py` is foundational. `syntax/` owns surface expression data.
`semantics/` may depend on `objects.py` and syntax, but not on `structure/`,
`runtime/`, or `main.py`. `structure/` may depend on syntax and semantics because
it realizes validated semantic nodes, but it must not depend on `runtime/` or
`main.py`. `runtime/` owns native execution boundaries and must not depend on
syntax, semantics, structure, or orchestration.

## Layer responsibilities

### `objects.py` — type-level ground
- Thin constructors for object shapes: `ProductType(l, r)` → `TypePair`, `SumType(l, r)` → `TypeEither`, `ListType(a)` → `TypeList`, `MaybeType(a)` → `TypeMaybe`
- `Monad(type_ctor, bind_name, pure_name)` descriptors: `MAYBE`, `LIST`
- No Hydra term construction; no imports from other unialg modules

### `syntax/expressions.py` — expression syntax
- `MorphismExpr` sealed ADT: Identity, Copy, Delete, Literal, First, Second, Left, Right, Absurd, Assoc, Symmetry, DistributeLeft, DistributeRight, MonadicEmbed, ContextualBinary (Compose, SharedCompose, Parallel, Pair, Case), PolyFmap, SelfRef, AlgExpr, Cata, Ana, Prim, DomainPrim, BackendPrim
- `PolyExpr` sealed ADT: Zero, One, Id, Const, Sum, Prod, Exp(base: PolyExpr, body), PolyCompose, List, Maybe
- `pretty` display via singledispatch
- Frozen dataclasses; no Hydra term imports

### `semantics/morphisms.py` — algebraic construction, morphism level
- `Morphism(node, param, monad, aux_primitives)` — typed handle with plain/para/lax/lax-para modes
- `dom_of` / `cod_of` / `signature` — type derivation from `MorphismExpr`
- `MorphismError(TypeError)` with `check(a, b, msg)` — single error class
- Combinators: `compose`, `par`, `pair`, `case` — via `_contextual_binary`; `distribute_left(a,b,c)`, `distribute_right(a,b,c)` — distributivity isos; `merge(a)` — codiagonal `A+A→A`
- Point constructor: `lit(value, A) : Unit -> A`; contextual use is assembled as `delete(X) >> lit(value, A) : X -> A`
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
- Scalar point builders: `scalar_literal`, `literal_point` for integer, float, boolean, and string Hydra values
- Projection/injection terms: `pair_first`, `pair_second`, `pair_swap`, `either_swap`, `pairs_bimap`, `eithers_bimap`, `eithers_either`, `left_injection`, `right_injection`, `absurd`
- List/maybe terms: `lists_cons`, `lists_empty`, `lists_foldr`, `lists_map`, `lists_uncons`, `maybes_maybe`, `maybes_nothing`, `maybes_just`
- `optimize_term` — peephole simplifier

### `structure/realize.py` — DSL-to-Hydra translation
- `realize(node: MorphismExpr, _prims: list | None = None)` → raw Hydra term
- `realize_term(node, _prims=None)` → typed `TTerm` handle
- `poly_action_term(body, h, monad)` — functor action realization
- Handles the deferred-node pattern: `SelfRef` resolves to a named primitive; `Cata`/`Ana` create and register a `Primitive` in `_prims`
- Realizes `Literal` only after semantic typing has resolved its codomain and native scalar payload
- Does not evaluate terms — that is `main.py`'s job

### `structure/recursion.py` — fixed-point optic bridge
- `recursive_carrier(*, functor, carrier, unroll, roll)` — wraps Python callables as `Prim` morphisms and returns a carrier `Optic`
- `fixed_point_optic` — lower-level shim for direct raw-term injection; retained for performance review

### `extensions.py` — domain extension registry and finalize contract
- `DomainProtocol(construct, construct_expr, refs, finalize)` — registration interface for domain modules
- `register_keyword(keyword, handler)`, `register_expr_form(name, handler)`, `register_domain(tag, protocol)` — self-registration at import time
- **Finalize contract:** `DomainProtocol.finalize` is a whole-morphism rewrite hook, not limited to domain-specific elaboration. It receives a fully constructed `Morphism` and an env dict; it returns a rewritten `Morphism`. It runs after all morphisms in a program are resolved and type-checked, before realization. It sees the complete morphism tree. Every registered domain's finalize hook is called on every morphism in the program, in registration order.
- Finalize is called by `finalize_domain_morphisms()` in `semantics/_construct_helpers.py`
- Current user: tensor fusion (`normalize_contracts` in `tensors/fusion.py`) — fuses adjacent contractions then decomposes all `DomainPrim` nodes into substrate morphisms
- No Hydra term construction; no imports from `structure/`, `runtime/`, or `main.py`

### `tensors/` — tensor extension (semiring-based contractions)
- `tensors/notation.py` — surface notation: `Equation`, `AlignmentPlan`, `SemiringDecl`, `ContractExpr`
- `tensors/semirings.py` — `Semiring` dataclass (carrier, plus/times/zero/one, adjoint, reduce fields); `op_env()` to select product/fold/seed ops
- `tensors/semantics.py` — `ContractSpec`, `resolve_semiring(decl, op_morphisms) → Semiring`, `contract_morphism(...)` returning a lazy `DomainPrim`, `_strip_exp` to remove `ExpType` wrappers
- `tensors/primitives.py` — `compile_contract_spec` lowers a `ContractSpec` into a substrate `Morphism` tree
- `tensors/fusion.py` — `normalize_contracts`, `_par_to_optic`, shape-based fusion
- Tensor extension registers itself via `tensors/__init__.py` finalize hook; `DomainPrim` nodes emitted here are rewritten before `realize`
- Import rule: `tensors/` may depend on `semantics/` and `objects.py`; must not depend on `structure/`, `runtime/`, or `main.py`

### `runtime/` — native boundary and backend codecs
- `runtime/backend.py` loads backend specs and registers Hydra primitives for native operations.
- `runtime/codecs.py` owns type-directed Hydra term codecs; `runtime/boundary.py` owns Python value ↔ Hydra term boundary behavior and native handle storage.
- Runtime modules are execution-boundary support only; they do not construct semantic morphism trees.

### `main.py` — orchestration and execution boundary
- `compile_program(src, ...)` parses source, constructs the semantic expression tree, realizes the final route/map, and packages it as a `CompiledProgram`.
- `compile_morphism(m, ...)` realizes an already-built `Morphism` and records auxiliary primitives.
- `lower(m, graph, _extra_prims=None)` extracts a Hydra term without evaluating it.
- `run(m, arg, ctx, graph)` realizes, augments the graph, applies one raw Hydra argument, and reduces.

### `__init__.py` — public surface
- Aggregates exports; no logic

## Invariants

1. `compose(f, g)` is defined iff `f.cod() == g.dom()`; mismatch raises `MorphismError` at construction time.
2. `Literal(text, value, cod)` denotes `Unit -> cod`; syntax holds text, semantics resolves typed values, and only `structure/` builds Hydra literal terms.
3. `identity(A).dom() == identity(A).cod() == A`
4. `fst(AB).dom() == AB`, `fst(AB).cod() == AB.value.first`
5. `snd(AB).dom() == AB`, `snd(AB).cod() == AB.value.second`
6. `par(f, g).dom() == ProductType(f.dom(), g.dom())`, `par(f, g).cod() == ProductType(f.cod(), g.cod())`
7. Types are native `hydra.core.Type` variants; equality is Hydra type equality.
8. Import direction follows ownership: `semantics/` does not import `structure/`, `runtime/`, or `main.py`; `structure/` may import syntax/semantics for realization but does not import `runtime/` or `main.py`; `runtime/` does not import syntax/semantics/structure/main.
9. `Optic.__post_init__` validates that `functor.unapply()` succeeds on both `forward.cod()` and `backward.dom()`.
10. `poly_fmap(F, h).dom() == F.apply(h.dom())`, `poly_fmap(F, h).cod() == F.apply(h.cod())`
11. For recursive optics, `fp.carrier` is present and marks `μF`.
12. `fp.forward.dom() == fp.carrier`, `fp.forward.cod() == fp.functor.apply(fp.carrier)`
13. `fp.backward.dom() == fp.functor.apply(fp.carrier)`, `fp.backward.cod() == fp.carrier`
14. `cata(fp, alg)` and `ana(fp, coalg)` preserve the `param` and `monad` of their algebra/coalgebra.
15. `hylo(fp, coalg, alg)` uses `compose(..., shared_context=True)`, so matching non-unit params are shared, not duplicated.
16. `Cata`/`Ana` nodes are created by `semantics/optics.py` with no Hydra imports; `structure/realize.py` materializes them.
17. The `_prims` list accumulator in `realize` is the mechanism for registering recursion-created `Primitive` objects; `main.run()` and `CompiledProgram.run()` augment the graph with collected primitives before reduction.
18. `DomainProtocol.finalize` is a whole-morphism rewrite hook. It runs after all semantic construction is complete, before realization. Every registered domain's finalize hook is applied to every morphism in registration order. Finalize may restructure the morphism tree but must preserve dom/cod and param/monad invariants.
19. Combinator laws (identity, associativity, product/coproduct universal properties, bifunctor, functorial action) hold at every layer that implements them. See `docs/COMBINATOR_LAWS.md`.

## Do not redo
- Do not redo Hydra API exploration (already settled)
- Do not redo the typed morphism design (stable)
- Do not redo the product boundary design (fst/snd/par is the settled interface)
- Do not redo the unified optic design (Optic(functor, forward, backward) is sealed)
- Do not redo the deferred-node pattern (SelfRef/AlgExpr/Cata/Ana nodes are the settled separation between semantics and structure)
- Do not move cata/ana/hylo back to structure/; they are pure semantic functions that return deferred nodes
- Do not reintroduce SpaceT or parallel type hierarchies (eliminated)
- Do not reintroduce `space.py`, `hydra_primitives.py`, or `actions.py`; their responsibilities now live in `objects.py`, `structure/terms.py`, `semantics/functors.py`, `semantics/optics.py`, and `structure/realize.py`.
- Do not expose copy, delete, or fanout yet (deliberate boundary)
