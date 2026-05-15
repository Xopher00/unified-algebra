# Design Decisions

## Semantic Contract 2 is sealed (2026-05-05)
The following are frozen design decisions. Do not reopen them.
Typed objects · typed morphisms · identity · composition · product boundaries ·
projections · parallel composition.

## Composition order is diagrammatic
`compose(f, g)` means "f first, then g" (diagrammatic / left-to-right).
This required calling `P.compose(g.term, f.term)` (reversed) at the Hydra level,
because Hydra's `P.compose(a, b)` is right-to-left (a ∘ b means b first).
Do not change this inversion without updating the module docstring.

## Objects are native Hydra Type values
The project no longer maintains a parallel `SpaceT` hierarchy. Object-level
types are native `hydra.core.Type` variants directly: `TypePair`, `TypeEither`,
`TypeUnit`, `TypeVoid`, `TypeFunction`, `TypeList`, `TypeMaybe`, and
`TypeVariable`.

`ProductType(l, r)` and `SumType(l, r)` in `space.py` are thin constructors for
`TypePair(PairType(l, r))` and `TypeEither(EitherType(l, r))`. Equality is
Hydra type equality, so product and sum order remains structural and
order-sensitive.

## par is implemented as pair(f ∘ fst, g ∘ snd)
`par(f, g) : A × C → B × D` is not a primitive. It is derived from projections
and an internal pair construction. The general pairing/fanout/copy/delete API
from Hydra is not exposed publicly. This keeps the public surface aligned with
the categorical product structure without committing to a cartesian closed interface.

## Pairing, copy, delete, fanout are not public
Only `fst`, `snd`, and `par` are exported. Copy, delete, fanout, and the raw
`_pair_morphism_term` helper remain internal. This is a deliberate boundary:
the public API expresses product morphisms, not arbitrary cartesian combinators.

## Lowering and evaluation are separated
`lower(m)` extracts the term without evaluating it. `run(m, arg, ctx, graph)`
evaluates. These are separate functions to allow the algebraic layer to be
tested and reasoned about without a live Hydra evaluation context.

## Hydra composition and term API
The project uses `hydra.dsl.meta.phantoms` (`P`) for term construction and
`hydra.reduction.reduce_term` for evaluation. The composition inversion is
documented in `morphism.py`. This is a stable interface assumption.

## Coproducts are tracked by TypeEither

At the type boundary, coproducts are native `TypeEither(EitherType(A, B))`.
Morphism composition and optic/functor validation use that Hydra type structure
directly.

At the term boundary, realization currently uses Hydra's Either term
constructors and `hydra.lib.eithers` eliminators. The important contract is that
type identity is preserved by the `Morphism` dom/cod layer; term construction is
the backend realization of that already-checked boundary.

## Lax Para algebra is implemented via Semantic Contract 5 (2026-05-07)

### Monad is a three-field descriptor, not an abstract typeclass

`Monad(type_ctor, bind_name, pure_name)` carries only what is needed for type and
term construction: the Hydra type constructor plus the two Hydra primitive names
(`bind`, `pure`). `fmap`, `join`, and strength are not stored — they are either
derivable from bind+pure or built inline during lax composition. This keeps the
descriptor minimal and avoids encoding monad laws that the Python layer cannot
enforce.

### Tensorial strength is not an explicit morphism

Lax Para composition does not require a separate `strength : P × T(A) → T(P × A)`
morphism. The `bind` primitive plus a lambda that captures the `q` parameter handles
the threading naturally:
```
h(qp, a) = bind (f(p, a)) (λb. g(q, b))
```
where `q` is captured from the pair destructuring above. This eliminates an entire class
of morphisms that would otherwise need to be defined per-monad.

### All monad primitives are named Hydra terms

`hydra.lib.maybes.{bind,pure,map,compose}`, `hydra.lib.lists.{bind,pure,map,concat}`,
and `hydra.lib.eithers.{bind,map}` exist as resolvable named references in the Hydra
graph (`TermVariable` resolved at reduction time). No new primitives were added.

### Two concrete monads are sealed: MAYBE and LIST

`MAYBE = Monad("Maybe", Name("hydra.lib.maybes.bind"), Name("hydra.lib.maybes.pure"))`
`LIST  = Monad("List",  Name("hydra.lib.lists.bind"),  Name("hydra.lib.lists.pure"))`

Either monad (error propagation) is deferred — `pure` requires injecting `Right` which
needs a specific error space; the design is not yet decided.

### LaxParaMorphism.__post_init__ validates underlying boundary

The post-init check enforces `underlying.dom == ProductSpace(param, dom)` and
`underlying.cod == MonadSpace(monad.tag, cod)`. This mirrors `ParaMorphism`'s validation
and catches construction errors at definition time rather than evaluation time.

## Optics are unified as polynomial functor optics (2026-05-09)

All optics (Lens, Prism, Traversal, height-2) are a single `Optic(functor, forward, backward)` dataclass.
The action is always `compose(forward, poly_fmap(F, h), backward)` — no type dispatch, no subclasses.

- `forward: S → F(A)` decomposes source into F-shaped container
- `backward: F(B) → T` reconstructs target from F-shaped container
- Lens = `Prod(Id(), Const(residue))`, Prism = `Sum(Id(), Const(residue))`, Traversal = arbitrary polynomial F
- Focus and replacement derived via strict `functor.unapply()` (Hydra type unification plus validated inverse of `apply_poly`)
- For simple lenses/prisms where S = F(A), forward and backward are identity morphisms
- Height-2 requires no structural change — just deeper polynomial bodies in the functor

This was chosen over: (a) separate Lens/Prism/Traversal subclasses with per-type actions,
(b) two-functor optics with explicit bridge morphisms. The single-functor design gives
uniform action, prepares for algebra/coalgebra reuse (`F(A) → A` is an algebra,
`A → F(A)` is a coalgebra — the optic's forward/backward are already these shapes),
and eliminates type-variant pattern matching.

## Para uses ProductSpace as its underlying boundary
`Para(f)` where `f : P × A → B` is not a new input convention. `P × A` is
`ProductSpace(P, A)`, which already exists in the sealed contract. Para is
a semantic wrapper that separates the left component (parameter space) from
the right component (input space) at construction time. No new space types,
no changes to lowering, no changes to Hydra term construction are required.

## Type-directed codec architecture (2026-05-15)

The emitter codec layer is type-directed: the JSON backend spec declares `arg_type` and
`result_type` as Hydra type specs; `load_spec` derives `TermCoder`s automatically via
`type_from_spec` + `coder_for_type`. No named codec registry; no framework knowledge in
`codecs.py`.

**Architecture invariants:**
- `codecs.py` knows Hydra types and plain Python values (int, float, str, bool, bytes, list,
  tuple, None, Left/Right). It does NOT import numpy, torch, cupy, jax, or shape/dtype policy.
- `backend.py` loads specs, resolves import paths, registers Hydra primitives. No
  framework-specific dispatch table.
- Backend JSON `"path"` points to a callable that accepts/returns universal Python
  representation. For framework-strict backends, `"path"` points to a wrapper that converts
  on entry and exit; the codec layer is uninvolved.

**Supported type specs:** `"FLOAT"`, `"INT"`, `"STRING"`, `"BOOL"`, `"BINARY"`, `"UNIT"`,
and compound forms: `{"list": T}`, `{"pair": [A, B]}`, `{"either": [L, R]}`, `{"maybe": T}`.

**Tensor operations:** Array-typed ops (matmul, tensor contraction) must use distinct logical
op names and either (a) backend wrapper functions that accept/return nested Python lists, or
(b) a future tensor plugin with a typed `TENSOR` kind. They must not modify the `arg_type` of
existing scalar ops like `add` or `multiply`.

## Route parameters are morphism-valued lexical variables (2026-05-15)

`route f(x, y) = body` declares a parameterized morphism. `x` and `y` are morphism-valued
lexical variables — they can be composed, paired, or applied anywhere in the body.
`f(a, b)` in a route body explicitly instantiates it.

**Binding uses Hydra's native lambda calculus, not parse-time substitution.**
- Parser produces `MorphismApp(fun=<body with Refs>, args=(a, b), param_names=("x","y"))`.
- Realization maps `Ref(name)` → `TermVariable(Name(name))`, wraps parameterized bodies
  in `TermLambda`, and lowers `MorphismApp` to curried `TermApplication`.
- Hydra's `reduce_term` performs beta reduction.
- `run()` only handles data input. Never param binding.

**Parser invariants:**
- Param names shadow ALL builtins (`x`, `id`, `copy`, `fst`, etc.) — the lexical-params
  set is checked FIRST in the morphism grammar NAME handler, before builtins and env lookup.
- `f(a, b)` in body position produces a `MorphismApp` AST node. Parser preserves structure,
  does not substitute.
- Declaration syntax (`route f(x) = ...`) and call syntax (`f(arg)`) are distinguished
  by parser context — declaration in `parse_program`, call in morphism grammar `_nud`.

**Rejected alternative:** Parse-time Ref substitution (inline body rewriting). Tested and
proven redundant — Hydra's lambda/beta path produces identical results. Substitution was
removed to avoid a parallel binding mechanism.

**TypeVariable integration:**
- `signature(Ref("x"), frozenset({"x"}))` → `(TypeVariable(Name("x")), TypeVariable(Name("x")))`
- `free_variables_in_type(TypeVariable(Name("x")))` → `{Name("x")}` — trackable
- The existing `split_input`/`_mk_child_call` infrastructure activates when
  `ContextualBinary.param != TypeUnit()`. TypeVariable on the param field activates it.
  This is the path for the semantic layer's typed interpretation of parameterized routes.
