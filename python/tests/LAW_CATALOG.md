# Law Catalog

This file lists the semantic laws we want tests to cover for the current core.
It is a checklist for pytest examples and Hypothesis properties, not an
implementation spec.

## expressions.py

### ADT Structure

- Every concrete `MorphismExpr` node is a `MorphismExpr`.
- Every concrete `PolyExpr` node is a `PolyExpr`.
- Structural equality is class-sensitive:
  - `Zero() != One()`
  - `Sum(F, G) != Sum(G, F)` unless the children are equal by structure
  - `Prod(F, G) != Prod(G, F)` unless the children are equal by structure
- Base classes are not valid semantic expressions:
  - `pretty(MorphismExpr())` raises `ValueError`
  - `pretty(PolyExpr())` raises `ValueError`

### Pretty Laws

- Polynomial pretty names:
  - `pretty(Zero()) == "0"`
  - `pretty(One()) == "1"`
  - `pretty(Id()) == "X"`
  - `pretty(Const(S)) == show_type(S)`
- Polynomial composition:
  - `pretty(Sum(F, G)) == pretty(F) + " + " + pretty(G)`
  - `pretty(Prod(F, G))` parenthesizes sum children
  - `pretty(Exp(S, F))` renders `S` with `show_type` and parenthesizes sum/product bodies
- Morphism pretty names:
  - `Identity -> "id"`
  - `Copy -> "copy"`
  - `Delete -> "!"`
  - `First -> "π₁"`
  - `Second -> "π₂"`
  - `Left -> "ι₁"`
  - `Right -> "ι₂"`
  - `Absurd -> "absurd"`
  - `Assoc -> "assoc"`
  - `Prim -> "prim"`
- Morphism composition formatting:
  - `Compose(f, g)` renders as `(f ; g)`
  - `Parallel(f, g)` renders as `(f × g)`
  - `Pair(f, g)` renders as `⟨f, g⟩`
  - `Case(f, g)` renders as `[f, g]`
  - `MonadicEmbed(f, T)` renders as `η(f)`

## morphisms.py

### Type Constructors

- `ProductType(A, B)` is structurally `TypePair(PairType(A, B))`.
- `SumType(A, B)` is structurally `TypeEither(EitherType(A, B))`.
- Product and sum constructors are order-sensitive:
  - `ProductType(A, B) != ProductType(B, A)` when `A != B`
  - `SumType(A, B) != SumType(B, A)` when `A != B`

### Primitive Morphism Type Laws

- `identity(A) : A -> A`
- `copy(A) : A -> A × A`
- `delete(A) : A -> 1`
- `fst(A × B) : A × B -> A`
- `snd(A × B) : A × B -> B`
- `inl(A + B) : A -> A + B`
- `inr(A + B) : B -> A + B`
- `absurd(A) : 0 -> A`
- `distribute_left(A, B, C) : A×(B+C) -> (A×B)+(A×C)`
- `distribute_right(A, B, C) : (A+B)×C -> (A×C)+(B×C)`
- `merge(A) : A+A -> A`

### signature / dom_of / cod_of

- `signature(node) == (dom_of(node), cod_of(node))`
- `Prim(raw, A, B)` reports `(A, B)` and preserves `raw` only as payload.
- Contextual nodes report their stored `dom` and `cod`.
- `MonadicEmbed(f, T)` has:
  - domain `dom_of(f)`
  - codomain `T(cod_of(f))`
- Unknown/base expression nodes raise `TypeError`.

### Morphism Wrapper Laws

- Plain morphism:
  - if `param == Unit` and `monad is None`, `m.dom()` is `dom_of(m.node)`
  - if `param == Unit` and `monad is None`, `m.cod()` is `cod_of(m.node)`
- Parametric morphism:
  - raw domain must be `param × A`
  - `m.dom()` returns `A`
  - wrong raw domain raises `MorphismError`
- Lax morphism:
  - raw codomain must be `monad.wrap(B)`
  - `m.cod()` returns `B`
  - wrong raw codomain raises `MorphismError`
- `node_in(None)` returns the node unchanged.
- `node_in(same_monad)` returns the node unchanged.
- `node_in(target_monad)` embeds a plain node as `MonadicEmbed`.
- `node_in(other_monad)` rejects already-lax morphisms with a different monad.
- `to_lax(monad)` preserves aux primitives and param.

### Composition and Product/Sum Laws

- `compose(f, g)` is valid iff `f.cod() == g.dom()`.
- If valid:
  - `compose(f, g).dom() == f.dom()`
  - `compose(f, g).cod() == g.cod()`
  - aux primitives are concatenated left-to-right.
- `identity` is neutral for type structure:
  - `compose(identity(A), f)` has the same dom/cod as `f`
  - `compose(f, identity(B))` has the same dom/cod as `f`
- Composition is associative for type structure when all parts compose.
- `par(f, g)` has:
  - domain `f.dom() × g.dom()`
  - codomain `f.cod() × g.cod()`
- `pair(f, g)` is valid iff `f.dom() == g.dom()`.
- If valid:
  - domain is `f.dom()`
  - codomain is `f.cod() × g.cod()`
- `case(f, g)` is valid iff `f.cod() == g.cod()`.
- If valid:
  - domain is `f.dom() + g.dom()`
  - codomain is `f.cod()`

### Params and Monads

- Combined param law:
  - `Unit` with `P` gives `P`
  - `P` with `Unit` gives `P`
  - `P` with `Q` gives `Q × P`
- Contextual combinators route the combined param consistently:
  - right child receives the right-side/older param
  - left child receives the left-side/newer param
- Plain morphisms auto-embed into the target monad when composed with lax morphisms.
- Combining two different non-`None` monads raises `MorphismError`.
- Lax `compose` sequences through bind.
- Lax `pair` and `par` sequence both effects and rebuild a product.
- Lax `case` preserves the branch shape while sequencing branch effects.

### PolyExpr / Functor Laws

- Constructors return the matching expression class:
  - `zero -> Zero`
  - `one -> One`
  - `id_ -> Id`
  - `const(S) -> Const(S)`
  - `sum_(F, G) -> Sum(F, G)`
  - `prod(F, G) -> Prod(F, G)`
  - `exp(S, F) -> Exp(S, F)`
- `apply_poly` object action:
  - `Id(A) = A`
  - `One(A) = 1`
  - `Zero(A) = 0`
  - `Const(S)(A) = S`
  - `Prod(F, G)(A) = F(A) × G(A)`
  - `Sum(F, G)(A) = F(A) + G(A)`
  - `Exp(S, F)(A) = S -> F(A)`
- `Functor.summands()` flattens top-level sums left-to-right.
- `Functor.x_arity()` counts `Id` occurrences.
- `Functor.consts()` collects `Const` spaces and `Exp` bases depth-first, left-to-right.
- `Functor(category="poset")` accepts only `body=Id`.
- `Functor(category="set")` accepts any `PolyExpr`.

## realize.py

Realization laws should be checked primarily by running realized terms against
small concrete Hydra values. Direct term-shape tests are allowed only for stable
primitive names or pass-through payloads.

### Primitive Morphism Realization

- `realize(Identity(A))(x) == x`
- `realize(Copy(A))(x) == (x, x)`
- `realize(Delete(A))(x) == unit`
- `realize(First(A × B))((a, b)) == a`
- `realize(Second(A × B))((a, b)) == b`
- `realize(Left(A + B))(a)` produces a left value containing `a`
- `realize(Right(A + B))(b)` produces a right value containing `b`
- `realize(Prim(raw, A, B)) is raw`
- `realize(Assoc(...))(((q, p), a)) == (q, (p, a))`
- `realize(DistributeLeft(A×(B+C), ...))(a, left(b)) == left((a, b))`
- `realize(DistributeLeft(A×(B+C), ...))(a, right(c)) == right((a, c))`
- `realize(DistributeRight((A+B)×C, ...))(left(a), c) == left((a, c))`
- `realize(DistributeRight((A+B)×C, ...))(right(b), c) == right((b, c))`
- Realizing an unknown/base `MorphismExpr` raises `TypeError`.

### Contextual Realization

- Plain `Compose(f, g)` runs as `g(f(x))`.
- Plain `Parallel(f, g)` runs as `(f(left), g(right))`.
- Plain `Pair(f, g)` runs as `(f(x), g(x))`.
- Plain `Case(f, g)` runs left inputs through `f` and right inputs through `g`.
- Parametric `Compose` threads the same param structure used by `morphisms._combine_param`.
- Parametric `Pair` and `Parallel` route each child its declared child param.
- Lax `Compose` uses monadic bind instead of direct application.
- Lax `Pair` and `Parallel` sequence both child effects and wrap the rebuilt product with `pure`.
- Lax `Case` maps branch effects back into the corresponding sum side.

## actions.py

### poly_fmap / polynomial action laws

- Plain functor action:
  - `Id.map(h) == h`
  - `One.map(h)` ignores input and returns unit
  - `Const(S).map(h)` is identity on `S`
  - `Prod(F, G).map(h)` maps both components
  - `Sum(F, G).map(h)` maps the active branch
  - `Zero.map(h)` is the unique map from void
  - `Exp(S, F).map(h)` post-composes inside the function result
- Lax/traversal action:
  - `Id.traverse(h) == h`
  - `One.traverse(h)` returns `pure(unit)`
  - `Const(S).traverse(h)` returns `pure(s)`
  - `Prod(F, G).traverse(h)` sequences left and right effects and rebuilds the product
  - `Sum(F, G).traverse(h)` sequences only the active branch and rebuilds the same branch
  - `Exp` with a monad raises `TypeError`
- `poly_fmap(functor, h)` type law:
  - domain is `apply_poly(body, h.dom())`
  - codomain is `apply_poly(body, h.cod())`
  - if `h` is parametric, raw domain is `h.param × apply_poly(body, h.dom())`
  - if `h` is lax, raw codomain is `h.monad.wrap(apply_poly(body, h.cod()))`
  - aux primitives are preserved
  - `param` and `monad` are preserved

## Priority Order

1. Fill missing `morphisms.py` negative tests for wrapper invariants, monad mismatch, and unknown nodes.
2. Add direct runtime tests for `realize.py` primitive morphism encodings.
3. Add `poly_fmap` type laws and small runtime laws for `Id`, `One`, `Const`, `Prod`, and `Sum`.
4. Add param/lax runtime tests for contextual realization beyond the current smoke tests.
5. Add remaining expression structural equality and constructor tests if they prove useful.
