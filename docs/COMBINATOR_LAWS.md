# Combinator Laws

Algebraic laws for the core combinators, stated as laws first then shown at
each layer that implements them. This is the semantic target for the structural
normalization layer (Tier 4) and the reference for the cross-layer combinator
audit (Tier 2).

`≅` means equal dom/cod and equal observable behaviour.

---

## 1. Identity

**Law:** composition with identity is transparent.

```
compose(id, f) ≅ f
compose(f, id) ≅ f
```

| Layer | Form |
|-------|------|
| `morphisms.py` | `identity(A) : A → A` |
| `functors.py` | `Id` body — `apply_poly(Id, A) == A` |
| `optics.py` | `identity_optic(name, F, focus)` — both boundaries are identity |

---

## 2. Composition

**Law:** composition is associative.

```
compose(compose(f, g), h) ≅ compose(f, compose(g, h))
```

| Layer | Form |
|-------|------|
| `morphisms.py` | `compose(f, g)` — asserts `f.cod() == g.dom()` |
| `functors.py` | `F.compose(inner)` — `F∘G`, substitutes `G` for `Id` in `F` |
| `optics.py` | `outer.compose(inner)` — focus through outer then inner |

---

## 3. Product

**Law:** the product is characterised by its universal property — pairing
two morphisms with a common domain into a product codomain, with projections
that recover each component.

```
pair(f, g) : A → B × C         for f : A → B, g : A → C
fst(B × C) : B × C → B
snd(B × C) : B × C → C

compose(pair(f, g), fst) ≅ f
compose(pair(f, g), snd) ≅ g
pair(compose(h, fst), compose(h, snd)) ≅ h     for h : A → B × C
```

| Layer | Form |
|-------|------|
| `morphisms.py` | `pair(f, g)` — `Pair` node; `_fst`, `_snd` — `First`, `Second` nodes |
| `functors.py` | `Prod(F, G)` body — `apply_poly(Prod(F,G), A) == ProductType(F(A), G(A))` |
| `optics.py` | `Lens` — `F = Prod(Id, Const(residue))` |

---

## 4. Coproduct

**Law:** the coproduct is characterised by its universal property — case
analysis over two morphisms with a common codomain, with injections that
embed each component.

```
case(f, g) : A + B → C         for f : A → C, g : B → C
inl(A + B) : A → A + B
inr(A + B) : B → A + B

compose(inl, case(f, g)) ≅ f
compose(inr, case(f, g)) ≅ g
case(compose(inl, h), compose(inr, h)) ≅ h     for h : A + B → C
```

| Layer | Form |
|-------|------|
| `morphisms.py` | `case(f, g)` — `Case` node; `_inl`, `_inr` — `Left`, `Right` nodes |
| `functors.py` | `Sum(F, G)` body — `apply_poly(Sum(F,G), A) == SumType(F(A), G(A))` |
| `optics.py` | `Prism` — `F = Sum(Id, Const(residue))` |

---

## 5. Parallel (bifunctor)

**Law:** parallel applies two independent morphisms to the two sides of a
product. It is functorial in each argument.

```
par(f, g) : A × C → B × D     for f : A → B, g : C → D

par(id, id) ≅ id
par(compose(f, f'), compose(g, g')) ≅ compose(par(f, g), par(f', g'))
```

| Layer | Form |
|-------|------|
| `morphisms.py` | `par(f, g)` — `Parallel` node |
| `functors.py` | `Prod(F, G)` body evaluated on a single type (diagonal of bifunctor) |
| `optics.py` | `op1.par(op2)` — parallel optic over shared carrier |

---

## 6. Interlayer functors

`poly_fmap` and `optic.act` are not combinators — they are functors that
relate layers to each other. They lift morphisms through structure.

```
poly_fmap(F, id)             ≅ id
poly_fmap(F, compose(f, g))  ≅ compose(poly_fmap(F, f), poly_fmap(F, g))

poly_fmap(F, h).dom() == F.apply(h.dom())
poly_fmap(F, h).cod() == F.apply(h.cod())
```

| What | Where | Role |
|------|-------|------|
| `poly_fmap(F, h)` | `functors.py` | lift morphism through polynomial functor |
| `Functor.map(h)` | `functors.py` | delegates to `poly_fmap(self, h)` |
| `optic.act(h)` | `optics.py` | `compose(forward, functor.map(h), backward)` |

Body-level type action for `poly_fmap(F, h)` where `h : A → B`:

| Body | dom | cod |
|------|-----|-----|
| `Id` | `A` | `B` |
| `One` | `1` | `1` |
| `Zero` | `0` | `0` |
| `Const(S)` | `S` | `S` |
| `Prod(F, G)` | `F(A) × G(A)` | `F(B) × G(B)` |
| `Sum(F, G)` | `F(A) + G(A)` | `F(B) + G(B)` |
| `Exp(S, F)` | `S → F(A)` | `S → F(B)` |

---

## 7. Copy / delete (comonoid)

```
copy(A) : A → A × A
delete(A) : A → 1

compose(copy, fst) ≅ id
compose(copy, snd) ≅ id
compose(copy, par(delete, id)) ≅ id
compose(copy, par(id, delete)) ≅ id
compose(copy, par(copy, id)) ≅ compose(copy, par(id, copy))     # coassociativity
```

---

## 8. Merge (codiagonal)

```
merge(A) : A + A → A
merge(A) ≅ case(id, id)

compose(inl, merge) ≅ id
compose(inr, merge) ≅ id
```

---

## 9. Frobenius structure (copy/merge interaction)

Copy (`δ`: 1→2) and merge (`μ`: 2→1) together with delete (`ε`: 1→0) and
unit injection (`η`: 0→1) form the four generators of a Frobenius algebra
over each type. The laws below are the normalization targets for the
structural rewrite layer (Tier 4).

**Frobenius equation** — copy and merge slide through each other:

```
compose(par(merge, id), par(id, copy))
    ≅ compose(copy, merge)
    ≅ compose(par(id, merge), par(copy, id))
```

**Special condition** — copy then merge is identity:

```
compose(copy(A), merge(A)) ≅ id
```

**Snake equations** — unit/counit zigzags collapse:

```
compose(inl, merge) ≅ id
compose(inr, merge) ≅ id
compose(copy, fst)  ≅ id
compose(copy, snd)  ≅ id
```

**Spider theorem** — any connected diagram built from copy, merge, delete,
unit, and swap with *m* inputs and *n* outputs equals any other connected
diagram with the same *m* inputs and *n* outputs. The canonical form is a
single spider node (*m*, *n*).

This means the arrow notation symbol encodes the normal form directly:
the net count of `>` minus `<` determines the spider type.

| Symbol | Generator | Morphism | Type |
|--------|-----------|----------|------|
| `<>` | δ (comultiply) | `copy(A)` | `A → A × A` |
| `><` | μ (multiply) | `merge(A)` | `A + A → A` |
| `!` | ε (counit) | `delete(A)` | `A → 1` |
| `\|0` | η (unit) | `inl(A+A)` | `A → A + A` |

**Cross-layer correspondence:**

| Layer | `<>` (copy/diverge) | `><` (merge/converge) |
|-------|--------------------|-----------------------|
| `morphisms.py` | `copy(A)` | `merge(A)` |
| tensor layer | index duplication (diagonal) | contraction/trace via semiring `times` |
| functor layer | diagonal into `Prod` | sum collapse from `Sum` |

The Frobenius equation is the algebraic reason distributivity works:
`distribute_left` and `distribute_right` are the mixed interaction between
the product-side copy/merge and the sum-side copy/merge.

---

## 10. Symmetry / associativity

```
symmetry(A × B) : A × B → B × A
symmetry(A + B) : A + B → B + A

compose(symmetry(A×B), symmetry(B×A)) ≅ id

assoc((A×B)×C) : (A × B) × C → A × (B × C)
```

---

## 11. Distributivity

```
distribute_left(A, B, C)  : A × (B + C) → (A × B) + (A × C)
distribute_right(A, B, C) : (A + B) × C → (A × C) + (B × C)
```

---

## 12. Recursion schemes

```
cata(fp, alg)  : μF → A      where alg : F(A) → A
ana(fp, coalg) : A → μF      where coalg : A → F(A)

hylo(fp, coalg, alg) ≅ compose(ana(fp, coalg), cata(fp, alg))

cata(fp, alg) ≅ compose(unroll, compose(poly_fmap(F, cata(fp, alg)), alg))
```

| Layer | Form |
|-------|------|
| `optics.py` | `cata(fp, alg)`, `ana(fp, coalg)`, `hylo(fp, coalg, alg)` |
| `optics.py` | `algebra(fp, alg, i)` — unified builder for cata (i=0) and ana (i=1) |

---

## Cross-layer audit (2026-05-21)

### Semantic vs structural inventory

At the **semantic level** (morphisms.py, functors.py, optics.py), combinators
are typed constructions that check domains, codomains, and monad/param
context. At the **structural level** (realize.py, terms.py), each semantic
node becomes a Hydra term. The structural vocabulary is complete — the
semantic vocabulary has gaps.

#### Semantic level

| Combinator | morphisms.py | functors.py | optics.py |
|---|---|---|---|
| product intro | `pair(f,g)` via `_contextual_binary(Pair)` | `Prod(F,G)` PolyExpr ctor | Lens = `Prod(Id, Const(R))` implicit |
| product elim | `_fst(ab)` → `First`, `_snd(ab)` → `Second` | **gap** | `forward` decomposes |
| sum elim | `case(f,g)` via `_contextual_binary(Case)` | `Sum(F,G)` PolyExpr ctor | Prism = `Sum(Id, Const(R))` implicit |
| sum intro | `_inl(ab)` → `Left`, `_inr(ab)` → `Right` | **gap** | `backward` recomposes |
| compose | `compose(f,g)` check `f.cod()==g.dom()` | `compose_poly` substitutes `Id` | `_compose_optic` |
| parallel | `par(f,g)` via `_contextual_binary(Parallel)` | `Prod` on single type | `Optic.par` |
| copy | `_copy(A)` → `Copy(A)` | **gap** (implicit `Prod(Id,Id)`) | **gap** |
| merge | `merge(A)` = `case(id,id)` | **gap** (implicit `Sum` collapse) | **gap** |
| delete | `_delete(A)` → `Delete(A)` | `One` body | **gap** |
| absurd | `absurd(A)` → `Absurd(A)` | `Zero` body | **gap** |
| swap | `_symmetry(dom)` → `Symmetry(dom,cod)` | **gap** | **gap** |
| assoc | `_assoc(dom)` → `Assoc(dom,cod)` | **gap** | **gap** |
| distribute | `distribute_left/right` | **gap** | **gap** |

#### Structural level (realize.py → terms.py)

| Combinator | Plain Hydra term | Monadic Hydra term |
|---|---|---|
| product intro | `P.pair(f(x), g(x))` | `lift2_effect(monad, λl r. pair(l,r), f(x), g(x))` |
| product elim | `pairs.first` / `pairs.second` primitives | n/a |
| sum elim | `eithers.bimap(f, g)` | `eithers.either(λl. map_effect(left, f(l)), ...)` |
| sum intro | `λx. Terms.left(x)` / `λx. Terms.right(x)` | n/a |
| compose | `g(f(x))` | `bind(monad, f(x), g)` |
| parallel | `pairs.bimap(f, g)` | `λx. pair_effects(monad, f(fst(x)), g(snd(x)))` |
| copy | `λx. pair(x, x)` | n/a |
| swap | `λp. pair(snd(p), fst(p))` | n/a |
| distribute_left | `λp. eithers.bimap(λb. pair(fst(p),b), λc. pair(fst(p),c))(snd(p))` | n/a |

The Hydra vocabulary (`pairs.first`, `pairs.second`, `pairs.bimap`, `eithers.bimap`,
`eithers.either`, `Terms.left`, `Terms.right`) covers every structural combinator.
The gap is **semantic**: functors and optics lack named combinators for operations
their structure implicitly contains.

#### Interlayer functors

`poly_fmap` and `optic.act` lift morphisms through structure. They are functors
between layers, not combinators within a layer.

| PolyExpr body | Plain action | Monadic action |
|---|---|---|
| `Id` | `h` itself | `h` itself |
| `One` | `constant(unit())` | `pure(monad, unit())` |
| `Const(S)` | `identity` | `pure(monad, x)` |
| `Prod(F,G)` | `pairs.bimap(F(h), G(h))` | `pair_effects(monad, F(h)(fst(x)), G(h)(snd(x)))` |
| `Sum(F,G)` | `eithers.bimap(F(h), G(h))` | `eithers.either(map_effect per branch)` |
| `Exp(S,F)` | `λg. λs. F(h)(g(s))` | **error** — not traversable |

### Findings

1. **The structural layer is complete.** Every combinator has a Hydra term.
   realize.py + terms.py handle plain, parametric, and monadic modes for all
   contextual binary nodes.

2. **The semantic layer has systematic gaps at the functor and optic levels.**
   Product introduction/elimination and coproduct introduction/elimination
   exist as morphism-layer combinators and as Hydra primitives, but the
   functor and optic layers encode them implicitly in their type structure
   (`Prod`/`Sum` bodies, Lens/Prism shapes) rather than exposing named
   combinators.

3. **Interlayer functors are not combinators.** `poly_fmap` and `optic.act`
   relate layers to each other. They should not appear in the combinator
   vocabulary alongside compose, pair, case, etc.

4. **The Frobenius generators (copy, merge, delete, absurd) exist only at
   the morphism layer.** At the functor level they are implicit in `Prod(Id,Id)`,
   `Sum` collapse, `One`, and `Zero`. At the optic level they do not appear.

5. **Structural isomorphisms (swap, assoc, distribute) exist only at the
   morphism layer.** The functor and optic layers have no equivalent.

---

## Cross-layer coverage and gaps

### Product structure

| | `morphisms.py` | `functors.py` | `optics.py` |
|---|---|---|---|
| introduction | `pair(f, g) : A → B×C` | `Prod(F, G)` body | **gap** — Lens encodes product focus but no `optic_pair` |
| elimination | `fst : A×B → A`, `snd : A×B → B` | **gap** — no functor projection | `forward` (implicit in Lens shape) |
| bifunctor | `par(f, g) : A×C → B×D` | `Prod` applied = `F(A)×G(A)` | `Optic.par(other)` |
| diagonal | `copy(A) : A → A×A` | **gap** — implicit as `Prod(Id, Id)` | **gap** |
| terminal | `delete(A) : A → 1` | `One` body | **gap** |

### Sum structure

| | `morphisms.py` | `functors.py` | `optics.py` |
|---|---|---|---|
| elimination | `case(f, g) : A+B → C` | `Sum(F, G)` body | **gap** — Prism encodes sum focus but no `optic_case` |
| introduction | `inl : A → A+B`, `inr : B → A+B` | **gap** — no functor injection | `backward` (implicit in Prism shape) |
| codiagonal | `merge(A) : A+A → A` | **gap** — implicit in `Sum` collapse | **gap** |
| initial | `absurd : 0 → A` | `Zero` body | **gap** |

### Structural isomorphisms

| | `morphisms.py` | `functors.py` | `optics.py` |
|---|---|---|---|
| swap | `_symmetry(A×B)`, `_symmetry(A+B)` | **gap** | **gap** |
| assoc | `_assoc((A×B)×C)` | **gap** | **gap** |
| distribute | `distribute_left/right` | **gap** | **gap** |

### Composition and identity

| | `morphisms.py` | `functors.py` | `optics.py` |
|---|---|---|---|
| compose | `compose(f, g)` | `F.compose(G)` | `outer.compose(inner)` |
| identity | `identity(A)` | `Id` body / `id_()` | `identity_optic` |

### Interlayer functors (NOT combinators)

| What | Where | Role |
|------|-------|------|
| `poly_fmap(F, h)` | `functors.py` | lift morphism through polynomial functor |
| `Functor.map(h)` | `functors.py` | delegates to `poly_fmap` |
| `optic.act(h)` | `optics.py` | decompose → lift → recompose |
