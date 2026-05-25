# unialg DSL Lexicon

Quick reference for every token in the unialg surface language. All entries
reflect the current parser implementation. Use this when writing new programs.

---

## Keywords

Reserved words. Cannot be used as morphism or shape names.

| Keyword      | Role |
|--------------|------|
| `let`        | Morphism binding declaration |
| `shape`      | Functor, carrier, or optic declaration |
| `load`       | Backend import; `load extension` for extension loading |
| `fix`        | Marks a shape body as a recursive carrier |
| `by`         | Separates optic kind annotation from boundary morphisms |
| `view`       | Separates source from focus type in `lens` and `prism` declarations |
| `lens`       | Optic kind: view-based lens |
| `prism`      | Optic kind: view-based prism |
| `traversal`  | Optic kind: traversal without view annotation |

---

## Structural Morphism Atoms

Atoms that stand alone as morphism expressions.

| Token(s) | Type | Meaning |
|----------|------|---------|
| `id`, `identity`, `x` | `A → A` | Identity |
| `copy` | `A → A × A` | Diagonal copy |
| `dup(n)` | `A → A×…×A` | Named n-fold copy, `n >= 2` |
| `*n` (prefix, n ≥ 2) | `A → A×…×A` | n-fold copy |
| `delete`, `drop`, `del`, `!`, `1` | `A → 1` | Terminal map (delete) |
| `absurd`, `0` | `0 → A` | Initial map |
| `'<payload>'` | `1 → A` | Typed literal point; requires a receiving typed primitive site |
| `assoc` | `(A × B) × C → A × (B × C)` | Product associativity |
| `sym`, `symmetry` | `A × B → B × A` | Product symmetry (swap) |
| `distl` | `A × (B + C) → (A × B) + (A × C)` | Distribute left |
| `distr` | `(A + B) × C → (A × C) + (B × C)` | Distribute right |
| `merge` | `A + A → A` | Codiagonal |
| `\|0` (prefix) | `A → A + B` | Left injection |
| `\|1` (prefix) | `B → A + B` | Right injection |
| `[0]` (standalone) | `A × B → A` | Left projection |
| `[1]` (standalone) | `A × B → B` | Right projection |

---

## Morphism Operators

Higher precedence binds tighter. The "Requires" column states what `f` and
`g` must be for the expression to type-check.

| Symbol | Prec | Result type | Requires |
|--------|------|-------------|---------|
| `f & g` | 70 | `A → B × C` | `f : A → B`, `g : A → C` — same domain |
| `f \|\| g` | 65 | `A × B → C × D` | `f : A → C`, `g : B → D` — split product input |
| `f && g` | 65 | `A + B → C + D` | `f : A → C`, `g : B → D` — map each branch |
| `f >> g` | 60 | `A → C` | `f : A → B`, `g : B → C` — codomains match |
| `f >>>> g` | 60 | `A → C` | same as `>>`, but `f` and `g` share one parameter copy |
| `f \| g` | 50 | `A + B → C` | `f : A → C`, `g : B → C` — same codomain |

**Input shape at a glance:**

```
f & g     — ONE value of type A; copied; f and g each receive their own copy
f || g    — PRODUCT  A × B; f receives A, g receives B
f && g    — COPRODUCT A + B; f handles left branch, g handles right branch
f | g     — COPRODUCT A + B; both branches map to the SAME output type C
```

`||` and `&&` look symmetric: both take two morphisms over each side of a
type. The difference is input shape — product vs. coproduct.

`&` and `|` look symmetric: both take two morphisms with a shared type. The
difference is output shape — `&` shares the domain (copy input), `|` shares
the codomain (merge branches).

Useful identities:

```
f & g   = copy >> (f || g)    # same input, then map both copies
merge   = id | id             # collapse either coproduct branch
```

For a case expression such as `zero_case | succ_case : 1 + A -> A`,
`zero_case : 1 -> A` must be a supplied constant morphism. `delete` cannot
serve as that branch because its direction is `A -> 1`.

**Postfix operators:**

| Symbol | Meaning |
|--------|---------|
| `f[0]` | `f >> [0]` |
| `f[1]` | `f >> [1]` |
| `f*n` | `f >> *n` |
| `F{f}` | Functor map or optic action for shape `F` applied to morphism `f` |

---

## Recursion Scheme Atoms

| Token | Meaning |
|-------|---------|
| `roll[F]` | `F(μF) → μF` |
| `unroll[F]` | `μF → F(μF)` |
| `cata[F](alg)` | Fold: `μF → B` given `alg : F(B) → B` |
| `ana[F](coalg)` | Unfold: `A → νF` given `coalg : A → F(A)` |
| `hylo[F](coalg, alg)` | Unfold then fold: `A → B` |

---

## Monadic Lift

| Token | Meaning |
|-------|---------|
| `pure[Maybe](f)` | Lift `f` into the `Maybe` monad |
| `pure[List](f)` | Lift `f` into the `List` monad |

---

## Functor Expression Atoms

Used inside `shape` bodies.

| Token | Meaning |
|-------|---------|
| `x`, `id` | Recursion variable / identity functor |
| `0` | Initial functor |
| `1` | Terminal functor |
| `const[T]` | Constant functor at backend type `T` (STRING, INT, FLOAT, BOOL, BINARY, List[T], Maybe[T]) |
| `<name>` | Named shape reference |
| `List[F]` | List type constructor over `F` |
| `Maybe[F]` | Maybe type constructor over `F` |
| `Exp[F, G]` | Exponential functor `F → G` |
| `Rose[F]` | Rose tree functor over `F` |
| `Tree[F]` | Binary tree functor over `F` |

**Functor operators** (tightest to loosest):

| Symbol | Prec | Meaning |
|--------|------|---------|
| `F & G` | 70 | Product of functors |
| `F \| G` | 60 | Sum of functors |
| `F >> G` | 50 | Functor composition |

---

## Literal Payloads

Single-quoted scalars passed to typed primitive arguments.

| Value | Resolves to | When receiving argument is |
|-------|-------------|---------------------------|
| `'-1'`, `'42'` | `INT` point | `INT` |
| `'0.5'`, `'1e-3'` | `FLOAT` point | `FLOAT` |
| `'true'`, `'false'` | `BOOL` point | `BOOL` |
| `'hello'`, `'-1'` | `STRING` text | `STRING` |

A literal cannot appear without a typed receiving site.

---

## Tensor Extension (requires `load extension tensors`)

### `algebra` Declaration

```
algebra Name(plus=op, times=op, zero=val, one=val)
algebra Name(plus=op, times=op, zero=val, one=val, adjoint=op)
```

Declares a named semiring. `op` is a morphism name; `val` is a morphism name
or a numeric literal (`0.0`, `1.0`, `-inf`).

### `contract` Expression

| Token | Result type | Meaning |
|-------|-------------|---------|
| `contract[Name]("eq")` | `A → B` | Semiring contraction along index equation |
| `contract[Name, adjoint]("eq")` | `A → B` | Adjoint (transposed) contraction |

`eq` is an einsum-style string: `"ij,j->i"` means sum over `j`, keeping `i`.
Indices in the inputs but absent from the output are contracted (summed over).

---

## Names

Starts with a letter or `_`, followed by letters, digits, `_`, or `.`.
Dots are allowed to support backend names like `reduce.add`.
