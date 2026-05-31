# unialg DSL Syntax

Programmer-facing reference for the unialg surface language. Examples here
reflect current parser behaviour, not long-term design targets. For a flat
token-by-token reference see `docs/LEXICON.md`.

---

## Program Structure

A program is a sequence of declarations. Whitespace and newlines are
insignificant. Comments start with `#` and run to end of line.

```
program     ::= declaration*
declaration ::= load_decl | let_decl | shape_decl
```

**Top-level forms:**

```unialg
load <backend-name>
load extension <extension-name>

let <name> = <morphism>
let <name>(<param>, ...) = <morphism>

shape <name> = <functor>
shape <name> = fix <functor>
shape <name> : <functor> <-> <functor> by <morphism> / <morphism>
shape lens      <name> : <functor> view <functor> by <morphism> / <morphism>
shape prism     <name> : <functor> view <functor> by <morphism> / <morphism>
shape traversal <name> : <functor> by <morphism> / <morphism>
```

**Reserved words:** `let`, `shape`, `load`, `fix`, `by`, `view`, `lens`,
`prism`, `traversal`.

When compiling a whole program the default output is the final non-parametric
`let` binding in source order. A different binding can be selected:

```python
from unialg import compile_program

program = compile_program("""
shape NatF = 1 | x
shape Nat  = fix NatF
let zero = |0 >> roll[Nat]
let one  = zero >> |1 >> roll[Nat]
""", target="one")
result = program.run()
```

---

## `load` — Backend Import

`load` makes a backend's primitive morphisms available by name.

```unialg
load numpy
let f = tanh
let g = add(exp, tanh)
let h = exp >> log
```

Backend primitives behave as ordinary morphism references. Multi-argument
primitives use the same application syntax as parameterized morphisms:

```unialg
load numpy
let f    = add(exp, tanh)
let norm = softmax(x, '-1')
```

---

## `let` — Morphism Binding

`let` binds a name to a morphism expression.

```unialg
let f = id
let g = copy >> (id || f)
```

**Parameterized bindings** accept morphism arguments:

```unialg
let twice(step)    = step >> step
let compose2(a, b) = a >> b

let two_succ = twice(succ)
let f        = compose2(exp, tanh)
```

Arguments are morphism expressions, not values.

**Literal payloads** pass scalar constants to typed primitive sites:

```unialg
let configured(axis) = softmax(x, axis)
let f       = configured('-1')
let columns = reduce.add(x, '0')
```

Single-quoted strings resolve to the type declared by the receiving primitive
argument (`INT`, `FLOAT`, `BOOL`, or `STRING`). A quoted literal bound
without a receiving site is rejected:

```unialg
let axis = '-1'   # invalid — no typed receiving site
```

---

## Morphism Expressions

Atom tokens and their types are listed in `docs/LEXICON.md` under
"Structural Morphism Atoms". The sections below cover how atoms combine.

### Operators and Type Constraints

The key question for each binary operator is: what must `f` and `g` be?

| Symbol | Prec | Result | f must be | g must be |
|--------|------|--------|-----------|-----------|
| `f & g` | 70 | `A → B × C` | `A → B` | `A → C` (same domain as f) |
| `f \|\| g` | 65 | `A × B → C × D` | `A → C` | `B → D` |
| `f && g` | 65 | `A + B → C + D` | `A → C` | `B → D` |
| `f >> g` | 60 | `A → C` | `A → B` | `B → C` (g's domain = f's codomain) |
| `f >>>> g` | 60 | `A → C` | `A → B` | `B → C` (shared parameter) |
| `f \| g` | 50 | `A + B → C` | `A → C` | `B → C` (same codomain as f) |

**Choosing the right operator:**

- Use `&` when you have **one value** and need two results from it.
  `f & g` copies implicitly: `f & g : A → B × C`. Categorically,
  it is the diagonal followed by component-wise action:
  `f & g = copy >> (f || g)`.

- Use `||` when you have a **product** and want to transform each component
  independently. `f || g : A × B → C × D`.

- Use `&&` when you have a **coproduct** and want to map both branches,
  keeping the branch structure. `f && g : A + B → C + D`.

- Use `|` when you have a **coproduct** and want to eliminate it into a
  single type. `f | g : A + B → C` — both `f` and `g` must produce `C`.

- Use `>>` to sequence: the codomain of `f` must equal the domain of `g`.

**Examples:**

```unialg
# & — fan out one value into a pair
let label_and_delete = id & delete   # A → A × 1
let paired = f & g                    # equals copy >> (f || g)

# || — transform a product component by component
let swap_then_map = sym >> (f || g)  # if sym : A × B → B × A

# && — map both branches of a sum without collapsing
let lift_both = |0 >> (f && g)       # inject then bimap (contrived; shows &&)

# | — case analysis: eliminate a sum into one type
# zero_const must be a supplied constant morphism 1 → A
let fold_step = zero_const | succ_step  # 1 + A → A, if succ_step : A → A

# >> — sequence
let chain = f >> g >> h
```

**Shared-context composition (`>>>>`):**

`f >>>> g` composes `f` and `g` so they share one copy of the current
parameter rather than each receiving separate copies. Use when a
self-reference and an algebra must share the same parameter — `hylo` uses
this internally. In normal programs, use `>>`.

### Postfix Operators

| Symbol | Equivalent | Meaning |
|--------|-----------|---------|
| `f[0]` | `f >> [0]` | Apply `f` then project left |
| `f[1]` | `f >> [1]` | Apply `f` then project right |
| `f*n` | `f >> *n` | Apply `f` then copy n times |
| `F{f}` | — | Functor map or optic action for shape `F` applied to morphism `f` |

### Parametric Application

```unialg
let f(a, b) = a >> b
let g = f(exp, tanh)
```

Multi-argument backend primitives use the same syntax:

```unialg
load numpy
let h = multiply(tanh, exp)
```

### Recursion Scheme Atoms

```unialg
roll[F]               # F(μF) → μF
unroll[F]             # μF → F(μF)
cata[F](alg)          # μF → B,  alg : F(B) → B
ana[F](coalg)         # A → νF,  coalg : A → F(A)
hylo[F](coalg, alg)   # A → B
```

### Writing a Cata Algebra

`cata[F](alg)` folds a recursive structure of type `μF` to a result type `B`.
The algebra `alg` must have type `F(B) → B` — it consumes one unrolled layer
of structure (with all recursive children already reduced to `B`) and produces
a single `B`.

**How to determine the algebra type:**

1. Write out `F(B)` by substituting `B` for `x` in the functor body.
2. Write a morphism `F(B) → B` using the regular morphism operators.

For `NatF = 1 | x`, `F(B) = 1 + B`. The algebra handles two cases:

- `zero_case : 1 → B` — produces the base value from the unit input
- `succ_case : B → B` — transforms an already-reduced recursive child
- Combined: `zero_case | succ_case : 1 + B → B`

**Writing `1 → B`:** the unit type `1` carries no information, so any
morphism `1 → B` is a constant — it always returns the same `B` regardless
of input. There is no structural atom for this; it must come from a backend
primitive declared with that type. `delete : A → 1` and `absurd : 0 → A` are
NOT valid here — they have the wrong types.

```unialg
load numpy

shape NatF = 1 | x
shape Nat  = fix NatF

# zero_const : 1 → INT  (a backend primitive that returns the constant 0)
# succ_step  : INT → INT (a backend primitive that adds 1)
let fold_to_int = cata[Nat](zero_const | succ_step)
```

When the result type `B` is itself `1` (the unit), `id` is the only valid
`1 → 1` morphism, making it the degenerate base case:

```unialg
let fold_to_unit = cata[Nat](id | delete)   # B = 1; collapses structure entirely
```

For `PairF = x & x`, `F(B) = B × B`. The algebra takes two already-reduced
children and combines them — no `|` needed since there is no sum:

```unialg
shape PairF  = x & x
shape BinRec = fix PairF

# combine : B × B → B, e.g. a backend add primitive
let bin_fold = cata[BinRec](combine)
```

**Pattern:** whenever `F` is a sum, write the algebra with `|` matching each
summand. Whenever `F` is a product, the algebra takes a product and combines
the components. Sum cases that map from `1` require a backend constant.

### Injections and Case Analysis

`|0` and `|1` are **injections** (constructors into a sum):

```
|0 : A → A + B   # inject a value into the left branch
|1 : B → A + B   # inject a value into the right branch
```

`f | g` is **case analysis** (eliminator of a sum):

```
f | g : A + B → C   # requires f : A → C and g : B → C
```

These are dual: injections build sums, case analysis destructs them. A
catamorphism algebra for `F = A | B` is always a `f | g` expression.

### Monadic Lift

```unialg
pure[Maybe](f)
pure[List](f)
```

`pure[M](f)` lifts morphism `f` into monad `M`. Available monads: `Maybe`,
`List`.

---

## `shape` — Structural Declarations

`shape` names polynomial functors, recursive carriers, and optics.

### Polynomial Functor

```unialg
shape MaybeF = 1 | x
shape PairF  = x & x
shape Nested = MaybeF >> List[x]
```

### Recursive Carrier

```unialg
shape NatF = 1 | x
shape Nat  = fix NatF
```

`fix` creates a nominal recursive carrier and a canonical focus used by
`roll[Nat]`, `unroll[Nat]`, `cata[Nat](...)`, `ana[Nat](...)`. The functor
body can be written inline:

```unialg
shape Nat = fix (1 | x)
```

### Generic Optic

```unialg
shape self : Id <-> Id by id / id
```

The first name on the source side names the optic functor. Annotations are
syntax; full type-level interpretation is not complete. Default kind is
`optic`.

### Lens

```unialg
shape lens myLens : SomeF view FocusF by forward / backward
```

`view` separates the functor type from the focus type.

### Prism

```unialg
shape prism myPrism : SomeF view FocusF by forward / backward
```

Same surface syntax as lens; different algebraic interpretation.

### Traversal

```unialg
shape traversal myTrav : F by forward / backward
```

No `view` annotation; `F` is the traversal functor directly.

### Optic Alias (Composition)

An alias built from existing focus names and `>>` composes optics
diagrammatically: `outer >> inner` focuses through `outer` then `inner`.

```unialg
shape NatF       = 1 | x
shape Nat        = fix NatF
shape two_layers = Nat >> Nat

let inspect_twice = unroll[two_layers]
```

---

## Polynomial Functor Expressions

Functor expressions appear in `shape` bodies and `fix` operands.

### Atoms

| Token | Meaning |
|-------|---------|
| `x`, `id` | Recursion variable / identity functor |
| `0` | Initial functor |
| `1` | Terminal functor |
| `const[T]` | Constant functor at backend type `T` (see below) |
| `<name>` | Named shape reference |
| `List[F]` | List type constructor over `F` |
| `Maybe[F]` | Maybe type constructor over `F` |
| `Exp[F, G]` | Exponential functor `F → G` |
| `Rose[F]` | Rose tree functor over `F` |
| `Tree[F]` | Binary tree functor over `F` |

**`const[T]`** embeds a backend carrier type as a constant functor —
`const[T](A) = T` regardless of `A`. The type name `T` is one of
`STRING`, `INT`, `FLOAT`, `BOOL`, `BINARY`, or a compound like
`List[STRING]` or `Maybe[INT]`. This enables functor bodies that include
backend types:

```unialg
shape EventF = const[List[STRING]] & const[STRING]   # chord list × melody note
shape PhraseF = 1 | (EventF & x)                     # nil | (event, rest)
shape Phrase = fix PhraseF
```

### Operators (tightest to loosest)

| Symbol | Prec | Meaning |
|--------|------|---------|
| `F & G` | 70 | Product of functors |
| `F \| G` | 60 | Sum of functors |
| `F >> G` | 50 | Functor composition |

`>>` binds looser than `|` and `&`; parenthesize when needed:

```unialg
shape F = (1 | x) >> Maybe[x]
```

### Built-in Functor Shapes

The following functor constructors are built-in and do not require a `shape`
declaration. Prefer them over manually writing equivalent expressions.

| Token | Expands to | Meaning |
|-------|-----------|---------|
| `List[F]` | `List[F(x)]` | List of `F`-structured elements |
| `Maybe[F]` | `1 \| F(x)` | Optional `F`-structured element (nothing or just) |
| `Rose[F]` | `F(x) & List[x]` | Rose tree: a payload node plus a list of children |
| `Tree[F]` | `1 + Rose[F](x)` | n-ary tree: leaf or (payload + list of children) |

Use these directly in functor bodies and `fix` expressions:

```unialg
# Manual Maybe (avoid this when Maybe[x] suffices)
shape ManualMaybe = 1 | x

# Built-in shapes used directly in recursive carriers
shape RecursiveList = fix List[x]
shape RoseTree      = fix Rose[x]

# Rose tree catamorphism: algebra takes a payload and a list of
# already-reduced children (a List[B]) and produces B. There is no leaf arm.
let rose_step  = [0]                   # Rose(B) = B & List[B] → B  (keep payload)
let rose_fold  = cata[RoseTree](rose_step)
```

`Tree[x]` includes an explicit leaf (`1`) case; `Rose[x]` does not. Choose
`Rose[x]` when every node carries a value; choose `Tree[x]` when leaves are
empty.

---

## Extension Loading

`load extension` registers an extension's keywords and declaration handlers.

```unialg
load extension tensors
```

Extensions may introduce new declaration forms unavailable in core syntax.

---

## Tensor Extension

Loaded with `load extension tensors`. Requires a numeric backend (e.g.
`load numpy`). Provides two new forms: `algebra` declarations and `contract`
expressions.

### `algebra` — Semiring Declaration

Declares a named semiring used for tensor contractions.

```unialg
algebra Name(plus=add_op, times=mul_op, zero=0.0, one=1.0)
algebra Name(plus=add_op, times=mul_op, zero=0.0, one=1.0, adjoint=adj_op)
```

| Field | Required | Meaning |
|-------|----------|---------|
| `plus` | yes | Addition operation (morphism name) |
| `times` | yes | Multiplication operation (morphism name) |
| `zero` | yes | Additive identity (morphism name or numeric literal) |
| `one` | yes | Multiplicative identity (morphism name or numeric literal) |
| `adjoint` | no | Adjoint/transpose operation (morphism name) |

`zero` and `one` may be floating-point literals (`0.0`, `1.0`, `-inf`) or
morphism names. When a morphism name is given the value is resolved at
realization time.

### `contract` — Tensor Contraction

Applies a semiring contraction along the index pattern given by an
einsum-style equation string.

```unialg
contract[Name]("equation")
contract[Name, adjoint]("equation")
```

The equation string follows einsum notation: comma-separated input index
labels, `->`, output index labels. Indices absent from the output are summed.

```unialg
load extension tensors
load numpy

algebra Tropical(plus=minimum, times=add, zero=inf, one=0.0)

# matrix-vector product under the tropical semiring
let shortest_path = contract[Tropical]("ij,j->i")

# standard matrix multiply using the real semiring
algebra Real(plus=add, times=mul, zero=0.0, one=1.0)
let matvec = contract[Real]("ij,j->i")

# batched outer product
let outer = contract[Real]("bi,bj->bij")
```

`contract[Name, adjoint]("eq")` uses the semiring's declared `adjoint`
operation instead of `times`, transposing the contraction.

---

## Grammar Reference

```
program     ::= declaration*

declaration ::= load_decl
              | let_decl
              | shape_decl
              | algebra_decl            (only after "load extension tensors")

load_decl   ::= "load" NAME
              | "load" "extension" NAME

let_decl    ::= "let" NAME params? "=" morphism
params      ::= "(" NAME ("," NAME)* ")"

shape_decl  ::= "shape" NAME "=" shape_rhs
              | "shape" NAME ":" optic_annotation
              | "shape" "lens"      NAME ":" functor "view" functor "by" morphism "/" morphism
              | "shape" "prism"     NAME ":" functor "view" functor "by" morphism "/" morphism
              | "shape" "traversal" NAME ":" functor "by" morphism "/" morphism

algebra_decl ::= "algebra" NAME "(" algebra_field ("," algebra_field)* ")"
algebra_field ::= ("plus" | "times" | "zero" | "one" | "adjoint") "=" (NAME | NUMBER)

shape_rhs   ::= "fix" functor | functor

optic_annotation ::= functor "<->" functor "by" morphism "/" morphism

morphism    ::= atom
              | morphism "&"    morphism
              | morphism "||"   morphism
              | morphism "&&"   morphism
              | morphism ">>"   morphism
              | morphism ">>>>" morphism
              | morphism "|"    morphism
              | morphism "[" INT "]"
              | morphism "*" INT
              | NAME "{" morphism "}"
              | "cata"  "[" NAME "]" "(" morphism ")"
              | "ana"   "[" NAME "]" "(" morphism ")"
              | "hylo"  "[" NAME "]" "(" morphism "," morphism ")"
              | "pure"  "[" NAME "]" "(" morphism ")"
              | "roll"   "[" NAME "]"
              | "unroll" "[" NAME "]"
              | "contract" "[" NAME "]" "(" STRING ")"
              | "contract" "[" NAME "," "adjoint" "]" "(" STRING ")"
              | "|" INT
              | "[" INT "]"
              | "*" INT
              | "(" morphism ")"
              | NAME "(" morphism ("," morphism)* ")"
              | NAME
              | QUOTED

functor     ::= atom
              | functor "&"  functor
              | functor "|"  functor
              | functor ">>" functor
              | NAME "[" functor "]"
              | NAME "[" functor "," functor "]"
              | "(" functor ")"

atom (functor) ::= "x" | "id" | "0" | "1" | NAME
```

---

## Limitations

- **Shape parameters** (`shape ListF(a) = 1 | a & x`) are reserved syntax
  but currently rejected.
- **Optic annotations** are not a full type language. Semantic construction
  uses the first source-side name as the optic functor.
- **Shorthand optic form** (`shape root : F <-> G` without `by`) is not
  accepted.
- **`;`** is explicitly rejected. Use `>>` for composition.
