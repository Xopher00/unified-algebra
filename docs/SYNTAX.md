# unialg DSL Syntax

This document describes the current surface syntax accepted by the unialg
parser. It is meant as a programmer-facing reference: examples here should
match the implementation, not just the long-term design direction.

The DSL has three top-level declaration forms:

```unialg
load numpy

shape NatF = 1 | x
shape Nat = fix NatF

let zero = |0 >> roll[Nat]
let succ = |1 >> roll[Nat]
let two = zero >> succ >> succ
```

`load` imports backend primitives, `shape` names structural objects, and `let`
names morphisms.

## Program Structure

A program is a sequence of declarations. Whitespace and newlines are
insignificant; comments start with `#` and run to the end of the line.

```unialg
load <backend-name>
let <name> = <morphism>
let <name>(<param>, ...) = <morphism>
shape <name> = <shape>
shape <name> : <source> <-> <target> by <forward> / <backward>
```

Names may contain letters, digits, underscores, and dots after the first
character. The reserved words are `load`, `let`, `shape`, `fix`, and `by`.

When compiling a whole program, the default output is the final non-parametric
`let` binding in source order. A caller can choose a different binding with the
`target=` argument:

```python
from unialg import compile_program

program = compile_program("""
shape NatF = 1 | x
shape Nat = fix NatF
let zero = |0 >> roll[Nat]
let one = zero >> |1 >> roll[Nat]
""", target="one")

result = program.run()
```

## `load`

`load` makes a backend's primitive morphisms available by name.

```unialg
load numpy
let f = tanh
let g = add(exp, tanh)
```

Backend primitives can be used as ordinary morphism references:

```unialg
let f = exp >> log
```

Multi-argument backend primitives use the same application syntax as
parameterized morphisms:

```unialg
let f = add(exp, tanh)
```

This means: apply `exp` and `tanh` to the same input, then call `add` on the two
results.

## `let`

`let` defines a morphism.

```unialg
let f = id
let g = copy >> (id || f)
```

Parameterized `let` definitions bind morphism parameters:

```unialg
let twice(step) = step >> step
let two_succ = twice(succ)
```

The arguments to a parameterized morphism are themselves morphism expressions.
This is useful for higher-level definitions:

```unialg
let compose2(a, b) = a >> b
let f = compose2(exp, tanh)
```

## Morphism Expressions

Morphism expressions denote arrows. They are parsed first as syntax and then
resolved by semantic construction, so most atoms start with placeholder types.

### Morphism Atoms

| Syntax | Meaning |
| --- | --- |
| `id`, `identity`, `x` | identity morphism |
| `copy` | copy, `A -> A & A` |
| `dup(n)` | n-fold copy, `n >= 2` |
| `*n` | prefix shorthand for n-fold copy |
| `delete`, `drop`, `del`, `!`, `1` | delete, `A -> 1` |
| `absurd`, `0` | absurd morphism, `0 -> A` |
| `assoc` | associativity isomorphism |
| `sym`, `symmetry` | symmetry isomorphism |
| `distl` | distributivity `A × (B + C) → (A × B) + (A × C)` |
| `distr` | distributivity `(A + B) × C → (A × C) + (B × C)` |
| `merge` | codiagonal `A + A → A` |
| `[0]` | left projection |
| `[1]` | right projection |
| `|0` | left injection |
| `|1` | right injection |
| `<name>` | named morphism or backend primitive |

### Morphism Operators

Operators are listed from tightest to loosest binding.

| Syntax | Operation | Example |
| --- | --- | --- |
| `f[0]`, `f[1]` | compose with projection | `copy[0]` |
| `f*2` | compose with copy power | `f*2` |
| `f & g` | pair, same input to both sides | `exp & tanh` |
| `f || g` | parallel product | `exp || tanh` |
| `f >> g` | sequential composition | `exp >> log` |
| `f >>>> g` | shared-context composition | `step >>>> fold` |
| `f | g` | case over sums | `zero | succ` |

Composition is diagrammatic: `f >> g` means run `f`, then run `g`.

`&` and `||` are different. `f & g` duplicates the same visible input into both
branches. `f || g` expects a product input and sends the left component through
`f` and the right component through `g`.

### Parametric Application

```unialg
let f(a, b) = a >> b
let g = f(exp, tanh)
```

Application substitutes morphism arguments into the declared parameter slots.
The same syntax is also used for backend primitives whose arity is greater than
one:

```unialg
load numpy
let f = multiply(tanh, exp)
```

### Functor and Optic Action

Curly braces apply a shape action to a morphism:

```unialg
shape MaybeF = 1 | x
let lifted = MaybeF{id}
```

For polynomial functors this is functorial mapping. For optics it is optic
action: decompose, map through the optic functor, and reconstruct.

The identity functor can be written as `x{f}`.

### Recursion and Carrier Boundaries

Recursive carriers define a canonical focus. The focus name is the carrier name
unless an alias is declared.

```unialg
shape NatF = 1 | x
shape Nat = fix NatF

let zero = |0 >> roll[Nat]
let succ = |1 >> roll[Nat]
let inspect = unroll[Nat]
```

Recursive schemes use square brackets to select a focus:

```unialg
let fold_nat = cata[Nat](zero | succ)
let unfold_nat = ana[Nat](coalgebra)
let transform = hylo[Nat](coalgebra, algebra)
```

The arities are:

```unialg
cata[focus](algebra)
ana[focus](coalgebra)
hylo[focus](coalgebra, algebra)
```

### Monadic Lift

The built-in monads currently exposed by name are `Maybe` and `List`.

```unialg
let safe = pure[Maybe](id)
let many = pure[List](id)
```

`pure[Monad](f)` lifts a morphism into the named monad. Composition rules then
thread the monadic context through the term during realization.

## `shape`

`shape` names structural objects. The current parser recognizes three practical
forms:

```unialg
shape <name> = <polynomial-functor>
shape <name> = fix <polynomial-functor>
shape <name> : <source> <-> <target> by <forward> / <backward>
```

Shape parameters are reserved syntax but are not implemented yet:

```unialg
shape ListF(a) = 1 | a & x   # currently rejected
```

Use named non-parameterized shapes for now.

## Polynomial Functor Expressions

Polynomial functor expressions describe one layer of structure.

### Functor Atoms

| Syntax | Meaning |
| --- | --- |
| `x`, `id` | recursion variable / identity functor |
| `0` | initial functor |
| `1` | terminal functor |
| `<name>` | named shape reference |
| `List[F]` | list type constructor over `F` |
| `Maybe[F]` | maybe type constructor over `F` |
| `Exp[F, G]` | exponential functor `F → G` |

### Functor Operators

Operators are listed from tightest to loosest binding.

| Syntax | Operation |
| --- | --- |
| `F & G` | product of functors |
| `F | G` | sum of functors |
| `F >> G` | diagrammatic functor composition |

Examples:

```unialg
shape MaybeF = 1 | x
shape PairF = x & x
shape Nested = MaybeF >> List[x]
```

Because `>>` binds looser than `|` and `&` in functor expressions, use
parentheses when the intended grouping is not obvious:

```unialg
shape F = (1 | x) >> Maybe[x]
```

## Recursive Carriers

A carrier is a named fixed point of a polynomial functor:

```unialg
shape NatF = 1 | x
shape Nat = fix NatF
```

This creates a nominal carrier type and an implicit canonical focus between the
carrier and one unrolled functor layer. That focus is what `roll[Nat]`,
`unroll[Nat]`, `cata[Nat](...)`, and `ana[Nat](...)` use.

You can also write the functor inline:

```unialg
shape Nat = fix (1 | x)
```

## Optics and Focuses

Explicit optics use `<->` and `by`:

```unialg
shape Id = x
shape self : Id <-> Id by id / id
let folded = cata[self](id)
```

The source side of the optic declaration must include the functor name used for
the optic. The full source and target annotations are accepted as syntax, but
they are not yet a complete type language. Today, semantic construction uses the
first name on the source side as the optic functor.

An optic alias can be declared with ordinary `shape =` syntax when the expression
is made only from existing focus names and `>>`:

```unialg
shape NatF = 1 | x
shape Nat = fix NatF
shape two_layers = Nat >> Nat

let inspect_twice = unroll[two_layers]
```

Optic composition is diagrammatic: `outer >> inner` means focus through `outer`,
then through `inner`.

## Grammar Sketch

This is an informal grammar, intended to clarify the accepted surface forms.

```text
program      ::= declaration*
declaration  ::= load_decl | let_decl | shape_decl | algebra_decl

load_decl    ::= "load" NAME
               | "load" "extension" NAME

let_decl     ::= "let" NAME params? "=" morphism
params       ::= "(" NAME ("," NAME)* ")"

shape_decl   ::= "shape" NAME "=" shape_rhs
               | "shape" NAME ":" optic_annotation

algebra_decl ::= "algebra" NAME "(" algebra_field ("," algebra_field)* ")"
algebra_field ::= ("plus" | "times" | "zero" | "one") "=" NAME_OR_FLOAT
                | "adjoint" "=" NAME

shape_rhs    ::= "fix" functor
               | functor

optic_annotation ::= tokens_until_bidir "<->" tokens_until_by
                     "by" morphism "/" morphism
```

Morphism expression grammar:

```text
morphism ::= atom
           | morphism "[" ("0" | "1") "]"
           | morphism "*" INT
           | morphism "&" morphism
           | morphism "||" morphism
           | morphism ">>" morphism
           | morphism ">>>>" morphism
           | morphism "|" morphism

atom     ::= NAME
           | NAME "(" morphism ("," morphism)* ")"
           | NAME "{" morphism "}"
           | "cata" "[" NAME "]" "(" morphism ")"
           | "ana" "[" NAME "]" "(" morphism ")"
           | "hylo" "[" NAME "]" "(" morphism "," morphism ")"
           | "roll" "[" NAME "]"
           | "unroll" "[" NAME "]"
           | "pure" "[" NAME "]" "(" morphism ")"
           | "contract" "[" NAME "]" "(" STRING ")"
           | "contract" "[" NAME "," "adjoint" "]" "(" STRING ")"
           | "[" ("0" | "1") "]"
           | "|" ("0" | "1")
           | "!" | "0" | "1" | "* INT"
           | "(" morphism ")"
```

Functor expression grammar:

```text
functor ::= atom
          | functor "&" functor
          | functor "|" functor
          | functor ">>" functor

atom    ::= "x" | "id" | "0" | "1" | NAME
          | "List" "[" functor "]"
          | "Maybe" "[" functor "]"
          | "Exp" "[" functor "," functor "]"
          | "(" functor ")"
```

## Current Limitations

These are implementation limits, not desired long-term semantics:

- Shape parameters such as `shape ListF(a) = ...` are reserved but rejected.
- Explicit optic annotations are not a full type language yet; they currently
  identify the optic functor by the first source-side name.
- The shorthand optic form without `by`, such as `shape root : F <-> G`, is not
  accepted yet.
- User-defined type aliases and type-level interpretation are not complete.
- `;` is rejected; use `>>` for composition.

