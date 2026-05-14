# Surface Syntax Reference

Current reference for the ASCII surface language implemented by `unialg.syntax`.

This document is descriptive, not aspirational. It records what the parser accepts
today, what those forms mean, and what is intentionally still outside the language.
Use it as the checkpoint before adding optics, carriers, recursion syntax, types,
or tensor notation.

## Implementation Map

| Concern | File |
|---------|------|
| Tokenization | `src/unialg/syntax/_lex.py` |
| Operator metadata and precedence | `src/unialg/syntax/_ops.py` |
| Morphism Pratt grammar | `src/unialg/syntax/_morphism_grammar.py` |
| Functor Pratt grammar | `src/unialg/syntax/_functor_grammar.py` |
| Program parser | `src/unialg/syntax/parse.py` |
| Syntax node ADTs | `src/unialg/syntax/expressions.py` |
| End-to-end compilation | `src/unialg/main.py` |
| Regression tests | `tests/syntax/test_pratt.py`, `tests/test_load_program.py` |

The parser builds `MorphismExpr` and `PolyExpr` trees. It does not type-check.
Semantic validation belongs to `semantics/morphisms.py`, `semantics/functors.py`,
and later realization.

## Program Syntax

A source program is a sequence of top-level declarations:

```text
load BACKEND
map NAME = <functor-expr>
route NAME = <morphism-expr>
```

Declarations are parsed top to bottom.

- `load BACKEND` imports backend primitive morphisms into the route environment.
- `map NAME = ...` defines a named polynomial functor body.
- `route NAME = ...` defines a named morphism expression.
- Later definitions can refer to earlier `map` and `route` names.
- Unknown route names become unresolved `Ref` nodes.
- Unknown functor names in functor position become unresolved `PolyRef` nodes.

Reserved top-level words are:

```text
load
map
route
```

They cannot be used as ordinary expression names.

## Morphism Syntax

Morphism expressions describe arrows. Composition order is diagrammatic:

```text
f >> g
```

means "run `f`, then run `g`".

### Morphism Operators

From highest to lowest precedence:

| Syntax | Form | Meaning |
|--------|------|---------|
| `m[0]` | postfix | Project first component after `m` |
| `m[1]` | postfix | Project second component after `m` |
| `m^n` | postfix | Run `m`, then copy result into `n` copies |
| `m?0` | postfix | Run `m`, then inject result into the left case |
| `m?1` | postfix | Run `m`, then inject result into the right case |
| `^n` | prefix | Copy input into `n` copies, `n >= 2` |
| `?0` | prefix | Left sum injection |
| `?1` | prefix | Right sum injection |
| `f & g` | infix | Pair: same input goes to both branches |
| `f || g` | infix | Parallel: product input is split into separate lanes |
| `f >> g` | infix | Compose: first `f`, then `g` |
| `f | g` | infix | Case: branch on a sum input |

All binary operators are left-associative.

### Built-In Morphism Atoms

| Syntax | Meaning |
|--------|---------|
| `x`, `id`, `identity` | Identity |
| `!`, `delete`, `drop` | Delete to unit |
| `copy` | Binary copy |
| `fst` | First projection |
| `snd` | Second projection |
| `inl` | Left injection |
| `inr` | Right injection |
| `absurd` | Unique morphism from void |
| `assoc` | Associativity isomorphism placeholder |
| `sym`, `symmetry` | Symmetry isomorphism placeholder |

`copy`, `fst`, `snd`, `inl`, and `inr` are still accepted as names because they
are useful while the notation settles. The newer symbolic forms are:

```text
^2      # copy once: A -> A x A
m[0]    # run m, then first projection
m[1]    # run m, then second projection
?0      # left injection
?1      # right injection
```

### Pair Versus Parallel

`&` and `||` are deliberately different.

```text
f & g
```

uses the same input for both branches. Programming analogy:

```python
lambda x: (f(x), g(x))
```

```text
f || g
```

expects a pair and runs each side independently. Programming analogy:

```python
lambda pair: (f(pair[0]), g(pair[1]))
```

### Copy Power

`^n` is not hard-coded to a fixed arity. It expands structurally.

```text
^2
^3
^5
```

`^1` is rejected. Copy power begins at `2` because `^1` would just be identity.

As postfix syntax:

```text
morph^3
```

means:

```text
morph >> ^3
```

### Projection Postfix

Projection only accepts indexes `0` and `1`:

```text
m[0]
m[1]
(f & g)[0]
```

These are product projections, not arbitrary tuple/list indexing.

### Sum Injection Syntax

Case injections only accept tags `0` and `1`:

```text
?0
?1
m?0
m?1
```

They construct values of a sum/coproduct. Conceptually:

```python
("left", value)
("right", value)
```

The actual runtime representation is Hydra `Either`.

### Morphism Examples

```text
route square = ^2 >> multiply
route tanh_square = tanh^2 >> multiply
route first_branch = (tanh & square)[0]
route tagged = square?0
route choose = tagged >> (tanh | exp)
```

With a backend loaded:

```text
load numpy

map Pair = x & x

route gated_activation = Pair{tanh} >> multiply
```

## Functor Syntax

Functor expressions describe polynomial shapes. They are used in `map`
declarations and in functor action forms like `F{morphism}`.

### Functor Operators

From highest to lowest precedence:

| Syntax | Meaning |
|--------|---------|
| `F*` | List-like functor over `F` |
| `F & G` | Product functor |
| `F | G` | Sum functor |

All functor operators are left-associative.

### Functor Atoms

| Syntax | Meaning |
|--------|---------|
| `0` | Zero / impossible functor |
| `1` | Unit functor |
| `x` | Identity functor / recursive hole |
| `Name` | Previously defined functor, or unresolved `PolyRef` |
| `(F)` | Grouping |

Only integers `0` and `1` are valid as functor atoms.

### Functor Examples

```text
map Maybe = 1 | x
map List = x*
map Pair = x & x
map MaybeList = 1 | x*
map Streams = x* & x*
```

## Functor Action In Morphism Syntax

Functor action maps a morphism through a functor shape.

```text
F{m}
```

means `F(m)`.

Named functors come from earlier `map` declarations:

```text
map Tokens = x*
route tokenwise = Tokens{token_ffn}
```

Inline identity-list action is available through `x*{...}`:

```text
route seq_ffn = x*{token_ffn}
```

Backend/einsum-style references can use bracket syntax:

```text
E[name]
E[name]{m}
```

For `E[name]{m}`, if `name` is a previously defined functor, that functor body is
used. Otherwise the parser records a `PolyRef(name)`.

### Functor Action Examples

```text
map Nat = 1 | x

route zero = ! >> ?0
route successor = ?1

route one = zero >> successor
route two = one >> Nat{successor}
route three = two >> Nat{Nat{successor}}
route count = three
```

This represents Peano-style data structurally:

```text
zero  = left unit
one   = right zero
two   = right one
three = right two
```

It does not yet express a general recursive counter. It only constructs finite
successor structure using current functor and morphism syntax.

## Execution Boundary

`compile_program(src)` parses and lowers the final `route` in a program.

With a backend loaded, `compiled.run(...)` encodes Python scalar arguments using
the first loaded backend coder and decodes the final value:

```python
compiled = compile_program(src)
compiled.run(0.1, 0.8)
```

For pure structural programs with no arguments, `compiled.run()` starts from
`Unit` and decodes ordinary Hydra value terms into Python structures:

| Hydra value | Python display |
|-------------|----------------|
| unit | `None` |
| pair | `(left, right)` |
| list | `[item, ...]` |
| left sum | `("left", value)` |
| right sum | `("right", value)` |

Example output for `three` above:

```python
("right", ("right", ("right", ("left", None))))
```

## Current Limits

These are not syntax today:

- Type annotations.
- Literal values as morphism atoms.
- Numeric arithmetic without backend primitives.
- Predicates or conditionals beyond coproduct `case`.
- General tuple/list indexing; only product `[0]` and `[1]`.
- Named parameters or parameter binding.
- Lambda syntax.
- Local `let` expressions.
- Forward references that must resolve during parsing.
- User-defined backend primitives.
- Optic declarations.
- Carrier declarations.
- `roll` / `unroll` syntax.
- Surface `ana`, `cata`, or `hylo`.
- A general recursive counting function.

The Python API already has `Optic`, `ana`, `cata`, and `hylo`. The surface
language does not yet have a way to name a carrier optic or provide the required
fixed-point boundaries. That is the next design boundary, not an existing feature.

## Kernel Reload Note

In notebooks, parser changes require a kernel restart or module reload. If the
notebook still reports an error like:

```text
unconsumed input: '?0...'
```

then the kernel is still using an older loaded copy of `unialg.syntax._lex`.
Restart the kernel before rerunning syntax examples.

