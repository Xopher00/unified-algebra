# Unified Algebra DSL — Syntax Reference

This document is the authoritative lexicon for the `.ua` DSL.
It is derived from `src/unialg/parser/_grammar.py` and must be kept in sync with it.

---

## Whitespace and Comments

- **Inline whitespace** (spaces, tabs) is ignored around tokens.
- **Newlines** are significant — they terminate declarations and attribute lines.
- **Blank lines** (empty, whitespace-only, or comment-only) are ignored everywhere.
- **Comments** begin with `#` and extend to end of line.

```
# this is a comment
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)  # inline comment
```

---

## Tokens

### Identifiers
Start with a letter or `_`, followed by letters, digits, or `_`.

```
hidden   real   ffn_layer   _tmp
```

### String literals
Enclosed in double quotes. Used for einsum subscript strings.

```
"ij,j->i"   "bij,bj->bi"   "i,i->i"
```

### Number literals
Integer or decimal. Special values `inf` and `-inf` are supported.

```
0.0   1.0   float('inf')   -inf   42
```

---

## Operators and Punctuation

| Token | Role |
|-------|------|
| `>>`  | Sequential composition (`seq` chain, `lens_seq` chain) |
| `->`  | Sort signature separator (`dom -> cod`) |
| `<->` | Bidirectional sort signature (`dom <-> cod`, lenses only) |
| `:`   | Type annotation — separates a declaration name from its signature |
| `=`   | Assignment — separates a name or key from its value |
| `~`   | Template marker — prefix on `op` name; marks the op as a template for parametric instantiation |
| `+`   | Residual marker — suffix on `seq` name; adds a skip connection via the semiring's plus |
| `\|`  | Branch separator — inline branch list in `branch` and `lens_branch` declarations |
| `(`   | Open argument list |
| `)`   | Close argument list |
| `[`   | Open template instantiation or subscript |
| `]`   | Close template instantiation or subscript |
| `,`   | Separator in argument lists |

---

## Declaration Forms

Declarations are separated by newlines (blank lines and comments allowed between them).
Attribute blocks are indented with at least one space or tab.

---

### `algebra`

Declares a semiring with named binary operations and identity elements.

```
algebra <name>(plus=<ident>, times=<ident>, zero=<num>, one=<num>)
```

The backend must provide functions named by `plus` and `times`.
`zero` and `one` are the additive and multiplicative identities.

**Examples:**
```
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
algebra fuzzy(plus=maximum, times=minimum, zero=0.0, one=1.0)
```

---

### `spec`

Declares a named tensor type associated with an algebra.

```
spec <name>(<algebra-name>)
spec <name>(<algebra-name>, batched)
```

`batched` adds a leading independent batch dimension at resolution time.

**Examples:**
```
spec hidden(real)
spec output(real)
spec hidden_batched(real, batched)
```

---

### `op`

Declares a morphism: a typed tensor operation.

```
op <name> : <dom> -> <cod>
  einsum = "<subscript>"
  algebra = <algebra-name>
```

For nonlinear (pointwise) ops, use `nonlinearity` instead of `einsum`:

```
op <name> : <dom> -> <cod>
  nonlinearity = <ident>
```

Attributes are indented key-value pairs. `einsum` and `nonlinearity` are mutually exclusive.
`algebra` is required for einsum ops; omit it for nonlinearities.

| Attribute | Type | Meaning |
|---|---|---|
| `einsum` | string | Einsum subscript string |
| `algebra` | ident | Algebra name (required for einsum ops) |
| `nonlinearity` | ident | Pointwise operation name |

**Examples:**
```
op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu

op softmax : hidden -> output
  nonlinearity = softmax
```

#### Template ops

An op declared with the `~` prefix is a **template**: it is not registered as a concrete op itself, but can be instantiated with a prefix to produce distinct concrete ops with separate weight keys in the Hydra graph.

**Declaration:**
```
op ~proj : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
```

**Instantiation** — use `name[prefix]` in a `seq` chain or `branch` inline list:
```
seq qk : hidden -> hidden = proj[q] >> proj[k]

branch kv : hidden -> hidden = proj[q] | proj[k] | proj[v]
  merge = add_merge
```

Each `proj[q]`, `proj[k]`, `proj[v]` expands to concrete ops `q_proj`, `k_proj`, `v_proj` respectively, sharing the same einsum/specs/algebra but with distinct names (= distinct weight tensors at runtime).

Rules:
- Template refs may appear anywhere plain op names appear in `seq` `>>` chains and `branch` `|` inline lists.
- Using the same template ref twice in the same program (e.g. `proj[q] >> proj[q]`) creates exactly one concrete op — not a duplicate.
- Referencing an undeclared template name raises `ValueError`.
- Template ops do not appear in `spec.ops`; only their concrete instantiations do.

---

### `seq`

Declares a sequential composition of ops, applied left-to-right.

```
seq <name> : <dom> -> <cod> = <op1> >> <op2> >> ... >> <opN>
```

`op1` receives the input; `opN` produces the output.
All op names must be declared before the `seq`. Template refs (`name[prefix]`) may appear in the chain alongside plain op names.

An optional indented attribute block may follow the chain:

| Attribute  | Type  | Meaning |
|------------|-------|---------|
| `algebra`  | ident | Name of the algebra whose `plus` operation performs the skip connection. Required when `+` suffix is used. |

#### Residual `seq`

Appending `+` to the `seq` name adds a skip connection: `output = seq(x) ⊕ x` using the algebra's plus.

```
seq <name>+ : <dom> -> <cod> = <op1> >> <op2> >> ... >> <opN>
  algebra = <algebra-name>
```

**Examples:**
```
seq layer : hidden -> hidden = linear >> relu
seq ffn : hidden -> hidden = linear1 >> relu1 >> linear2 >> relu2 >> linear3

# Residual / skip connection
seq resblock+ : hidden -> hidden = linear >> relu
  algebra = real
```

---

### `branch`

Declares a parallel composition: branches applied to the same input, then merged.
Branches are listed inline, separated by `|`.

```
branch <name> : <dom> -> <cod> = <op1> | <op2> | ... | <opN>
  merge = <merge-op>
```

All branch ops receive the same input. The merge op receives a list of branch outputs. Template refs (`name[prefix]`) may appear in the branch list alongside plain op names.

**Example:**
```
branch attention : hidden -> hidden = query | key | value
  merge = softmax_combine
```

---

### `scan`

Declares a catamorphism (scan / left-recursive accumulation).

```
scan <name> : <dom> -> <state>
  step = <op-name>
```

The step op is the recurrent cell, applied at each element.

**Example:**
```
scan rnn : hidden -> hidden
  step = layer
```

---

### `unroll`

Declares an anamorphism (unroll / iterated state transition).

```
unroll <name> : <dom> -> <state>
  step = <op-name>
  steps = <integer>
```

The step op is applied `steps` times starting from an initial state.
Produces a list of intermediate states.

| Attribute | Type    | Default | Meaning |
|-----------|---------|---------|---------|
| `step`    | ident   | —       | State transition op |
| `steps`   | integer | —       | Number of unroll steps |

**Example:**
```
unroll stream : hidden -> hidden
  step = transition
  steps = 10
```

---

### `fixpoint`

Declares a fixpoint iteration (converge-to-epsilon).

```
fixpoint <name> : <sort>
  step = <op-name>
  predicate = <op-name>
  epsilon = <number>
  max_iter = <integer>
```

The step op is iterated until the predicate returns a value ≤ epsilon,
or max_iter is reached. The signature is a single sort (state → state).

| Attribute   | Type   | Default | Meaning |
|-------------|--------|---------|---------|
| `step`      | ident  | —       | State transition op |
| `predicate` | ident  | —       | Convergence check: state → float |
| `epsilon`   | number | 1e-6    | Convergence threshold |
| `max_iter`  | number | 100     | Maximum iterations |

**Example:**
```
fixpoint converge : hidden
  step = step_eq
  predicate = residual_eq
  epsilon = 0.001
  max_iter = 50
```

---

### `lens`

Declares a bidirectional morphism pairing a forward and backward op.

```
lens <name> : <dom> <-> <cod>
  fwd = <op-name>
  bwd = <op-name>
```

Sort constraints: `fwd.domain == bwd.codomain`, `fwd.codomain == bwd.domain`.

**Example:**
```
lens backprop : hidden <-> hidden
  fwd = linear
  bwd = linear_bwd
```

---

### `lens_seq`

Declares a sequential composition of lenses, forming a bidirectional path.

```
lens_seq <name> : <dom> <-> <cod> = <lens1> >> <lens2> >> ... >> <lensN>
```

All lens names must be declared before the `lens_seq`.

**Example:**
```
lens_seq deep_backprop : hidden <-> hidden = backprop1 >> backprop2
```

---

### `lens_branch`

Declares a parallel composition of lenses, forming a bidirectional branch.
Branches are listed inline, separated by `|`.

```
lens_branch <name> : <dom> <-> <cod> = <lens1> | <lens2> | ... | <lensN>
  merge = <merge-lens>
```

`merge` names a lens whose forward op merges branch forward-outputs and whose backward op merges branch backward-outputs.

All lens names must be declared before the `lens_branch`.

The assembled result is two bound terms:
- `ua.branch.<name>.fwd` — forward branch: branch forwards + merge forward
- `ua.branch.<name>.bwd` — backward branch: branch backwards + merge backward

**Example:**
```
lens_branch attention : hidden <-> hidden = backprop1 | backprop2
  merge = merge_lens
```

---

## Declaration Order

The parser resolves names in dependency order:

1. `algebra` — no dependencies
2. `spec` — depends on algebras by name
3. `op` — depends on specs and algebras by name
4. `seq`, `branch`, `scan`, `unroll`, `fixpoint` — depend on ops by name
5. `lens` — depends on ops by name
6. `lens_seq` — depends on lenses by name
7. `lens_branch` — depends on lenses by name

Forward references are not supported. Declare dependencies before use.

---

## Full Example

```
# Algebra
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)

# Specs
spec hidden(real)
spec output(real)

# Ops
op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

op relu : hidden -> hidden
  nonlinearity = relu

op linear_bwd : hidden -> hidden
  einsum = "ji,i->j"
  algebra = real

# Template op
op ~proj : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real

# Seq
seq ffn : hidden -> hidden = linear >> relu

# Residual seq
seq resblock+ : hidden -> hidden = linear >> relu
  algebra = real

# Branch with template instantiation
branch attention : hidden -> hidden = proj[q] | proj[k] | proj[v]
  merge = softmax_combine

# Scan (catamorphism)
scan rnn : hidden -> hidden
  step = layer

# Unroll (anamorphism)
unroll stream : hidden -> hidden
  step = transition
  steps = 10

# Fixpoint
fixpoint converge : hidden
  step = step_eq
  predicate = residual_eq
  epsilon = 0.001
  max_iter = 100

# Lens
lens backprop : hidden <-> hidden
  fwd = linear
  bwd = linear_bwd

# Lens seq
lens_seq deep_backprop : hidden <-> hidden = backprop >> backprop

# Lens branch
lens_branch bidi : hidden <-> hidden = backprop1 | backprop2
  merge = merge_lens
```

---

## Maintenance Protocol

**This file must stay in sync with `src/unialg/parser/_grammar.py`.**

Rules:
1. Any change to `_grammar.py` that adds, removes, or renames a keyword, operator, attribute, or declaration form **must** include a corresponding update to this file in the same commit.
2. Any change to `SYNTAX.md` that adds a new construct **must** be backed by a corresponding parser implementation and tests before merging.
3. When in doubt: `_grammar.py` is ground truth for what parses; `SYNTAX.md` is ground truth for what is intended to parse. Divergence is a bug.

The canonical test suite for parser/syntax alignment is `tests/test_parser.py`.
Run it after any parser change: `uv run --python 3.12 --extra dev python -m pytest tests/test_parser.py -v`.
