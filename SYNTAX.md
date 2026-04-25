# Unified Algebra DSL — Syntax Reference

This document is the authoritative lexicon for the `.ua` DSL.
It is derived from `src/unified_algebra/parser.py` and must be kept in sync with it.

---

## Whitespace and Comments

- **Inline whitespace** (spaces, tabs) is ignored around tokens.
- **Newlines** are significant — they terminate declarations and attribute lines.
- **Blank lines** (empty, whitespace-only, or comment-only) are ignored everywhere.
- **Comments** begin with `#` and extend to end of line.

```
# this is a comment
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)  # inline comment
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
| `>>`  | Sequential composition (path chain, lens_path chain) |
| `->`  | Sort signature separator (`dom -> cod`) |
| `<->` | Bidirectional sort signature (`dom <-> cod`, lenses only) |
| `:`   | Type annotation — separates a declaration name from its signature |
| `=`   | Assignment — separates a name or key from its value |
| `(`   | Open argument list |
| `)`   | Close argument list |
| `[`   | Open branch list |
| `]`   | Close branch list |
| `,`   | Separator in argument lists and branch lists |

---

## Declaration Forms

Declarations are separated by newlines (blank lines and comments allowed between them).
Attribute blocks are indented with at least one space or tab.

---

### `semiring`

Declares a semiring with named binary operations and identity elements.

```
semiring <name>(plus=<ident>, times=<ident>, zero=<num>, one=<num>)
```

The backend must provide functions named by `plus` and `times`.
`zero` and `one` are the additive and multiplicative identities.

**Examples:**
```
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
semiring tropical(plus=minimum, times=add, zero=inf, one=0.0)
semiring fuzzy(plus=maximum, times=minimum, zero=0.0, one=1.0)
```

---

### `sort`

Declares a named tensor type associated with a semiring.

```
sort <name>(<semiring-name>)
sort <name>(<semiring-name>, batched)
```

`batched` adds a leading independent batch dimension at resolution time.

**Examples:**
```
sort hidden(real)
sort output(real)
sort hidden_batched(real, batched)
```

---

### `equation`

Declares a morphism: a typed tensor operation.

```
equation <name> : <dom> -> <cod>
  einsum = "<subscript>"
  semiring = <semiring-name>
```

For nonlinear (pointwise) equations, use `nonlinearity` instead of `einsum`:

```
equation <name> : <dom> -> <cod>
  nonlinearity = <ident>
```

Attributes are indented key-value pairs. `einsum` and `nonlinearity` are mutually exclusive.
`semiring` is required for einsum equations; omit it for nonlinearities.

| Attribute | Type | Meaning |
|---|---|---|
| `einsum` | string | Einsum subscript string |
| `semiring` | ident | Semiring name (required for einsum equations) |
| `nonlinearity` | ident | Pointwise operation name |
| `template` | boolean | If `true`, this equation is a template for parametric instantiation |

**Examples:**
```
equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real

equation relu : hidden -> hidden
  nonlinearity = relu

equation softmax : hidden -> output
  nonlinearity = softmax
```

#### Template equations

An equation declared with `template = true` is a **template**: it is not registered as a concrete equation itself, but can be instantiated with a prefix to produce distinct concrete equations with separate weight keys in the Hydra graph.

**Declaration:**
```
equation proj : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real
  template = true
```

**Instantiation** — use `name[prefix]` in a `path` chain or `fan` branch list:
```
path qk : hidden -> hidden = proj[q] >> proj[k]

fan kv : hidden -> hidden
  branches = [proj[q], proj[k], proj[v]]
  merge = add_merge
```

Each `proj[q]`, `proj[k]`, `proj[v]` expands to concrete equations `q_proj`, `k_proj`, `v_proj` respectively, sharing the same einsum/sorts/semiring but with distinct names (= distinct weight tensors at runtime).

Rules:
- Template refs may appear anywhere plain equation names appear in `path` `>>` chains and `fan` `branches = [...]` lists.
- Using the same template ref twice in the same program (e.g. `proj[q] >> proj[q]`) creates exactly one concrete equation — not a duplicate.
- Referencing an undeclared template name raises `ValueError`.
- Template equations do not appear in `spec.equations`; only their concrete instantiations do.

---

### `path`

Declares a sequential composition of equations, applied left-to-right.

```
path <name> : <dom> -> <cod> = <eq1> >> <eq2> >> ... >> <eqN>
```

`eq1` receives the input; `eqN` produces the output.
All equation names must be declared before the path. Template refs (`name[prefix]`) may appear in the chain alongside plain equation names.

An optional indented attribute block may follow the path chain:

| Attribute   | Type    | Meaning |
|-------------|---------|---------|
| `residual`  | boolean | If `true`, adds the original input back via the semiring's plus: `output = path(x) ⊕ x` |
| `semiring`  | ident   | Name of the semiring whose `plus` operation performs the skip connection. Required when `residual = true`. |

**Examples:**
```
path layer : hidden -> hidden = linear >> relu
path ffn : hidden -> hidden = linear1 >> relu1 >> linear2 >> relu2 >> linear3

# Residual / skip connection
path resblock : hidden -> hidden = linear >> relu
  residual = true
  semiring = real
```

---

### `fan`

Declares a parallel composition: branches applied to the same input, then merged.

```
fan <name> : <dom> -> <cod>
  branches = [<eq1>, <eq2>, ..., <eqN>]
  merge = <merge-eq>
```

All branch equations receive the same input. The merge equation receives a list of branch outputs. Template refs (`name[prefix]`) may appear in the branch list alongside plain equation names.

**Example:**
```
fan attention : hidden -> hidden
  branches = [query, key, value]
  merge = softmax_combine
```

---

### `fold`

Declares a catamorphism (fold / left-recursive accumulation).

```
fold <name> : <dom> -> <state>
  step = <eq-name>
```

The step equation is the recurrent cell, applied at each element.

**Example:**
```
fold rnn : hidden -> hidden
  step = layer
```

---

### `unfold`

Declares an anamorphism (unfold / iterated state transition).

```
unfold <name> : <dom> -> <state>
  step = <eq-name>
  n_steps = <integer>
```

The step equation is applied `n_steps` times starting from an initial state.
Produces a list of intermediate states.

**Example:**
```
unfold stream : hidden -> hidden
  step = transition
  n_steps = 10
```

---

### `fixpoint`

Declares a fixpoint iteration (converge-to-epsilon).

```
fixpoint <name> : <sort>
  step = <eq-name>
  predicate = <eq-name>
  epsilon = <number>
  max_iter = <integer>
```

The step equation is iterated until the predicate returns a value ≤ epsilon,
or max_iter is reached. The signature is a single sort (state → state).

| Attribute   | Type   | Default | Meaning |
|-------------|--------|---------|---------|
| `step`      | ident  | —       | State transition equation |
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

Declares a bidirectional morphism pairing a forward and backward equation.

```
lens <name> : <dom> <-> <cod>
  fwd = <eq-name>
  bwd = <eq-name>
```

Sort constraints: `fwd.domain == bwd.codomain`, `fwd.codomain == bwd.domain`.

**Example:**
```
lens backprop : hidden <-> hidden
  fwd = linear
  bwd = linear_bwd
```

---

### `lens_path`

Declares a sequential composition of lenses, forming a bidirectional path.

```
lens_path <name> : <dom> <-> <cod> = <lens1> >> <lens2> >> ... >> <lensN>
```

All lens names must be declared before the `lens_path`.

**Example:**
```
lens_path deep_backprop : hidden <-> hidden = backprop1 >> backprop2
```

---

### `lens_fan`

Declares a parallel composition of lenses, forming a bidirectional fan.

```
lens_fan <name> : <dom> <-> <cod>
  branches = [<lens1>, <lens2>, ..., <lensN>]
  merge = <merge-lens>
```

`branches` lists lens names applied in parallel to the same input.
`merge` names a lens whose forward equation merges branch forward-outputs and whose backward equation merges branch backward-outputs.

All lens names must be declared before the `lens_fan`.

The assembled result is two bound terms:
- `ua.fan.<name>.fwd` — forward fan: branch forwards + merge forward
- `ua.fan.<name>.bwd` — backward fan: branch backwards + merge backward

**Example:**
```
lens_fan attention : hidden <-> hidden
  branches = [backprop1, backprop2]
  merge = merge_lens
```

---

## Declaration Order

The parser resolves names in dependency order:

1. `semiring` — no dependencies
2. `sort` — depends on semirings by name
3. `equation` — depends on sorts and semirings by name
4. `path`, `fan`, `fold`, `unfold`, `fixpoint` — depend on equations by name
5. `lens` — depends on equations by name
6. `lens_path` — depends on lenses by name
7. `lens_fan` — depends on lenses by name

Forward references are not supported. Declare dependencies before use.

---

## Full Example

```
# Semiring
semiring real(plus=add, times=multiply, zero=0.0, one=1.0)
semiring tropical(plus=minimum, times=add, zero=inf, one=0.0)

# Sorts
sort hidden(real)
sort output(real)

# Equations
equation linear : hidden -> hidden
  einsum = "ij,j->i"
  semiring = real

equation relu : hidden -> hidden
  nonlinearity = relu

equation linear_bwd : hidden -> hidden
  einsum = "ji,i->j"
  semiring = real

# Path
path ffn : hidden -> hidden = linear >> relu

# Lens
lens backprop : hidden <-> hidden
  fwd = linear
  bwd = linear_bwd

# Lens path
lens_path deep_backprop : hidden <-> hidden = backprop >> backprop
```

---

## Maintenance Protocol

**This file must stay in sync with `src/unified_algebra/parser.py`.**

Rules:
1. Any change to `parser.py` that adds, removes, or renames a keyword, operator, attribute, or declaration form **must** include a corresponding update to this file in the same commit.
2. Any change to `SYNTAX.md` that adds a new construct **must** be backed by a corresponding parser implementation and tests before merging.
3. When in doubt: `parser.py` is ground truth for what parses; `SYNTAX.md` is ground truth for what is intended to parse. Divergence is a bug.

The canonical test suite for parser/syntax alignment is `tests/test_parser.py`.
Run it after any parser change: `uv run --python 3.12 --extra dev python -m pytest tests/test_parser.py -v`.
