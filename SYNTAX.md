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
| `~>`  | Merge chain — sequences merge steps in `branch` declarations (stack-machine semantics) |
| `->`  | Sort signature separator (`dom -> cod`) |
| `<->` | Bidirectional sort signature (`dom <-> cod`, lenses only) |
| `:`   | Type annotation — separates a declaration name from its signature |
| `=`   | Assignment — separates a name or key from its value |
| `~`   | Template marker — prefix on `op` name in a declaration (marks op as template); also a call-site prefix on op references (`~name`) to create a fresh instance with unique weights |
| `*`   | Adjoint marker — suffix on an op reference at a call site; invokes that step using the adjoint contraction. Requires the algebra to declare `residual=`. |
| `+`   | Residual/skip marker — suffix on a `seq` name (declaration-time skip over whole seq); also a call-site suffix on op references (`name+`) for a per-step skip connection: `output = op(x) ⊕ x` using the semiring's plus |
| `\|`  | Branch separator — inline branch list in `branch` and `lens_branch` declarations |
| `&`   | Bimap separator — separates the two component ops in a `parallel` declaration |
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

### `define`

Declares a custom operation inline as an expression over existing backend operations.

```
define unary <name>(<param>) = <expr>
define binary <name>(<param1>, <param2>) = <expr>
```

The expression language supports:
- **Literals**: numbers (`0.0`, `1.0`, `-inf`)
- **Variables**: the declared parameter names
- **Function calls**: `fn(arg1, arg2)` — names resolve against the backend's unary (1-arg) or binary (2-arg) operation tables
- **Infix operators**: `+` `-` `*` `/` desugar to `add`/`subtract`/`multiply`/`divide` with standard precedence
- **Parenthesized grouping**: `(expr)`
- **Unary minus**: `-expr` desugars to `neg(expr)`

Unary defines register as nonlinearities (usable in `op` declarations). Binary defines register as both elementwise and reduction operations (usable as semiring `plus`/`times` in `algebra` declarations).

`define` declarations must appear before any `algebra` or `op` that references them. Multiple defines are allowed; later defines may reference earlier ones.

**Examples:**
```
define unary clamp(x) = minimum(1.0, maximum(0.0, x))
define binary smooth_max(a, b) = log(exp(a) + exp(b))
define unary leaky_relu(x) = maximum(0.0, x) * 0.1 + x * 0.9
```

**Semiring usage:**
```
define binary smooth_max(a, b) = log(exp(a) + exp(b))
algebra lse(plus=smooth_max, times=add, zero=-inf, one=0.0)
```

---

### `import`

Specifies the backend for the program. Must appear before any other declaration. Available backends: `numpy`, `jax`, `pytorch`, `cupy`.

```
import <backend-name>
```

When `import` is present, `parse_ua(text)` can be called without a backend argument. A backend passed explicitly to `parse_ua(text, backend)` overrides the import.

**Example:**
```
import numpy
```

---

### `algebra`

Declares a semiring with named binary operations and identity elements.

```
algebra <name>(plus=<ident>, times=<ident>, zero=<num>, one=<num>[, strategy=<ident>][, residual=<ident>])
```

The backend must provide functions named by `plus` and `times`.
`zero` and `one` are the additive and multiplicative identities.

The optional `strategy` names a contraction hook registered in `CONTRACTION_REGISTRY`. When present, it replaces the default single-pass algorithm (align → ⊗ elementwise → ⊕ reduce) with the named implementation. Register hooks via `CONTRACTION_REGISTRY["name"] = fn` in Python before the program runs. The hook signature is `fn(compute_sum, backend, params) → tensor`.

The legacy `contraction=` key is equivalent and still accepted; `strategy=` takes precedence if both are given.

The optional `residual` names a binary operation that serves as the algebraic residual (right adjoint of `times`). When an `op` with `adjoint = true` is resolved against this algebra, the contraction uses `residual` elementwise and `times_reduce` instead of the forward `times`/`plus_reduce` pair. Both `strategy` and `residual` are optional and independent; a user may supply either, both, or neither.

**Examples:**
```
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
algebra fuzzy(plus=maximum, times=minimum, zero=0.0, one=1.0)
algebra logprob(plus=logaddexp, times=add, zero=-inf, one=0.0, strategy=my_hook)
algebra fuzzy_residuated(plus=maximum, times=minimum, zero=0.0, one=1.0, residual=godel_impl)
```

---

### `spec`

Declares a named tensor type associated with an algebra.

```
spec <name>(<algebra-name>)
spec <name>(<algebra-name>, batched)
spec <name>(<algebra-name>[, batched], axes=[<axis>, ...])
```

Where `<axis>` is either `<ident>` (unsized) or `<ident>:<integer>` (sized).

`batched` adds a leading independent batch dimension at resolution time.

The optional `axes` declares ordered named axes for the sort. Each axis is a name optionally followed by `:` and an integer dimension size. When declared:
- Einsum output rank must match the codomain sort's axis count.
- Einsum last-input rank must match the domain sort's axis count (earlier inputs are weight parameters).
- Graph edges validate axis name compatibility between connected sorts.
- When both sides of a graph edge declare a size for the same axis, the sizes must match. Unsized axes skip the dimension check.

Sized and unsized axes may be mixed freely (e.g., `axes=[batch, feature:128]`). Axis order corresponds to einsum subscript position. When `batched` is set, declared axes do NOT include the batch dimension — it is prepended automatically.

Sorts without `axes` skip all axis validation (backward compatible).

**Examples:**
```
spec hidden(real)
spec output(real)
spec hidden_batched(real, batched)
spec hidden(real, axes=[batch, feature])
spec hidden(real, axes=[batch, feature:128])
spec output(real, axes=[batch, classes:10])
spec hidden_batched(real, batched, axes=[feature:128])
```

---

### `op`

Declares a morphism: a typed tensor operation.

```
op <name> : <dom> -> <cod>
  einsum = "<subscript>"
  algebra = <algebra-name>
```

Three call-site suffixes/prefixes modify how an op is invoked inside `seq` chains and `branch` lists. The op itself is declared normally; the annotation appears only at the reference.

| Annotation | Syntax | Effect | Synthetic name |
|---|---|---|---|
| Adjoint | `op*` | Invokes via residual contraction: `residual_elementwise` + `times_reduce`. Requires `residual=` on the algebra. | `op__adj` |
| Skip (residual) | `op+` | Wraps op with a skip connection: `output = op(x) ⊕ x` using the semiring's `plus`. | `op__res` |
| Fresh instance | `~op` | Creates a distinct copy of the op with a unique name (separate weight tensor). Auto-numbered: first `~op` → `op__0`, second → `op__1`, … | `op__0`, `op__1`, … |

All three can be combined: `~op*` = fresh adjoint instance; `~op+` = fresh with skip; `~op[prefix]` = fresh via template prefix.

```
seq adj_path : mat -> mat = residuate*
seq skip_path : hidden -> hidden = linear+
seq heads : hidden -> hidden = ~proj[q] | ~proj[k] | ~proj[v]
```

Each annotation creates and registers a synthetic equation on first use; subsequent uses of the same annotation reuse it (except `~` which always creates a fresh one).

The codomain `<cod>` can be a single sort name or a **product sort** `(<sort1>, <sort2>[, ...])` for multi-value output (e.g., Viterbi returning values + argmax indices):

```
op <name> : <dom> -> (<sort1>, <sort2>)
  einsum = "<subscript>"
  algebra = <algebra-name>
```

Product sort codomains create a `ProductSort` — a right-nested Hydra pair type. The contraction hook (via `contraction_fn` on the algebra) returns a tuple matching the product structure.

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
| `inputs` | ident list | Comma-separated list of upstream op names whose outputs feed into this op. Establishes typed DAG edges for sort compatibility validation and topological ordering. Example: `inputs = linear` or `inputs = query, key`. |

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
Names in the chain can be op names, template refs (`name[prefix]`), or composition names (`branch`, `seq`, etc.). All referenced names must be declared before the `seq`.

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

# Composing compositions: seq referencing a branch
seq block : hidden -> hidden = attn_head >> ffn
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

#### Merge chains

Multiple merge steps can be chained with `~>` to interleave contractions and nonlinearities:

```
branch <name> : <dom> -> <cod> = <op1> | <op2> | ... | <opN>
  merge = <step1> ~> <step2> ~> ... ~> <stepK>
```

Each step is either a declared `op` name or a bare nonlinearity name (resolved via the backend). Steps execute as a stack machine over the branch outputs:
- Branch outputs form the initial stack (leftmost = top)
- Each step consumes N elements from the top (N = einsum operand count for ops, 1 for nonlinearities)
- Each step produces 1 output, pushed to top; remaining elements carry through
- The chain must reduce the stack to exactly 1 element

`~>` vs `>>`: `>>` is point-to-point piping (single tensor in/out). `~>` is stack-machine sequencing over multiple tensors, where unconsumed tensors carry through to later steps.

**Examples:**
```
branch pair : hidden -> hidden = relu | tanh_act
  merge = hadamard

branch head : hidden -> hidden = proj[q] | proj[k] | proj[v]
  merge = score ~> softmax ~> mix
```

In the second example, `softmax` is a bare nonlinearity name — no `op` declaration needed. Stack trace: `[Q,K,V] → score(Q,K) → [scores,V] → softmax(scores) → [probs,V] → mix(probs,V) → [output]`.

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
  residual = <sort-name>
```

Sort constraints: `fwd.domain == bwd.codomain`, `fwd.codomain == bwd.domain`.

The optional `residual` attribute names a sort for auxiliary data carried from the forward pass to the backward pass (e.g., backpointer tables for Viterbi, stabilizers for logsumexp). When any lens in a `lens_seq` declares a residual, the composition uses optic threading: the forward pass collects residuals into a list, and the backward pass consumes them in reverse order.

**Examples:**
```
lens backprop : hidden <-> hidden
  fwd = linear
  bwd = linear_bwd

lens viterbi_step : hidden <-> hidden
  fwd = viterbi_fwd
  bwd = viterbi_bwd
  residual = indices
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
lens_branch bidi_pair : hidden <-> hidden = backprop1 | backprop2
  merge = merge_lens
```

---

### `parallel`

Declares a bimap / monoidal product of two morphisms: routes the two components of a pair independently.

Categorical reading: `f ⊗ g : (A, B) → (C, D)`

```
parallel <name> : <dom> -> <cod> = <left-op> & <right-op>
```

`left-op` receives the first component of the input pair; `right-op` receives the second component. Both must be declared ops. `dom` and `cod` are sort names — they annotate the input/output types of the composition. The result is a new composed op that takes a Python tuple `(a, b)` and returns `(left-op(a), right-op(b))`.

Both named ops must already be declared before the `parallel` declaration.

The assembled result is a single bound term `ua.parallel.<name>` (native path) or a Hydra primitive that wraps `hydra.lib.pairs.bimap` (term fallback path).

**Example:**
```
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec feat(real)
spec label(real)

op encode : feat -> feat
  nonlinearity = relu

op decode : label -> label
  nonlinearity = relu

parallel bimap_pair : feat -> label = encode & decode
```

---

## Declaration Order

The parser resolves names in dependency order:

0. `define` — no dependencies (uses built-in backend ops or earlier defines)
1. `algebra` — may reference define names for plus/times
2. `spec` — depends on algebras by name
3. `op` — depends on specs, algebras, and optionally define names for nonlinearity
4. `seq`, `branch`, `parallel`, `scan`, `unroll`, `fixpoint` — depend on ops by name
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
spec hidden(real, axes=[feature])
spec output(real, axes=[classes])

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
branch multi_head : hidden -> hidden = proj[q] | proj[k] | proj[v]
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

The canonical test suite for parser/syntax alignment is `tests/unit/test_parser.py` and `tests/negative/test_parser_errors.py`.
Run them after any parser change: `uv run --python 3.12 --extra dev python -m pytest tests/unit/test_parser.py tests/negative/test_parser_errors.py -v`.
