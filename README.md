# unialg

unialg is a typed DSL for building algebraic programs as compositional
morphisms, polynomial functors, optics, and recursion schemes. It lowers those
programs to [Hydra](https://github.com/CategoricalData/hydra) terms for
reduction, while native backends such as NumPy can supply concrete primitive
operations at the runtime boundary.

The project is currently a research prototype. The core goal is to make
programs that are usually scattered across tensor code, recursion code, and
architecture glue feel like one algebraic language:

```unialg
load numpy

let f = add(exp, tanh)
```

That program applies `exp` and `tanh` to the same input, then combines the
results with `add`.

## Quick Start

Requires Python 3.12+.

```bash
uv sync --extra dev
uv run pytest -q
```

Compile and run a small backend-backed program:

```python
import numpy as np
from unialg import compile_program

program = compile_program("""
load numpy
let f = add(exp, tanh)
""")

x = np.array([1.0, 2.0, 3.0])
y = program.run(x)
```

The DSL also supports pure structural programs without importing a native
backend:

```python
from unialg.syntax.parse import parse_program

program = parse_program("""
shape NatF = 1 | x
shape Nat = fix NatF

let zero = |0 >> roll[Nat]
let succ = |1 >> roll[Nat]
let one = zero >> succ
""")
```

For the full surface language, see [docs/SYNTAX.md](docs/SYNTAX.md).

## What The DSL Has

Top-level declarations:

```unialg
load numpy
shape NatF = 1 | x
shape Nat = fix NatF
let f = id
let parameterized(step) = step >> step
```

Morphism composition:

```unialg
let sequential = exp >> log
let paired = exp & tanh
let parallel = exp || tanh
let branch = zero | succ
let shared = f >>>> g
```

Polynomial functors and actions:

```unialg
shape MaybeF = 1 | x
let lifted = MaybeF{id}
```

Recursive carriers and recursion schemes:

```unialg
shape NatF = 1 | x
shape Nat = fix NatF

let zero = |0 >> roll[Nat]
let succ = |1 >> roll[Nat]
let folded = cata[Nat](zero | succ)
let unfolded = ana[Nat](coalgebra)
let transformed = hylo[Nat](coalgebra, algebra)
```

Monadic lifting:

```unialg
let safe = pure[Maybe](id)
let many = pure[List](id)
```

## Design

unialg is organized around a strict layer split:

```text
objects.py                 type constructors and monad descriptors
syntax/                    parser and pure syntax nodes
semantics/                 typed morphisms, functors, optics, recursion
structure/                 Hydra term vocabulary and realization
runtime/                   native backend boundary and codecs
main.py                    compile/run orchestration
```

The main compilation path is:

```text
source text
  -> parse_program
  -> construct_program
  -> realize_normalized
  -> Hydra reduction
  -> decoded Python value
```

The parser stays deliberately simple: it builds syntax trees with unresolved
references. Semantic construction resolves names, checks types, constructs full
morphism trees, and only then realization lowers to Hydra terms.

For architecture details, see:

- [docs/LAYER_REFERENCE.md](docs/LAYER_REFERENCE.md)
- [docs/ARCHITECTURE_CONTRACT.md](docs/ARCHITECTURE_CONTRACT.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)

## Current Status

Implemented and tested:

- `let`, `shape`, and `load` program syntax
- typed morphism composition, pairing, parallel product, and case
- parameterized morphisms
- backend primitive loading through runtime specs
- polynomial functor object and arrow action
- unified polynomial optics
- recursive carriers with `roll` and `unroll`
- `cata`, `ana`, and `hylo`
- `Maybe` and `List` monadic lifting
- Hydra lowering and runtime execution boundary

Still evolving:

- shape parameters such as `shape ListF(a) = ...`
- full type interpretation in explicit optic annotations
- tensor equation syntax and semiring-backed contractions
- broader examples and public-facing tutorials

## Testing

```bash
uv run pytest -q
```

Focused test areas:

```bash
uv run pytest tests/syntax -q
uv run pytest tests/semantics -q
uv run pytest tests/properties -q
```

## Research Context

The project is influenced by work on categorical deep learning, Para, optics,
polynomial functors, recursion schemes, semiring semantics, and Hydra's
categorical data model. The practical design constraint is simple: keep the DSL
surface compact while preserving enough categorical structure for programs to
compose, lower, and execute predictably.

Useful references include:

- Brendan Fong, David Spivak, and collaborators on applied category theory
- Gavranovic, *Fundamental Components of Deep Learning*
- Categorical Deep Learning work on Para and compositional architectures
- Optics and lenses as a uniform account of bidirectional structure
- Hehner's *Unified Algebra*, the namesake inspiration for a compact algebraic notation
