# Explore — Architecture Verification Layer

This layer lets you express a machine-learning architecture as a categorical
specification (a functor algebra or coalgebra), lower it through the UniAlg
pipeline to runnable Python, and verify the result against a canonical library
implementation — all in one command.

If you want to experiment with architectures from the categorical deep learning
literature (Gavranović et al. 2024), or verify that a new specification round-trips
through code generation correctly, this is where you work.

---

## How it works

There are two distinct phases, called Arm A and Arm B.

**Arm A — symbolic law checks** run entirely in Haskell. They verify properties of
the functor grammar and the seed catalogue using symbolic term reduction: things like
"the functor F(X) = 1 + A×X has arity 1" or "the seed `seqCata` maps to the
CataArch architecture class." These checks are confirmations — if they pass, the
claim holds by construction, not by sampling.

**Arm B — differential testing** uses the Haskell pipeline to generate Python code
for each architecture, then runs a pytest harness that compares the generated
functions against library-native implementations (TensorFlow, PyTorch) on random
inputs drawn by Hypothesis. A pass means "no counterexample found" — it is strong
evidence of correctness, not a proof.

Both arms run in sequence when you run:

```bash
cabal test explore-test --test-show-details=direct
```

---

## Directory layout

```
explore/
├── ExploreTest.hs          # Main test driver (Haskell entry point)
├── gen-catalogue.hs        # Script to regenerate Catalogue.hs (see below)
├── conftest.py             # pytest setup — discovers arch.py files automatically
│
├── archs/                  # One subdirectory per architecture
│   ├── seq_rnn/            # Folding RNN — F(X) = 1 + (A × X)
│   │   ├── SeqRnn.hs       #   Haskell: functor definition, seed entries
│   │   ├── arch.py         #   Python: backends, reference impl, Hypothesis tests
│   │   ├── __init__.py     #   (empty — required so pytest treats this as a package)
│   │   └── generated/      #   Generated Python (written on each test run)
│   │       ├── numpy/      #     numpy backend
│   │       ├── tensorflow/ #     TensorFlow backend
│   │       └── torch/      #     PyTorch backend
│   ├── tree_rnn/           # Recursive NN — F(X) = A + (X × X)
│   ├── stream_rnn/         # Unfolding RNN — F(X) = A × X
│   └── moore/              # Moore machine — F(X) = O × (I → X)
│
└── support/
    ├── haskell/            # Shared Haskell modules (part of the explore library)
    │   ├── Catalogue.hs    # GENERATED — list of all arch seeds (do not edit)
    │   ├── Seed.hs         # SeedEntry type and contraction helper
    │   ├── Laws.hs         # Symbolic law checks (Arm A)
    │   ├── Grammar.hs      # Value-level functor AST and enumeration
    │   └── Generate.hs     # PolyF classification and seed matching
    └── python/
        └── backends.py     # Backend abstraction (TFBackend, TorchBackend, NumpyBackend)
```

---

## The architectures

Each architecture is defined by a polynomial endofunctor `F`. The fixed point of `F`
gives you the recursive data structure the architecture folds or unfolds over. The
seed catalogue currently includes four:

| Architecture | Functor F(X) | Kind | Library native |
|---|---|---|---|
| `seqCata` | `1 + (A × X)` | catamorphism (fold) | `tf.keras.layers.SimpleRNN(activation='linear')` |
| `seqCataTanh` | `1 + (A × X)` | catamorphism (fold) | `torch.nn.RNN(nonlinearity='tanh')` |
| `treeCata` | `A + (X × X)` | catamorphism (fold) | no exact library native — constrained linear invariant |
| `streamAna` | `A × X` | anamorphism (unfold) | no library native — structural test only |
| `mooreCata` | `O × (I → X)` | anamorphism (unfold) | no library native — structural test only |

`seqCata` and `seqCataTanh` are the same functor shape with different nonlinearities.
They are tested against different backends because TF's SimpleRNN uses linear
activation and PyTorch's RNN uses tanh.

### How the RNN algebra is written

In `SeqRnn.hs`, the algebra is expressed as a case match over the functor's data:

```haskell
\case
  InL (Const ())                    -> s0            -- empty list: return initial state
  InR (Pair (Const a) (Identity s)) ->               -- cons cell: apply RNN step
    add (add (contraction real "hi,i->h" wIn a)
             (contraction real "hj,j->h" wRec s)) b
```

- `InL / InR` correspond to the sum `1 + (A × X)` — the two constructors of the list functor
- `contraction "hi,i->h" wIn a` is Einstein summation: computes `h_t = W_in · x_t`
- The semiring `real` specifies that addition is `+` and multiplication is `×`

This expression lowers through UniAlg to concrete tensor ops in each backend.

### Fold direction and weight alignment

The generated catamorphism is a right fold (from tail to head). Library RNNs are
left folds (from head to tail). The `arch.py` reference implementations reverse the
input sequence before passing it to the library module so both folds agree.

Weight layout in the generated code matches each library's convention directly:

| Weight | Generated shape | TF kernel | PyTorch |
|---|---|---|---|
| `wIn` | `[hidden, input]` | transposed to `[input, hidden]` | used as-is |
| `wRec` | `[hidden, hidden]` | transposed to `[hidden, hidden]` | used as-is |
| `b` | `[hidden]` | used as-is | used as `bias_ih`; `bias_hh` zeroed |

---

## What happens when you run the tests

1. **Seed smoke check** — every seed in the catalogue generates non-empty Python with
   a `def` statement. Basic pipeline sanity.

2. **Moore gate** — the Moore machine seed is lowered and evaluated. This exercises
   the Exp functor path and checks that the pipeline produces a callable.

3. **Arm A — symbolic laws** — 14 checks covering functor grammar enumeration, arity,
   classification, and seed-to-class mapping. All run in Haskell with no randomness.

4. **Module generation** — for each architecture and each backend (numpy, tensorflow,
   torch), the Haskell pipeline generates a Python module and writes it to
   `archs/<arch>/generated/<backend>/`.

5. **black formatting** — the generated Python is formatted in place.

6. **Arm B — differential tests** — pytest discovers every `arch.py` under `archs/`
   and runs the test classes inside. Each test uses Hypothesis to draw random weight
   matrices and input sequences, calls both the generated function and the
   library-native reference, and asserts `allclose` within a tolerance of `1e-4`.

---

## The backend system

`backends.py` defines a `Backend` base class with three concrete subclasses:

| Backend | Framework | Used by |
|---|---|---|
| `NumpyBackend` | numpy | `stream_rnn`, `moore` (structural tests only) |
| `TFBackend` | TensorFlow | `seq_rnn` (linear), `tree_rnn` |
| `TorchBackend` | PyTorch | `seq_rnn` (tanh), `tree_rnn` |

Each backend owns the full tensor lifecycle for its tests — tensor creation, random
input generation, the reference call, and the `allclose` comparison. Tensors never
cross backends during a test.

`backend.framework` gives you the lazily imported native library (`tf`, `torch`, or
`numpy`) when you need to call library-specific APIs in a reference implementation.

`BackendSpec` is the glue object in `arch.py` that binds a backend to a specific
generated module and an optional reference callable:

```python
BackendSpec(
    TFBackend(),
    module="seed.seq",       # resolves to generated/tensorflow/seed/seq.py
    fn="fold_seq",           # function name in that module
    reference=_tf_reference  # callable to compare against, or None
)
```

When `reference=None`, the test only checks that the generated function runs and
returns finite values (structural test). When `reference` is provided, the test
additionally asserts numerical agreement against the library-native output.

---

## Adding a new architecture

Here is the complete workflow. As a running example, assume you are adding a Mealy
machine with functor `F(X) = O × (I → X)`.

### 1. Write the Haskell seed

Create `explore/archs/mealy/Mealy.hs`:

```haskell
module Mealy (MealyF, mealyCata, backendSeeds) where

import UniAlg
import Seed (SeedEntry(..), ArchClass(..))

type MealyF o i = Product (Const (TTerm o)) (Exp (TTerm i))

mealyCata :: SeedEntry
mealyCata = SeedEntry "mealyCata" AnaArch $
  anaModule @(MealyF Tensor Tensor)
    "seed.mealy" "mealy_step"
    [Namespace "numpy"] [] $ \[] ->
      \s -> (s, \_inp -> s)

backendSeeds :: [(String, SeedEntry)]
backendSeeds = [("numpy", mealyCata)]
```

### 2. Register in cabal

In `unialg.cabal`, under `library explore`, add two lines:

```cabal
  hs-source-dirs:
    ...
    explore/archs/mealy       -- new source directory

  exposed-modules:
    ...
    Mealy                     -- new module name
```

### 3. Regenerate the catalogue

From `haskell/`:

```bash
runghc explore/gen-catalogue.hs
```

This scans `explore/archs/`, derives module names from directory names
(snake_case → PascalCase), and rewrites `support/haskell/Catalogue.hs`. You never
edit `Catalogue.hs` by hand.

### 4. Write the Python test

Create `explore/archs/mealy/arch.py` and an empty `explore/archs/mealy/__init__.py`:

```python
"""
Mealy machine — MealyF ana.

Functor:   F(X) = O × (I → X)
Structural test only.
"""
import numpy as np
import pytest
from backends import BackendSpec, NumpyBackend, arch_generated_root

GENERATED_ROOT = arch_generated_root(__file__)

BACKENDS = [
    BackendSpec(NumpyBackend(), module="seed.mealy", fn="mealy_step", reference=None)
]

@pytest.fixture(params=BACKENDS, ids=lambda s: s.backend.name)
def spec(request):
    return request.param

class TestMealy:
    def test_output_structure(self, spec):
        mealy_step = spec.load(GENERATED_ROOT)
        pair = (np.float64(1.0), lambda inp: np.float64(inp + 1.0))
        result = mealy_step(pair)
        assert isinstance(result, tuple) and len(result) == 2
```

For architectures with a library-native equivalent, provide a `reference` callable
that accepts `(backend, *inputs)` and returns a tensor using `backend.framework`.
See `seq_rnn/arch.py` for a full example with TensorFlow and PyTorch.

### 5. Run

```bash
cabal test explore-test --test-show-details=direct
```

The new arch is picked up automatically in both Arm A (via the catalogue) and Arm B
(via pytest's `arch.py` file discovery). No further changes are needed.

---

## Adding a new backend

1. Subclass `Backend` in `support/python/backends.py` and implement:
   - `random_vector(draw, dim)` — returns a native tensor of shape `(dim,)`
   - `random_matrix(draw, rows, cols)` — returns a native tensor of shape `(rows, cols)`
   - `fill_vector(dim, value)` — returns a constant tensor (used for zero initial states)
   - `allclose(a, b, atol)` — backend-native approximate equality
   - `is_finite(tensor)` — True if all elements are finite
   - `framework` property — lazily imports and returns the native library object

2. Add it to the `BACKENDS` list in any `arch.py` that should run on the new backend.

3. Add the backend name to `backendSeeds` in the relevant Haskell arch module so the
   Haskell pipeline generates a Python module for it.

No changes to the Haskell pipeline are needed — backends are a pure Python concern.

---

## Interpreting test results

**Arm A passes** → the categorical specification is internally consistent: the functor
has the expected arity, the architecture class is correctly identified, and each seed
maps to its declared class.

**Arm B passes** → no counterexample found on the sampled inputs. Because Hypothesis
shrinks failures to minimal examples, a failure message shows the exact weights and
input sequence that caused the mismatch, making it straightforward to trace which
generated op differs from the reference.

**Structural tests pass** (reference=None) → the generated function exists, runs
without error, and returns finite values. No claim is made about numerical equivalence
with any external library.

A complete pass looks like:

```
=== All explore gates: PASSED ===
Test suite explore-test: PASS
```

---

## Common tasks

**Run only the Python harness** (after modules are already generated):

```bash
cd haskell && .venv/bin/python -m pytest explore/archs/ -v --tb=short
```

**Run only one architecture's tests:**

```bash
.venv/bin/python -m pytest explore/archs/seq_rnn/ -v
```

**Inspect a generated module** — after a test run, generated Python lives at:

```
explore/archs/<arch>/generated/<backend>/seed/<module>.py
```

For example: `explore/archs/seq_rnn/generated/tensorflow/seed/seq.py`
