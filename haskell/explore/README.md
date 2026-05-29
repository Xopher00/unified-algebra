# Explore — Architecture Exploration and Evaluation Layer

## Purpose

Explore is a verification layer built **on top of** the UniAlg API. It answers
the question: does a categorical architecture specification, lowered through the
UniAlg pipeline, produce code that is behaviorally indistinguishable from the
canonical library implementation of that architecture?

It does this by:
1. Defining architecture specifications as functor algebras/coalgebras (the **seed catalogue**)
2. Lowering them to runnable Python via the UniAlg codegen pipeline
3. Differentially testing the generated code against real library modules
   (`tf.keras.layers.SimpleRNN`, `torch.nn.RNN`, etc.) on random inputs

A differential pass means "no counterexample found on sampled inputs" — not
proof of semantic equality. Only the symbolic law checks (Arm A) yield
"confirmed" results.

## What it contains

```
explore/
├── Explore/                    # Haskell modules (cabal target: explore)
│   ├── Archs.hs               # Architecture functor aliases (SeqF, RTreeF, StreamF, MooreF)
│   ├── Generate.hs            # PolyF classification and seed matching
│   ├── Grammar.hs             # Value-level functor AST and bounded enumeration
│   ├── Laws.hs                # Symbolic law checks (grammar, classification, seed mapping)
│   └── Seed.hs                # Seed catalogue — architecture specs using contractions
├── test/
│   └── ExploreTest.hs         # Test driver: generates modules, runs Arm A + Arm B
├── generated/                  # Generated Python (written by ExploreTest, formatted by black)
│   ├── seed/                   #   numpy backend (default)
│   ├── tf/seed/                #   tensorflow backend
│   └── torch/seed/             #   torch backend
├── backends.py                 # Backend abstraction (TFBackend, TorchBackend)
├── strategies.py               # Hypothesis strategies for random tensor inputs
├── harness.py                  # Arm B: differential tests against library-native modules
├── conftest.py                 # pytest path setup
└── README.md                   # this file
```

## How to run

From `haskell/`:

```bash
cabal test explore-test --test-show-details=direct
```

This runs the full pipeline:
1. **Arm A** — Symbolic law checks on the functor grammar, classification, and seed mapping (34 checks)
2. **Module generation** — Lowers each seed spec to Python for numpy, tensorflow, and torch backends
3. **Arm B** — Hypothesis-driven differential tests against library-native modules

To run only the Python harness directly (after modules are generated):

```bash
PYTHONPATH="explore/generated/tf:explore/generated/torch:explore" \
  ../.venv/bin/python3 -m pytest explore/harness.py -v
```

## Seed catalogue

Each seed is a named architecture from the categorical deep learning literature
(Gavranović et al. 2024), expressed as a functor algebra or coalgebra:

| Seed | Functor F | Form | Architecture | Library native |
|------|-----------|------|-------------|----------------|
| `seqCata` | `1 + A×X` (SeqF) | algebra | Folding RNN (linear) | `tf.keras.layers.SimpleRNN` |
| `seqCataTanh` | `1 + A×X` (SeqF) | algebra | Folding RNN (tanh) | `torch.nn.RNN` |
| `treeCata` | `A + X²` (RTreeF) | algebra | Recursive NN | novel (no library native) |
| `streamAna` | `O×X` (StreamF) | coalgebra | Unfolding RNN | novel |
| `mooreCata` | `O×(I→X)` (MooreF) | coalgebra | Moore machine | novel |

### How the RNN cell works

The folding RNN algebra uses real tensor contractions from `UniAlg.Domain.Tensors`:

```haskell
-- h_t = W_in · x_t + W_rec · h_{t-1} + b          (seqCata, linear)
-- h_t = tanh(W_in · x_t + W_rec · h_{t-1} + b)    (seqCataTanh, for torch)

\case InL (Const ())                    -> s0
      InR (Pair (Const a) (Identity s)) ->
        add (add (contraction "hi,i->h" wIn a)
                 (contraction "hj,j->h" wRec s)) b
```

- `contraction "hi,i->h"` compiles an Einstein notation equation via `applyEquation`
- It lowers to `expand_dims` + `transpose` + `multiply` + `reduce.add` per backend
- `add`, `tanh` remain elementwise — contraction expresses only index summation
- Weights `wIn [hidden, input]`, `wRec [hidden, hidden]`, `b [hidden]` copy
  directly into library modules with no adapter

### Weight alignment

The generated code uses the same weight layout as the library modules:

| Weight | Generated shape | TF SimpleRNN | torch.nn.RNN |
|--------|----------------|-------------|--------------|
| W_in | `[hidden, input]` | `kernel = wIn.T` | `weight_ih_l0 = wIn` |
| W_rec | `[hidden, hidden]` | `recurrent_kernel = wRec.T` | `weight_hh_l0 = wRec` |
| b | `[hidden]` | `bias = b` | `bias_ih_l0 = b` |

TF transposes because SimpleRNN convention is `[input, hidden]` for kernel.

### Fold direction

The generated catamorphism is a **right fold** (processes from tail to head).
Library RNNs are **left folds** (process head to tail). The harness reverses
the input element order before feeding to the library module so that
`left_fold(reversed) == right_fold(original)`.

## Backend abstraction

Tests are backend-polymorphic. Each backend (`TFBackend`, `TorchBackend`)
handles:

- **Tensor generation** — `random_vector`, `random_matrix`, `zeros_vector` produce
  native tensors (numpy for TF, torch.Tensor for torch). No runtime crossing.
- **Structure building** — `make_seq` builds `Left`/`Right` structures with native tensors
- **Reference execution** — `run_reference_rnn` runs the library-native module with
  direct weight copy and reversed elements
- **Comparison** — `allclose` uses the backend's native comparison

Tests are parameterized via `@pytest.fixture(params=[TFBackend(), TorchBackend()])`.

## How to extend

### Adding a new architecture seed

1. Define the functor alias in `Explore/Archs.hs` (if new)
2. Add the seed entry in `Explore/Seed.hs`:
   ```haskell
   mySeed :: SeedEntry
   mySeed = SeedEntry "mySeed" CataArch $
     recModule @(MyF Tensor)
       "seed.mine" "my_func"
       [Namespace "numpy"] ["w1", "w2"] $ \[w1, w2] ->
         ( id
         , \case ... )
   ```
3. Add it to the `seeds` list
4. If it has a library native equivalent, add a reference in `backends.py`
5. If it needs new input shapes, add a strategy in `strategies.py`
6. Add a test class in `harness.py`

### Adding a new backend

1. Subclass `Backend` in `backends.py`
2. Implement `random_vector`, `random_matrix`, `zeros_vector`, `allclose`
3. Implement `load_fold_seq`, `load_fold_tree` (import from the generated module)
4. Implement `run_reference_rnn` (run the library's native module)
5. Add an entry in `ExploreTest.hs` to generate modules for the new backend
6. Add the backend instance to `ALL_BACKENDS` in `harness.py`

### Using contractions in algebras

Use the `contraction` helper for any matrix-vector or matrix-matrix product:

```haskell
contraction "hi,i->h" w x    -- matrix-vector: W @ x
contraction "ij,jk->ik" a b  -- matrix-matrix: A @ B
contraction "bi,i->b" w x    -- batched matvec
```

This calls `applyEquation` from `UniAlg.Domain.Tensors` with the real semiring.
Keep `add`, `tanh`, `multiply` for elementwise operations — contraction expresses
only index summation, never addition or nonlinearity.

### Extending enumeration

`Explore/Grammar.hs` defines a value-level functor grammar (`PolyF`) with bounded
enumeration. To explore beyond the seed set:

1. Call `enumerate depth` to get all `PolyF` up to the given depth
2. For each, `classifyPolyF` determines the architecture class
3. The type-application bridge (connecting value-level `PolyF` to compile-time
   `recModule @T`) is currently hand-paired in `Seed.hs`. General depth-N
   enumeration would require a Template Haskell splice — documented in
   `Generate.hs` but not yet built.
