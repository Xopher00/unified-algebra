# unified-algebra

A Python DSL for expressing any machine learning architecture as morphisms
between typed tensors. The algebra is parameterised by a semiring V — swapping
V changes the semantics (real arithmetic, tropical/max-plus, fuzzy,
probabilistic, etc.). The DSL compiles to
[Hydra](https://github.com/CategoricalData/hydra) terms/types directly — there
is no separate AST or translation layer.

**Docs:** [DSL syntax reference](SYNTAX.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)

## Goals

- Any ML architecture expressible as compositions of typed tensor equations
- Hydra-first: DSL terms and types are Hydra `Term`/`Type` — one AST, no translation layer
- Backend-agnostic: same architecture runs on numpy, PyTorch, JAX, or CuPy
- Semiring-parameterised: swap V to change semantics without changing the architecture
- Bidirectional: lenses pair forward and backward equations for gradient computation, path recovery, or any semiring-parameterised adjoint

## Current status

366 tests passing. Implemented capabilities:

- **Backend** — `Backend` ABC with concrete subclasses `NumpyBackend`, `PytorchBackend`,
  `JaxBackend`, `CupyBackend`. Provides binary ops (elementwise + reduction),
  unary ops, structural ops, and `compile()` / `while_loop()` hooks for JIT/iteration.
  Users can extend with custom ops at runtime.
- **Semiring** — user-defined semirings as Hydra record terms (real, tropical, fuzzy, logaddexp, etc.).
  Auto-validates the 7 semiring axioms at `resolve()` time using a user-supplied `(bottom, top)`
  domain — invalid declarations are rejected before they reach the contraction engine.
- **Contraction** — generalised einsum over arbitrary semirings with optional blocked
  (chunked) execution for memory efficiency. `CompiledEinsum` is the structured
  representation; batch dimensions are inserted via `prepend_batch_var()` (no string surgery).
- **Sorts** — named tensor types bound to a semiring, with optional batching (`Sort("hidden", sr, batched=True)`).
  Batch dimensions are auto-prepended at resolution time. Sort/rank junction checking at composition boundaries.
- **Equations** — the unified construct: simultaneously a morphism and a tensor equation.
  Supports `param_slots` for user-defined parametric nonlinearities (e.g. temperature-scaled softplus).
- **Composition** — sequential (`path`) and parallel (`fan`, list-based, unbounded arity).
  Validated for sort junction consistency.
- **Recursion** — fold (catamorphism) via Hydra's `foldl`, unfold (anamorphism) via custom `unfold_n`.
  Weight tying is automatic.
- **Parameter tying** — shared parameters across morphisms declared via the `.ua` DSL, enabling weight sharing beyond fold boundaries.
- **Lenses** — bidirectional morphisms pairing forward and backward equations.
  `lens_path` composes forward left-to-right, backward right-to-left. Semiring-agnostic.
- **Height-2 lenses (optics)** — residual-threading lenses that collect residuals in the
  forward pass and inject them in the backward pass, enabling multi-lens optic composition.
- **Product sorts** — cartesian products of sorts as right-nested `TypePair`, enabling
  morphisms that consume or produce multiple tensors simultaneously.
- **Fixpoint** — convergence iteration via Hydra primitives for architectures like
  Bellman-Ford and concept lattice closure.
- **Dynamic hyperparameters** — named bound terms rebindable between reductions via `rebind_hyperparams`.
- **Graph assembly** — DAG wiring with topological ordering, cycle detection,
  sort/rank junction validation. Fan-out and diamond patterns supported.
- **Parser** — `.ua` surface syntax compiles directly to Hydra terms via `parse_ua()`.
  See [`SYNTAX.md`](SYNTAX.md) for the full DSL reference.
- **Program** — `compile_program()` wraps the full parse-to-execution pipeline as a
  callable `Program` object.
- **JIT compilation** — `runtime/compiler.py` walks the Hydra graph once at compile
  time and emits native Python closures, bypassing `reduce_term` and wire format on
  the hot path. Compiles paths, lens-paths, fans, folds, unfolds, fixpoints, and
  single equations. JAX gets `lax.while_loop` fusion of fixpoint iterations.
- **N-ary equations** — equations with `n_params + n_inputs > 3` (e.g. attention
  Q/K/V with parameters) automatically list-pack into Hydra prim slots; the native
  callable stays variadic across all arities.

## Installation

Requires Python 3.12+. Hydra is fetched automatically as a dependency.

```bash
git clone https://github.com/Xopher00/unified-algebra
cd unified-algebra
uv pip install -e ".[dev]"
```

## Usage

```python
import numpy as np
from unialg import (
    Semiring, Sort, Equation, NumpyBackend,
    PathSpec, compile_program,
)

backend = NumpyBackend()

# 1. Define a semiring — user chooses operations, backend provides implementations.
#    resolve() (called transitively by compile_program) auto-validates the 7
#    semiring axioms against `(bottom, top)`-sampled triplets.
real_sr = Semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
hidden = Sort("hidden", real_sr)

# 2. Declare equations (morphisms)
eq_relu = Equation("relu", None, hidden, hidden, nonlinearity="relu")
eq_tanh = Equation("tanh", None, hidden, hidden, nonlinearity="tanh")

# 3. Compose into a path and compile to a callable Program
program = compile_program(
    [eq_relu, eq_tanh], backend=backend,
    specs=[PathSpec("act", ["relu", "tanh"], hidden, hidden)],
)

# 4. Call directly on native arrays — no Hydra encode/decode on the hot path
x = np.array([-1.0, 0.0, 0.5, 1.0])
output = program("act", x)
# tanh(relu(x)) = [0.0, 0.0, 0.462, 0.762]
```

Same architecture, different semiring — swap to tropical (min-plus):

```python
tropical_sr = Semiring("tropical", plus="minimum", times="add",
                       zero=float("inf"), one=0.0)
```

Restricted-domain semirings declare their domain via `(bottom, top)`:

```python
fuzzy_sr = Semiring("fuzzy", plus="maximum", times="minimum",
                    zero=0.0, one=1.0, bottom=0.0, top=1.0)
```

## Testing

```bash
uv run --python 3.12 --extra dev python -m pytest tests/ -v        # 366 tests
uv run --python 3.12 --extra dev python -m pytest tests/test_contraction.py  # one module
```

## Architecture

```
src/unialg/
    backend.py       — Backend ABC + NumpyBackend, PytorchBackend, JaxBackend, CupyBackend.
                       NumpyApiBackend is the shared base for numpy-API-compatible backends.
    terms.py         — tensor_coder(backend), Hydra record-view descriptors, literal helpers

    algebra/         — sort, semiring (auto-validates laws), equation/morphism,
                       contraction (CompiledEinsum), lens
    assembly/        — graph assembly, DAG validation, path/fan/fold/unfold/fixpoint/lens
                       composition, Hydra primitive resolution, specs
    parser/          — .ua DSL grammar (_grammar.py) and name resolution (_resolver.py)
    runtime/         — compile_program(), Program execution wrapper, JIT compiler
                       (compiler.py — emits native closures bypassing reduce_term)
```

## Research basis

| Paper | Relevance |
|---|---|
| Gavranovic, B. (2024). *Fundamental Components of Deep Learning.* PhD thesis. | Primary reference: Para, weighted optics, Lens_A-coalgebra, architecture survey |
| Gavranovic et al. (2024). *Categorical Deep Learning.* ICML. | Position paper: CDL as algebraic theory of all architectures |
| Capucci et al. (2024). *On a Fibrational Construction for Optics, Lenses, and Dialectica.* MFPS. | Lenses, optics, Dialectica unified as dialenses |
| Cruttwell et al. (2022). *Categorical Foundations of Gradient-Based Learning.* ESOP. | Backprop as lens |
| Dudzik & Veličković (2022). *GNNs are Dynamic Programmers.* NeurIPS. | Algorithmic alignment via structure |
| Lewis et al. (2025). *Filter Equivariant Functions.* NeurIPS DiffCoAlg workshop. | Symmetric length-general extrapolation |
| Hehner, E. (2007). *Unified Algebra.* | Project namesake: unifying boolean, number, set algebra |
| Lawvere, F.W. (1973). *Metric spaces, generalized logic, and closed categories.* | V-enriched foundation |
| Domingos, P. (2025). *Tensor Logic: The Language of AI.* | Tensor equations as universal construct |
| Maragos et al. (2021). *Tropical Geometry and Machine Learning.* | Max-plus/min-plus as concrete V instances |
| Shen & Tang (2022). *Isbell adjunctions and Kan adjunctions via quantale-enriched two-variable adjunctions.* | Quantale enrichment, representation theorems |
| Schultz, Spivak, Vasilakopoulou & Wisnesky (2016). *Algebraic Databases.* | Multi-sorted algebraic theories, Hydra's theoretical basis |
