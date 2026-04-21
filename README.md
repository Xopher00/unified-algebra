# unified-algebra

A Python DSL for expressing any machine learning architecture as morphisms
between typed tensors. The algebra is parameterised by a semiring V — swapping
V changes the semantics (real arithmetic, tropical/max-plus, fuzzy,
probabilistic, etc.). The DSL compiles to
[Hydra](https://github.com/CategoricalData/hydra) terms/types directly — there
is no separate AST or translation layer.

## Goals

- Any ML architecture expressible as compositions of typed tensor equations
- Hydra-first: DSL terms and types are Hydra `Term`/`Type` — one AST, no translation layer
- Backend-agnostic: same architecture runs on numpy, PyTorch, or JAX
- Semiring-parameterised: swap V to change semantics without changing the architecture
- Bidirectional: lenses pair forward and backward equations for gradient computation, path recovery, or any semiring-parameterised adjoint

## Current status

279 tests passing across 14 phases:

- **Backend** — abstraction over numpy/PyTorch providing binary ops (elementwise + reduction),
  unary ops, structural ops. Users can extend with custom ops at runtime.
- **Semiring** — user-defined semirings as Hydra record terms (real, tropical, fuzzy, logaddexp, etc.).
- **Contraction** — generalised einsum over arbitrary semirings with optional blocked
  (chunked) execution for memory efficiency.
- **Sorts** — named tensor types bound to a semiring, with optional batching (`sort("hidden", sr, batched=True)`).
  Batch dimensions are auto-prepended at resolution time. Sort/rank junction checking at composition boundaries.
- **Equations** — the unified construct: simultaneously a morphism and a tensor equation.
  Supports `param_slots` for user-defined parametric nonlinearities (e.g. temperature-scaled softplus).
- **Composition** — sequential (`path`) and parallel (`fan`, list-based, unbounded arity).
  Validated for sort junction consistency.
- **Recursion** — fold (catamorphism) via Hydra's `foldl`, unfold (anamorphism) via custom `unfold_n`.
  Weight tying is automatic.
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
from unified_algebra import (
    semiring, sort, equation, numpy_backend, assemble_graph,
    PathSpec, path, lens, lens_path,
)
from hydra.context import Context
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

backend = numpy_backend()
cx = Context(trace=(), messages=(), other=FrozenDict({}))

# 1. Define a semiring — user chooses operations, backend provides implementations
real_sr = semiring("real", plus="add", times="multiply", zero=0.0, one=1.0)
hidden = sort("hidden", real_sr)

# 2. Declare equations (morphisms)
eq_relu = equation("relu", None, hidden, hidden, nonlinearity="relu")
eq_tanh = equation("tanh", None, hidden, hidden, nonlinearity="tanh")

# 3. Compose into a path and assemble the graph
graph = assemble_graph(
    [eq_relu, eq_tanh], backend,
    specs=[PathSpec("act", ["relu", "tanh"], hidden, hidden)],
)

# 4. Execute via Hydra reduction
from unified_algebra import tensor_coder
coder = tensor_coder()
x = np.array([-1.0, 0.0, 0.5, 1.0])
x_enc = coder.decode(None, x).value

result = reduce_term(cx, graph, True, apply(var("ua.path.act"), x_enc))
output = coder.encode(None, None, result.value).value
# tanh(relu(x)) = [0.0, 0.0, 0.462, 0.762]
```

Same architecture, different semiring — swap to tropical (min-plus):

```python
tropical_sr = semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)
```

## Testing

```bash
uv run --python 3.12 --extra dev python -m pytest tests/ -v   # 279 tests
uv run --python 3.12 --extra dev python -m pytest tests/test_phase1.py  # one phase
```

## Architecture

```
src/unified_algebra/
    backend.py       — Backend class: numpy/PyTorch ops
    semiring.py      — Semiring declaration + resolution
    contraction.py   — Generalised einsum with blocked execution
    sort.py          — Named tensor types, coders, junction checking
    morphism.py      — Equation declaration + resolution
    composition.py   — Path, fan, lens, lens_path composition as Hydra lambda terms
    recursion.py     — Fold, unfold, fixpoint via Hydra primitives
    graph.py         — Graph assembly, rebind_hyperparams, NamedTuple specs
    validation.py    — DAG resolution, pipeline validation
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
