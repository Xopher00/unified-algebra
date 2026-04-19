# unified-algebra

A domain-specific language for expressing any machine learning architecture,
based on ideas from category theory.

Architectures are written as morphisms between typed tensors. The algebra of
composition is parameterised by a semiring V — changing V changes what
computation means (real arithmetic, tropical, fuzzy, probabilistic, etc.).
Built on the [Hydra](https://github.com/CategoricalData/hydra) functional
programming framework.

## Goals

- Any ML architecture expressible in a compact declarative `.ua` source file
- Hydra-first: DSL terms and types are Hydra `Term`/`Type` — one AST, no translation layer
- Backend-agnostic: same architecture runs on numpy, PyTorch, or JAX
- Semiring-parameterised: swap V to change semantics without changing the architecture
- Batch and streaming evaluation modes

## Current status

The core algebra is implemented and tested (57 tests passing):

- **Backend** — abstraction over numpy/PyTorch providing binary ops (elementwise + reduction),
  unary ops, structural ops, and constants. Users can add custom ops at runtime.
- **Semiring** — user-defined semirings as Hydra record terms. The user names two binary
  operations (⊕, ⊗) and their identities; the backend provides the implementations.
- **Contraction** — generalised einsum over arbitrary semirings. Parses an equation string,
  aligns tensors via broadcasting, applies ⊗ elementwise, reduces contracted axes with ⊕.
- **Sorts** — named tensor types bound to a semiring with identity encoded in Hydra
  `TypeVariable`s (`ua.sort.hidden:real`). Includes a tensor `TermCoder` for lossless
  numpy↔Hydra round-trips and sort/rank junction checking at composition boundaries.
- **Equations** — the unified construct: simultaneously a morphism (typed domain→codomain)
  and a tensor equation (einsum + semiring + optional nonlinearity). Variable arity.
  Registered as Hydra `Primitive`s via `prim1`/`prim2`/`prim3` and callable through
  `reduce_term`.
- **Graph assembly** — DAG-based equation wiring with topological ordering, cycle detection,
  sort junction validation, and rank checking. Fan-out and diamond patterns supported.

## Installation

Requires Python 3.12+ and [Hydra](https://github.com/CategoricalData/hydra)
(CategoricalData — not the Facebook config framework):

```bash
git clone https://github.com/CategoricalData/hydra
git clone https://github.com/Xopher00/unified-algebra
pip install -e "unified-algebra[dev]"
```

## Usage

```python
import numpy as np
from unified_algebra.backend import numpy_backend
from unified_algebra.semiring import semiring, resolve_semiring
from unified_algebra.contraction import compile_equation, semiring_contract

backend = numpy_backend()

# Define a semiring — user chooses the operations, backend provides them
real = resolve_semiring(
    semiring("real", plus="add", times="multiply", zero=0.0, one=1.0),
    backend,
)

# Same equation, different semirings
eq = compile_equation("ij,j->i")
W = np.array([[1.0, 2.0], [3.0, 4.0]])
x = np.array([1.0, 1.0])

result = semiring_contract(eq, [W, x], real, backend)
# array([3., 7.])  — standard matrix-vector multiply

# Switch to tropical (min-plus) — same equation, different algebra
tropical = resolve_semiring(
    semiring("tropical", plus="minimum", times="add", zero=float("inf"), one=0.0),
    backend,
)
result = semiring_contract(eq, [W, x], tropical, backend)
# array([2., 4.])  — shortest-path style computation
```

## Testing

```bash
uv run --python 3.12 --extra dev python -m pytest tests/ -v
```

## Research basis

| Paper | Relevance |
|---|---|
| Lawvere, F.W. (1973). *Metric spaces, generalized logic, and closed categories.* Rend. Sem. Mat. Fis. Milano XLIII, 135–166. | V-enriched foundation; morphisms as V-functors, fans as V-category products |
| Gavranovic et al. *Categorical Deep Learning as Algebraic Theory.* | Parametric morphisms in 2-categories (§3); initial algebra / final coalgebra for recursive architectures (§5) |
| Maragos, Charisopoulos & Theodosis (2021). *Tropical Geometry and Machine Learning.* Proc. IEEE. | Max-plus and min-plus semirings as concrete V instances |
| Domingos, P. (2025). *Tensor Logic: The Language of AI.* | Tensor equations as the universal construct; semiring-parameterised einsum |
| Bělohlávek, R. (2000). *Fuzzy Formal Concept Analysis.* | Complete lattice fixpoints; fuzzy formal contexts and concept lattices |
| Tarski, A. (1955). *A lattice-theoretical fixpoint theorem and its applications.* Pacific J. Math. 5(2), 285–309. | Convergence guarantees for fixpoint computations |
| Shen & Tang (2022). *Isbell adjunctions and Kan adjunctions via quantale-enriched two-variable adjunctions.* | Quantale-enriched categories; representation theorems |
| Green et al. *Provenance Semirings.* | Semiring provenance for fixpoint logic |
