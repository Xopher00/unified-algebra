# unified-algebra

A typed DSL for wiring algebraic tensor programs. The algebra is
parameterised by a semiring V — swapping V changes the semantics (real
arithmetic, tropical/max-plus, fuzzy, probabilistic, etc.). The DSL
compiles to [Hydra](https://github.com/CategoricalData/hydra) terms/types
directly — there is no separate AST or translation layer.

**Docs:** [Architecture](ARCHITECTURE.md) · [DSL syntax reference](SYNTAX.md) · [Examples](examples/) · [Changelog](CHANGELOG.md)

## Design

- **Hydra-first** — DSL declarations are Hydra `Term`/`Type`. One AST, no translation layer.
- **Operation-agnostic** — semantics come from declared semirings and backends. The framework never assumes real arithmetic.
- **Backend-agnostic** — same program runs on numpy, PyTorch, JAX, or CuPy.
- **Semiring-parameterised** — swap V to change what contraction means without changing the wiring. Algebras optionally declare `leq=` for partial-order semantics (fuzzy, tropical).
- **Bidirectional** — lenses pair forward and backward morphisms for any semiring-parameterised adjoint. `lens_seq` composes lenses with optic threading.
- **Compositional IR** — `cell` declarations express seq (`>`), par (`&`), lens (`~`), copy/delete/identity, catamorphism, and anamorphism as a single typed Pratt-parsed expression.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design contract.

## Quick start

The primary interface is `parse_ua()`, which compiles `.ua` source text to a callable `Program`:

```python
import numpy as np
from unialg import parse_ua, NumpyBackend

prog = parse_ua('''
algebra real(plus=add, times=multiply, zero=0.0, one=1.0)
spec hidden(real)

op linear : hidden -> hidden
  einsum = "ij,j->i"
  algebra = real
''', NumpyBackend())

W = np.array([[1.0, 2.0], [3.0, 4.0]])
x = np.array([1.0, 1.0])
result = prog('linear', W, x)   # [3.0, 7.0]
```

Same equation, different semiring — tropical (min-plus) turns matrix-vector product into shortest-path relaxation:

```python
prog = parse_ua('''
algebra tropical(plus=minimum, times=add, zero=inf, one=0.0)
spec nodes(tropical)

op relax : nodes -> nodes
  einsum = "ij,j->i"
  algebra = tropical
''', NumpyBackend())
```

See [`examples/`](examples/) for 7 runnable demonstrations covering semiring contraction, path composition, fan/merge, batching, custom operations, attention-like patterns, and residual connections.

## Installation

Requires Python 3.12+. Hydra is fetched automatically as a dependency.

```bash
git clone https://github.com/Xopher00/unified-algebra
cd unified-algebra
uv pip install -e ".[dev]"
```

## Testing

```bash
uv run --python 3.12 --extra dev python -m pytest tests/ -v              # all tests
uv run --python 3.12 --extra dev python -m pytest tests/unit/ -v         # unit tests
uv run --python 3.12 --extra dev python -m pytest tests/semantics/ -v    # end-to-end semantics
uv run --python 3.12 --extra dev python -m pytest tests/architectures/ -v  # architecture patterns
```

Test suite is organised by purpose:

```
tests/
  unit/           component isolation (sort, equation, contraction, assembly)
  parser/         DSL grammar and name resolution
  semantics/      Hydra reduce_term equivalence, lens/optic threading, end-to-end execution
  architectures/  pattern demonstrations (message passing, shortest path, attention, etc.)
  negative/       error handling and invalid input
  backend/        multi-backend parity
  integration/    cross-layer integration tests
  regression/     regression guards for previously fixed bugs
```

## Module map

```
src/unialg/
    algebra/         sorts, semirings (law checking), equations, contraction engine
    assembly/        graph construction, DAG validation, morphism compilation, Program runtime
    morphism/        typed morphisms, lenses, lens sequential composition, functors, algebra homomorphisms
    parser/          .ua grammar + name resolution
    backend.py       Backend ABC + concrete backends (numpy, pytorch, jax, cupy)
    terms.py         Hydra record-view helpers, tensor encoding
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and core abstractions.

## Research basis

| Paper | Relevance |
|---|---|
| Gavranovic, B. (2024). *Fundamental Components of Deep Learning.* PhD thesis. | Primary reference: Para, weighted optics, architecture survey |
| Gavranovic et al. (2024). *Categorical Deep Learning.* ICML. | Algebraic theory of architectures |
| Capucci et al. (2024). *On a Fibrational Construction for Optics, Lenses, and Dialectica.* MFPS. | Lenses, optics, Dialectica unified as dialenses |
| Cruttwell et al. (2022). *Categorical Foundations of Gradient-Based Learning.* ESOP. | Bidirectional morphisms (backprop as lens) |
| Dudzik & Veličković (2022). *GNNs are Dynamic Programmers.* NeurIPS. | Algorithmic alignment via semiring structure |
| Lewis et al. (2025). *Filter Equivariant Functions.* NeurIPS DiffCoAlg workshop. | Symmetric length-general extrapolation |
| Hehner, E. (2007). *Unified Algebra.* | Project namesake: unifying boolean, number, set algebra |
| Lawvere, F.W. (1973). *Metric spaces, generalized logic, and closed categories.* | V-enriched foundation |
| Domingos, P. (2025). *Tensor Logic: The Language of AI.* | Tensor equations as universal construct |
| Maragos et al. (2021). *Tropical Geometry and Machine Learning.* | Max-plus/min-plus as concrete V instances |
| Shen & Tang (2022). *Isbell adjunctions and Kan adjunctions via quantale-enriched two-variable adjunctions.* | Quantale enrichment, representation theorems |
| Schultz, Spivak, Vasilakopoulou & Wisnesky (2016). *Algebraic Databases.* | Multi-sorted algebraic theories, Hydra's theoretical basis |
