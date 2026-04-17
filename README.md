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
- Installable Python package
- Batch and streaming evaluation modes

## Installation

Requires [Hydra](https://github.com/CategoricalData/hydra) installed locally:

```bash
git clone https://github.com/CategoricalData/hydra
pip install -e hydra/heads/python/

pip install git+https://github.com/Xopher00/unified-algebra
```

## Status

Early development.

## Research basis

| Paper | Relevance |
|---|---|
| Lawvere, F.W. (1973). *Metric spaces, generalized logic, and closed categories.* Rend. Sem. Mat. Fis. Milano XLIII, 135–166. | V-enriched foundation; morphisms as V-functors, fans as V-category products |
| Gavranovic et al. *Categorical Deep Learning as Algebraic Theory.* | Parametric morphisms in 2-categories (§3); initial algebra / final coalgebra for recursive architectures (§5) |
| Maragos, Charisopoulos & Theodosis (2021). *Tropical Geometry and Machine Learning.* Proc. IEEE. | Max-plus and min-plus semirings as concrete V instances |
| Bělohlávek, R. (2000). *Fuzzy Formal Concept Analysis.* | Complete lattice fixpoints; fuzzy formal contexts and concept lattices |
| Tarski, A. (1955). *A lattice-theoretical fixpoint theorem and its applications.* Pacific J. Math. 5(2), 285–309. | Convergence guarantees for fixpoint computations |
| Lambek & Scott. *Isbell adjunctions and Kan adjunctions on quantales.* | Quantale-enriched categories; representation theorems |
| Green et al. *Provenance Semirings.* | Semiring provenance for fixpoint logic |
