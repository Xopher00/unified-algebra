# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Parser and program compiler for the `.ua` DSL surface syntax, compiling to Hydra terms/types
- Architecture-specific test suite covering autoencoder, feedforward, residual, RNN, and shortest-path patterns
- Backend abstraction for numpy and PyTorch tensor operations with configurable binary (elementwise + reduce) and unary operators
- Semiring-parameterized einsum contraction with support for real, tropical (max-plus), fuzzy, and probabilistic algebras
- Blocked (chunked) tensor contraction for memory-efficient reduction via `block_size` parameter
- Named tensor types (sorts) with optional batching support and automatic batch dimension handling
- Named axes on sorts with optional size annotations and rank/axis compatibility validation at composition edges
- Parametric equations via `param_slots` for user-defined scalar hyperparameters in tensor operations
- Sequential (path) and parallel (fan) composition of tensor operations with unbounded arity list-based fan merges
- Bidirectional morphisms (lenses) pairing forward and backward equations for gradient computation, path recovery, and likelihood propagation
- Lens sequential composition (`lens_seq`) with optic threading for residual sort propagation
- Fold (catamorphism) and unfold (anamorphism) for recursive architectures via Hydra foldl and custom unfold_n
- Fixpoint iteration construct with convergence predicate, epsilon threshold, and max-iteration limit
- Adjoint contraction via `*` call-site suffix — invokes `residual_elementwise + times_reduce` when algebra declares `residual=`
- Partial order semantics via `leq=` field in algebra declarations, with reflexivity/transitivity law checking
- `define` declarations for inline custom unary and binary operations in `.ua` DSL
- `functor` declarations with polynomial expression sub-grammar (`0`, `1`, `X`, `+`, `&`, `@`)
- `cell` composition IR with Pratt-parsed operators (`>` seq, `&` par, `~` lens, `^[A]` copy, `![A]` delete, `_[A]` identity, `>[F]` cata, `<[F]` ana)
- Dynamic parameter binding with `rebind_params` for injectable and rebindable named parameters between reductions
- DAG validation including topological sort and cycle detection for pipeline composition
- Sort and rank junction checking across morphism compositions to enforce type consistency
- Hydra-native morphism resolution via `Primitive` terms with full integration into Hydra's reduction engine and standard library
- Parameter tying (`share`) in DSL for weight sharing across morphism instances
- JIT compilation path via `backend.compile(fn)` with seq/par fixpoint fusion

### Changed
- Reorganized codebase into `algebra`, `assembly`, `parser`, `runtime` subpackages based on import clustering
- Refactored parser into `_grammar`, `_resolver` submodules with cleaner separation of concerns
- Consolidated morphism, equation, and semiring classes into shared dataclass abstractions to reduce redundant code
- Assembly layer restructured: validation extracted to `_validation.py`, morphism registration to `_morphism_compile.py`
- Graph construction migrated to `graph_with_primitives` + `elements_to_graph` two-step Hydra API
- DSL keyword `op` replaces prior morphism declaration keywords for ML-practitioner readability

### Removed
- Stale and redundant test files from prior package structure
- `composition/` and `resolve/` subpackages (absorbed into `assembly/` and `algebra/`)

### Fixed
- Python dependency resolution issues with Hydra kernel installation
- Sort axis dimension mismatch validation now correctly skips unsized axes
