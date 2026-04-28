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
- Parametric equations via `param_slots` for user-defined scalar hyperparameters in tensor operations
- Sequential (path) and parallel (fan) composition of tensor operations with unbounded arity list-based fan merges
- Bidirectional morphisms (lenses) pairing forward and backward equations for gradient computation, path recovery, and likelihood propagation
- Fold (catamorphism) and unfold (anamorphism) for recursive architectures via Hydra foldl and custom unfold_n
- Dynamic parameter binding with `rebind_params` for injectable and rebindable named parameters between reductions
- DAG validation including topological sort and cycle detection for pipeline composition
- Sort and rank junction checking across morphism compositions to enforce type consistency
- Hydra-native morphism resolution via `Primitive` terms with full integration into Hydra's reduction engine and standard library
- Parameter tying support in DSL for weight sharing and tied parameters across morphisms

### Changed
- Reorganized codebase into `algebra`, `assembly`, `parser`, `runtime` subpackages based on import clustering
- Refactored parser into `_grammar`, `_resolver` submodules with cleaner separation of concerns
- Consolidated morphism, equation, and semiring classes into shared dataclass abstractions to reduce redundant code
- Consolidated redundant code across DSL operations; extended composition to tensor operation chaining

### Removed
- Stale and redundant test files from prior package structure

### Fixed
- Python dependency resolution issues with Hydra kernel installation
