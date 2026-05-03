# Layer Boundary Contract

See `ARCHITECTURE.md` for the full design contract and module map.

---

## Five-Layer Architecture

| Layer | File | Responsibility |
|-------|------|---------------|
| 1. Syntax | `parser/_grammar.py` | Parse `.ua` text to typed AST nodes |
| 2. Resolution | `parser/_resolver.py` | Name resolution → typed morphism construction |
| 3. Algebraic construction | `morphism/` | Build algebraic/categorical terms (TypedMorphism) |
| 4. Graph registration | `assembly/` | Lower morphisms to Hydra graph (primitives + bound_terms) |
| 5. Execution | `assembly/program.py` | `reduce_term` only — single execution path |

Each layer's boundary is sharp. Grammar imports nothing from `algebra/` or `morphism/`. Assembly registration is separate from validation. `Program.__call__` always uses `reduce_term`; there is no compiled-function bypass.

---

## 1. What the `.ua` parser accepts

Eight top-level declaration forms, in dependency order:

- `import <backend>` — names the backend; must appear first.
- `define unary|binary <name>(<params>) = <expr>` — custom op from expression; no dependencies.
- `algebra <name>(plus=, times=, zero=, one=, ...)` — semiring declaration; may reference `define` names.
- `spec <name>(<algebra>[, batched][, axes=[...]])` — named tensor type; depends on algebras.
- `op <name> : <dom> -> <cod>` — typed tensor morphism (einsum or nonlinearity); depends on specs/algebras.
- `share <name> : <op1>, <op2>, ...` — structural weight-tying assertion across ops.
- `functor <name> : <poly_expr>` — polynomial functor; body atoms are `0`, `1`, `X`, `+`, `&`, `@`.
- `cell <name> : <sort_sig> = <cell_expr>` — composition IR; Pratt-parsed infix expression.

`cell` expression operators and prefix forms:

| Syntax | Meaning |
|---|---|
| `f > g` | Sequential composition (left-assoc, prec 60) |
| `f & g` | Parallel bimap (left-assoc, prec 70) |
| `f ~ g` or `f ~ g *[Sort]` | Lens pairing; optional residual sort (prec 50) |
| `_[Sort]` or `id[Sort]` | Identity morphism |
| `^[Sort]` or `copy[Sort]` | Copy morphism |
| `![Sort]` or `drop[Sort]` | Delete morphism |
| `>[F](args)` or `fold[F](args)` | Catamorphism |
| `<[F](args)` or `unfold[F](args)` | Anamorphism |
| `<number>` | Literal constant |
| `<name>` | Equation reference |

---

## 2. What the resolver guarantees

- Every name in the source is resolved or a `ValueError` is raised.
- Returns a `UASpec` with fields: `semirings`, `sorts`, `equations`, `defines`, `cells`,
  `share_groups`, `backend_name`.
- `share_groups` is validated at parse time (all ops exist, same domain algebra).
  NOTE: the check that all ops in a share group appear in at least one `seq` is not yet enforced.
  `share_groups` is wired in `compile_program`: non-canonical ops are aliased to the canonical
  param's term before `assemble_graph` is called.
- `?` and `@` modifiers parse without error but the resolver raises `NotImplementedError` — see §6.

---

## 3. What `TypedMorphism` guarantees

- Wraps a Hydra `Term` with explicit `domain_sort` and `codomain_sort`.
- Boundary sorts are `Sort`, `ProductSort`, or any Hydra `Type` variant.
- `value` (inherited from `core.TypeFunction`) is always the normalized Hydra function type
  `domain.type_ -> codomain.type_`.
- Does NOT validate that the enclosed term correctly implements the morphism — that is the
  caller's responsibility at assembly time.

---

## 4. What assembly is allowed to do

- Registration only. `assemble_graph` receives resolved equations and compiled callables and
  returns a Hydra `Graph`.
- Schema validation (`validate_pipeline` / `unify_type_constraints`) runs inside
  `_resolve_equations` before the graph is built.
- `assemble_graph` must not make semantic decisions: no new type rules, no cell semantics,
  no contraction logic.
- KNOWN DEVIATIONS: two semantic decisions currently live in assembly and are slated for
  migration. (1) The adjoint (transpose) rewrite in `_equation_resolution.py`'s
  `compile_equation` constructs a transposed equation for backward passes — this belongs in
  `algebra/` and is tracked as P-A1 (being implemented concurrently). (2) Lens residual
  threading in `_morphism_compile.py`'s `_try_lens_seq` encodes optic sequencing semantics
  (forward collects residuals, backward consumes in reverse) that logically belongs in
  `morphism/lens.py`. Both will be moved once the algebra-layer lens interfaces stabilise.

---

## 5. What the execution path is

- Single canonical path: `encode(arg) → reduce_term(graph, term) → decode(result)`.
- `compiled_fns` bypass is removed. Every entry point goes through `reduce_term`.
- `share_groups` weight aliasing is live: `compile_program` aliases non-canonical op params
  to the canonical op's param term before `assemble_graph` is called.

---

## 6. Syntax parsed but intentionally unsupported

- `?` — masked/guarded equation modifier (`name?` at a `cell` call site). Parsed by grammar;
  resolver raises `NotImplementedError`. Semantic TBD: zero-mask vs NaN-mask vs conditional
  vs projection.
- `@` — functor composition (`F @ G` in a `functor` body). Parsed by grammar as `poly_compose`;
  resolver raises `NotImplementedError`. Deferred. Blocked on PolyExpr compose variant in
  `morphism/functor.py` and expansion logic in `algebra_hom.py`.
