# Next Steps

## Done: Revise tests

Tests revised. Current live suite: 240 passing, 6 skipped. No references to old
API remain in live tests; legacy tests quarantined under
`tests/regression/stale_old_api/`.

## Done: Functor cleanup

- `Functor.apply(space)` added as instance method (object action)
- `Functor.compose(inner)` added as instance method (functor composition)
- Three `functor_*` compatibility wrappers removed
- `poly_fmap` signature changed to `(functor: Functor, h: Morphism)`, uses `functor.apply()` internally
- `Functor.map(h)` added as a convenience method; `poly_fmap` remains the semantic free function in `semantics/functors.py` and returns a deferred `PolyFmap` node. Hydra-backed realization happens later in `structure/realize.py`.

## Done: Optics layer (unified polynomial optic)

- Single `Optic(functor, forward, backward)` dataclass replaces former Lens/Prism/Traversal subclasses
- `forward: S → F(A)` decomposes source; `backward: F(B) → T` reconstructs target
- Uniform action: `compose(forward, poly_fmap(F, h), backward)` — implemented as `Optic.act()`
- `focus`/`replacement` derived via strict `functor.unapply()` (Hydra unification plus validated inverse of `apply_poly`)
- `Functor.unapply(fa)` builds `F(A)` with a Hydra type variable, asks `hydra.unification.unify_types` to solve `F(A) = fa`, then performs round-trip validation
- Lens = `Prod(Id, Const(residue))`, Prism = `Sum(Id, Const(residue))`, Traversal = arbitrary polynomial
- For simple lenses/prisms where S = F(A), forward and backward are identity morphisms
- Convenience constructors: `fst_lens(a, b)`, `snd_lens(a, b)`, `left_prism(a, b)`, `right_prism(a, b)`
- Height-2 optics need no structural change — just deeper polynomial bodies
- Unified Hypothesis property tests cover type laws and rejection laws
- No Hydra term construction in `optics.py`; action methods return semantic morphisms and realization stays in `structure/realize.py`

## Next: tensor operations (revised 2026-05-17)

### What was tried and why it failed

Three distinct attempts all broke down in the same ways:

1. **Parallel expression language** — each attempt invented `TensorExpr`, `TensorVar`, `ContractExpr`, or `TensorSemiring` as a new syntax tree running beside `MorphismExpr`. This duplicates the semantic layer.
2. **Python-native evaluator** — each attempt ended with a `_RUNTIME` dict (op name → numpy function) that bypasses Hydra entirely. This defeats the purpose of the backend/Hydra machinery.
3. **`ContractSpec` embedded in `Prim`** — the most mature attempt (notebook cells 66–89) correctly used `Morphism` as the carrier but then stuffed semantic metadata into `Prim(raw, dom, cod)`, which takes a raw Hydra `Term`, not a dataclass.

### What the Hydra API survey ruled out (2026-05-13)

- Hydra has **no built-in einsum/tensor operations**. `hydra.lib.math` is scalar-only.
- `hydra.reduction.contract_term` is beta-reduction cleanup — unrelated to tensor contraction despite the name.
- `hydra.parsers` is a Hydra-internal parser combinator library — not a surface syntax parser for user expressions.
- Creating new `MorphismExpr` subclasses needs stronger justification: Hydra's `TermPrimitive` + correctly named `BackendPrimitive` registration may be sufficient representation.

### Hard constraints for any implementation

- **No new expression language under `tensors/`** — new semantic files belong in `semantics/`, new structure files in `structure/`, following the existing layer split.
- **No Python-native evaluator** — execution must go through `run()` → `structure/realize.py` → Hydra reduction.
- **No new `MorphismExpr` subclass without clear justification** — exhaust `Prim` + named primitive first.
- **Explore Hydra before adding** — use `pkgutils`/`importlib`/`inspect` to verify a capability does not already exist.
- **`tensors/semirings.py`** and **`tensors/contractions.py`** are the current tensor-specific helper modules. Keep tensor-specific code there unless it needs to become general syntax/semantics/structure machinery.
- Do not recreate old parallel tensor expression modules such as `tensorexpressions.py` or `equations.py`.

### Correct layer mapping (to validate before implementing)

| Concern | Layer | Correct file | Status |
|---------|-------|--------------|--------|
| Subscript parsing (`"ij,jk->ik"`) | Syntax | `syntax/expressions.py` or keep as `str` | Unclear — may not need a node |
| Semiring dataclass | Tensor semantics | `tensors/semirings.py` | Exists ✓ |
| `from_backend` factory + `op_env` | Semantics | `tensors/semirings.py` | Missing |
| Tensor type `ExpType(I, A)` | Semantics | `semantics/morphisms.py` (helper) | Missing |
| `contract_morphism(sr, eq) → Morphism` | Tensor semantics | `tensors/contractions.py` | Exists/experimental |
| Contract fusion rewrite | Structure | `structure/` | Missing (notebook only) |
| Contraction kernel registration | Structure | `structure/` | Missing |

### Before writing any code

1. Do not recreate old parallel tensor expression modules.
2. Keep tensor-specific helpers inside `tensors/` unless a helper becomes broadly useful enough to promote into syntax/semantics/structure.
3. For each piece, ask: does Hydra already provide this? Check with `importlib`/`inspect` before implementing.


## Next: Clarify the core API flow

The current code has a natural but partly scattered flow:

```text
Expr node -> typed semantic wrapper -> type action -> structure realization -> main run
```

For morphisms, that flow is:

- `MorphismExpr` in `expressions.py` — pure syntax tree
- `Morphism` in `morphisms.py` — typed arrow handle
- `signature` / `dom_of` / `cod_of` — type derivation
- `realize` in `realize.py` — Hydra interpretation
- `lower` / `run` — execution boundary

For polynomial functors, the parallel flow is currently less explicit:

- `PolyExpr` in `expressions.py` — pure syntax tree
- `Functor` in `functors.py` — named descriptor with `apply(space)` object action method
- `functors.apply_poly` — internal recursive implementation of F(A)
- `Functor.unapply(fa)` — inverse object action, using Hydra unification to solve `F(A) = fa`
- `poly_fmap(functor, h)` in `semantics/functors.py` — arrow action `h -> F(h)`, takes a `Functor` and returns a deferred node

Refactor direction:

- Do **not** collapse `Morphism` and `Functor` into one universal class; morphisms are arrows, functors are object-and-arrow transformers.
- Keep `MorphismExpr` and `PolyExpr` as pure ADT syntax.
- Keep `Morphism` as the typed semantic handle for arrows.
- `Functor` upgraded: `apply(space)` is now an instance method; `summands()`, `x_arity()`, `consts()` are methods.
- `Functor.unapply(fa)` now uses Hydra type unification instead of a hand-written structural inverse walker.
- `poly_fmap(functor, h)` in `semantics/functors.py` takes a `Functor`, uses `functor.apply()` internally, and stays as a free semantic function. `Functor.map(h)` delegates to it.
- Import-order-sensitive monkey patches remain removed; no layer boundary violations.

Current reader-facing shape:

```python
Maybe = Functor("Maybe", Sum(One(), Id()))

Maybe.apply(INT)              # object action: F(A)
poly_fmap(Maybe, add1)        # arrow action: F(f), deferred until realization
```

## Next: Optics — remaining work

The unified `Optic` handles Lens, Prism, Traversal, and height-2 cases structurally. Remaining:

- **Runtime behavioral tests** — lens get/set, set/get, set/set laws; prism review/preview roundtrips (these require `realize`/`lower`/`run`)

## Done: Recursion schemes over carrier optics

The old `recursion.py` (with `rec`, `Inductive`, `LIST_IND`, `MAYBE_IND`, `AlgebraError`)
moved to `.bak`. The current API is generic over any carrier optic:

```python
Optic(functor, forward, backward, carrier=mu)
```

The fixed-point boundary is supplied by the optic:

- `fp.forward = unroll : μF → F(μF)` (destructor — peel one layer)
- `fp.backward = roll : F(μF) → μF` (constructor — wrap one layer)
- `fp.carrier = μF`
- No `FixedPoint` subclass — plain `Optic` is sufficient

Built-in carrier helpers, if added, are adapters which produce this optic boundary. They are not core semantics.

`cata`, `ana`, and `hylo` are implemented for plain, para, lax, and lax-para `Morphism` algebras/coalgebras. Parameter context is shared by calling contextual combinators with `shared_context=True`; effects are sequenced with the existing monad `bind`.

**`cata(fp, alg)`** — catamorphism / fold

```
cata(fp, alg) : P × μF → T(A)
             = compose(act_forward(fp, self), alg, shared_context=True)
             = compose(compose(fp.forward, poly_fmap(F, self)), alg, shared_context=True)
```

`alg` may be plain (`F(A) → A`), para (`P × F(A) → A`), lax (`F(A) → T(A)`), or lax-para (`P × F(A) → T(A)`). `act_forward` performs `forward ∘ poly_fmap`; the recursive self-reference has the same `param` and `monad` as `alg`.

For the list functor `F(X) = 1 + (E × X)`, `alg : F(A) → A` decomposes into:
- nil branch: `alg(Left(())) : A` — the base value
- cons branch: `(e: E, acc: A) → alg(Right(e, acc)) : A` — the step function

**`ana(fp, coalg)`** — anamorphism / unfold

```
ana(fp, coalg) : P × A → T(μF)
              = compose(coalg, act_backward(fp, self), shared_context=True)
              = compose(coalg, compose(poly_fmap(F, self), fp.backward), shared_context=True)
```

`coalg` may be plain (`A → F(A)`), para (`P × A → F(A)`), lax (`A → T(F(A))`), or lax-para (`P × A → T(F(A))`). `act_backward` performs `poly_fmap ∘ backward`; the recursive self-reference has the same `param` and `monad` as `coalg`.

**`hylo(fp, coalg, alg)`** — hylomorphism

```
hylo(fp, coalg, alg) = compose(ana(fp, coalg), cata(fp, alg), shared_context=True)
```

The shared-context composition is what keeps a lax-para hylo at `P × A → T(B)` instead of expanding into `P × P × A → T(B)`.

## Done: Runtime recursion smoke tests

Runtime tests now prove that the recursive primitive wiring reduces.

The smoke carrier uses `F(X) = Unit + X` and takes the terminating `Unit` branch. This keeps the functor valid for `Optic` validation while avoiding an infinite self-call:

- `cata(fp, alg)(value) == expected`
- `ana(fp, coalg)(seed) == expected`
- `hylo(fp, coalg, alg)(seed) == expected`

The lax-para runtime cases verify that the shared parameter is supplied once:

```text
P × A -> T(B)
```

not:

```text
P × P × A -> T(B)
```

This also caught and fixed a runtime bug: recursive primitives must register their actual raw function type as their Hydra `TypeScheme`; a dummy `Unit` scheme makes Hydra treat plain recursive primitives as nullary.

## Done: List carrier adapter

`list_carrier(element)` is implemented as a convenience constructor:

```python
Optic(functor=F, forward=unroll, backward=roll, carrier=mu)
```

It represents Hydra lists as:

```text
μX. 1 + (A × X)
```

Runtime coverage verifies:

- carrier boundary type laws
- `cata(list_carrier(INT), sum_alg)` sums a concrete Hydra list
- `ana(list_carrier(INT), countdown_coalg)` builds a concrete Hydra list
- `hylo(list_carrier(INT), countdown_coalg, sum_alg)` unfolds and folds in one pass

This adapter did not change the core recursion semantics.

## Next: Non-list carrier adapters

Maybe/tree adapters should also be convenience constructors, but they need a little design care:

- `Maybe(A)` is structurally `1 + A`, a constant polynomial with no `Id`; current `Optic` validation expects a functor position it can `unapply`.
- Tree carriers need an agreed recursive shape and carrier encoding before writing roll/unroll.

Do not weaken the core recursion semantics for these. Add only the adapter support needed to produce the same carrier optic boundary cleanly.

## Next: Algebra structure above recursion

`Morphism` remains the algebra/coalgebra representation; no separate `ParaAlgebra` wrapper is needed for this layer. The next algebraic layer is about named relationships between algebras.

Typed maps between algebras that commute with the algebra structure. Needed for relating model components (encoder/decoder adjointness, residual connections as natural transformations). `AlgebraHom(f, src, tgt)` where `f: Morphism` is the carrier map.

Design questions:
- Does `AlgebraHom` live in `recursion.py` or a new `algebra.py`?
- How are coherence cells `ε_A`, `δ_A` represented for lax cases?
- Does the optic structure make algebra/coalgebra typing constraints more explicit?

## Next: Semiring tensor structure

For multi-headed and branching architectures. Tensor products of morphisms, bilinearity. See `docs/ALGEBRA.md` / `claude-mdtopics/ALGEBRA.md` for context.

## Watch: Strength and distributivity

Lax para composition handles parameter threading with shared context plus `bind`/lambda capture. No explicit strength morphism is part of the current semantics.

## Tensor contraction: open design questions (2026-05-17)

These gaps must be resolved before any tensor contraction code is written.
Ordered by how much they block forward progress.

### 1. Index type encoding — DECISION REQUIRED (blocks everything)

`tensor_type(index, element)` and the dom/cod of
`contract_morphism` in `tensors/contractions.py` both depend on what Hydra
`Type` represents an index set.

Options:
- **Nominal** — `Name("I")`, `Name("J")` as opaque phantom types.  Sufficient
  for checking that `contract_morphism` output type matches a downstream
  morphism's input type. Does not encode size.
- **TypeVariable** — polymorphic; `contract_morphism` returns a scheme not a
  monotype.  Clean but harder to instantiate in practice.
- **Nat-indexed** — encodes concrete sizes statically.  Precise but requires
  size arithmetic in the type system; more than is needed now.

Nominal is the likely starting point.  Decide before touching `tensor_type()`
  or any `contract_morphism` dom/cod.

### 2. Structural tensor ops not in backend specs — needs placement decision

Any future tensor-lowering helper will need `expand_dims` and `transpose` for the
alignment step (unsqueeze + permute input tensors to `output_vars ++ reduced_vars`
dim order).  These are **not** in `tensors/backends/*.json`, which only carries
elementwise binary, unary, and reduce ops.

Options:
- Add them to the existing JSON specs under a `"structural"` kind.
- Register them separately in the tensor helper outside the JSON
  spec mechanism (direct `register_backend_primitive` calls).
- Handle alignment at the Hydra term level using existing pair/list primitives
  rather than delegating to the backend.

Decide before adding new tensor-lowering code.

### 3. Batching via poly_fmap needs grounding

The claim is that `bij,bjk->bik` = `poly_fmap(BatchFunctor, contract_morphism(sr, "ij,jk->ik"))`.
The existing functor machinery handles **polynomial** endofunctors (Prod, Sum,
Const, Id, Comp).  A function-space functor `(B → -)` is exponential, not
polynomial.

For finite batch dimensions it collapses to `Prod(Id, Id, ..., Id)` — a
product functor — which `poly_fmap` can handle.  But this requires:
- The batch index type to be finite and explicitly encoded as a product.
- A clear statement of what the `BatchFunctor` PolyExpr looks like.

Decide whether batching is handled by poly_fmap over a product functor, or
by a separate mechanism (e.g. explicit loop / vmap primitive), before
implementing or broadening `tensors/contractions.py`.

### 4. CompiledEquation has no layer home

The equation parser (`compile_einsum` from `unified-algebra/algebra/contraction.py`)
is pure data — no backend coupling.  It parses a string like `"ij,jk->ik"` into
`input_vars`, `output_vars`, `reduced_vars` index structures.

Candidate placements:
- `syntax/` — it is string parsing, which is syntax-layer work.
- Private helper inside `tensors/contractions.py` — keeps it close to its
  only consumer.

Decide before implementing `contract_morphism`.

### 5. Post-contraction nonlinearity — hook needed

The old `Equation` had an optional nonlinearity applied after contraction
(`contract_and_apply`).  This is common in practice: sigmoid/relu/softmax after
a semiring contraction.  In the new system, this is just morphism composition
(`compose(contract_morphism(sr, eq), nl)`), but `contract_morphism` should note
this as the intended pattern so users know not to bake it into the semiring.

No implementation needed now — note the pattern in `tensors/contractions.py`.

### 6. No Semiring law-checking equivalent

The old `Semiring.check_laws` verified associativity, commutativity, identity,
annihilation, and distributivity on scalar samples before the semiring was used.
The new `Semiring` has no equivalent.  Useful for custom and research semirings.

Add as a future method on `Semiring` in `tensors/semirings.py`, gated behind
a flag (default off for performance, on for development/research use).

### 7. Import linter boundary rules not updated

Import-linter contracts now include the current `tensors/semirings.py` and
`tensors/contractions.py` modules. If new tensor modules are added later, update
the contracts before merging implementation code.

## Deferred

- Surface syntax / grammar (no timeline)
- Backend expansion beyond current Hydra primitives
- `CarrierExpr` — a syntax-layer expression type for declaratively describing carrier roll/unroll structure, so recursive carriers (List, Maybe, RoseTree, user-defined) do not require hand-written Hydra plumbing. Design is unsettled: the expression ADT must be general enough to not enumerate per-carrier primitives, and the derivation of `unroll`/`roll` from that description must stay clean. Do not implement until the design is clear.

## Historical Reference

`/home/scanbot/unified-algebra/src/unialg` is a prior, ad-hoc version of this project. It may be useful for understanding old experiments, but it is a **reference only**:

- Do not copy code or port abstractions wholesale.
- Do not resurrect `_RecordView`; it created invisible structural coupling by backing domain objects with Hydra record terms.
- Do not reintroduce a manual `TypedMorphism.kind`-style tag; the current ADT structure is clearer.
- Treat the old `algebra_hom` bridge as a cautionary example: it exposed a broad functor surface while only executing a narrow subset. Avoid shipping similarly incomplete abstractions as stable API.
