# Test Organization

The trusted test suite is built around **pytest** and **Hypothesis**.

Pytest gives readable examples and regression checks. Hypothesis gives breadth
by generating many valid `Type`, `PolyExpr`, and `Morphism`-shaped cases from
small reusable strategies.

Use `LAW_CATALOG.md` as the checklist for semantic laws we intend to cover.

## Trusted Folders

- `semantics/` — hand-written behavioral contracts for the DSL. These should be readable examples of what the API means.
- `properties/` — Hypothesis tests for algebraic laws and typing invariants.
- `negative/` — expected rejection paths: type mismatches, invalid params, incompatible monads, unsupported encodings.
- `regression/` — bug reproductions and reviewed tests promoted after a real failure.
- `unit/` — small substrate/helper checks. Hydra API probes live here because they validate local backend behavior, not unialg semantics.
- `support/` — reusable pytest fixtures, Hypothesis strategies, and test helpers. This is not a test collection target.

Run the trusted suite:

```sh
.venv/bin/pytest tests
```

Run only property tests:

```sh
.venv/bin/pytest tests/properties -m property
```

## Hypothesis

Prefer Hypothesis when a behavior is a law over many valid values:

- `apply_poly(Id(), A) == A`
- `apply_poly(Prod(F, G), A) == apply_poly(F, A) × apply_poly(G, A)`
- `compose(f, g)` is valid exactly when `f.cod() == g.dom()`
- `pair(f, g)` is valid exactly when domains match
- `case(f, g)` is valid exactly when codomains match

Reusable generators belong in `tests/support/strategies.py`. Keep them small
and semantic. Do not ask Hypothesis to invent raw Hydra programs unless a test
also provides the execution fixture needed to interpret them.

## Pytest Examples

Use ordinary pytest tests when a concrete example explains the API better than
a property:

- notebook-facing examples
- lowering/runtime smoke tests
- exact error messages
- known bug reproductions

## Pynguin

Pynguin is a scouting tool only. It is not the trusted testing strategy for this
project.

Observed local behavior from the old module layout:

- `unialg.syntax.expressions` can produce useful seed tests, but output still needs review.
- the removed `unialg.space` module produced an empty file.
- the old `unialg.morphisms` target failed during Python 3.12 bytecode instrumentation with
  `RuntimeError: Failed to compute stacksize, got negative size`.

Raw generated Pynguin output belongs under `.pynguin/`, which is ignored. Promote
only meaningful, reviewed contracts into `semantics/`, `properties/`,
`negative/`, or `regression/`.

If Pynguin is useful for a narrow target, expose a wrapper under
`tests/scouting/pynguin_targets/` and generate against that wrapper:

```sh
env PYNGUIN_DANGER_AWARE=1 PYTHONPATH=src:tests/scouting .venv/bin/python -m pynguin \
  --project-path tests/scouting \
  --module-name pynguin_targets.morphisms_core \
  --output_path .pynguin/morphisms_core \
  --maximum-search-time 20 \
  --no-rich
```

To run that quarantined generated output directly, keep the scouting wrapper on
`PYTHONPATH`:

```sh
env PYTHONPATH=src:tests/scouting .venv/bin/pytest .pynguin/morphisms_core -q
```

## Legacy Tests

`regression/stale_old_api/` contains old tests for the pre-current API
(`Space`, `ProductSpace`, `ParaMorphism`, `CompositionError`,
`unialg.morphism.*`). They are historical reference material only and intentionally do not
use the `test_*.py` naming pattern.
