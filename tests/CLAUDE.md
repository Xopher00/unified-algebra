## Testing approach

Use **pytest** and **Hypothesis** as the trusted testing strategy for this
project.

Start by identifying the semantic contract:

* what inputs are valid
* what inputs must be rejected
* what type/domain/codomain result must be produced
* what algebraic law should hold

Then choose the smallest test style that expresses that contract:

* pytest examples for concrete API behavior, runtime smoke tests, exact
  rejections, and regressions
* Hypothesis property tests for laws over many valid `Type`, `PolyExpr`, and
  `Morphism`-shaped values

Keep Hypothesis strategies in `tests/support/strategies.py` so new tests reuse
the same generators instead of rebuilding them locally.

Pynguin is only a scouting tool. Generated tests are evidence, not authority.
Do not commit raw generated suites as trusted tests. Promote only reviewed
contracts into `semantics/`, `properties/`, `negative/`, or `regression/`.

Do not change implementation merely to satisfy generated tests if the generated
tests encode accidental behavior.
