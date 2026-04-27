"""Verify semiring axioms via Semiring.validate_laws using the backend's ops.

The validator pulls the user's named ops out of the backend and evaluates
each of the 7 semiring axioms on caller-supplied scalar triplets. No parallel
operator definitions, no parallel arithmetic — whatever the backend says
"add" or "minimum" or "logaddexp" means IS what's checked.
"""

import pytest

from unialg import NumpyBackend, Semiring


@pytest.fixture
def backend():
    return NumpyBackend()


# Triplets of unbounded reals — work for real, tropical, max-plus, log-prob.
_REAL_SAMPLES = [
    (1.0, 2.0, 3.0), (-1.5, 0.5, 2.5),
    (0.0, 1.0, -1.0), (5.0, -2.0, 4.0),
]

# Fuzzy is only a semiring on [0, 1].
_FUZZY_SAMPLES = [(0.1, 0.5, 0.8), (0.0, 0.3, 1.0), (0.7, 0.7, 0.2)]


# ---------------------------------------------------------------------------
# Canonical semirings — must validate
# ---------------------------------------------------------------------------

class TestCanonicalSemirings:

    def test_real(self, backend):
        Semiring("real", "add", "multiply", 0.0, 1.0).validate_laws(backend, _REAL_SAMPLES)

    def test_tropical(self, backend):
        Semiring("tropical", "minimum", "add", float("inf"), 0.0).validate_laws(backend, _REAL_SAMPLES)

    def test_max_plus(self, backend):
        Semiring("maxplus", "maximum", "add", float("-inf"), 0.0).validate_laws(backend, _REAL_SAMPLES)

    def test_logprob(self, backend):
        Semiring("logprob", "logaddexp", "add", float("-inf"), 0.0).validate_laws(backend, _REAL_SAMPLES)

    def test_fuzzy(self, backend):
        Semiring("fuzzy", "maximum", "minimum", 0.0, 1.0).validate_laws(backend, _FUZZY_SAMPLES)
