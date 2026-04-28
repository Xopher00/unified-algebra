"""Verify semiring axioms via Semiring.check_laws using the backend's ops.

The checker pulls the user's named ops out of the backend and evaluates
each of the 7 semiring axioms on caller-supplied scalar triplets. No parallel
operator definitions, no parallel arithmetic — whatever the backend says
"add" or "minimum" or "logaddexp" means IS what's checked.
"""

import pytest

from unialg import Semiring


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
        Semiring("real", "add", "multiply", 0.0, 1.0).check_laws(backend, _REAL_SAMPLES)

    def test_tropical(self, backend):
        Semiring("tropical", "minimum", "add", float("inf"), 0.0).check_laws(backend, _REAL_SAMPLES)

    def test_max_plus(self, backend):
        Semiring("maxplus", "maximum", "add", float("-inf"), 0.0).check_laws(backend, _REAL_SAMPLES)

    def test_logprob(self, backend):
        Semiring("logprob", "logaddexp", "add", float("-inf"), 0.0).check_laws(backend, _REAL_SAMPLES)

    def test_fuzzy(self, backend):
        Semiring("fuzzy", "maximum", "minimum", 0.0, 1.0).check_laws(backend, _FUZZY_SAMPLES)


# ---------------------------------------------------------------------------
# leq: ordering relation as the meet of the induced order
# ---------------------------------------------------------------------------

class TestLeqOrdering:
    """For an ordered semiring, the named `leq` op is the meet:
    a ≤ b iff meet(a, b) ≈ a. The law check verifies reflexivity and
    transitivity hold under this interpretation.
    """

    def test_fuzzy_with_leq_minimum(self, backend):
        # Fuzzy on [0,1]: a ≤ b iff min(a,b) = a.
        sr = Semiring("fuzzy", "maximum", "minimum", 0.0, 1.0, leq="minimum")
        sr.check_laws(backend, _FUZZY_SAMPLES)

    def test_tropical_with_leq_minimum(self, backend):
        # Tropical (min,+): a ≤ b iff min(a,b) = a (small absorbed).
        sr = Semiring("tropical", "minimum", "add", float("inf"), 0.0, leq="minimum")
        sr.check_laws(backend, _REAL_SAMPLES)

    def test_real_with_broken_leq_rejected(self, backend):
        # Real semiring with leq=add: meet(a,b)=a+b, so meet(0,0)=0 makes
        # reflexivity hold at 0 only — but generally not. Should fail.
        sr = Semiring("real", "add", "multiply", 0.0, 1.0, leq="add")
        with pytest.raises(ValueError, match="leq"):
            sr.check_laws(backend, _REAL_SAMPLES)

    def test_resolved_carries_leq(self, backend):
        sr = Semiring("fuzzy", "maximum", "minimum", 0.0, 1.0, leq="minimum")
        resolved = sr.resolve(backend, samples=_FUZZY_SAMPLES)
        assert resolved.leq_name == "minimum"
        assert resolved.leq_elementwise is not None
        # Smoke-check: meet(0.3, 0.7) = 0.3
        assert abs(resolved.leq_elementwise(0.3, 0.7) - 0.3) < 1e-9

    def test_no_leq_means_none(self, backend):
        sr = Semiring("real", "add", "multiply", 0.0, 1.0)
        resolved = sr.resolve(backend, samples=_REAL_SAMPLES)
        assert resolved.leq_name is None
        assert resolved.leq_elementwise is None
