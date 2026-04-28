"""Catamorphism unit tests: fold and unfold lambda term structure."""

import numpy as np
import pytest

import hydra.core as core
from hydra.core import Name

from unialg.assembly.compositions import FoldComposition, UnfoldComposition
from conftest import encode_array


# ---------------------------------------------------------------------------
# Fold: structure
# ---------------------------------------------------------------------------

class TestFoldStructure:

    def test_fold_returns_name_and_lambda(self, hidden, coder):
        init = encode_array(coder, np.zeros(3))
        name, term = FoldComposition("rnn", "step", init).to_lambda()
        assert name == Name("ua.fold.rnn")
        assert isinstance(term.value, core.Lambda)

    def test_fold_name_prefix(self, hidden, coder):
        init = encode_array(coder, np.zeros(3))
        name, _ = FoldComposition("test", "step", init).to_lambda()
        assert name.value == "ua.fold.test"


# ---------------------------------------------------------------------------
# Unfold: structure
# ---------------------------------------------------------------------------

class TestUnfoldStructure:

    def test_unfold_returns_name_and_lambda(self, hidden):
        name, term = UnfoldComposition("stream", "step", 3).to_lambda()
        assert name == Name("ua.unfold.stream")
        assert isinstance(term.value, core.Lambda)
