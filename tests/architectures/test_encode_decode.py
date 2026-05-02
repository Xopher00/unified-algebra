"""Encode-decode pattern: bidirectional path composition.

An autoencoder is a bidirectional morphism:
  forward (encoder): input -> latent   (compression)
  backward (decoder): latent -> input  (reconstruction)

The lens primitive pairs these two directions as a single compositional
unit. This test file uses morphism.lens(...) + NamedCell + assemble_graph.

Tests:
  1. test_single_lens_autoencoder_assembles  -- graph assembly succeeds
  2. test_forward_encodes                    -- encoder primitive maps input -> latent
  3. test_backward_decodes                   -- decoder primitive maps latent -> input
  4. test_deep_autoencoder_assembles         -- two-lens deep autoencoder assembles
  5. test_semiring_polymorphism              -- tropical semiring lens works unchanged
"""

import numpy as np
import pytest

from hydra.core import Name
from hydra.dsl.python import Right

from unialg import (
    NumpyBackend, Semiring, Sort,
    Equation,
)
from unialg.assembly.graph import assemble_graph
from unialg.assembly._equation_resolution import resolve_equation
from conftest import encode_array, decode_term, assert_reduce_ok


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def real_sr():
    return Semiring("ae_real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def tropical_sr():
    return Semiring("ae_tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def input_sort(real_sr):
    """High-dimensional input space (dim=6)."""
    return Sort("ae_input", real_sr)


@pytest.fixture
def latent_sort(real_sr):
    """Low-dimensional latent/bottleneck space (dim=3)."""
    return Sort("ae_latent", real_sr)


@pytest.fixture
def hidden_sort(real_sr):
    """Intermediate hidden space for deep autoencoder (dim=4)."""
    return Sort("ae_hidden", real_sr)


# ===========================================================================
# TestAutoencoder
# ===========================================================================

class TestAutoencoder:
    """Express an autoencoder using equation pairs (encoder/decoder)."""

    def test_single_autoencoder_assembles(
        self, real_sr, input_sort, latent_sort, backend
    ):
        """Autoencoder assembles: encoder and decoder equation primitives registered."""
        eq_enc = Equation("ae1_enc", "ij,j->i", input_sort, latent_sort, real_sr)
        eq_dec = Equation("ae1_dec", "ij,j->i", latent_sort, input_sort, real_sr)

        graph, *_ = assemble_graph([eq_enc, eq_dec], backend)

        assert Name("ua.equation.ae1_enc") in graph.primitives
        assert Name("ua.equation.ae1_dec") in graph.primitives

    def test_forward_encodes(
        self, cx, real_sr, input_sort, latent_sort, backend, coder
    ):
        """Encoder primitive: W (3x6), x (6,) -> z (3,) = W @ x."""
        eq_enc = Equation("ae2_enc", "ij,j->i", input_sort, latent_sort, real_sr)
        prim, *_ = resolve_equation(eq_enc, backend)

        W = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = prim.implementation(cx, None, (
            encode_array(coder, W),
            encode_array(coder, x),
        ))
        assert isinstance(result, Right)
        z = decode_term(coder, result.value)

        np.testing.assert_allclose(z, W @ x)
        assert z.shape == (3,)

    def test_backward_decodes(
        self, cx, real_sr, input_sort, latent_sort, backend, coder
    ):
        """Decoder primitive: W_dec (6x3), z (3,) -> x_hat (6,)."""
        eq_dec = Equation("ae3_dec", "ij,j->i", latent_sort, input_sort, real_sr)
        prim, *_ = resolve_equation(eq_dec, backend)

        W_dec = np.eye(6, 3)
        z = np.array([1.0, 2.0, 3.0])

        result = prim.implementation(cx, None, (
            encode_array(coder, W_dec),
            encode_array(coder, z),
        ))
        assert isinstance(result, Right)
        x_hat = decode_term(coder, result.value)

        np.testing.assert_allclose(x_hat, W_dec @ z)
        assert x_hat.shape == (6,)

    def test_deep_autoencoder_assembles(
        self, real_sr, input_sort, hidden_sort, latent_sort, backend, coder
    ):
        """Deep autoencoder: four equations (enc1, enc2, dec1, dec2) all register."""
        eq_enc1 = Equation("ae4_enc1", "ij,j->i", input_sort, hidden_sort, real_sr)
        eq_enc2 = Equation("ae4_enc2", "ij,j->i", hidden_sort, latent_sort, real_sr)
        eq_dec1 = Equation("ae4_dec1", "ij,j->i", latent_sort, hidden_sort, real_sr)
        eq_dec2 = Equation("ae4_dec2", "ij,j->i", hidden_sort, input_sort, real_sr)

        graph, *_ = assemble_graph(
            [eq_enc1, eq_enc2, eq_dec1, eq_dec2], backend,
        )

        for name in ("ae4_enc1", "ae4_enc2", "ae4_dec1", "ae4_dec2"):
            assert Name(f"ua.equation.{name}") in graph.primitives

    def test_semiring_polymorphism(
        self, cx, tropical_sr, backend, coder
    ):
        """Tropical semiring identity equation: output == input."""
        trop_input = Sort("ae_trop_input", tropical_sr)
        trop_latent = Sort("ae_trop_latent", tropical_sr)

        eq_enc = Equation("ae6_enc", "i->i", trop_input, trop_latent, tropical_sr)
        eq_dec = Equation("ae6_dec", "i->i", trop_latent, trop_input, tropical_sr)

        from unialg import compile_program
        prog = compile_program([eq_enc, eq_dec], backend=backend)

        assert Name("ua.equation.ae6_enc") in prog.graph.primitives
        assert Name("ua.equation.ae6_dec") in prog.graph.primitives

        x = np.array([1.0, 3.0, 2.0])
        z = prog("ae6_enc", x)
        np.testing.assert_allclose(z, x)

        x_hat = prog("ae6_dec", x)
        np.testing.assert_allclose(x_hat, x)
