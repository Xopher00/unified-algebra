"""Autoencoder tests: autoencoder as a lens primitive.

An autoencoder is a bidirectional morphism:
  forward (encoder): input → latent   (compression)
  backward (decoder): latent → input  (reconstruction)

The lens primitive pairs these two directions as a single compositional
unit. This test file expresses autoencoders entirely with the existing
lens/lens_path DSL — no new DSL code is needed.

Single autoencoder:
  encoder: "ij,j->i" — linear projection from input_dim to latent_dim
  decoder: "ij,j->i" — linear projection from latent_dim to input_dim
  lens("autoencoder", forward="encoder", backward="decoder")

Deep autoencoder (two encoding layers, two decoding layers):
  encoder1: input  → hidden   ("ij,j->i")
  encoder2: hidden → latent   ("ij,j->i")
  decoder1: latent → hidden   ("ij,j->i")
  decoder2: hidden → input    ("ij,j->i")
  Two lenses composed via LensPathSpec.

Tests:
  1. test_single_lens_autoencoder_assembles  — graph assembly succeeds
  2. test_forward_encodes                    — encoder primitive maps input → latent
  3. test_backward_decodes                   — decoder primitive maps latent → input
  4. test_deep_autoencoder_assembles         — two-lens deep autoencoder assembles
  5. test_semiring_polymorphism              — tropical semiring lens works unchanged
"""

import numpy as np
import pytest

from hydra.context import Context
from hydra.core import Name
from hydra.dsl.python import FrozenDict, Right
from hydra.dsl.terms import apply, var
from hydra.reduction import reduce_term

from unialg import (
    numpy_backend, Semiring, sort, tensor_coder,
    Equation,
    lens, validate_lens,
    assemble_graph, LensPathSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return numpy_backend()


@pytest.fixture
def cx():
    return Context(trace=(), messages=(), other=FrozenDict({}))


@pytest.fixture
def coder():
    return tensor_coder()


@pytest.fixture
def real_sr():
    return Semiring("ae_real", plus="add", times="multiply", zero=0.0, one=1.0)


@pytest.fixture
def tropical_sr():
    return Semiring("ae_tropical", plus="minimum", times="add", zero=float("inf"), one=0.0)


@pytest.fixture
def input_sort(real_sr):
    """High-dimensional input space (dim=6)."""
    return sort("ae_input", real_sr)


@pytest.fixture
def latent_sort(real_sr):
    """Low-dimensional latent/bottleneck space (dim=3)."""
    return sort("ae_latent", real_sr)


@pytest.fixture
def hidden_sort(real_sr):
    """Intermediate hidden space for deep autoencoder (dim=4)."""
    return sort("ae_hidden", real_sr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_array(coder, arr):
    result = coder.decode(None, np.ascontiguousarray(arr, dtype=np.float64))
    assert isinstance(result, Right)
    return result.value


def decode_term(coder, term):
    result = coder.encode(None, None, term)
    assert isinstance(result, Right)
    return result.value


def assert_reduce_ok(cx, graph, term):
    result = reduce_term(cx, graph, True, term)
    assert isinstance(result, Right), f"reduce_term returned Left: {result}"
    return result.value


# ===========================================================================
# TestAutoencoder
# ===========================================================================

class TestAutoencoder:
    """Express an autoencoder using the lens primitive."""

    # -----------------------------------------------------------------------
    # Test 1: single lens autoencoder assembles
    # -----------------------------------------------------------------------

    def test_single_lens_autoencoder_assembles(
        self, real_sr, input_sort, latent_sort, backend
    ):
        """Single lens autoencoder with encoder/decoder equations assembles correctly.

        The encoder maps input → latent ("ij,j->i") and the decoder maps
        latent → input ("ij,j->i"). Wrapping them in a lens and assembling
        with LensPathSpec should succeed without errors and register both
        the forward and backward path bound_terms.
        """
        # encoder: input_sort → latent_sort  (compression)
        # decoder: latent_sort → input_sort  (reconstruction)
        eq_enc = Equation("ae1_enc", "ij,j->i", input_sort, latent_sort, real_sr)
        eq_dec = Equation("ae1_dec", "ij,j->i", latent_sort, input_sort, real_sr)
        ae_lens = lens("ae1", "ae1_enc", "ae1_dec")

        # Weight matrices for path params
        W_enc = np.random.randn(3, 6)   # latent_dim=3, input_dim=6
        W_dec = np.random.randn(6, 3)   # input_dim=6, latent_dim=3
        coder = tensor_coder()
        W_enc_term = encode_array(coder, W_enc)
        W_dec_term = encode_array(coder, W_dec)

        graph = assemble_graph(
            [eq_enc, eq_dec], backend,
            lenses=[ae_lens],
            specs=[LensPathSpec(
                "ae1_pipe",
                ["ae1"],
                input_sort,
                input_sort,
                params={
                    "ae1_enc": [W_enc_term],
                    "ae1_dec": [W_dec_term],
                },
            )],
        )

        assert Name("ua.path.ae1_pipe.fwd") in graph.bound_terms
        assert Name("ua.path.ae1_pipe.bwd") in graph.bound_terms

    # -----------------------------------------------------------------------
    # Test 2: forward encodes (encoder primitive maps input → latent)
    # -----------------------------------------------------------------------

    def test_forward_encodes(
        self, cx, real_sr, input_sort, latent_sort, backend, coder
    ):
        """Run just the encoder equation; verify output has latent dimension.

        The encoder equation "ij,j->i" contracts a weight matrix W (latent x input)
        against input x (input,) to produce z (latent,).
        """
        eq_enc = Equation("ae2_enc", "ij,j->i", input_sort, latent_sort, real_sr)
        prim = eq_enc.resolve(backend)

        # W: 3 x 6,  x: 6  →  z: 3
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

        # W @ x selects first three components
        np.testing.assert_allclose(z, W @ x)
        assert z.shape == (3,), f"Expected latent dim 3, got shape {z.shape}"

    # -----------------------------------------------------------------------
    # Test 3: backward decodes (decoder primitive maps latent → input)
    # -----------------------------------------------------------------------

    def test_backward_decodes(
        self, cx, real_sr, input_sort, latent_sort, backend, coder
    ):
        """Run just the decoder equation; verify output has input dimension.

        The decoder equation "ij,j->i" contracts a weight matrix W_dec
        (input x latent) against latent z (latent,) to reconstruct x_hat (input,).
        """
        eq_dec = Equation("ae3_dec", "ij,j->i", latent_sort, input_sort, real_sr)
        prim = eq_dec.resolve(backend)

        # W_dec: 6 x 3,  z: 3  →  x_hat: 6
        W_dec = np.eye(6, 3)   # first 3 columns of identity
        z = np.array([1.0, 2.0, 3.0])

        result = prim.implementation(cx, None, (
            encode_array(coder, W_dec),
            encode_array(coder, z),
        ))
        assert isinstance(result, Right)
        x_hat = decode_term(coder, result.value)

        np.testing.assert_allclose(x_hat, W_dec @ z)
        assert x_hat.shape == (6,), f"Expected input dim 6, got shape {x_hat.shape}"

    # -----------------------------------------------------------------------
    # Test 4: deep autoencoder assembles (two lenses via lens_path)
    # -----------------------------------------------------------------------

    def test_deep_autoencoder_assembles(
        self, real_sr, input_sort, hidden_sort, latent_sort, backend, coder
    ):
        """Two-lens composition for a deep autoencoder assembles correctly.

        Architecture:
          encoder1: input  → hidden  (lens ae_deep1 forward)
          encoder2: hidden → latent  (lens ae_deep2 forward)
          decoder1: latent → hidden  (lens ae_deep2 backward)
          decoder2: hidden → input   (lens ae_deep1 backward)

        Composed as LensPathSpec(["ae_deep1", "ae_deep2"]).
        Forward:  encoder1 then encoder2  (input → hidden → latent)
        Backward: decoder1 then decoder2  (latent → hidden → input, reversed order)
        """
        # encoder1: input → hidden  (4 x 6)
        eq_enc1 = Equation("ae4_enc1", "ij,j->i", input_sort, hidden_sort, real_sr)
        # encoder2: hidden → latent  (3 x 4)
        eq_enc2 = Equation("ae4_enc2", "ij,j->i", hidden_sort, latent_sort, real_sr)
        # decoder1: latent → hidden  (4 x 3)
        eq_dec1 = Equation("ae4_dec1", "ij,j->i", latent_sort, hidden_sort, real_sr)
        # decoder2: hidden → input   (6 x 4)
        eq_dec2 = Equation("ae4_dec2", "ij,j->i", hidden_sort, input_sort, real_sr)

        lens1 = lens("ae_deep1", "ae4_enc1", "ae4_dec2")
        lens2 = lens("ae_deep2", "ae4_enc2", "ae4_dec1")

        W1 = np.random.randn(4, 6)   # enc1: hidden x input
        W2 = np.random.randn(3, 4)   # enc2: latent x hidden
        W3 = np.random.randn(4, 3)   # dec1: hidden x latent
        W4 = np.random.randn(6, 4)   # dec2: input x hidden

        W1_t = encode_array(coder, W1)
        W2_t = encode_array(coder, W2)
        W3_t = encode_array(coder, W3)
        W4_t = encode_array(coder, W4)

        graph = assemble_graph(
            [eq_enc1, eq_enc2, eq_dec1, eq_dec2], backend,
            lenses=[lens1, lens2],
            specs=[LensPathSpec(
                "ae_deep_pipe",
                ["ae_deep1", "ae_deep2"],
                input_sort,
                input_sort,
                params={
                    "ae4_enc1": [W1_t],
                    "ae4_enc2": [W2_t],
                    "ae4_dec1": [W3_t],
                    "ae4_dec2": [W4_t],
                },
            )],
        )

        assert Name("ua.path.ae_deep_pipe.fwd") in graph.bound_terms
        assert Name("ua.path.ae_deep_pipe.bwd") in graph.bound_terms

    # -----------------------------------------------------------------------
    # Test 5: semiring polymorphism
    # -----------------------------------------------------------------------

    def test_semiring_polymorphism(
        self, cx, tropical_sr, backend, coder
    ):
        """Same autoencoder lens structure works with the tropical semiring.

        The tropical semiring (min, +) gives min-plus linear algebra.
        A "ij,j->i" contraction under tropical is:
          z_i = min_j(W_ij + x_j)
        The encoder and decoder are structurally identical to the real case;
        only the semiring arithmetic differs.

        We use "i->i" (identity contraction — no reduction) to allow exact
        numeric verification independent of matrix dimensions.
        """
        trop_input = sort("ae_trop_input", tropical_sr)
        trop_latent = sort("ae_trop_latent", tropical_sr)

        # Identity contraction: no weight, no reduction — output equals input
        eq_enc = Equation("ae6_enc", "i->i", trop_input, trop_latent, tropical_sr)
        eq_dec = Equation("ae6_dec", "i->i", trop_latent, trop_input, tropical_sr)
        ae_lens = lens("ae6", "ae6_enc", "ae6_dec")

        graph = assemble_graph(
            [eq_enc, eq_dec], backend,
            lenses=[ae_lens],
            specs=[LensPathSpec(
                "ae6_pipe",
                ["ae6"],
                trop_input,
                trop_input,
            )],
        )

        assert Name("ua.path.ae6_pipe.fwd") in graph.bound_terms
        assert Name("ua.path.ae6_pipe.bwd") in graph.bound_terms

        # Under tropical "i->i": output == input (identity)
        x = np.array([1.0, 3.0, 2.0])
        x_enc = encode_array(coder, x)

        z_term = assert_reduce_ok(cx, graph, apply(var("ua.path.ae6_pipe.fwd"), x_enc))
        z = decode_term(coder, z_term)
        np.testing.assert_allclose(z, x)

        x_hat_term = assert_reduce_ok(cx, graph, apply(var("ua.path.ae6_pipe.bwd"), x_enc))
        x_hat = decode_term(coder, x_hat_term)
        np.testing.assert_allclose(x_hat, x)
