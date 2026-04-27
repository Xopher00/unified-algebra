"""Fuzzy closure via fixpoint iteration with Boltzmann temperature.

Demonstrates the ua_tensors/core/fixpoint.py pattern:
  - State = (R, T) where R is the relation matrix, T is temperature
  - Step: smooth contraction at current T, then derive new T from energy
  - Energy: sum of squared state change (dynamic error)
  - Temperature: derived from energy via Boltzmann formula

Uses the contraction engine directly with temperature-parameterized ops.
The fixpoint iteration is a plain Python loop mirroring ua_tensors'
FixpointIterator.run() — the compiled fixpoint path would work identically
if wired through a product-sort step op.
"""

import numpy as np
import pytest
from unialg import NumpyBackend, compile_einsum, semiring_contract
from unialg.algebra.semiring import Semiring


def _smooth_fuzzy_contract(A, B, T, backend):
    """Run "ij,jk->ik" with smooth fuzzy ops at temperature T."""
    def smooth_min(a, b):
        return -T * np.logaddexp(-a / T, -b / T)

    def smooth_max_reduce(tensor, axis):
        if isinstance(axis, (tuple, list)):
            result = tensor
            for ax in sorted(axis, reverse=True):
                result = smooth_max_reduce(result, ax)
            return result
        n = tensor.shape[axis]
        idx = [slice(None)] * tensor.ndim
        idx[axis] = 0
        result = tensor[tuple(idx)]
        for i in range(1, n):
            idx[axis] = i
            result = T * np.logaddexp(result / T, tensor[tuple(idx)] / T)
        return result

    sr = Semiring.Resolved(
        name="smooth_fuzzy",
        plus_name="smooth_max", times_name="smooth_min",
        plus_elementwise=lambda a, b: T * np.logaddexp(a/T, b/T),
        plus_reduce=smooth_max_reduce,
        times_elementwise=smooth_min,
        times_reduce=lambda x, axis: -smooth_max_reduce(-x, axis),
        zero=float('-inf'), one=float('inf'),
    )
    einsum = compile_einsum("ij,jk->ik")
    return semiring_contract(einsum, [A, B], sr, backend)


def _energy(new, old):
    """Dynamic error: sum of squared state change."""
    return float(np.sum(np.abs(new - old) ** 2))


def _update_temp(energy, state, eps):
    """Boltzmann-derived temperature from energy and state magnitude."""
    clipped = np.clip(state, eps, 1.0)
    log_mean = float(np.mean(np.log(clipped)))
    if log_mean == 0:
        return abs(energy)
    n = state.size
    return abs(-energy / (n * log_mean))


class TestFuzzyClosure:
    """Transitive closure via fixpoint iteration with adaptive temperature."""

    def test_closure_converges(self):
        """Iterated smooth join converges on a small relation."""
        backend = NumpyBackend()

        # 3-node relation: 0→1 (0.8), 1→2 (0.7), 0→2 (0.3)
        E = np.array([[0.0, 0.8, 0.3],
                      [0.0, 0.0, 0.7],
                      [0.0, 0.0, 0.0]])

        R = E.copy()
        T = 1.0
        eps = 1e-3
        max_iter = 50
        energies = []

        for i in range(max_iter):
            new_R = _smooth_fuzzy_contract(R, E, T, backend)
            # Merge with original (SmoothMax)
            merged = T * np.logaddexp(new_R / T, E / T)
            np.fill_diagonal(merged, 0)

            e = _energy(merged, R)
            energies.append(e)
            T = _update_temp(e, R, eps)
            R = merged

            if e <= eps:
                break

        # Must converge within max_iter
        assert energies[-1] <= eps, f"Did not converge: final energy={energies[-1]}"
        # Energy must decrease monotonically (with tolerance for numerical noise)
        for j in range(2, len(energies)):
            assert energies[j] <= energies[j-2] + 0.1, \
                f"Energy not decreasing at step {j}: {energies[j]} > {energies[j-2]}"

    def test_closure_finds_transitive_paths(self):
        """Closure discovers indirect paths through the relation."""
        backend = NumpyBackend()

        # Chain: 0→1 (0.9), 1→2 (0.8) — no direct 0→2
        E = np.array([[0.0, 0.9, 0.0],
                      [0.0, 0.0, 0.8],
                      [0.0, 0.0, 0.0]])

        R = E.copy()
        T = 1.0
        eps = 1e-3

        for _ in range(30):
            new_R = _smooth_fuzzy_contract(R, E, T, backend)
            merged = T * np.logaddexp(new_R / T, E / T)
            np.fill_diagonal(merged, 0)
            e = _energy(merged, R)
            T = _update_temp(e, R, eps)
            R = merged
            if e <= eps:
                break

        # Transitive path: 0→1→2 should have strength min(0.9, 0.8) ≈ 0.8
        assert R[0, 2] > 0.5, f"Expected transitive path 0→2, got {R[0, 2]}"
        # Direct edges preserved
        assert R[0, 1] >= 0.85

    def test_temperature_decreases_with_convergence(self):
        """Temperature falls as the state stabilizes."""
        backend = NumpyBackend()

        E = np.array([[0.0, 0.6, 0.0, 0.0],
                      [0.0, 0.0, 0.5, 0.0],
                      [0.0, 0.0, 0.0, 0.4],
                      [0.0, 0.0, 0.0, 0.0]])

        R = E.copy()
        T = 1.0
        eps = 1e-3
        temps = [T]

        for _ in range(30):
            new_R = _smooth_fuzzy_contract(R, E, T, backend)
            merged = T * np.logaddexp(new_R / T, E / T)
            np.fill_diagonal(merged, 0)
            e = _energy(merged, R)
            T = _update_temp(e, R, eps)
            temps.append(T)
            R = merged
            if e <= eps:
                break

        # Temperature should generally decrease as system converges
        assert temps[-1] < temps[0], \
            f"Final T ({temps[-1]}) should be less than initial T ({temps[0]})"

    def test_hard_at_convergence(self):
        """When converged, low T means the result is close to hard max-min."""
        backend = NumpyBackend()

        E = np.array([[0.0, 0.9, 0.3],
                      [0.0, 0.0, 0.8],
                      [0.0, 0.0, 0.0]])

        # Run smooth closure to convergence
        R_smooth = E.copy()
        T = 1.0
        eps = 1e-3
        for _ in range(50):
            new_R = _smooth_fuzzy_contract(R_smooth, E, T, backend)
            merged = T * np.logaddexp(new_R / T, E / T)
            np.fill_diagonal(merged, 0)
            e = _energy(merged, R_smooth)
            T = _update_temp(e, R_smooth, eps)
            R_smooth = merged
            if e <= eps:
                break

        # Run hard closure for comparison
        R_hard = E.copy()
        for _ in range(50):
            new_R = np.array([
                [max(min(R_hard[i, j], E[j, k]) for j in range(3))
                 for k in range(3)]
                for i in range(3)
            ])
            merged = np.maximum(new_R, E)
            np.fill_diagonal(merged, 0)
            if np.allclose(merged, R_hard):
                break
            R_hard = merged

        # At convergence, smooth result should be close to hard
        np.testing.assert_allclose(R_smooth, R_hard, atol=0.15)
