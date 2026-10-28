"""Tests for cits.cits_rcit (RCIT kernel CI test variant)."""
import numpy as np
import pytest

import cits
from cits import cits_rcit
from cits.rcit import rcit_test

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def _nonlinear_chain(T=1500, seed=0):
    """0 -> 1 -> 2 with nonlinear (sin) couplings, uniform noise; 3 independent."""
    rng = np.random.default_rng(seed)
    X = np.zeros((4, T))
    e = rng.uniform(0, 1, size=(4, T))
    for t in range(1, T):
        X[0, t] = e[0, t]
        X[1, t] = 3 * np.sin(2 * X[0, t - 1]) + e[1, t]
        X[2, t] = 3 * np.sin(X[1, t - 1]) + e[2, t]
        X[3, t] = e[3, t]
    truth = np.zeros((4, 4), dtype=int)
    truth[0, 1] = truth[1, 2] = 1
    return X, truth


def test_exported_and_dispatched():
    assert "cits_rcit" in dir(cits)
    assert "rcit" in cits._METHODS


def test_rcit_test_dependence_and_independence():
    rng = np.random.default_rng(1)
    x = rng.normal(size=600)
    assert rcit_test(x, np.sin(2 * x) + 0.2 * rng.normal(size=600), n_perm=50) < 0.05
    z = rng.normal(size=600)
    a = z + 0.5 * rng.normal(size=600)
    b = z + 0.5 * rng.normal(size=600)
    assert rcit_test(a, b, n_perm=50) < 0.05          # marginally dependent
    assert rcit_test(a, b, z=z, n_perm=50) > 0.05     # independent given z


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed (pip install cits[rcit])")
def test_gamma_recovers_nonlinear_chain_cpu():
    X, truth = _nonlinear_chain()
    B = cits_rcit(X, alpha=0.05, tau=1, seed=0, device="cpu")
    off = ~np.eye(4, dtype=bool)
    assert np.array_equal(B[off], truth[off])


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed (pip install cits[rcit])")
def test_gamma_reproducible_and_run_dispatch():
    X, _ = _nonlinear_chain(T=900, seed=2)
    B1 = cits_rcit(X, seed=3, device="cpu", max_cond_size=5)
    B2 = cits.run(X, "rcit", seed=3, device="cpu", max_cond_size=5)
    assert np.array_equal(B1, B2)


def test_perm_reference_recovers_nonlinear_chain():
    X, truth = _nonlinear_chain(T=900)
    B = cits_rcit(X, null="perm", n_perm=50, max_cond_size=2, seed=0)
    off = ~np.eye(4, dtype=bool)
    assert np.array_equal(B[off], truth[off])


def test_validation_errors():
    with pytest.raises(ValueError):
        cits_rcit(np.zeros(10))
    X = np.random.default_rng(0).normal(size=(3, 300))
    X[0, 5] = np.nan
    with pytest.raises(ValueError):
        cits_rcit(X)
    with pytest.raises(ValueError):
        cits_rcit(np.random.default_rng(0).normal(size=(3, 300)), null="bogus")
