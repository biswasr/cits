"""
Tests for the three CITS entry points.

  1. base CITS (cits.methods.cits_full)      -- CPU, always runs
  2. GPU CITS (cits.cits_gpu)                 -- needs cuPC; skipped otherwise
  3. Contemporaneous CITS (cits.cits_contemporaneous)      -- needs cuPC; skipped otherwise

The GPU / contemporaneous tests skip (rather than fail) when cuPC is unavailable,
so the suite passes on a CPU-only machine. A GPU is NOT required for the base
tests to pass.
"""

import numpy as np
import pytest


def _toy_series(p=4, T=800, seed=0):
    """Small linear lag-1 chain X0 -> X2 <- X1, X2 -> X3, shape (p, T)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((p, T))
    X[:, 0] = rng.standard_normal(p)
    for t in range(1, T):
        X[0, t] = rng.standard_normal() + 1.0
        X[1, t] = rng.standard_normal() - 1.0
        X[2, t] = 2.0 * X[0, t - 1] + 1.0 * X[1, t - 1] + rng.standard_normal()
        X[3, t] = 2.0 * X[2, t - 1] + rng.standard_normal()
        for extra in range(4, p):
            X[extra, t] = rng.standard_normal()
    return X


def _cupc_available():
    """True iff the CITS fork of cuPC (no level cap) can be loaded."""
    try:
        from cits._cupc_wrapper import cupc_info
        return bool(cupc_info()["uncapped"])
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  Import / export tests (no GPU needed)
# ---------------------------------------------------------------------------

def test_import_cits():
    import cits
    exported = [n for n in dir(cits) if not n.startswith("__")]
    for name in ("methods", "simulate_timeseries", "cits_gpu", "cits_contemporaneous"):
        assert name in exported, f"{name} not exported from cits ({exported})"


def test_base_entrypoint_importable():
    from cits import methods
    assert hasattr(methods, "cits_full")
    assert hasattr(methods, "cits_full_weighted")


def test_gpu_entrypoint_importable():
    import cits
    assert callable(cits.cits_gpu)


def test_contemporaneous_entrypoint_importable():
    import cits
    assert callable(cits.cits_contemporaneous)


def test_set_cupc_dir_exported():
    import cits
    assert callable(cits.set_cupc_dir)
    assert "set_cupc_dir" in cits.__all__


def test_set_cupc_dir_rejects_dir_without_skeleton(tmp_path):
    """set_cupc_dir must validate that Skeleton.so is present (no GPU needed)."""
    import cits
    with pytest.raises(FileNotFoundError) as excinfo:
        cits.set_cupc_dir(str(tmp_path))
    assert "Skeleton.so" in str(excinfo.value)


# ---------------------------------------------------------------------------
#  Base CITS runs on CPU
# ---------------------------------------------------------------------------

def test_base_cits_runs_cpu():
    from cits import methods
    X = _toy_series()
    adj = methods.cits_full(X, 1, 0.05)
    adj = np.asarray(adj)
    assert adj.shape == (X.shape[0], X.shape[0])


# ---------------------------------------------------------------------------
#  GPU / contemporaneous: skip when cuPC unavailable
# ---------------------------------------------------------------------------

def test_gpu_cits_runs_if_cupc():
    if not _cupc_available():
        pytest.skip("cuPC (Skeleton.so) unavailable; set CUPC_DIR and build it")
    import cits
    X = _toy_series()
    B = cits.cits_gpu(X, alpha=0.05, tau=1)
    B = np.asarray(B)
    assert B.shape == (X.shape[0], X.shape[0])


def test_contemporaneous_cits_runs_if_cupc():
    if not _cupc_available():
        pytest.skip("cuPC (Skeleton.so) unavailable; set CUPC_DIR and build it")
    import cits
    X = _toy_series()
    B = cits.cits_contemporaneous(X, alpha=0.05, tau=1)
    B = np.asarray(B)
    assert B.shape == (X.shape[0], X.shape[0])


def test_gpu_call_raises_clearly_without_cupc():
    """When cuPC is unavailable, calling cits_gpu must raise a clear error
    mentioning cuPC / CUPC_DIR (not a bare ImportError from `import cits`)."""
    if _cupc_available():
        pytest.skip("cuPC available; clear-error path not exercised")
    import cits
    X = _toy_series()
    with pytest.raises(Exception) as excinfo:
        cits.cits_gpu(X, alpha=0.05, tau=1)
    msg = str(excinfo.value)
    assert "cuPC" in msg or "CUPC_DIR" in msg or "Skeleton.so" in msg


# ---------------------------------------------------------------------------
#  CPU backend: always runs, no GPU required
# ---------------------------------------------------------------------------

def test_cpu_skeleton_runs():
    """Pure-numpy PC-stable skeleton runs with no GPU and returns a
    symmetric (p, p) 0/1 adjacency."""
    from cits._pc_skeleton_cpu import pc_skeleton_cpu
    rng = np.random.default_rng(1)
    U = rng.standard_normal((300, 6))
    G, sep, inactive, level = pc_skeleton_cpu(U, alpha=0.05)
    assert G.shape == (6, 6)
    assert np.array_equal(G, G.T)
    assert np.all(np.diag(G) == 0)


def test_contemporaneous_cpu_runs_without_gpu():
    """cits_contemporaneous(backend='cpu') returns a (p, p) result with no GPU."""
    import cits
    X = _toy_series()
    B = cits.cits_contemporaneous(X, alpha=0.05, tau=1, backend='cpu')
    B = np.asarray(B)
    assert B.shape == (X.shape[0], X.shape[0])


def test_cpu_matches_cupc_skeleton():
    """The CPU PC-stable skeleton must equal the cuPC skeleton on the same
    data (skipped when cuPC is unavailable)."""
    if not _cupc_available():
        pytest.skip("cuPC (Skeleton.so) unavailable; equivalence not checked")
    from cits._pc_skeleton_cpu import pc_skeleton_cpu
    from cits._cupc_wrapper import pc_skeleton_cupc
    from cits.gpu import _build_chi_nonoverlap
    from cits import simulate_timeseries
    X, _, _ = simulate_timeseries.simulate('lingauss1', noise=1.0, T=2000)
    U = _build_chi_nonoverlap(X, tau=1)
    G_cpu, _, _, _ = pc_skeleton_cpu(U, alpha=0.05)
    G_gpu, _, _, _ = pc_skeleton_cupc(U, alpha=0.05)
    assert np.array_equal(G_cpu, G_gpu)


def test_contemporaneous_bad_backend_raises():
    import cits
    X = _toy_series()
    with pytest.raises(ValueError):
        cits.cits_contemporaneous(X, alpha=0.05, tau=1, backend='nonsense')


def test_deprecated_versionb_names_still_work():
    """v1.9.0 names (cits_versionb, method='versionb') warn and give the same result."""
    import warnings
    import cits
    from cits import simulate_timeseries
    X, _, _ = simulate_timeseries.simulate('lingauss1', noise=1.0, T=600)
    new = cits.cits_contemporaneous(X, alpha=0.05, tau=1, backend='cpu')
    with pytest.warns(DeprecationWarning):
        old = cits.cits_versionb(X, alpha=0.05, tau=1, backend='cpu')
    with pytest.warns(DeprecationWarning):
        via_run = cits.run(X, 'versionb', alpha=0.05, tau=1, backend='cpu')
    assert np.array_equal(np.nan_to_num(new), np.nan_to_num(old))
    assert np.array_equal(np.nan_to_num(new), np.nan_to_num(via_run))
