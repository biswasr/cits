"""
Tests for the v1.8 usability features: run() dispatch, input validation,
cite(), and the plotting helpers. CPU-safe (GPU-dependent paths are skipped
when cuPC is unavailable).
"""

import logging

import numpy as np
import pytest


def _toy(p=4, T=800, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((p, T))
    X[:, 0] = rng.standard_normal(p)
    for t in range(1, T):
        X[0, t] = rng.standard_normal() + 1.0
        X[1, t] = rng.standard_normal() - 1.0
        X[2, t] = 2.0 * X[0, t - 1] + X[1, t - 1] + rng.standard_normal()
        X[3, t] = 2.0 * X[2, t - 1] + rng.standard_normal()
    return X


# ----------------------------- run() facade ------------------------------

def test_run_exported_and_dispatches_base():
    import cits
    assert callable(cits.run)
    X = _toy()
    adj = np.asarray(cits.run(X, 'base', tau=1, alpha=0.05))
    assert adj.shape == (X.shape[0], X.shape[0])


def test_run_unknown_method_lists_valid():
    import cits
    with pytest.raises(ValueError) as e:
        cits.run(_toy(), 'nope')
    msg = str(e.value)
    assert all(m in msg for m in ('base', 'gpu', 'contemporaneous'))


# --------------------------- input validation ----------------------------

def test_validate_nan_raises():
    from cits._common import validate_X
    with pytest.raises(ValueError) as e:
        validate_X(np.array([[1.0, np.nan], [2.0, 3.0]]), 't')
    assert 'non-finite' in str(e.value)


def test_validate_1d_raises():
    from cits._common import validate_X
    with pytest.raises(ValueError) as e:
        validate_X(np.arange(10), 't')
    assert '2D' in str(e.value)


def test_validate_transpose_and_constant_warn(caplog):
    from cits._common import validate_X
    with caplog.at_level(logging.WARNING, logger='cits'):
        validate_X(np.random.randn(50, 4), 't')          # p>T
        Z = np.random.randn(4, 200); Z[2, :] = 3.0        # constant row
        validate_X(Z, 't')
    text = " ".join(r.getMessage() for r in caplog.records)
    assert 'timepoints' in text          # transpose warning
    assert 'constant' in text            # dead-row warning


# -------------------------------- cite() ---------------------------------

def test_cite_runs(capsys):
    import cits
    cits.cite()
    out = capsys.readouterr().out
    assert 'arxiv.org/abs/2508.01920' in out.lower()


# ------------------------------- plotting ---------------------------------

def test_plot_functions_exist():
    import cits
    assert callable(cits.plot_graph)
    assert callable(cits.plot_matrix)


def test_plot_graph_and_matrix_run(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cits

    A = np.array([[0, 0.8, 0, 0],
                  [0, 0, -0.5, 0],
                  [0, 0.6, 0, 1.2],
                  [0, 0, 1.2, 0]], float)  # reciprocal 2<->3
    A[1, 3] = np.nan                        # skeleton-only edge
    groups = {0: 'X', 1: 'X', 2: 'Y', 3: 'Y'}

    cits.plot_graph(A, labels=list('abcd'), groups=groups, title='g')
    g_png = tmp_path / "g.png"
    plt.savefig(g_png); plt.close()
    assert g_png.exists() and g_png.stat().st_size > 0

    cits.plot_matrix(A, labels=list('abcd'), groups=groups, title='m')
    m_png = tmp_path / "m.png"
    plt.savefig(m_png); plt.close()
    assert m_png.exists() and m_png.stat().st_size > 0
