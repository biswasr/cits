"""Tests for the PC search without a cap on the conditioning-set size (cits 1.10.0), on the CPU and on the CITS
fork of cuPC, and for the handling of an outdated cuPC library.

Graph used for levels above 14: x -> m_1..m_15 -> y. The only set that separates x and y is all 15 mediators,
so PC-stable must remove x - y at level 15. Mediator pairs separate given x (level 1).
"""
import os
import warnings

import numpy as np
import pytest

N_MED = 15  # mediators; x - y is removed at level N_MED
X_NODE, Y_NODE = 0, N_MED + 1
ALPHA_SYNTH = 1e-6  # synthetic GPU checks against the true graph


def _mediator_oracle(a, b, S):
    """d-separation in x -> m_i -> y (x = 0, m_i = 1..15, y = 16)."""
    S = set(S)
    pair = {a, b}
    meds = set(range(1, N_MED + 1))
    if pair == {X_NODE, Y_NODE}:
        return meds <= S
    if X_NODE in pair or Y_NODE in pair:
        return False  # x - m_i and m_i - y are adjacent
    return X_NODE in S and Y_NODE not in S  # two mediators: blocked at x, collider at y


def _expected_mediator_skeleton():
    p = N_MED + 2
    G = np.zeros((p, p), dtype=int)
    for m in range(1, N_MED + 1):
        G[X_NODE, m] = G[m, X_NODE] = 1
        G[m, Y_NODE] = G[Y_NODE, m] = 1
    return G


def _mediator_data(N=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    M = x[:, None] + rng.standard_normal((N, N_MED))
    y = M.sum(1) / np.sqrt(N_MED) + rng.standard_normal(N)
    return np.column_stack([x, M, y])


def _fork_available():
    try:
        from cits._cupc_wrapper import cupc_info
        return bool(cupc_info()["uncapped"])
    except Exception:
        return False


@pytest.fixture
def fresh_wrapper(monkeypatch):
    """The cuPC wrapper with its per-process state reset (restored after the test)."""
    from cits import _cupc_wrapper as W
    for name in ("_LIB", "_LIB_INFO", "_OUTDATED_OK"):
        monkeypatch.setattr(W, name, None)
    monkeypatch.delenv("CITS_ALLOW_OUTDATED_CUPC", raising=False)
    return W


# --------------------------------------------------------------------------- CPU, no GPU needed

def test_cpu_default_has_no_level_cap():
    import inspect
    from cits._pc_skeleton_cpu import pc_skeleton_cpu
    from cits.gpu import cits_gpu
    from cits._cupc_wrapper import pc_skeleton_cupc
    for fn in (pc_skeleton_cpu, pc_skeleton_cupc, cits_gpu):
        assert inspect.signature(fn).parameters["max_level"].default is None


def test_cpu_removes_an_edge_at_level_15(monkeypatch):
    """With a d-separation oracle, the CPU search must continue past level 14 and remove x - y at level 15
    with all mediators as the separating set."""
    import cits._pc_skeleton_cpu as cpu
    monkeypatch.setattr(cpu, "_independent", lambda x, y, S, data, alpha: _mediator_oracle(x, y, S))
    X = np.random.default_rng(0).standard_normal((100, N_MED + 2))
    G, sep, _, level = cpu.pc_skeleton_cpu(X, alpha=0.05)
    assert np.array_equal(G, _expected_mediator_skeleton())
    assert set(sep[(X_NODE, Y_NODE)]) == set(range(1, N_MED + 1))
    assert sep[(1, 2)] == (X_NODE,)
    assert level == N_MED + 1


def test_cpu_max_level_still_caps_on_request(monkeypatch):
    import cits._pc_skeleton_cpu as cpu
    monkeypatch.setattr(cpu, "_independent", lambda x, y, S, data, alpha: _mediator_oracle(x, y, S))
    X = np.random.default_rng(0).standard_normal((100, N_MED + 2))
    G, _, _, level = cpu.pc_skeleton_cpu(X, alpha=0.05, max_level=14)
    assert level == 14
    assert G[X_NODE, Y_NODE] == 1  # needs level 15


def test_cpu_stops_when_no_test_can_remove_an_edge(monkeypatch):
    """With T samples, a level with T - level - 3 <= 0 cannot remove an edge, so the search stops there. With 8
    samples and 7 variables that is level 5. Without the rule the search would run to level 6."""
    import cits._pc_skeleton_cpu as cpu
    monkeypatch.setattr(cpu, "_independent", lambda x, y, S, data, alpha: False)
    X = np.random.default_rng(3).standard_normal((8, 7))
    G, _, _, level = cpu.pc_skeleton_cpu(X, alpha=0.05)
    assert level == 8 - 3
    assert G.sum() == 7 * 6


def test_cpu_return_sep_sets_false():
    from cits._pc_skeleton_cpu import pc_skeleton_cpu
    X = np.random.default_rng(1).standard_normal((300, 6))
    _, sep, _, _ = pc_skeleton_cpu(X, alpha=0.05, return_sep_sets=False)
    assert sep is None


# --------------------------------------------------------------------------- outdated library, no GPU needed

def _pretend_loaded(W, monkeypatch, **info):
    base = dict(path="/somewhere/Skeleton.so", version=None, uncapped=False, dev_build=False)
    base.update(info)
    monkeypatch.setattr(W, "_LIB", object())
    monkeypatch.setattr(W, "_LIB_INFO", base)


def test_outdated_library_stops_a_script_with_rebuild_steps(fresh_wrapper, monkeypatch):
    W = fresh_wrapper
    _pretend_loaded(W, monkeypatch)
    monkeypatch.setattr(W, "_interactive", lambda: False)
    with pytest.raises(W.OutdatedCupcError) as e:
        W._confirm_outdated()
    msg = str(e.value)
    assert "level 14" in msg and "git clone" in msg and "nvcc" in msg and "CITS_ALLOW_OUTDATED_CUPC" in msg


def test_outdated_library_prompt_yes_continues_once(fresh_wrapper, monkeypatch, capsys):
    W = fresh_wrapper
    _pretend_loaded(W, monkeypatch)
    monkeypatch.setattr(W, "_interactive", lambda: True)
    answers = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    with pytest.warns(UserWarning):
        assert W._confirm_outdated() is True
    assert "git clone" in capsys.readouterr().err  # the rebuild steps were shown before the question
    assert W._confirm_outdated() is True  # remembered for the session; input() is not called again


def test_outdated_library_prompt_no_raises_and_is_remembered(fresh_wrapper, monkeypatch):
    W = fresh_wrapper
    _pretend_loaded(W, monkeypatch)
    monkeypatch.setattr(W, "_interactive", lambda: True)
    calls = []
    monkeypatch.setattr("builtins.input", lambda prompt="": calls.append(prompt) or "n")
    with pytest.raises(W.OutdatedCupcError):
        W._confirm_outdated()
    with pytest.raises(W.OutdatedCupcError):
        W._confirm_outdated()
    assert len(calls) == 1


def test_outdated_library_env_override_warns(fresh_wrapper, monkeypatch):
    W = fresh_wrapper
    _pretend_loaded(W, monkeypatch)
    monkeypatch.setattr(W, "_interactive", lambda: False)
    monkeypatch.setenv("CITS_ALLOW_OUTDATED_CUPC", "1")
    with pytest.warns(UserWarning, match="level 14"):
        assert W._confirm_outdated() is True


def test_development_build_is_always_refused(fresh_wrapper, monkeypatch):
    """A build with ext_* functions but no version has a different Skeleton() signature and cannot run."""
    W = fresh_wrapper
    _pretend_loaded(W, monkeypatch, dev_build=True)
    monkeypatch.setenv("CITS_ALLOW_OUTDATED_CUPC", "1")
    with pytest.raises(W.OutdatedCupcError, match="development build"):
        W._confirm_outdated()


def test_cupc_info_is_exported():
    import cits
    assert callable(cits.cupc_info)
    assert issubclass(cits.OutdatedCupcError, RuntimeError)
    assert "cupc_info" in cits.__all__ and "OutdatedCupcError" in cits.__all__


# --------------------------------------------------------------------------- GPU (CITS fork of cuPC)

def test_gpu_removes_an_edge_at_level_15():
    """Real data from x -> m_1..m_15 -> y. The small alpha makes a false keep of a truly independent pair
    negligible, while every true dependence stays far beyond the threshold (smallest statistic about 9)."""
    if not _fork_available():
        pytest.skip("CITS fork of cuPC unavailable")
    from cits._cupc_wrapper import pc_skeleton_cupc
    G, sep, _, level = pc_skeleton_cupc(_mediator_data(), alpha=ALPHA_SYNTH)
    assert np.array_equal(G, _expected_mediator_skeleton())
    assert set(sep[(X_NODE, Y_NODE)]) == set(range(1, N_MED + 1))
    assert level == N_MED + 1


def test_gpu_max_level_caps_on_request():
    if not _fork_available():
        pytest.skip("CITS fork of cuPC unavailable")
    from cits._cupc_wrapper import pc_skeleton_cupc
    G, _, _, level = pc_skeleton_cupc(_mediator_data(), alpha=ALPHA_SYNTH, max_level=14)
    assert level == 14
    assert G[X_NODE, Y_NODE] == 1


def test_gpu_matches_cpu_and_sep_sets_separate():
    """Random sparse linear-Gaussian graph: same skeleton as the CPU, and every GPU separating set passes the
    CPU test for its pair."""
    if not _fork_available():
        pytest.skip("CITS fork of cuPC unavailable")
    from cits._cupc_wrapper import pc_skeleton_cupc
    from cits._pc_skeleton_cpu import pc_skeleton_cpu, _independent
    rng = np.random.default_rng(7)
    p, N = 24, 3000
    W_ = np.triu((rng.random((p, p)) < 0.18) * rng.uniform(0.5, 1.0, (p, p)), 1)
    X = np.zeros((N, p))
    for j in range(p):
        X[:, j] = X @ W_[:, j] + rng.standard_normal(N)
    G_gpu, sep_gpu, _, lv_gpu = pc_skeleton_cupc(X, alpha=0.05)
    G_cpu, sep_cpu, _, lv_cpu = pc_skeleton_cpu(X, alpha=0.05)
    assert np.array_equal(G_gpu, G_cpu)
    assert lv_gpu == lv_cpu
    assert set(sep_gpu) == set(sep_cpu)
    data = np.ascontiguousarray(X.T)
    for (i, j), S in sep_gpu.items():
        if S != sep_cpu[(i, j)]:
            assert _independent(i, j, S, data, 0.05)


@pytest.mark.skipif(os.environ.get("CITS_SLOW_TESTS") != "1", reason="slow (about 4 min on the CPU); set CITS_SLOW_TESTS=1")
def test_gpu_matches_cpu_on_a_star_beyond_level_14():
    """Hub with 16 parents: the search reaches level 16 on both backends with identical skeletons."""
    if not _fork_available():
        pytest.skip("CITS fork of cuPC unavailable")
    from cits._cupc_wrapper import pc_skeleton_cupc
    from cits._pc_skeleton_cpu import pc_skeleton_cpu
    rng = np.random.default_rng(0)
    L = rng.standard_normal((600, 16))
    X = np.column_stack([0.5 * L.sum(1) + rng.standard_normal(600), L])
    G_gpu, _, _, lv_gpu = pc_skeleton_cupc(X, alpha=0.05)
    G_cpu, _, _, lv_cpu = pc_skeleton_cpu(X, alpha=0.05)
    assert np.array_equal(G_gpu, G_cpu) and lv_gpu == lv_cpu == 16


def _toy_lag_series(p=4, T=800, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((p, T))
    X[:, 0] = rng.standard_normal(p)
    for t in range(1, T):
        X[0, t] = rng.standard_normal() + 1.0
        X[1, t] = rng.standard_normal() - 1.0
        X[2, t] = 2.0 * X[0, t - 1] + 1.0 * X[1, t - 1] + rng.standard_normal()
        X[3, t] = 1.5 * X[2, t - 1] + rng.standard_normal()
    return X


def test_contemporaneous_auto_falls_back_to_cpu_on_outdated_library(fresh_wrapper, monkeypatch, caplog):
    W = fresh_wrapper
    _pretend_loaded(W, monkeypatch)
    monkeypatch.setattr(W, "_interactive", lambda: False)
    monkeypatch.setattr(W, "is_cupc_available", lambda: True)
    import cits
    with caplog.at_level("WARNING", logger="cits"):
        B = cits.cits_contemporaneous(_toy_lag_series(), alpha=0.05, tau=1, backend="auto")
    assert np.asarray(B).shape == (4, 4)
    assert "git clone" in caplog.text and "CPU backend" in caplog.text


def test_contemporaneous_explicit_cupc_raises_on_outdated_library(fresh_wrapper, monkeypatch):
    W = fresh_wrapper
    _pretend_loaded(W, monkeypatch)
    monkeypatch.setattr(W, "_interactive", lambda: False)
    import cits
    with pytest.raises(W.OutdatedCupcError, match="git clone"):
        cits.cits_contemporaneous(_toy_lag_series(), alpha=0.05, tau=1, backend="cupc")
