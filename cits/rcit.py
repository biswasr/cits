"""
cits.rcit -- CITS with the randomized conditional independence test (RCIT).

RCIT (Strobl, Zhang & Visweswaran 2019) approximates the kernel (HSIC)
conditional-independence test with random Fourier features, so CITS can detect
nonlinear and non-Gaussian dependence at practical cost. This is the CI test
the CITS paper uses for its non-linear autoregressive and spiking-network
benchmarks.

Entry point: :func:`cits_rcit`.

Windowing follows the paper's Methods: non-overlapping windows of 2*tau + 1
time points (``N = T // (2*tau + 1)`` samples). For each candidate lagged edge
``X_u(t - s) -> X_v(t)`` (s = 1..tau), CITS searches for a separating set among
the other variables in the window, in increasing size, and keeps the edge if
none is found. The result is the rolled lagged adjacency (self-lags included).

Two null distributions:

* ``null='gamma'`` (default; the paper's setting): Strobl-2019 gamma moment
  match, batched with torch. Runs on a CUDA GPU when available, otherwise on
  CPU. Requires ``torch`` (``pip install cits[rcit]``).
* ``null='perm'``: permutation null, pure NumPy reference implementation
  (slow; no torch needed).

The computational core is copied from the code that produced the paper's
results, so it reproduces them exactly for the same arguments and seed. The
paper's benchmarks used ``K=25`` and ``max_cond_size=None`` (no cap: search
all conditioning-set sizes), which is the default; only its Markov-order
analysis used ``max_cond_size=5``.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist

__all__ = ["cits_rcit", "rcit_test"]

_RIDGE_LAMBDA = 1e-2
_DEFAULT_BATCH = 4096


# ---------------------------------------------------------------------------
# Windowing (2*tau + 1 time points per sample, non-overlapping)
# ---------------------------------------------------------------------------

def _window_samples(X: np.ndarray, tau: int) -> np.ndarray:
    """Time-windowed samples ``chi`` of shape ``(p*(2*tau+1), T // (2*tau+1))``.

    Row block ``k`` (rows ``k*p .. k*p + p - 1``) holds time slice ``k`` of each
    window; slice ``2*tau`` is the target time ``t``.
    """
    p, T = X.shape
    w = 2 * tau + 1
    N = T // w
    chi = np.zeros((p * w, N))
    for i in range(N):
        chi[:, i] = X[:, w * i: w * (i + 1)].T.reshape(p * w)
    return chi


def _candidate_edges(p: int, tau: int):
    """Candidate lagged edges in window coordinates: (i_chi, j_chi, u, v)."""
    t_target = 2 * tau
    cands = []
    for v in range(p):
        for u in range(p):
            for t1 in range(tau, 2 * tau):
                i_idx = t1 * p + u
                j_idx = t_target * p + v
                if i_idx != j_idx:
                    cands.append((i_idx, j_idx, u, v))
    return cands


def _validate(X, tau):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array of shape (p, T); got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    if int(tau) < 1:
        raise ValueError("tau must be >= 1")
    p, T = X.shape
    if T // (2 * tau + 1) < 10:
        raise ValueError(
            f"too few time points: T={T} gives {T // (2 * tau + 1)} windows of "
            f"length 2*tau+1={2 * tau + 1}")
    return X


# ---------------------------------------------------------------------------
# NumPy reference RCIT (permutation null)
# ---------------------------------------------------------------------------

def _median_bandwidth(X: np.ndarray) -> float:
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n > 200:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=200, replace=False)
        X = X[idx]
    med = np.median(pdist(X))
    if med < 1e-12:
        med = 1.0
    return float(med)


def _rff_embed(X: np.ndarray, K: int, sigma: float, rng) -> np.ndarray:
    if X.ndim == 1:
        X = X[:, None]
    N, d = X.shape
    W = rng.normal(scale=1.0 / sigma, size=(d, K))
    b = rng.uniform(0, 2 * np.pi, size=K)
    proj = X @ W + b
    phi = np.empty((N, 2 * K))
    phi[:, :K] = np.cos(proj)
    phi[:, K:] = np.sin(proj)
    phi /= np.sqrt(K)
    return phi


def _residualize(phi: np.ndarray, phi_z: np.ndarray,
                 ridge_lambda: float = _RIDGE_LAMBDA) -> np.ndarray:
    K = phi_z.shape[1]
    A = phi_z.T @ phi_z + ridge_lambda * np.eye(K)
    B = np.linalg.solve(A, phi_z.T @ phi)
    return phi - phi_z @ B


def _Tstat(phi_x: np.ndarray, phi_y: np.ndarray) -> float:
    N = phi_x.shape[0]
    Sxy = (phi_x.T @ phi_y) / N
    return N * float(np.sum(Sxy ** 2))


def rcit_test(x, y, z=None, K: int = 25, n_perm: int = 100, seed: int = 0) -> float:
    """RCIT p-value for H0: x independent of y given z (permutation null).

    Parameters
    ----------
    x, y : array_like, shape (N,) or (N, d)
    z : array_like, shape (N, d_z), or None for a marginal test
    K : int
        Number of random Fourier features per variable.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = x.shape[0]
    if y.shape[0] != N:
        raise ValueError("x and y must have the same number of rows")
    rng = np.random.default_rng(seed)
    phi_x = _rff_embed(x, K, _median_bandwidth(x), rng)
    phi_y = _rff_embed(y, K, _median_bandwidth(y), rng)
    phi_x = phi_x - phi_x.mean(axis=0, keepdims=True)
    phi_y = phi_y - phi_y.mean(axis=0, keepdims=True)
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z[:, None]
        if z.shape[0] != N:
            raise ValueError("z must have the same number of rows as x")
        phi_z = _rff_embed(z, K, _median_bandwidth(z), rng)
        phi_z = phi_z - phi_z.mean(axis=0, keepdims=True)
        phi_x = _residualize(phi_x, phi_z)
        phi_y = _residualize(phi_y, phi_z)
    T_obs = _Tstat(phi_x, phi_y)
    count_geq = 1
    for _ in range(n_perm):
        idx = rng.permutation(N)
        if _Tstat(phi_x, phi_y[idx]) >= T_obs:
            count_geq += 1
    return count_geq / (n_perm + 1)


def _cits_rcit_perm(X, alpha, tau, K, n_perm, max_cond_size, seed, verbose):
    p, _ = X.shape
    chi = _window_samples(X, tau)
    n_feat = chi.shape[0]
    C = chi.T
    pval_cache: dict = {}

    def _ci_pvalue(i, j, S):
        a, b = (i, j) if i < j else (j, i)
        S_sorted = tuple(sorted(S))
        key = (a, b, S_sorted)
        if key in pval_cache:
            return pval_cache[key]
        local_seed = (seed * 1_000_003 + a * 9973 + b * 97
                      + sum(S_sorted) * 7 + len(S_sorted)) & 0x7FFFFFFF
        z = C[:, list(S_sorted)] if S_sorted else None
        pv = rcit_test(C[:, i], C[:, j], z=z, K=K, n_perm=n_perm, seed=local_seed)
        pval_cache[key] = pv
        return pv

    B = np.zeros((p, p), dtype=int)
    for (i_idx, j_idx, u, v) in _candidate_edges(p, tau):
        cand = [k for k in range(n_feat) if k != i_idx and k != j_idx]
        if _ci_pvalue(i_idx, j_idx, ()) > alpha:
            continue
        separated = False
        for l in range(1, max_cond_size + 1):
            if l > len(cand):
                break
            for S in itertools.combinations(cand, l):
                if _ci_pvalue(i_idx, j_idx, S) > alpha:
                    separated = True
                    break
            if separated:
                break
        if not separated:
            B[u, v] = 1
    if verbose:
        print(f"[cits_rcit perm] retained lagged edges = {int(B.sum())}", flush=True)
    return B


# ---------------------------------------------------------------------------
# Batched torch RCIT (gamma null): CUDA or CPU
# ---------------------------------------------------------------------------

def _import_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "cits_rcit with null='gamma' requires PyTorch (pip install cits[rcit] "
            "or see https://pytorch.org). Use null='perm' for the NumPy-only "
            f"reference implementation. Original error: {e!r}") from e
    return torch


def _torch_median_bandwidth(torch, X, max_samples=200, seed=0):
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X.index_select(0, torch.as_tensor(idx, device=X.device, dtype=torch.long))
    else:
        X_sub = X
    d = torch.cdist(X_sub, X_sub)
    iu = torch.triu_indices(d.shape[0], d.shape[1], offset=1, device=X.device)
    med = torch.median(d[iu[0], iu[1]]).item()
    if not math.isfinite(med) or med < 1e-12:
        med = 1.0
    return float(med)


def _torch_rff_embed(torch, x, K, sigma, seed, device):
    if x.ndim == 1:
        x = x[:, None]
    N, d = x.shape
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    W = torch.randn((d, K), generator=g, device=device, dtype=x.dtype) * (1.0 / float(sigma))
    b = torch.rand((K,), generator=g, device=device, dtype=x.dtype) * (2.0 * math.pi)
    proj = x @ W + b
    phi = torch.empty((N, 2 * K), device=device, dtype=x.dtype)
    phi[:, :K] = torch.cos(proj)
    phi[:, K:] = torch.sin(proj)
    phi.div_(math.sqrt(K))
    return phi


def _precompute_rff_cache(torch, C, K, seed, device):
    N, V = C.shape
    PHI = torch.empty((V, N, 2 * K), device=device, dtype=C.dtype)
    for v in range(V):
        x_v = C[:, v]
        sigma = _torch_median_bandwidth(torch, x_v, seed=0)
        sub_seed = (int(seed) * 1_000_003 + int(v) * 9973 + 11) & 0x7FFFFFFF
        phi_v = _torch_rff_embed(torch, x_v, K, sigma, sub_seed, device)
        PHI[v] = phi_v - phi_v.mean(dim=0, keepdim=True)
    return PHI


def _batched_residualize(torch, phi, phi_z, ridge_lambda=_RIDGE_LAMBDA):
    B, N, M = phi_z.shape
    A = torch.bmm(phi_z.transpose(1, 2), phi_z)
    A = A + ridge_lambda * torch.eye(M, device=A.device, dtype=A.dtype).expand(B, M, M)
    rhs = torch.bmm(phi_z.transpose(1, 2), phi)
    try:
        Bsol = torch.linalg.solve(A, rhs)
    except RuntimeError:
        Bsol = torch.linalg.solve(A.double(), rhs.double()).to(phi.dtype)
    return phi - torch.bmm(phi_z, Bsol)


def _batched_gamma_pvalues(torch, phi_x, phi_y):
    B, N, _ = phi_x.shape
    Sxy = torch.bmm(phi_x.transpose(1, 2), phi_y) / N
    T_obs = N * (Sxy ** 2).sum(dim=(1, 2))
    Cxx = torch.bmm(phi_x.transpose(1, 2), phi_x) / N
    Cyy = torch.bmm(phi_y.transpose(1, 2), phi_y) / N
    tr_Cxx = torch.diagonal(Cxx, dim1=1, dim2=2).sum(dim=1)
    tr_Cyy = torch.diagonal(Cyy, dim1=1, dim2=2).sum(dim=1)
    tr_Cxx2 = (Cxx * Cxx).sum(dim=(1, 2))
    tr_Cyy2 = (Cyy * Cyy).sum(dim=(1, 2))
    mu = tr_Cxx * tr_Cyy
    var = 2.0 * tr_Cxx2 * tr_Cyy2
    bad32 = (~torch.isfinite(mu)) | (~torch.isfinite(var)) | (~torch.isfinite(T_obs))
    if bad32.any() and phi_x.dtype != torch.float64:
        idx_bad = torch.nonzero(bad32, as_tuple=False).squeeze(-1)
        px = phi_x.index_select(0, idx_bad).to(torch.float64)
        py = phi_y.index_select(0, idx_bad).to(torch.float64)
        Sb = torch.bmm(px.transpose(1, 2), py) / N
        Tb = N * (Sb ** 2).sum(dim=(1, 2))
        Cxb = torch.bmm(px.transpose(1, 2), px) / N
        Cyb = torch.bmm(py.transpose(1, 2), py) / N
        mu[idx_bad] = (torch.diagonal(Cxb, dim1=1, dim2=2).sum(dim=1)
                       * torch.diagonal(Cyb, dim1=1, dim2=2).sum(dim=1)).to(phi_x.dtype)
        var[idx_bad] = (2.0 * (Cxb * Cxb).sum(dim=(1, 2))
                        * (Cyb * Cyb).sum(dim=(1, 2))).to(phi_x.dtype)
        T_obs[idx_bad] = Tb.to(phi_x.dtype)
    bad = (~torch.isfinite(mu)) | (~torch.isfinite(var)) | (var <= 0) | (mu <= 0) | (T_obs <= 0)
    shape = (mu * mu) / var.clamp_min(1e-300)
    rate = mu / var.clamp_min(1e-300)
    pvals = torch.special.gammaincc(shape.to(torch.float64),
                                    (rate * T_obs).to(torch.float64)).to(phi_x.dtype)
    pvals = torch.where(bad, torch.ones_like(pvals), pvals)
    pvals = pvals.clamp_(min=1e-300, max=1.0)
    return torch.where(torch.isfinite(pvals), pvals, torch.ones_like(pvals))


def _batched_rcit_pvalues(torch, PHI, i_idx, j_idx, S_idx):
    phi_x = PHI.index_select(0, i_idx)
    phi_y = PHI.index_select(0, j_idx)
    if S_idx is None or S_idx.shape[-1] == 0:
        return _batched_gamma_pvalues(torch, phi_x, phi_y)
    B, l = S_idx.shape
    phi_z = PHI.index_select(0, S_idx.reshape(-1))
    N, twoK = phi_z.shape[1], phi_z.shape[2]
    phi_z = phi_z.view(B, l, N, twoK).permute(0, 2, 1, 3).reshape(B, N, l * twoK)
    return _batched_gamma_pvalues(torch, _batched_residualize(torch, phi_x, phi_z),
                                  _batched_residualize(torch, phi_y, phi_z))


def _key(i, j, S):
    a, b = (i, j) if i < j else (j, i)
    return (a, b, tuple(sorted(S)))


def _flush(torch, items, PHI, cache, device):
    if not items:
        return
    l = len(items[0][2])
    i_idx = torch.tensor([w[0] for w in items], device=device, dtype=torch.long)
    j_idx = torch.tensor([w[1] for w in items], device=device, dtype=torch.long)
    if l == 0:
        S_idx = None
    else:
        S_arr = np.empty((len(items), l), dtype=np.int64)
        for k, w in enumerate(items):
            S_arr[k, :] = w[2]
        S_idx = torch.as_tensor(S_arr, device=device, dtype=torch.long)
    pv = _batched_rcit_pvalues(torch, PHI, i_idx, j_idx, S_idx).detach().to("cpu").numpy()
    for k, w in enumerate(items):
        cache[_key(w[0], w[1], w[2])] = float(pv[k])


def _cits_rcit_gamma(X, alpha, tau, K, max_cond_size, seed, device, batch, dtype, verbose):
    torch = _import_torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    tdtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    p, _ = X.shape
    chi = _window_samples(X, tau)
    n_feat = chi.shape[0]
    C = torch.as_tensor(chi.T, device=dev, dtype=tdtype).contiguous()
    PHI = _precompute_rff_cache(torch, C, K, seed, dev)
    cache: dict = {}
    cands = _candidate_edges(p, tau)
    if not cands:
        return np.zeros((p, p), dtype=int)

    pending, seen = [], set()
    for (i, j, _u, _v) in cands:
        k = _key(i, j, ())
        if k in seen:
            continue
        seen.add(k)
        pending.append((i, j, ()))
        if len(pending) >= batch:
            _flush(torch, pending, PHI, cache, dev)
            pending = []
    _flush(torch, pending, PHI, cache, dev)

    n_cand = len(cands)
    alive = np.zeros(n_cand, dtype=bool)
    separated = np.zeros(n_cand, dtype=bool)
    for k, (i, j, _u, _v) in enumerate(cands):
        if cache[_key(i, j, ())] > alpha:
            separated[k] = True
        else:
            alive[k] = True

    for l in range(1, max_cond_size + 1):
        if not alive.any() or l > n_feat - 2:
            break
        pending, seen = [], set()
        for k in np.flatnonzero(alive):
            i, j, _u, _v = cands[k]
            cand = [m for m in range(n_feat) if m != i and m != j]
            if len(cand) < l:
                continue
            for S in itertools.combinations(cand, l):
                key = _key(i, j, S)
                if key in cache or key in seen:
                    continue
                seen.add(key)
                pending.append((i, j, tuple(sorted(S))))
                if len(pending) >= batch:
                    _flush(torch, pending, PHI, cache, dev)
                    pending = []
        _flush(torch, pending, PHI, cache, dev)
        for k in np.flatnonzero(alive):
            i, j, _u, _v = cands[k]
            cand = [m for m in range(n_feat) if m != i and m != j]
            if len(cand) < l:
                continue
            for S in itertools.combinations(cand, l):
                key = _key(i, j, S)
                if key not in cache:
                    _flush(torch, [(i, j, tuple(sorted(S)))], PHI, cache, dev)
                if cache[key] > alpha:
                    separated[k] = True
                    alive[k] = False
                    break
        if verbose:
            print(f"[cits_rcit] |S|={l}: undecided={int(alive.sum())}, "
                  f"separated={int(separated.sum())}", flush=True)

    B = np.zeros((p, p), dtype=int)
    for k, (_i, _j, u, v) in enumerate(cands):
        if not separated[k]:
            B[u, v] = 1
    del PHI
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return B


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cits_rcit(X, alpha: float = 0.05, tau: int = 1, K: int = 25,
              max_cond_size: Optional[int] = None, seed: int = 0,
              null: str = "gamma", device: Optional[str] = None,
              n_perm: int = 100, batch: int = _DEFAULT_BATCH,
              dtype: str = "float32", verbose: bool = False) -> np.ndarray:
    """CITS lagged graph with the RCIT kernel conditional-independence test.

    Use this for nonlinear or non-Gaussian data such as spike counts; for
    linear-Gaussian data, the partial-correlation versions
    (:func:`cits.methods.cits_full`, :func:`cits.cits_gpu`) are faster.

    Parameters
    ----------
    X : array_like, shape (p, T)
        Time series, one row per variable.
    alpha : float
        Significance level of each conditional-independence test.
    tau : int
        Markov order (maximum lag). Windows have 2*tau + 1 time points.
    K : int
        Random Fourier features per variable (paper: 25).
    max_cond_size : int or None
        Maximum conditioning-set size searched. ``None`` (default) searches all
        sizes, as in the paper's benchmarks.
    seed : int
        Random seed for the Fourier features; results are reproducible for a
        fixed seed, device and dtype.
    null : {'gamma', 'perm'}
        ``'gamma'`` (default; paper setting): batched gamma-approximation null
        via torch. ``'perm'``: NumPy permutation null (slow, no torch).
    device : str or None
        Torch device for ``null='gamma'``, e.g. ``'cuda'``, ``'cuda:1'``,
        ``'cpu'``. ``None`` picks CUDA when available.
    n_perm : int
        Permutations for ``null='perm'``.
    batch : int
        Tests per batched kernel call (``null='gamma'``).
    dtype : {'float32', 'float64'}
        Torch precision for ``null='gamma'`` (paper: float32).
    verbose : bool
        Print progress.

    Returns
    -------
    numpy.ndarray, shape (p, p), int
        Rolled lagged adjacency: ``B[i, j] = 1`` if an edge ``i -> j`` (at some
        lag 1..tau) survived every test. Diagonal entries are self-lags.
    """
    X = _validate(X, tau)
    tau = int(tau)
    n_feat = X.shape[0] * (2 * tau + 1)
    m = n_feat - 2 if max_cond_size is None else int(max_cond_size)
    if m < 0:
        raise ValueError("max_cond_size must be >= 0 or None")
    if null in ("gamma",):
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        return _cits_rcit_gamma(X, alpha, tau, K, m, seed, device, batch, dtype, verbose)
    if null in ("perm", "permutation"):
        return _cits_rcit_perm(X, alpha, tau, K, n_perm, m, seed, verbose)
    raise ValueError(f"null must be 'gamma' or 'perm'; got {null!r}")
