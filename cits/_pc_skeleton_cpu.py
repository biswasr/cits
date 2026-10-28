"""
cits._pc_skeleton_cpu (internal)

Pure-numpy, neighbor-restricted, increasing-conditioning-size PC-stable
skeleton with a Fisher-z partial-correlation conditional-independence test.

This is the CPU counterpart of ``cits._cupc_wrapper.pc_skeleton_cupc``. It
implements the SAME neighbor-restricted PC-stable algorithm that cuPC runs
on the GPU (NOT the exponential powerset conditioning of the base
``cits.methods.cits_full``), so on the same data it recovers the SAME
skeleton as cuPC. The powerset search is intractable beyond ~10 variables;
this neighbor-restricted search is what makes both cuPC and this module scale.

The CI test reuses the v1.4-fixed ``cits.methods.partial_corr`` (regression
with an intercept, shift-invariant) so the partial correlations, and hence
the edge decisions, match cuPC's Fisher-z test:

    z    = 0.5 * log((1+r) / (1-r))
    stat = sqrt(N - |S| - 3) * |z|
    edge (i, j) is declared independent (removed) given S when the two-sided
    p-value 2*(1 - Phi(stat)) > alpha  (equivalently |z| <= the cuPC
    per-level threshold qnorm(1-alpha/2)/sqrt(N-|S|-3)).

Signature and return contract mirror ``pc_skeleton_cupc`` so the two are
drop-in interchangeable:

    pc_skeleton_cpu(X, alpha=0.05, max_level=None)
        -> (G, sep_sets, inactive_neurons, final_level)

There is no cap on the conditioning-set size by default (pure PC-stable). The
search stops when no variable has enough neighbours, or when N - level - 3 <= 0
(no test can then remove an edge), the same rule as the cuPC backend.

  X : (T, p) data matrix (samples x variables).
"""

from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy import stats

from .methods import partial_corr

def _independent(x, y, S, data, alpha):
    """Fisher-z partial-correlation CI test. True iff x _||_ y | S.

    data : (p, N). S : set of conditioning indices. Mirrors the arithmetic
    of cits.methods.cond_dep_pcorr (dependent iff p-value <= alpha), so the
    removal decision matches cuPC.
    """
    N = data.shape[1]
    df = N - len(S) - 3
    if df <= 0:
        # cuPC sets Th=0 in this regime -> edge kept (treated as dependent).
        return False
    r = partial_corr(x, y, set(S), data)
    if not np.isfinite(r):
        # Degenerate residuals: keep the edge (conservative, matches the
        # non-independence default).
        return False
    if r >= 1.0 or r <= -1.0:
        return False
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    stat = np.sqrt(df) * np.abs(z)
    pval = 2.0 * (1.0 - stats.norm.cdf(stat))
    return pval > alpha


def pc_skeleton_cpu(X, alpha: float = 0.05, max_level=None,
                    zero_var_tol: float = 1e-12, verbose: bool = False,
                    return_sep_sets: bool = True):
    """Neighbor-restricted PC-stable skeleton on data matrix X (CPU).

    Parameters
    ----------
    X : array_like, shape (T, p)
        Data matrix. Treated as N=T (approximately) independent samples.
    alpha : float
        Significance level for the Fisher-z partial-correlation CI test.
    max_level : int or None
        Max conditioning-set size. None (default) runs until no variable has
        enough neighbours, with no cap. An int stops after that level.
    zero_var_tol : float
        Columns with variance below this are flagged inactive; their edges
        are deterministically removed (same handling as pc_skeleton_cupc).
    verbose : bool
    return_sep_sets : bool
        Return the separating sets (default). With False, ``sep_sets`` is
        None, as in the cuPC backend.

    Returns
    -------
    G : np.ndarray, shape (p, p), int
        Symmetric undirected skeleton. G[i, j] = 1 if edge retained.
    sep_sets : dict[(i, j) -> tuple[int]]
        Separating set used to remove each removed edge (both orderings).
    inactive_neurons : np.ndarray
        Indices of inactive (zero-variance) neurons.
    final_level : int
        The first level at which no test could run or remove an edge, or
        ``max_level`` when the search stopped there.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, p); got {X.shape}")
    T, p = X.shape
    if T < 4:
        raise ValueError(f"T={T} too small (need T >= 4 for Fisher-z)")
    if p < 2:
        raise ValueError(f"p={p} too small")

    # Zero-variance guard (identical to pc_skeleton_cupc so skeletons match).
    col_var = X.var(axis=0)
    inactive_mask = col_var < zero_var_tol
    inactive_neurons = np.flatnonzero(inactive_mask)
    if inactive_neurons.size > 0:
        if verbose:
            print(f"[pc_skeleton_cpu] {inactive_neurons.size} inactive "
                  f"neurons; replacing with tiny noise", flush=True)
        rng = np.random.default_rng(0)
        X = X.copy()
        for j in inactive_neurons:
            X[:, j] = rng.standard_normal(T) * 1e-10

    # partial_corr expects data as (p, N).
    data = np.ascontiguousarray(X.T)

    # Full initial skeleton, no self-loops.
    G = np.ones((p, p), dtype=int)
    np.fill_diagonal(G, 0)

    sep_sets: dict = {}
    level = 0
    while True:
        if T - level - 3 <= 0:
            # No test at this level or above can remove an edge (cuPC uses Th=0 here).
            final_level = level
            break
        # PC-stable: snapshot every node's adjacency at the START of the
        # level, so edge-removal order within the level cannot change the
        # result (this is what makes the skeleton order-independent and
        # matches cuPC).
        neighbors = [set(np.flatnonzero(G[i])) for i in range(p)]

        # Present undirected edges (i < j) at the start of this level.
        edges = [(i, j) for i in range(p) for j in range(i + 1, p)
                 if G[i, j] != 0]

        testable = False
        for (i, j) in edges:
            if G[i, j] == 0:
                continue  # removed earlier this level
            removed = False
            # PC considers subsets of adj(x)\{y} from BOTH endpoints.
            for (x, y) in ((i, j), (j, i)):
                nb = neighbors[x] - {y}
                if len(nb) < level:
                    continue
                testable = True
                for S in combinations(sorted(nb), level):
                    if _independent(x, y, S, data, alpha):
                        G[i, j] = 0
                        G[j, i] = 0
                        sep_sets[(i, j)] = tuple(S)
                        sep_sets[(j, i)] = tuple(S)
                        removed = True
                        break
                if removed:
                    break

        if verbose:
            print(f"[pc_skeleton_cpu] level={level} edges_remaining="
                  f"{int(G.sum() // 2)}", flush=True)

        final_level = level
        if not testable:
            break
        if max_level is not None and level >= max_level:
            break
        level += 1

    # Force-remove edges incident on inactive neurons (defensive).
    if inactive_neurons.size > 0:
        for j in inactive_neurons:
            G[j, :] = 0
            G[:, j] = 0

    return G, (sep_sets if return_sep_sets else None), inactive_neurons, final_level
