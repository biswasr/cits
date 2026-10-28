"""
cits.gpu

GPU-accelerated CITS: the faithful scalable CITS-lag skeleton.

Matches the exact CITS algorithm (``cits.methods.cits_full`` with
``cond_dep='cond_dep_pcorr'``) on every axis EXCEPT the conditioning-set
search, which is replaced by the neighbor-restricted, increasing-size PC
search (cuPC on GPU). That single substitution is sound under the
faithfulness assumption CITS already requires: a valid separating set is
guaranteed to lie within the adjacency set, so the powerset traversal is
redundant and cuPC recovers the same skeleton under the oracle.

Matched to exact CITS:
  - NON-overlapping windows (as in cits.methods.data_transform), not overlapping.
  - Window width w = 2*(tau+1) and time-major layout c = t*p + v
    (identical to data_transform / cits_unrolled indexing).
  - Full initial adjacency -> conditioning pool is the whole window, exactly
    as in the CITS powerset (which conditions on all window variables).
  - Targeted extraction of lag edges into the present slice t = 2*tau+1 from
    source slices t1 in {tau+1, ..., 2*tau}, oriented by time -- identical to
    cits_unrolled's tested edge set and cits_rolled's collapse.

Only remaining difference from cits_full(pcorr): powerset conditioning ->
neighbor-restricted increasing-size search (cuPC, with no cap on the
conditioning-set size).

Requires a compiled cuPC ``Skeleton.so`` from the CITS fork
(https://github.com/biswasr/cupc) and a CUDA-capable GPU. See
``cits._cupc_wrapper`` (and the README "GPU setup (cuPC)" section) for the
``CUPC_DIR`` environment variable and build instructions.
"""
from __future__ import annotations
import numpy as np

from ._cupc_wrapper import pc_skeleton_cupc
from ._common import validate_X


def _build_chi_nonoverlap(X: np.ndarray, tau: int) -> np.ndarray:
    """Non-overlapping chi-stack, matching cits.methods.data_transform.

    X : (p, T). Returns U : (N, p*w) with w = 2*(tau+1), N = floor((T-w)/w),
    column index c = t*p + v (time-major), row i = window i.
    """
    p, T = X.shape
    w = 2 * (tau + 1)
    if T < 2 * w:
        raise ValueError(f"T={T} too small for tau={tau} (need T >= {2*w})")
    N = int((T - w) / w)
    U = np.empty((N, p * w), dtype=np.float64)
    for i in range(N):
        block = X[:, w * i: w * (i + 1)]          # (p, w)
        U[i, :] = block.T.reshape(p * w)          # time-major: c = t*p + v
    return U


def _lag_rolled_from_skeleton(X, skeleton_fn, alpha=0.05, tau=1,
                              max_level=None, verbose=False):
    """Faithful scalable CITS-lag rolled adjacency, using an arbitrary
    neighbor-restricted PC skeleton backend.

    ``skeleton_fn`` runs the skeleton on the non-overlapping chi-window and
    returns ``(G_unrolled, sep_sets, inactive, level)`` -- either
    ``cits._cupc_wrapper.pc_skeleton_cupc`` (GPU) or
    ``cits._pc_skeleton_cpu.pc_skeleton_cpu`` (CPU). Both implement the same
    neighbor-restricted PC-stable algorithm, so the extracted rolled
    adjacency is backend-independent.
    """
    p, T = X.shape
    U = _build_chi_nonoverlap(X, tau)             # (N, p*w), c = t*p + v

    G_unrolled, _sep, _inactive, _lvl = skeleton_fn(
        U, alpha=alpha, max_level=max_level, verbose=verbose, return_sep_sets=False)

    # Targeted extraction: edges (v1, t1) -> (v2, t_target) into the present
    # slice, sources in the recent-past window -- identical to cits_unrolled.
    t_target = 2 * tau + 1
    B = np.zeros((p, p), dtype=int)
    for v1 in range(p):
        for v2 in range(p):
            for t1 in range(tau + 1, 2 * tau + 1):   # tau+1 .. 2*tau
                c1 = t1 * p + v1
                c2 = t_target * p + v2
                if G_unrolled[c1, c2] != 0:
                    B[v1, v2] = 1
                    break
    return B


def cits_gpu(X: np.ndarray, alpha: float = 0.05, tau: int = 1,
             max_level=None, verbose: bool = False) -> np.ndarray:
    """Faithful scalable CITS-lag (rolled adjacency), Fisher-z via cuPC (GPU).

    GPU accelerator for the lagged CITS skeleton. It matches base CITS
    (``cits.methods.cits_full`` with the Gaussian partial-correlation test)
    except that the exponential powerset conditioning search is replaced by
    cuPC's neighbor-restricted PC-stable search on the GPU, which is sound
    under the faithfulness assumption CITS already requires. Scales to
    thousands of variables, limited by GPU memory. Requires a compiled cuPC ``Skeleton.so`` from the
    CITS fork and a CUDA-capable GPU (see the README "GPU setup (cuPC)" section).

    Parameters
    ----------
    X : np.ndarray, shape (p, T)
        Time series, p variables (neurons) by T time points. Same input
        convention as ``cits.methods.cits_full``.
    alpha : float
        Significance level for the Fisher-z partial-correlation test.
        Default 0.05.
    tau : int
        CITS Markovian order / maximum lag. Default 1 (paper default).
    max_level : int or None
        Max conditioning-set size. None (default) runs the PC search until no
        variable has enough neighbours, with no cap. An int stops after that level.
    verbose : bool

    Returns
    -------
    B : np.ndarray, shape (p, p), int
        Rolled lagged adjacency. B[i, j] = 1 iff variable i (at a past lag)
        is inferred to be a causal parent of variable j at the present time
        slice; 0 otherwise. Lagged (directed-in-time) edges only, no
        contemporaneous structure. Same contract as the source
        ``gpu_cits_lag_cupc_faithful``.
    """
    X = validate_X(X, "cits_gpu")
    return _lag_rolled_from_skeleton(
        X, pc_skeleton_cupc, alpha=alpha, tau=tau,
        max_level=max_level, verbose=verbose)


# Backwards-compatible alias matching the original source function name.
gpu_cits_lag_cupc_faithful = cits_gpu
