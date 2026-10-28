"""
cits._pc_raw (internal)

PC skeleton on per-trial calcium time series, operating on the
chi-subsampled contemporaneous slice (same data CITS sees at lag 0).

DESIGN NOTE on sample count:
  Initial implementation passed the FULL T raw samples (X.T) to the
  contemp skeleton, on the reasoning that more samples = more power.
  This was wrong on two fronts:

  (1) STATISTICAL: Fisher-z requires INDEPENDENT samples. Calcium imaging
      data has strong temporal autocorrelation (GCaMP decay ~ 500 ms);
      consecutive frames are NOT independent. Treating T raw samples as
      independent inflates Type I error and biases the skeleton toward
      false positives.

  (2) COMPUTATIONAL: Partial-correlation tests scale linearly with N
      (sample count). Full T was ~4x slower than the chi-subsampled
      version, and made PC the dominant compute cost (133s on a 47-neuron
      field -- intractable for the 290-neuron fields in the dataset).

  Resolution: subsample via the same data_transform that CITS uses
  internally, take the contemporaneous slice chi[t_target*p:(t_target+1)*p],
  and run PC on that. This is the EXACT same data CITS sees for its lag-0
  contemp tests. With tau=1, N = floor((T - 1) / 4); roughly T/4 samples.
  The samples ARE approximately independent (4-frame stride exceeds calcium
  autocorrelation).

Inputs:
  X         : (T, p) float64 raw calcium trace for one trial.
  alpha     : significance level for Fisher-z conditional-independence test.
  use_gpu   : whether to use GPU-batched residualization (default True).
  verbose   : print per-level diagnostics.

Outputs:
  A         : (p, p) int symmetric adjacency. A[i, j] = 1 iff edge (i, j)
              survives the skeleton phase. Diagonal 0.
  r_mat     : (p, p) float symmetric. |Pearson r| at l=0 for retained edges,
              0 for removed edges.
  sep_sets  : dict[(i, j) -> tuple[int]]. Separating set used to remove
              edge (i, j), if removed. Used by v-structure orientation in
              cits._pc_orientation.
  inactive_neurons : indices of zero-variance neurons.

Centering note:
  This function does NOT center X. The caller is responsible for any
  per-neuron centering that matches the downstream LSCM convention. PC's
  partial-correlation tests are invariant under constant shifts, so
  centering doesn't affect skeleton discovery, only edge weight scaling
  (which we don't use from this function -- LSCM refit happens in
  cits._lscm_refit).
"""

from __future__ import annotations
import numpy as np


def pc_skeleton_raw(X, alpha: float = 0.05, use_gpu: bool = True,
                    verbose: bool = False, zero_var_tol: float = 1e-12,
                    tau: int = 1, backend: str = 'cupc'):
    """Run PC skeleton on per-trial calcium series, chi-subsampled to the
    contemporaneous slice.

    Parameters
    ----------
    X : array_like, shape (T, p)
        Calcium trace for one trial. Pre-centering is recommended (the
        function does NOT auto-center).
    alpha : float
        Significance level for Fisher-z conditional-independence test.
    use_gpu : bool
        Whether to use GPU-batched residualization (only used by the
        'python' backend).
    verbose : bool
        Print per-level diagnostics.
    zero_var_tol : float
        Columns with variance below this threshold are flagged as inactive.
        Edges incident on such columns are deterministically removed.
    tau : int
        CITS Markovian order. Used to construct the chi sub-sampling.
        tau=1 (default) gives N = floor((T - 1) / 4) samples in the
        contemporaneous slice. tau=0 falls back to passing X.T as the
        chi_c matrix (full T samples; NOT independence-preserving, only
        useful for synthetic Gaussian smoke tests).
    backend : str
        'cupc' (default) uses the GPU cuPC skeleton. 'python' uses the
        original Python-loop PC (not bundled with this package).

    Returns
    -------
    A : np.ndarray, shape (p, p), int
        Symmetric undirected skeleton.
    r_mat : np.ndarray, shape (p, p), float
        |Pearson r| at l=0 for retained edges, 0 elsewhere.
    sep_sets : dict[(i, j) -> tuple[int]]
        Separating set used to remove each removed edge.
    inactive_neurons : np.ndarray, shape (n_inactive,), int
        Indices of neurons whose variance was below zero_var_tol.

    Notes
    -----
    The chi-subsampling matches what CITS internally uses for its lag-0
    contemp tests, so this PC pass is operating on the EXACT same data
    CITS sees at contemp lag -- the difference between CITS and this PC
    is only the alpha-level tests applied at the contemp slice.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, p); got shape {X.shape}")
    T, p = X.shape

    # Zero-variance guard
    col_var = X.var(axis=0)
    inactive_mask = col_var < zero_var_tol
    inactive_neurons = np.flatnonzero(inactive_mask)

    if inactive_neurons.size > 0 and verbose:
        print(f"[pc_skeleton_raw] {inactive_neurons.size} inactive neurons "
              f"detected (var < {zero_var_tol}): {inactive_neurons.tolist()}",
              flush=True)

    if inactive_neurons.size > 0:
        rng = np.random.default_rng(0)
        X = X.copy()
        for j in inactive_neurons:
            X[:, j] = rng.standard_normal(T) * 1e-10

    # Build chi_c via the same data_transform CITS uses internally, take
    # the contemporaneous slice. Same data as CITS's lag-0 view.
    if tau == 0:
        # Synthetic test path: just pass X.T as chi_c (full T samples;
        # use ONLY when you know samples are independent).
        chi_c = X.T
    else:
        from . import methods as cits_m
        # data_transform expects (p, T): f_raw orientation. We have X of
        # shape (T, p), so transpose first.
        chi_full = cits_m.data_transform(X.T, tau)
        # chi_full shape: ((2*tau+2)*p, N) where N = floor((T - 2*tau) / (2*tau+2)).
        # For tau=1: shape (4p, N).
        t_target = 2 * tau + 1
        chi_c = chi_full[t_target * p:(t_target + 1) * p, :]
        if verbose:
            print(f"[pc_skeleton_raw] X shape={X.shape}, chi_c shape={chi_c.shape}, "
                  f"tau={tau}, t_target={t_target}", flush=True)

    if backend == 'cupc':
        # cuPC (Zarebavani et al. 2020) -- ~800x faster than Python loop PC.
        # chi_c shape (p, N); cuPC expects (N, p).
        from ._cupc_wrapper import pc_skeleton_cupc
        A, sep_sets, _, _ = pc_skeleton_cupc(
            chi_c.T, alpha=alpha, verbose=verbose)
        # cuPC doesn't return r_mat; compute |corr| at l=0 for retained edges.
        R = np.corrcoef(chi_c)
        r_mat = np.where(A != 0, np.abs(R), 0.0)
        np.fill_diagonal(r_mat, 0.0)
    elif backend in ('python', 'cpu'):
        # Pure-numpy neighbor-restricted PC-stable skeleton (no GPU). Same
        # algorithm as cuPC, so it yields the same skeleton. chi_c is (p, N);
        # the CPU skeleton expects (N, p).
        from ._pc_skeleton_cpu import pc_skeleton_cpu
        A, sep_sets, _, _ = pc_skeleton_cpu(
            chi_c.T, alpha=alpha, verbose=verbose)
        R = np.corrcoef(chi_c)
        r_mat = np.where(A != 0, np.abs(R), 0.0)
        np.fill_diagonal(r_mat, 0.0)
    else:
        raise ValueError(f"unknown backend {backend!r}; "
                         f"expected 'cupc', 'cpu', or 'python'")

    if inactive_neurons.size > 0:
        for j in inactive_neurons:
            A[j, :] = 0
            A[:, j] = 0
            r_mat[j, :] = 0.0
            r_mat[:, j] = 0.0

    return A, r_mat, sep_sets, inactive_neurons
