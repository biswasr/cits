"""
cits.contemporaneous

CITS with contemporaneous edges.

The base CITS-lag skeleton (``cits.gpu.cits_gpu``) infers only LAGGED
(directed-in-time) causal edges. It omits contemporaneous (lag-0)
structure. Contemporaneous CITS augments the lagged skeleton with a contemporaneous PC
step and refits one coherent set of signed structural coefficients over the
UNION of lagged and contemporaneous parents.

Pipeline (per trial), given X of shape (p, T):
  1. Lagged skeleton: ``cits.gpu.cits_gpu(X, alpha, tau)`` -> rolled binary
     adjacency B_lag. For tau=1, B_lag[i, j] != 0 means neuron i at t-1 is a
     lag-1 parent of neuron j at t. Only the parent IDENTITY is used here;
     the signed weights are re-derived by the union LSCM refit below.
  2. Contemporaneous PC skeleton on the chi-subsampled lag-0 slice via
     ``cits._pc_raw.pc_skeleton_raw`` (same data CITS sees at lag 0).
  3. Orient the contemp skeleton with v-structure orientation
     (``cits._pc_orientation.orient_v_structures``). Meek propagation is
     OFF by default (``use_meek=False``): only collider-forced edges with
     direct conditional-independence evidence are oriented; everything else
     stays undirected and is handled by local IDA in the LSCM refit. Set
     ``use_meek=True`` to also apply Meek's R1-R3.
  4. Union: ``cits._union_cpdag.build_union`` merges the lagged parents and
     the contemp CPDAG into a single per-child parent specification plus a
     skeleton mask and an edge-type matrix.
  5. Signed LSCM refit: ``cits._lscm_refit.lscm_refit_cpdag`` fits one OLS
     per child on its union parent set. Directed edges get the signed beta;
     undirected contemp edges get a local-IDA conservative signed value (or
     NaN when sign-ambiguous).

By default (``weight_lagged_only=False``) the pipeline stops after step 5,
exactly reproducing the paper's montage pipeline (lagged-only edges stay in
the skeleton but carry no weight). This matches the Fig 5A magnitudes. Set
``weight_lagged_only=True`` to additionally OLS-weight the lagged-only edges.

Backends. Both the lagged skeleton and the contemp PC skeleton run through
the same neighbor-restricted PC-stable algorithm, available two ways:
  - CPU: pure numpy, works everywhere with no native dependency.
  - cuPC: GPU-accelerated, recommended above ~100 variables for speed (needs
    a compiled Skeleton.so and a CUDA GPU; see the README "GPU setup" note).
Both backends produce the same skeleton. ``backend='auto'`` (default) uses
cuPC when a working Skeleton.so is found, otherwise CPU.

Output convention for the signed weighted adjacency B:
    B[parent, child] = signed OLS beta   -- identifiable directed/undirected edge
    B[i, j] = 0                          -- non-edge
    B[i, j] = NaN                        -- skeleton-only edge (sign-ambiguous
                                            IDA multiset, or no valid orientation)
Downstream causal-weight analyses treat NaN as "edge present, no causal
weight": exclude from weighted analyses, include in skeleton-only analyses.
"""

from __future__ import annotations
import numpy as np

from .gpu import _lag_rolled_from_skeleton
from ._pc_raw import pc_skeleton_raw
from ._pc_orientation import (
    orient_v_structures,
    cpdag_from_skeleton,
    parents as _parents_in_G,
)
from ._union_cpdag import build_union
from ._lscm_refit import lscm_refit_cpdag, ols_beta_for_child
from ._common import log_info, log_warning, validate_X

# One-time backend notice (per process). Silenceable via CITS_QUIET=1.
_BACKEND_NOTICE_SHOWN = False

# Above this many variables, the cuPC GPU backend is recommended for speed.
# Practical guideline, not a hard rule -- benchmark your own setup.
_LARGE_P_THRESHOLD = 100


def _progress(stage):
    """Concise per-stage marker (only when verbose=True), via the logger."""
    log_info(f"[cits contemporaneous] {stage}")


def _cpu_large_p_msg(p):
    """Data-informed heads-up for large-p CPU runs. The ~33 s figure is the
    GPU (cuPC) number from the paper's scaling benchmark; no CPU-CITS runtime
    is claimed. ~100 is an approximate guideline."""
    return (f"cits: contemporaneous CITS on CPU with p={p} variables may be slow; the CPU "
            f"skeleton search scales steeply with p. In our scaling benchmark "
            f"the cuPC GPU backend inferred p=1000-variable graphs in ~33 s. "
            f"The GPU backend is recommended above ~{_LARGE_P_THRESHOLD} "
            f"variables (see README).")


def _concrete_backend(backend):
    """Decide the concrete backend ('cupc' or 'cpu') for a backend spec.

    Returns (name, reason) where reason is one of:
      'explicit'      -- user forced this backend
      'auto-found'    -- auto selected cuPC (working Skeleton.so found)
      'auto-notfound' -- auto fell back to CPU (no working cuPC)
    Raises the actionable FileNotFoundError for backend='cupc' when cuPC is
    not installed.
    """
    if backend not in ('auto', 'cpu', 'cupc'):
        raise ValueError(
            f"unknown backend {backend!r}; expected 'auto', 'cpu', or 'cupc'")

    if backend == 'cpu':
        return 'cpu', 'explicit'

    if backend == 'cupc':
        # Force GPU; surface the clear FileNotFoundError now if unavailable.
        from ._cupc_wrapper import _get_lib
        _get_lib()
        return 'cupc', 'explicit'

    # backend == 'auto'
    from ._cupc_wrapper import is_cupc_available
    if is_cupc_available():
        return 'cupc', 'auto-found'
    return 'cpu', 'auto-notfound'


def _backend_impl(name):
    """Concrete skeleton fn + pc_skeleton_raw backend string for a name."""
    if name == 'cupc':
        from ._cupc_wrapper import pc_skeleton_cupc
        return pc_skeleton_cupc, 'cupc'
    from ._pc_skeleton_cpu import pc_skeleton_cpu
    return pc_skeleton_cpu, 'cpu'


def _emit_backend_notice(name, reason, p):
    """One-time (per process) notice of which backend runs and why, routed
    through the package logger (log_info)."""
    global _BACKEND_NOTICE_SHOWN
    if _BACKEND_NOTICE_SHOWN:
        return
    large_p = p > _LARGE_P_THRESHOLD
    if reason == 'auto-found':
        log_info("cits: using cuPC GPU backend.")
        _BACKEND_NOTICE_SHOWN = True
    elif reason == 'auto-notfound':
        if large_p:
            log_info(_cpu_large_p_msg(p))
        else:
            log_info("cits: cuPC not found; using the CPU backend (fine up to "
                     "~100 variables; see README 'GPU setup' to enable GPU "
                     "acceleration).")
        _BACKEND_NOTICE_SHOWN = True
    elif reason == 'explicit' and name == 'cpu' and large_p:
        log_info(_cpu_large_p_msg(p))
        _BACKEND_NOTICE_SHOWN = True
    # explicit cupc, or explicit cpu with small p: no notice.


def _skeleton_stages(X, X_Tp, alpha, tau, name, verbose):
    """Run the two backend-dependent stages (lagged skeleton, contemp PC)
    with the concrete backend `name`. Returns (cits_lagged_B, pc_skel,
    sep_sets). Raises on a backend (e.g. cuPC/CUDA) failure."""
    skeleton_fn, raw_backend = _backend_impl(name)
    if verbose:
        _progress(f"1/4 lagged skeleton (p={X.shape[0]})")
    cits_lagged_B = _lag_rolled_from_skeleton(
        X, skeleton_fn, alpha=alpha, tau=tau, verbose=verbose)
    if verbose:
        _progress("2/4 contemporaneous PC")
    pc_skel, _pc_r0, sep_sets, _inactive = pc_skeleton_raw(
        X_Tp, alpha=alpha, tau=tau, backend=raw_backend, verbose=verbose)
    return cits_lagged_B, pc_skel, sep_sets


def cits_contemporaneous(X, alpha: float = 0.05, tau: int = 1, backend: str = 'auto',
                  use_meek: bool = False, weight_lagged_only: bool = False,
                  full_output: bool = False, verbose: bool = False):
    """CITS with contemporaneous edges: signed weighted adjacency over the
    union of lagged (CITS) and contemporaneous (PC) causal structure.

    This is the pipeline used for the paper's neural analyses. Unlike base
    CITS and ``cits_gpu`` (lagged edges only), contemporaneous CITS also recovers
    contemporaneous (within-time-slice) edges and assigns signed structural
    (LSCM) edge weights.

    Parameters
    ----------
    X : np.ndarray, shape (p, T)
        Time series, p variables (neurons) by T time points. Same input
        convention as ``cits.methods.cits_full`` and ``cits.cits_gpu``.
    alpha : float
        Significance level shared by the lagged skeleton and the contemp PC
        conditional-independence tests. Default 0.05 (paper default).
    tau : int
        CITS Markovian order / maximum lag. Default 1 (paper default). The
        union step currently supports tau=1 only.
    backend : str
        Skeleton backend for both the lagged and contemporaneous PC steps:
          'auto'  (default) -- use cuPC (GPU) if a working Skeleton.so is
                    found, otherwise fall back to the pure-numpy CPU skeleton.
          'cpu'   -- force the CPU skeleton (no GPU needed). Recommended for
                    small-to-moderate graphs; above ~100 variables the run
                    prints a one-time note recommending the cuPC backend.
          'cupc'  -- force the GPU skeleton; raises a clear error if cuPC is
                    unavailable, and does NOT silently fall back on a runtime
                    GPU failure (re-raises with guidance).
        Both backends implement the same algorithm and give the same skeleton.
        A one-time (per process) stderr notice states which backend is used
        and why; silence it with the env var CITS_QUIET=1. On a runtime cuPC
        GPU failure under 'auto', a warning is emitted and the run falls back
        to CPU.
    use_meek : bool
        If False (default), orient the contemp skeleton with v-structures
        only (no Meek propagation) -- the paper's orientation.
        If True, additionally apply Meek's R1-R3.
    weight_lagged_only : bool
        Controls whether lagged-ONLY edges (present in the lagged CITS graph
        but not in the contemporaneous PC skeleton) receive a signed OLS
        weight.
          False (default) -- reproduce the paper exactly: stop after the
            union LSCM refit, leaving lagged-only edges unweighted (0 in the
            weighted matrix; still present in the 'skeleton' / 'edge_type'
            outputs). The paper's Fig 5A magnitudes correspond to this.
          True -- also fit an OLS weight for each lagged-only edge on its
            union parent set (the fuller weighting). This adds small-magnitude
            nonzeros not present in the paper's montage pipeline.
    full_output : bool
        If False (default), return only the signed weighted adjacency B.
        If True, return a dict with keys 'weighted', 'skeleton',
        'edge_type', 'sign_ambiguous', 'lagged', 'cpdag'.
    verbose : bool

    Returns
    -------
    B : np.ndarray, shape (p, p), float   (when full_output=False)
        Signed weighted adjacency. B[parent, child] = signed OLS beta for an
        identifiable edge; 0 = non-edge; NaN = skeleton-only edge (edge
        present but sign-ambiguous, so no causal weight).
    result : dict                          (when full_output=True)
        'weighted'       : (p, p) float signed weighted adjacency (as above)
        'skeleton'       : (p, p) int8 union skeleton presence (directed+undirected)
        'edge_type'      : (p, p) int8 per-edge type
                           (0 none, 1 CITS-lagged, 2 directed PC-contemp,
                            3 undirected PC-contemp IDA-identifiable,
                            4 undirected PC-contemp sign-ambiguous)
        'sign_ambiguous' : (p, p) bool IDA sign-ambiguity mask
        'lagged'         : (p, p) int rolled lagged adjacency
        'cpdag'          : (p, p) int contemp PC CPDAG
    """
    X = validate_X(X, "cits_contemporaneous")
    X = np.ascontiguousarray(X, dtype=np.float64)
    p, T = X.shape

    if tau != 1:
        raise ValueError(
            "cits_contemporaneous currently supports tau=1 (the union step). For "
            "lagged-only inference at higher tau use cits_gpu(X, tau=...).")

    # Decide the concrete backend and emit the one-time notice.
    name, reason = _concrete_backend(backend)
    _emit_backend_notice(name, reason, p)

    # LSCM refit and PC skeleton use (T, p) orientation.
    X_Tp = np.ascontiguousarray(X.T)

    # ---- Stages 1-2 (backend-dependent), with runtime GPU-failure handling.
    try:
        cits_lagged_B, pc_skel, sep_sets = _skeleton_stages(
            X, X_Tp, alpha, tau, name, verbose)
    except Exception as e:  # noqa: BLE001 - backend/CUDA failures are broad
        if name != 'cupc':
            raise
        from ._cupc_wrapper import OutdatedCupcError
        if isinstance(e, OutdatedCupcError):
            # The message already says how to rebuild; keep it whole.
            if backend != 'auto':
                raise
            log_warning(f"cits: {e}\nFalling back to the CPU backend, which has no level cap.")
        else:
            first = (str(e).splitlines() or [''])[0][:200]
            short = f"{type(e).__name__}: {first}" if first else type(e).__name__
            if backend != 'auto':
                raise RuntimeError(
                    f"cits: cuPC GPU call failed ({short}). Fix your cuPC/CUDA "
                    f"setup or rerun with backend='cpu'.") from e
            log_warning(f"cits: cuPC GPU call failed ({short}); falling back "
                        f"to the CPU backend. See README 'GPU setup'.")
        name = 'cpu'
        cits_lagged_B, pc_skel, sep_sets = _skeleton_stages(
            X, X_Tp, alpha, tau, 'cpu', verbose)

    # ---- 3. Orient contemp skeleton (v-structures; optional Meek) ----
    if verbose:
        _progress("3/4 union")
    if use_meek:
        pc_G = cpdag_from_skeleton(pc_skel, sep_sets)
    else:
        pc_G = orient_v_structures(pc_skel, sep_sets)

    # ---- 4. Union of lagged parents + contemp CPDAG ----
    union_parents, union_skel, _edge_type = build_union(
        cits_lagged_B, pc_skel, pc_G, sign_amb_mat=None, tau=tau)

    # ---- 5. Signed LSCM refit on the union parent set ----
    if verbose:
        _progress("4/4 LSCM refit")
    union_B, union_sa = lscm_refit_cpdag(
        X_Tp, pc_G, extra_parents_per_child=union_parents, verbose=verbose)

    # Recompute edge_type now that sign-ambiguity is known.
    _, _, edge_type = build_union(
        cits_lagged_B, pc_skel, pc_G, sign_amb_mat=union_sa, tau=tau)

    # OPTIONAL fill of CITS-lagged-ONLY pairs (no contemp edge) into union_B.
    # The LSCM refit above iterates over PC-contemp CPDAG edges only, so
    # lagged-only edges are otherwise left at 0. The paper's montage pipeline
    # does NOT do this (it stops after lscm_refit_cpdag), so the default
    # weight_lagged_only=False reproduces the paper exactly.
    if weight_lagged_only:
        for child, lagged_parents in union_parents.items():
            cpdag_pa = _parents_in_G(pc_G, child)
            for (p_node, lag) in lagged_parents:
                if pc_skel[p_node, child] != 0:
                    # Contemp skeleton also has this pair; already handled.
                    continue
                full_parents = [(p_node, lag)]
                for k in cpdag_pa:
                    full_parents.append((k, 0))
                for (other_p, other_lag) in lagged_parents:
                    if (other_p, other_lag) == (p_node, lag):
                        continue
                    full_parents.append((other_p, other_lag))
                betas = ols_beta_for_child(X_Tp, child, full_parents)
                union_B[p_node, child] = betas.get((p_node, lag), np.nan)

    if not full_output:
        return union_B

    return {
        'weighted': union_B,
        'skeleton': union_skel.astype(np.int8),
        'edge_type': edge_type,
        'sign_ambiguous': union_sa,
        'lagged': cits_lagged_B,
        'cpdag': pc_G,
    }
