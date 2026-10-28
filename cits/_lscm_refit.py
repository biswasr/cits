"""
_lscm_refit.py

LSCM (linear structural causal model) refit utilities.

Given a CPDAG (from _pc_orientation.cpdag_from_skeleton) and the data matrix
X of shape (T, p), this module estimates one signed coefficient per edge:

  - DIRECTED edges (parent -> child in the CPDAG): OLS of child on its
    parent set in the CPDAG (plus any caller-supplied extra parents, e.g.,
    lagged parents in contemporaneous CITS). The coefficient on the parent is the LSCM
    beta. Under linear-Gaussian SEM with the chosen DAG correct, this is the
    structural causal effect.

  - UNDIRECTED edges in the CPDAG: local single-edge IDA (Maathuis, Kalisch &
    Buhlmann 2010, Nature Methods). Enumerate the two orientations of the
    single undirected edge; check each for consistency with the CPDAG (no
    new unshielded v-structures, no directed cycles); for each consistent
    orientation, refit OLS of the now-defined child on its parents
    (CPDAG-directed parents + extra parents + the edge's parent under the
    trial orientation). Return:
      * (smallest signed beta in absolute value, sign_ambiguous=False)
        when all valid orientations give the same sign -- this is the
        IDA-identifiable lower bound on the structural effect magnitude.
      * (np.nan, True) when signs differ OR no valid orientations exist
        (sign-ambiguous: no consistent structural parameter in the
        equivalence class; treat as skeleton-only edge for causal analyses).

  Methodological caveats explicitly documented:
    * Local single-edge IDA holds the rest of the CPDAG fixed when
      enumerating orientations of one undirected edge. When multiple
      undirected edges share endpoints, full joint enumeration over the MEC
      is exponential and not performed. This is the standard IDA
      approximation per Maathuis et al. 2010.
    * Sign-ambiguous edges (orientations give opposite signs) get NaN
      weight, NOT min-|beta|. Min-|beta| collapses to near zero with no
      structural interpretation when signs conflict; the multiset has no
      consistent causal parameter.

Convention for output B matrix:
  B[parent, child] = signed OLS beta for the directed edge parent -> child.
  For undirected edges, B[i, j] = B[j, i] = IDA-conservative signed value
  (or NaN when sign-ambiguous). The fact that an edge is undirected vs
  directed is NOT encoded in B alone -- downstream code consumes the CPDAG
  (or an edge-type matrix) to distinguish.

  Distinguishing NaN (skeleton-only edge) from 0 (non-edge) downstream:
    np.isnan(B[i, j])     -> skeleton-only edge, sign ambiguous
    B[i, j] == 0          -> non-edge
    B[i, j] != 0 and not nan -> identifiable signed coefficient
"""

from __future__ import annotations
import numpy as np

from ._pc_orientation import (
    directed_edges as _directed_edges,
    undirected_edges as _undirected_edges,
    parents as _parents,
)


# ──────────────────────────────────────────────────────────────────────────
#  OLS refit primitives
# ──────────────────────────────────────────────────────────────────────────

def ols_beta_for_child(X, child, parent_set):
    """Fit OLS of X[:, child] on the parent set with the given lag spec.

    Parameters
    ----------
    X : np.ndarray, shape (T, p), float
    child : int
        Index of child neuron (response).
    parent_set : list of (node, lag) tuples
        Each spec is one regressor. lag=0 means X[:, node] aligned with the
        response time t. lag=k>0 means X[:, node] at time t-k.

    Returns
    -------
    betas : dict[(node, lag) -> beta_value]
        OLS coefficient on each regressor. NaN entry if OLS failed (too few
        samples or singular design matrix).

    Notes
    -----
    Intercept is included in the regression but not returned.
    Response y is X[max_lag:, child]; the first max_lag rows are dropped
    to align all lagged regressors.
    """
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0]

    if len(parent_set) == 0:
        return {}

    max_lag = max(lag for _, lag in parent_set)
    T_eff = T - max_lag

    if T_eff <= len(parent_set) + 1:
        return {ps: np.nan for ps in parent_set}

    y = X[max_lag:, child].astype(np.float64)

    Z = np.empty((T_eff, len(parent_set)), dtype=np.float64)
    for col, (node, lag) in enumerate(parent_set):
        Z[:, col] = X[max_lag - lag: T - lag, node]

    Z_with_int = np.column_stack([np.ones(T_eff), Z])

    try:
        beta_full, _, _, _ = np.linalg.lstsq(Z_with_int, y, rcond=None)
        return {ps: float(beta_full[col + 1])
                for col, ps in enumerate(parent_set)}
    except np.linalg.LinAlgError:
        return {ps: np.nan for ps in parent_set}


# ──────────────────────────────────────────────────────────────────────────
#  Local IDA orientation enumeration
# ──────────────────────────────────────────────────────────────────────────

def _has_directed_path(G, source, target):
    """BFS for directed path source -> ... -> target, using strictly
    directed edges only (undirected edges NOT traversed)."""
    p = G.shape[0]
    if source == target:
        return True
    visited = {source}
    queue = [source]
    while queue:
        cur = queue.pop(0)
        for nxt in range(p):
            if nxt in visited:
                continue
            if G[cur, nxt] == 1 and G[nxt, cur] == 0:
                if nxt == target:
                    return True
                visited.add(nxt)
                queue.append(nxt)
    return False


def enumerate_local_orientations(G_cpdag, edge):
    """For an undirected edge (i, j), enumerate the orientations consistent
    with the rest of the CPDAG held fixed.

    A candidate orientation (parent -> child) is consistent iff:
      1. It does not introduce a new unshielded v-structure at `child`.
         An unshielded v-structure would be parent -> child <- k where k
         is a strict directed parent of child in the CPDAG and parent NOT
         adjacent to k. PC's orientation phase would have already detected
         this collider if it were supported by the data; so if PC left
         this edge undirected, no such collider should be forced. We check
         anyway because Meek's R1-R3 don't enumerate this case directly.
      2. It does not create a directed cycle. Adding parent -> child
         creates a cycle iff there is already a directed path
         child -> ... -> parent in the CPDAG (via directed edges only).

    Returns
    -------
    orientations : list of (parent, child) tuples
        Each entry is one valid orientation of the edge. Length 0, 1, or 2.
    """
    G = np.asarray(G_cpdag)
    i, j = edge
    p = G.shape[0]

    if not (G[i, j] == 1 and G[j, i] == 1):
        return []

    candidates = [(i, j), (j, i)]
    valid = []

    for (parent, child) in candidates:
        # Check 1: no new unshielded v-structure at child.
        new_vstruct = False
        for k in range(p):
            if k == parent or k == child:
                continue
            # k is a STRICT directed parent of child?
            if not (G[k, child] == 1 and G[child, k] == 0):
                continue
            # parent adjacent to k? (either directed or undirected)
            if G[parent, k] == 0 and G[k, parent] == 0:
                # NOT adjacent -- this would be an unshielded v-structure
                new_vstruct = True
                break
        if new_vstruct:
            continue

        # Check 2: no directed cycle.
        if _has_directed_path(G, child, parent):
            continue

        valid.append((parent, child))

    return valid


# ──────────────────────────────────────────────────────────────────────────
#  Local IDA conservative + full LSCM refit
# ──────────────────────────────────────────────────────────────────────────

def ida_conservative_for_undirected_edge(X, G_cpdag, edge,
                                         extra_parents_per_child=None):
    """For undirected edge (i, j), enumerate local orientations consistent
    with the CPDAG, compute OLS beta under each, and return:
      - signed entry with smallest |beta| if all valid orientations give
        the same sign (IDA-identifiable lower bound); sign_ambiguous=False.
      - np.nan if signs differ OR no valid orientations exist;
        sign_ambiguous=True.

    Parameters
    ----------
    X : np.ndarray, shape (T, p)
    G_cpdag : np.ndarray, shape (p, p), int
    edge : tuple (i, j)
    extra_parents_per_child : dict[int -> list[(node, lag)]] or None
        Optional caller-supplied extra parents (e.g., CITS-lagged parents
        for contemporaneous CITS). For lagged-only CITS, leave None.

    Returns
    -------
    beta_conservative : float
        Signed entry with smallest |beta|, or np.nan.
    sign_ambiguous : bool
    """
    valid = enumerate_local_orientations(G_cpdag, edge)
    if not valid:
        return np.nan, True

    betas = []
    for (parent, child) in valid:
        parent_set = [(parent, 0)]
        for k in _parents(G_cpdag, child):
            if k != parent:
                parent_set.append((k, 0))
        if extra_parents_per_child is not None:
            for spec in extra_parents_per_child.get(child, []):
                if spec not in parent_set:
                    parent_set.append(spec)

        beta_dict = ols_beta_for_child(X, child, parent_set)
        beta = beta_dict.get((parent, 0), np.nan)
        if not np.isnan(beta):
            betas.append(beta)

    if len(betas) == 0:
        return np.nan, True

    # Sign-ambiguity check: ignore exact-zero entries when classifying signs
    nonzero_signs = [np.sign(b) for b in betas if b != 0.0]
    if len(set(nonzero_signs)) > 1:
        return np.nan, True

    idx_min = int(np.argmin(np.abs(betas)))
    return float(betas[idx_min]), False


def lscm_refit_cpdag(X, G_cpdag, extra_parents_per_child=None,
                     verbose: bool = False):
    """Refit LSCM coefficients on a CPDAG.

    Directed edges: OLS of child on (CPDAG-directed parents + extra parents
    + edge's parent under the directed orientation).
    Undirected edges: local IDA conservative; placed symmetrically in B.

    Parameters
    ----------
    X : np.ndarray, shape (T, p)
    G_cpdag : np.ndarray, shape (p, p), int
        CPDAG from cpdag_from_skeleton.
    extra_parents_per_child : dict[int -> list[(node, lag)]] or None
        Optional extra regressors per child (e.g., CITS-lagged parents).
    verbose : bool

    Returns
    -------
    B : np.ndarray, shape (p, p), float
        Signed coefficient per edge. B[parent, child] = beta. For
        undirected edges, B[i, j] = B[j, i] = IDA-conservative or NaN.
    sign_amb : np.ndarray, shape (p, p), bool
        True for edges where the IDA enumeration produced sign-ambiguous
        results. Symmetric for undirected edges.
    """
    G_cpdag = np.asarray(G_cpdag)
    p = G_cpdag.shape[0]
    B = np.zeros((p, p), dtype=np.float64)
    sign_amb = np.zeros((p, p), dtype=bool)

    # ---- Directed edges ----
    for (parent, child) in _directed_edges(G_cpdag):
        parent_set = []
        for k in _parents(G_cpdag, child):
            parent_set.append((k, 0))
        if (parent, 0) not in parent_set:
            parent_set.append((parent, 0))
        if extra_parents_per_child is not None:
            for spec in extra_parents_per_child.get(child, []):
                if spec not in parent_set:
                    parent_set.append(spec)

        beta_dict = ols_beta_for_child(X, child, parent_set)
        beta = beta_dict.get((parent, 0), np.nan)
        B[parent, child] = beta

    # ---- Undirected edges (IDA conservative) ----
    for (i, j) in _undirected_edges(G_cpdag):
        beta_c, sa = ida_conservative_for_undirected_edge(
            X, G_cpdag, (i, j), extra_parents_per_child)
        B[i, j] = beta_c
        B[j, i] = beta_c
        sign_amb[i, j] = sa
        sign_amb[j, i] = sa

    if verbose:
        n_dir = len(_directed_edges(G_cpdag))
        n_und = len(_undirected_edges(G_cpdag))
        n_signamb = int(sign_amb.sum() // 2)
        print(f"[lscm_refit_cpdag] directed={n_dir}, undirected={n_und}, "
              f"sign_ambiguous={n_signamb}", flush=True)

    return B, sign_amb
