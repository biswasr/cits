"""
_union_cpdag.py

Combine CITS-lagged rolled adjacency with PC-contemporaneous CPDAG into
a single union of parent specifications, in a form consumable by
_lscm_refit.lscm_refit_cpdag and _lscm_refit.ols_beta_for_child.

For tau=1: each nonzero entry B[i, j] in the rolled CITS output is treated
as a lag-1 parent (i at t-1 -> j at t). This is exact, not approximate,
because CITS's internal unrolled graph for tau=1 has only one lag slot.

Outputs:
  union_extra_parents : dict[int -> list[(node, lag)]]
      For each child node j, the LAGGED extra parents to be included in
      the union LSCM refit. CITS-lagged parents are added as (i, 1).
      The PC-contemp directed parents and the IDA enumeration of
      undirected contemp edges are handled by _lscm_refit functions
      from the PC-contemp CPDAG; this dict only supplies the LAGGED part.

  pc_contemp_G_cpdag : np.ndarray (p, p) int
      Pass-through of the PC-contemp CPDAG. Returned for convenience so
      Contemporaneous-CITS drivers have a single object to hand to lscm_refit_cpdag.

  union_skeleton : np.ndarray (p, p) bool
      Binary mask of edges in the FULL union skeleton (PC-contemp skeleton
      OR CITS-lagged edges projected to (i, j) regardless of lag).
      Used by downstream skeleton-level analyses.

  edge_type_matrix : np.ndarray (p, p) int8
      Per-edge type (filled where union skeleton is True; 0 elsewhere):
        0 = non-edge
        1 = directed CITS-lagged only (no contemp edge at this pair)
        2 = directed PC-contemp (collider/Meek)
        3 = undirected PC-contemp, IDA-identifiable sign
        4 = undirected PC-contemp, IDA sign-ambiguous (skeleton only)
      For pairs with BOTH a CITS-lagged edge and a PC-contemp edge:
        the PC-contemp classification wins (2/3/4) because the lagged
        edge is then captured as an extra parent in the union LSCM, not
        as a separately-classified edge.

IDA consistency note (important):
  When the contemporaneous-CITS driver invokes lscm_refit_cpdag on the PC-contemp
  CPDAG WITH the lagged extra parents, IDA consistency is checked against
  the PC-contemp CPDAG ONLY -- the lagged parents are extra regressors,
  not orientation evidence.

  This is an APPROXIMATION, not a theoretically forced choice. In a true
  linear-Gaussian SVAR (structural vector autoregression), the conditional-
  independence relations of the FULL unrolled chi graph DO encode contemp
  orientations via Verma constraints, so lagged sep_sets could in principle
  disambiguate Markov-equivalent contemp edges. We do NOT use this
  information because our two-stage pipeline (CITS for lagged + PC for
  contemp) was not designed to jointly run PC's orientation phase on the
  full unrolled chi graph; doing so would require a substantial rewrite.
  The simplified approach is conservative: it leaves some orientation
  evidence on the table, biasing toward "more undirected edges" rather than
  fabricating orientations from incomplete evidence. Document explicitly
  in the methods.

  Local single-edge IDA (Maathuis et al. 2010) further holds the rest of
  the CPDAG fixed when enumerating orientations of one undirected edge;
  cross-edge parent-set interactions are not captured. This is the
  standard IDA approximation.
"""

from __future__ import annotations
import numpy as np


def build_union(cits_lagged_B, pc_contemp_skeleton, pc_contemp_G_cpdag,
                sign_amb_mat=None, tau: int = 1):
    """Build the union extra-parents dict and the edge-type bookkeeping.

    Parameters
    ----------
    cits_lagged_B : np.ndarray, shape (p, p)
        CITS rolled adjacency from the existing per-trial CSV file.
        For tau=1, B[i, j] != 0 iff neuron i at t-1 is a parent of neuron
        j at t.
    pc_contemp_skeleton : np.ndarray, shape (p, p), int or bool
        The PC-contemp UNDIRECTED skeleton (full skeleton before
        orientation; output `A` from pc_skeleton_raw, symmetric).
    pc_contemp_G_cpdag : np.ndarray, shape (p, p), int
        The PC-contemp CPDAG (after v-structures + Meek).
    sign_amb_mat : np.ndarray, shape (p, p), bool, optional
        Sign-ambiguous mask from lscm_refit_cpdag, if already computed.
        If None, edge_type_matrix can only populate types 0-3 (type 4 is
        merged into type 3). Pass after lscm_refit_cpdag has been run for
        the final edge classification.
    tau : int
        CITS Markovian order. tau=1 is the only currently-supported value.

    Returns
    -------
    union_extra_parents : dict[int -> list[(node, lag)]]
        Extra lagged parents per child. For tau=1, each entry is (i, 1).
    union_skeleton : np.ndarray, shape (p, p), bool
        Binary union skeleton mask.
    edge_type_matrix : np.ndarray, shape (p, p), int8
        Per-edge type (see module docstring for codes).
    """
    if tau != 1:
        raise NotImplementedError(
            f"build_union currently supports tau=1 only; got tau={tau}")

    cits_lagged_B = np.asarray(cits_lagged_B)
    pc_contemp_skel = np.asarray(pc_contemp_skeleton)
    G_cpdag = np.asarray(pc_contemp_G_cpdag)
    p = cits_lagged_B.shape[0]

    if pc_contemp_skel.shape != (p, p) or G_cpdag.shape != (p, p):
        raise ValueError(
            f"shape mismatch: cits={cits_lagged_B.shape}, "
            f"pc_skel={pc_contemp_skel.shape}, G={G_cpdag.shape}")

    # ── union_extra_parents: CITS-lagged parents at lag 1 ──────────────────
    union_extra_parents: dict[int, list[tuple[int, int]]] = {}
    for j in range(p):
        parents_j: list[tuple[int, int]] = []
        for i in range(p):
            if i == j:
                continue
            if cits_lagged_B[i, j] != 0:
                parents_j.append((i, 1))
        if parents_j:
            union_extra_parents[j] = parents_j

    # ── union_skeleton: PC-contemp skel OR CITS-lagged pair present ────────
    pc_skel_mask = (pc_contemp_skel != 0)
    np.fill_diagonal(pc_skel_mask, False)

    cits_pair_mask = np.zeros((p, p), dtype=bool)
    # CITS-lagged is directed (parent -> child); for the unordered pair
    # presence, mark both (i, j) and (j, i) if either is nonzero.
    cits_nz = cits_lagged_B != 0
    cits_pair_mask = cits_nz | cits_nz.T
    np.fill_diagonal(cits_pair_mask, False)

    union_skeleton = pc_skel_mask | cits_pair_mask
    np.fill_diagonal(union_skeleton, False)

    # ── edge_type_matrix ────────────────────────────────────────────────────
    edge_type_matrix = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if not union_skeleton[i, j]:
                continue
            in_pc = pc_skel_mask[i, j]
            in_cits = cits_pair_mask[i, j]
            if in_pc:
                # Determine PC orientation status
                if G_cpdag[i, j] == 1 and G_cpdag[j, i] == 0:
                    edge_type_matrix[i, j] = 2  # directed PC-contemp
                elif G_cpdag[j, i] == 1 and G_cpdag[i, j] == 0:
                    # i is the child of j in PC-contemp; type 2 from j's side,
                    # but the (i, j) cell still represents an edge from j -> i.
                    # We label by the cell's direction: parent -> child reading.
                    # Since i, j cell here is the "j -> i" reading (j is parent,
                    # because we're at row i, col j which is the target i),
                    # convention: edge_type for cell (parent, child).
                    edge_type_matrix[i, j] = 2
                else:
                    # Undirected in PC-contemp CPDAG
                    if sign_amb_mat is not None and sign_amb_mat[i, j]:
                        edge_type_matrix[i, j] = 4  # sign-ambiguous
                    else:
                        edge_type_matrix[i, j] = 3  # identifiable
            elif in_cits:
                # CITS-lagged only; direction is the (i, j) cell only if
                # cits_lagged_B[i, j] != 0 (i is parent of j at lag 1).
                # For pair-presence cells where i is NOT the parent, leave as 0
                # (the symmetric (j, i) cell handles the other direction).
                if cits_lagged_B[i, j] != 0:
                    edge_type_matrix[i, j] = 1

    return union_extra_parents, union_skeleton, edge_type_matrix
