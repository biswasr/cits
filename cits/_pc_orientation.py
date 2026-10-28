"""
_pc_orientation.py

PC orientation phase: v-structure detection + Meek's rules.

Operates on the output of _pc_raw.pc_skeleton_raw (or any PC skeleton):
takes an undirected symmetric adjacency A_skel and the sep_sets dict,
returns a CPDAG (mix of directed and undirected edges) as a directed
adjacency matrix with the convention:

    G[i, j] = 1  AND  G[j, i] = 0  -->  edge i -> j (directed)
    G[i, j] = 1  AND  G[j, i] = 1  -->  edge i -- j (undirected)
    G[i, j] = 0  AND  G[j, i] = 0  -->  no edge

V-structure rule (Spirtes, Glymour & Scheines 1993):
  For every unshielded triple i - k - j (i.e., i, k adjacent; k, j adjacent;
  i, j NOT adjacent), orient as i -> k <- j IF k NOT in sep_set(i, j).
  This is the only orientation forced by observational conditional-independence
  evidence alone.

Meek's rules (Meek 1995, R1-R3 -- the rules needed in the absence of
background knowledge):
  R1: i -> k and k -- j and i NOT adjacent to j   ==>   k -> j
      (orienting k -- j the other way would create a new v-structure at k)
  R2: i -> k -> j and i -- j                      ==>   i -> j
      (orienting i -- j the other way would create a directed cycle)
  R3: i -- k -> j, i -- l -> j, k NOT adjacent l, and i -- j  ==>  i -> j
      (any orientation of i -- j the other way forces a new v-structure
       at j via either k or l)
  R4 is typically not applied without background knowledge -- omitted here.

Convergence: rules are applied iteratively until no more orientations change
in a full pass.

References:
  Spirtes, Glymour, Scheines (1993, 2000) "Causation, Prediction, and Search"
  Meek (1995) "Causal inference and causal explanation with background knowledge"
"""

from __future__ import annotations
import numpy as np


def _is_adjacent(G, i, j):
    """True if there's any edge (directed or undirected) between i and j."""
    return bool(G[i, j] or G[j, i])


def _has_arrow_into(G, j, i):
    """True if there's a directed edge i -> j (i.e., G[i,j]=1 and G[j,i]=0)."""
    return bool(G[i, j] == 1 and G[j, i] == 0)


def _is_undirected(G, i, j):
    """True if i -- j is undirected (G[i,j]=1 AND G[j,i]=1)."""
    return bool(G[i, j] == 1 and G[j, i] == 1)


def orient_v_structures(A_skel, sep_sets):
    """Apply v-structure orientation rule to a PC skeleton.

    Two-pass implementation: first scan the WHOLE skeleton to collect every
    v-structure orientation (using the unmodified undirected skeleton, NOT
    the partially-oriented G), then apply them all at once with conflict
    detection.

    Conflict resolution: an edge (a -> b) gets marked as "directed into b"
    by every unshielded collider triple in which b is the collider. If both
    directions of an edge (a, b) get marked (i.e., a -> b from one triple
    AND b -> a from another), then b is a collider in one triple AND a is a
    collider in another. Standard PC handles this by leaving the edge
    UNDIRECTED in the CPDAG (it's a conflict the data can't resolve).

    Parameters
    ----------
    A_skel : np.ndarray, shape (p, p), int
        Symmetric skeleton adjacency from pc_skeleton_raw.
    sep_sets : dict[(i, j) -> tuple[int]]
        Separating set used to remove each removed edge (i, j).

    Returns
    -------
    G : np.ndarray, shape (p, p), int
        Partially directed adjacency. G[i,j]=1,G[j,i]=0 -> i->j; both 1 -> undirected.
    """
    A_skel = np.asarray(A_skel)
    p = A_skel.shape[0]
    if A_skel.shape != (p, p):
        raise ValueError(f"A_skel must be square; got {A_skel.shape}")

    # Symmetric skeleton: both G[i,j] and G[j,i] = 1 wherever A_skel has an edge.
    skel_sym = (A_skel != 0).astype(np.int8)
    np.fill_diagonal(skel_sym, 0)
    skel_sym = (skel_sym | skel_sym.T).astype(np.int8)

    # Collect arrows: arrow_into[a, b] = True means SOME triple wants to
    # orient a -> b (i.e., make b a collider through this pair).
    arrow_into = np.zeros((p, p), dtype=bool)

    # Scan unshielded triples using the ORIGINAL skeleton (not the mutated G).
    for k in range(p):
        nbrs_k = np.flatnonzero(skel_sym[k] != 0)
        if nbrs_k.size < 2:
            continue
        for a in range(len(nbrs_k)):
            for b in range(a + 1, len(nbrs_k)):
                i = int(nbrs_k[a])
                j = int(nbrs_k[b])
                # Must be UNSHIELDED in the skeleton
                if skel_sym[i, j] != 0:
                    continue
                S = sep_sets.get((i, j), sep_sets.get((j, i), None))
                if S is None:
                    continue
                if k in S:
                    continue
                # Triple (i, k, j) is a v-structure: orient i -> k and j -> k.
                arrow_into[i, k] = True
                arrow_into[j, k] = True

    # Apply orientations with conflict detection.
    # An edge (a, b) is in the skeleton iff skel_sym[a, b] = 1.
    # If only arrow_into[a, b] is True (not arrow_into[b, a]): a -> b.
    # If only arrow_into[b, a] is True: b -> a.
    # If BOTH arrow_into[a, b] AND arrow_into[b, a] are True: CONFLICT.
    #   Standard PC: leave undirected (the data is contradicting itself).
    # If NEITHER is True: undirected.
    G = skel_sym.copy()
    for a in range(p):
        for b in range(p):
            if a == b:
                continue
            if skel_sym[a, b] == 0:
                continue
            into_b = arrow_into[a, b]
            into_a = arrow_into[b, a]
            if into_b and not into_a:
                # a -> b: zero G[b, a]
                G[b, a] = 0
            elif into_a and not into_b:
                # b -> a: zero G[a, b]
                G[a, b] = 0
            else:
                # Both True -> conflict, leave undirected
                # Neither True -> already undirected
                pass

    return G


def apply_meek_rules(G, max_iters: int = 1000):
    """Iteratively apply Meek's R1-R3 propagation rules in place until
    no more changes (or max_iters reached).

    Parameters
    ----------
    G : np.ndarray, shape (p, p), int
        Partially directed adjacency from orient_v_structures.
    max_iters : int
        Safety cap on outer iterations. Typical convergence is in 1-5 iters.

    Returns
    -------
    G : np.ndarray
        Updated in place; returned for convenience.
    """
    p = G.shape[0]

    for it in range(max_iters):
        changed = False

        # Precompute neighbor lists for speed (Python iteration on numpy
        # rows is much faster via np.flatnonzero than scanning all p cols).
        # Strict directed parents of node j: set of i with G[i,j]=1, G[j,i]=0.
        # Undirected neighbors of i: set of k with G[i,k]=1, G[k,i]=1.
        directed_parents_of = [
            np.flatnonzero((G[:, j] == 1) & (G[j, :] == 0)).tolist()
            for j in range(p)
        ]
        undirected_neighbors = [
            np.flatnonzero((G[i, :] == 1) & (G[:, i] == 1)).tolist()
            for i in range(p)
        ]
        # adjacency matrix (any edge, directed or undirected)
        adj = ((G != 0) | (G.T != 0))

        # R1: i -> k and k -- j and i NOT adjacent to j  =>  k -> j
        for k in range(p):
            und_k = undirected_neighbors[k]
            dir_pa_k = directed_parents_of[k]
            if not und_k or not dir_pa_k:
                continue
            for j in und_k:
                if j == k:
                    continue
                if G[k, j] == 0 or G[j, k] == 0:
                    continue  # was already oriented earlier in this pass
                for i in dir_pa_k:
                    if i == k or i == j:
                        continue
                    if adj[i, j]:
                        continue
                    G[j, k] = 0
                    adj[k, j] = True  # already True but defensive
                    changed = True
                    break

        # R2: i -> k -> j and i -- j  =>  i -> j
        for i in range(p):
            und_i = undirected_neighbors[i]
            if not und_i:
                continue
            for j in und_i:
                if j == i:
                    continue
                if G[i, j] == 0 or G[j, i] == 0:
                    continue
                # need k with i -> k AND k -> j
                # k must be a directed CHILD of i AND directed PARENT of j
                dir_pa_j = directed_parents_of[j]
                children_of_i = [c for c in range(p)
                                  if G[i, c] == 1 and G[c, i] == 0]
                if not dir_pa_j or not children_of_i:
                    continue
                common = set(dir_pa_j) & set(children_of_i) - {i, j}
                if common:
                    G[j, i] = 0
                    changed = True

        # R3: i -- k -> j, i -- l -> j, k NOT adjacent l, and i -- j  =>  i -> j
        for i in range(p):
            und_i = undirected_neighbors[i]
            if len(und_i) < 2:
                continue
            for j in und_i:
                if j == i:
                    continue
                if G[i, j] == 0 or G[j, i] == 0:
                    continue
                # Candidates k: k in undirected_neighbors[i] AND k -> j
                cands = [k for k in und_i
                         if k != j and G[k, j] == 1 and G[j, k] == 0]
                if len(cands) < 2:
                    continue
                # Any pair (k, l) in cands with k NOT adjacent l?
                found = False
                for a in range(len(cands)):
                    if found:
                        break
                    ka = cands[a]
                    for b in range(a + 1, len(cands)):
                        kb = cands[b]
                        if not adj[ka, kb]:
                            G[j, i] = 0
                            changed = True
                            found = True
                            break

        if not changed:
            break

    return G


def cpdag_from_skeleton(A_skel, sep_sets):
    """Convenience wrapper: v-structure orientation then Meek's R1-R3.

    Returns
    -------
    G_cpdag : np.ndarray, shape (p, p), int
        Final CPDAG. G[i,j]=1, G[j,i]=0 -> i->j; both 1 -> undirected; both 0 -> none.
    """
    G = orient_v_structures(A_skel, sep_sets)
    G = apply_meek_rules(G)
    return G


def undirected_edges(G_cpdag):
    """Return list of (i, j) with i < j for which G[i,j] == 1 AND G[j,i] == 1.
    These are the Markov-equivalent edges that PC could not orient.
    """
    G = np.asarray(G_cpdag)
    p = G.shape[0]
    result = []
    for i in range(p):
        for j in range(i + 1, p):
            if G[i, j] == 1 and G[j, i] == 1:
                result.append((i, j))
    return result


def directed_edges(G_cpdag):
    """Return list of (parent, child) tuples for which G[parent,child] == 1
    AND G[child,parent] == 0. These are the orientation-forced edges.
    """
    G = np.asarray(G_cpdag)
    p = G.shape[0]
    result = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if G[i, j] == 1 and G[j, i] == 0:
                result.append((i, j))
    return result


def parents(G_cpdag, node):
    """Return list of strict parents of `node` in the CPDAG.

    A node i is a parent of `node` if G[i, node] == 1 AND G[node, i] == 0
    (i.e., a directed edge i -> node, not undirected).
    """
    G = np.asarray(G_cpdag)
    p = G.shape[0]
    return [i for i in range(p)
            if i != node and G[i, node] == 1 and G[node, i] == 0]


def undirected_neighbors(G_cpdag, node):
    """Return list of nodes connected to `node` by an undirected edge."""
    G = np.asarray(G_cpdag)
    p = G.shape[0]
    return [i for i in range(p)
            if i != node and G[i, node] == 1 and G[node, i] == 1]
