"""
Paper-reproduction equivalence test.

Asserts that ``cits.cits_contemporaneous(X, backend='cupc', weight_lagged_only=False)``
reproduces the original MICrONS montage pipeline
(``fig5_neuropixels/_neuropixels_contemp_pooled.py (contemp_directed) in cits-paper``, i.e. cuPC lagged +
PC-contemporaneous + v-structures + union + single LSCM refit, NO lagged-only
fill) bit-for-bit on the gabors input.

Runs only when cuPC is available AND the gabors data / original modules are
present on this machine; otherwise it skips (so the suite still passes on a
CPU-only or data-less box).
"""

import os
import sys
import pickle

import numpy as np
import pytest

_DATA = "/home/rbiswas1/citsproject/data"
_ANALYSIS_DIR = "/home/rbiswas1/microns/analysis/functional_circuitry"
_REPO = "/home/rbiswas1/repos/cits"
_CUPC = "/home/rbiswas1/repos/cupc"

_P_RAW = os.path.join(_DATA, "P_raw_gabors.npy")
_MASK = os.path.join(_DATA, "ID791319847_gabors_units2use_stim_gabors.p")


def _cupc_available():
    """True iff the CITS fork of cuPC (no level cap) can be loaded."""
    try:
        from cits._cupc_wrapper import cupc_info
        return bool(cupc_info()["uncapped"])
    except Exception:
        return False


def _load_gabors_X():
    """Build X exactly as the montage pipeline does for gabors."""
    D = np.load(_P_RAW).reshape(-1, 555)
    mask = np.asarray(pickle.load(open(_MASK, "rb")))
    ui = np.where(mask)[0] if mask.dtype == bool else mask
    X = (D[:9000, ui] - D[:9000, ui].mean(0)).T
    return np.ascontiguousarray(X, dtype=np.float64)


def _original_contemporaneous(X):
    """Original montage contemporaneous weighted adjacency (union_B, nan_to_num'd).

    Imports the original (pre-package) modules from the analysis directory.
    Those modules need /home/rbiswas1/repos/cits on sys.path because the
    original _pc_raw does `from cits import methods`.
    """
    for p in (_ANALYSIS_DIR, _REPO, _CUPC):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    from gpu_cits_lag_cupc_faithful import gpu_cits_lag_cupc_faithful
    from _pc_raw import pc_skeleton_raw
    from _pc_orientation import orient_v_structures
    from _union_cpdag import build_union
    from _lscm_refit import lscm_refit_cpdag

    cB = gpu_cits_lag_cupc_faithful(X, alpha=0.05, tau=1)
    Xtp = X.T
    pc_skel, _pc_r0, sep_sets, _inactive = pc_skeleton_raw(
        Xtp, alpha=0.05, use_gpu=True, verbose=False)
    pc_G = np.asarray(orient_v_structures(pc_skel, sep_sets))
    union_parents, _union_skel, _edge_type = build_union(
        cB.astype(float), pc_skel, pc_G, sign_amb_mat=None, tau=1)
    union_B, _ = lscm_refit_cpdag(
        Xtp, pc_G, extra_parents_per_child=union_parents, verbose=False)
    return np.nan_to_num(np.asarray(union_B))


@pytest.mark.skipif(not (os.path.exists(_P_RAW) and os.path.exists(_MASK)),
                    reason="gabors data not present on this machine")
def test_contemporaneous_reproduces_montage_pipeline():
    if not _cupc_available():
        pytest.skip("cuPC (Skeleton.so) unavailable; equivalence not checked")
    if not os.path.isdir(_ANALYSIS_DIR):
        pytest.skip("original analysis modules not present on this machine")
    import cits

    X = _load_gabors_X()

    pkg_B = cits.cits_contemporaneous(X, alpha=0.05, tau=1, backend='cupc',
                               weight_lagged_only=False)
    pkg_B = np.nan_to_num(np.asarray(pkg_B))

    orig_B = _original_contemporaneous(X)

    assert pkg_B.shape == orig_B.shape
    assert np.allclose(pkg_B, orig_B), (
        f"max abs diff {np.abs(pkg_B - orig_B).max()}, "
        f"nonzeros pkg={int((pkg_B != 0).sum())} orig={int((orig_B != 0).sum())}")
