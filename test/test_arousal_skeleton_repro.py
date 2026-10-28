"""Optional regression test: rebuild stored PC skeletons of a large calcium-imaging analysis exactly with the CITS
fork of cuPC. The stored runs searched past conditioning level 14, and one of the two removed an edge there.

It needs the stored inputs, so it runs only when CITS_AROUSAL_SKELETON_DIR points to a folder with
chi/<graph>/<job>.npz (array U, samples x variables) and supports/<graph>/<job>.npz (chi_edges, keep_cols, diag).
"""
import json
import os

import numpy as np
import pytest

BASE = os.environ.get("CITS_AROUSAL_SKELETON_DIR", "")
JOBS = [("s0", "s4_sc7_f1"), ("s0", "s4_sc7_f7")]


def _fork_available():
    try:
        from cits._cupc_wrapper import cupc_info
        return bool(cupc_info()["uncapped"])
    except Exception:
        return False


@pytest.mark.skipif(not BASE, reason="set CITS_AROUSAL_SKELETON_DIR to run")
@pytest.mark.parametrize("graph,job", JOBS)
def test_rebuilds_stored_skeleton_exactly(graph, job):
    if not _fork_available():
        pytest.skip("CITS fork of cuPC unavailable")
    from cits._cupc_wrapper import pc_skeleton_cupc
    U = np.load(os.path.join(BASE, "chi", graph, job + ".npz"))["U"]
    s = np.load(os.path.join(BASE, "supports", graph, job + ".npz"), allow_pickle=True)
    info = json.loads(str(s["diag"]))["fisher_chi"]
    keep = U.std(0) > 0
    assert np.array_equal(keep, s["keep_cols"])
    G, _, _, level = pc_skeleton_cupc(U[:, keep], alpha=info["alpha"], return_sep_sets=False)
    mine = {tuple(e) for e in np.argwhere(np.triu(G, 1)).tolist()}
    stored = {tuple(e) for e in s["chi_edges"].tolist()}
    assert mine == stored
    assert level == info["last_level"] + 1  # first level with no test left
