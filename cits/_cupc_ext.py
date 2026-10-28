"""
cits._cupc_ext (internal)

Driver for the uncapped GPU skeleton of the CITS fork of cuPC
(https://github.com/biswasr/cupc, ``cupc_cits_version() >= 1``).

The fork adds a per-level host API (``ext_*``) to cuPC. This module runs the
PC-stable search one level per call:

  - level 0 with cuPC's original marginal-test kernel,
  - every level l >= 1 with the generic kernel ``cal_Indep_ext``, which tests
    any conditioning-set size (no compile-time cap at 14),

until no variable has enough neighbours, or until no test can remove an edge
(N - l - 3 <= 0). Partial correlations come from an incremental Cholesky.
Decisions within a relative band of 1e-9 around the threshold use cuPC's exact
Fisher-z formula, so they equal the Fisher-z rule
``|0.5 log((1 + r) / (1 - r))| < |z_{alpha/2}| / sqrt(N - l - 3)``.
Each removal at a level l >= 1 is logged with its separating set.

The task layout and safety band are the ones validated on 229 production
skeletons (identical graphs, including 1,690 removals above level 14).
"""

from __future__ import annotations

import ctypes
from math import comb, tanh, sqrt

import numpy as np
from scipy.stats import norm

BAND = 1e-9  # relative safety band around tanh(th)^2; inside it the exact Fisher-z formula decides
STAT_NAMES = ('leaves', 'tests', 'band', 'nonpos_pivots', 'degenerate', 'tasks_skipped', 'removed',
              'tasks_run')

_P_D = ctypes.POINTER(ctypes.c_double)
_P_I = ctypes.POINTER(ctypes.c_int)
_P_LL = ctypes.POINTER(ctypes.c_longlong)
_P_ULL = ctypes.POINTER(ctypes.c_ulonglong)


def bind(lib):
    """Set the argument and return types of the ext_* functions on a loaded library."""
    lib.ext_create.restype = ctypes.c_void_p
    lib.ext_create.argtypes = [_P_D, ctypes.c_int, ctypes.c_int]
    lib.ext_destroy.restype = None
    lib.ext_destroy.argtypes = [ctypes.c_void_p]
    lib.ext_level0.restype = ctypes.c_int
    lib.ext_level0.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.ext_scan.restype = ctypes.c_int
    lib.ext_scan.argtypes = [ctypes.c_void_p, _P_I, _P_I]
    lib.ext_get_G.restype = ctypes.c_int
    lib.ext_get_G.argtypes = [ctypes.c_void_p, _P_I]
    lib.ext_level_new.restype = ctypes.c_int
    lib.ext_level_new.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_int, _P_I, _P_LL, _P_LL, ctypes.c_longlong,
                                  _P_ULL, ctypes.c_int, ctypes.c_int, ctypes.c_int, _P_I, _P_I, _P_I, _P_D,
                                  ctypes.c_longlong, _P_LL, _P_ULL, _P_I, _P_D, ctypes.c_int, ctypes.c_int]
    return lib


def _ptr(a, t):
    return a.ctypes.data_as(t)


def fisher_threshold(N, level, alpha):
    """Fisher-z threshold of one level, |z_{alpha/2}| / sqrt(N - level - 3), or 0 when N - level - 3 <= 0
    (no test can then remove an edge)."""
    df = N - level - 3
    if df <= 0:
        return 0.0
    return float(abs(norm.ppf(alpha / 2.0)) / sqrt(df))


def _band_bounds(th):
    t2 = tanh(th) ** 2
    return t2 * (1.0 - BAND), t2 * (1.0 + BAND)


def _binom_table(mmax, l):
    lim = 2 ** 63 - 1
    tab = np.zeros((mmax + 1, l + 1), dtype=np.uint64)
    for a in range(mmax + 1):
        for b in range(min(a, l) + 1):
            tab[a, b] = min(comb(a, b), lim)
    return tab


def _build_tasks(rowsizes, l, tmin=64, tmax=8192, target_tasks=20000, max_tasks=4_000_000):
    """Split the C(m_X, l) lexicographic subsets of every row with m_X >= l + 1 into contiguous rank ranges.
    Returns (task_row int32, task_start int64, task_count int64, total subsets, chunk) or None."""
    rows = np.flatnonzero(rowsizes >= l + 1)
    counts = [comb(int(m), l) for m in rowsizes[rows]]
    if not counts:
        return None
    if max(counts) >= 2 ** 62:
        raise OverflowError(
            f"conditioning level {l}: a variable has C(m, l) >= 2^62 candidate separating sets "
            f"(m = {int(rowsizes[rows].max())} neighbours). This enumeration is infeasible on any device.")
    total = sum(counts)
    T = min(tmax, max(tmin, -(-total // target_tasks)))
    T = max(T, -(-total // max_tasks))
    nt = np.array([-(-c // T) for c in counts], dtype=np.int64)
    task_row = np.repeat(rows.astype(np.int32), nt)
    offs = np.concatenate([np.arange(k, dtype=np.int64) for k in nt]) * np.int64(T)
    cnts = np.minimum(np.int64(T), np.repeat(np.array(counts, dtype=np.int64), nt) - offs)
    assert int(cnts.sum()) == total and (cnts > 0).all()
    return (np.ascontiguousarray(task_row), np.ascontiguousarray(offs), np.ascontiguousarray(cnts), total, T)


class _Session:
    """One correlation matrix on the GPU."""

    def __init__(self, lib, C):
        self.lib = lib
        self.C = np.ascontiguousarray(C, dtype=np.float64)
        self.p = int(self.C.shape[0])
        h = lib.ext_create(_ptr(self.C, _P_D), self.p, 0)
        if not h:
            need_gb = 24 * self.p * self.p / 1e9
            raise RuntimeError(
                f"cuPC could not allocate GPU memory for {self.p} variables (about {need_gb:.2f} GB "
                f"plus per-level buffers). Free GPU memory or choose another GPU with CUDA_VISIBLE_DEVICES.")
        self.h = ctypes.c_void_p(h)

    def close(self):
        if self.h is not None:
            self.lib.ext_destroy(self.h)
            self.h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def level0(self, th):
        rc = self.lib.ext_level0(self.h, float(th))
        if rc != 0:
            raise RuntimeError(f"cuPC level 0 failed (CUDA error, code {rc})")

    def scan(self):
        rows = np.zeros(self.p, dtype=np.int32)
        npr = np.zeros(1, dtype=np.int32)
        rc = self.lib.ext_scan(self.h, _ptr(rows, _P_I), _ptr(npr, _P_I))
        if rc != 0:
            raise RuntimeError(f"cuPC adjacency scan failed (CUDA error, code {rc})")
        return int(npr[0]), rows

    def get_G(self):
        G = np.zeros((self.p, self.p), dtype=np.int32)
        rc = self.lib.ext_get_G(self.h, _ptr(G, _P_I))
        if rc != 0:
            raise RuntimeError(f"cuPC could not copy the adjacency matrix (CUDA error, code {rc})")
        return G

    def level_new(self, level, th, rowsizes):
        tasks = _build_tasks(rowsizes, level)
        if tasks is None:
            return [], {}
        task_row, task_start, task_count, _total, _chunk = tasks
        mmax = int(rowsizes.max())
        bt = np.ascontiguousarray(_binom_table(mmax, level))
        cap = int(rowsizes.sum() // 2)
        log_xy = np.zeros(max(2 * cap, 2), dtype=np.int32)
        log_S = np.zeros(max(cap * level, 1), dtype=np.int32)
        log_rho = np.zeros(max(cap, 1), dtype=np.float64)
        nlog = np.zeros(1, dtype=np.int64)
        stats = np.zeros(len(STAT_NAMES), dtype=np.uint64)
        cfg = np.zeros(6, dtype=np.int32)
        ms = np.zeros(1, dtype=np.float64)
        t2lo, t2hi = _band_bounds(th)
        row_left = np.ascontiguousarray(rowsizes.astype(np.int32))
        rc = self.lib.ext_level_new(
            self.h, int(level), float(th), float(t2lo), float(t2hi), 0,
            _ptr(task_row, _P_I), _ptr(task_start, _P_LL), _ptr(task_count, _P_LL), int(len(task_row)),
            _ptr(bt, _P_ULL), int(bt.shape[0]), int(bt.shape[1]), mmax, _ptr(row_left, _P_I),
            _ptr(log_xy, _P_I), _ptr(log_S, _P_I), _ptr(log_rho, _P_D), cap, _ptr(nlog, _P_LL),
            _ptr(stats, _P_ULL), _ptr(cfg, _P_I), _ptr(ms, _P_D), 0, 0)
        if rc != 0:
            raise RuntimeError(f"cuPC conditioning level {level} failed (CUDA error, code {rc})")
        k = int(nlog[0])
        if k > cap:
            raise RuntimeError(f"cuPC conditioning level {level}: {k} removals logged, capacity {cap}")
        st = {nm: int(v) for nm, v in zip(STAT_NAMES, stats)}
        if st['removed'] != k:
            raise RuntimeError(f"cuPC conditioning level {level}: removal counter {st['removed']} differs "
                               f"from the log ({k})")
        xy = log_xy[:2 * k].reshape(k, 2)
        S = log_S[:k * level].reshape(k, level)
        return [(int(x), int(y), tuple(int(s) for s in S[i])) for i, (x, y) in enumerate(xy)], st


def skeleton(lib, C, N, alpha=0.05, max_level=None, return_sep_sets=True, verbose=False):
    """PC-stable skeleton of the correlation matrix ``C`` (p x p) from ``N`` samples, all levels on the GPU.

    Returns ``(G, sep_sets, final_level)``. ``G`` is the symmetric int32 adjacency with a zero diagonal.
    ``sep_sets`` maps each removed ordered pair (i, j) to its separating set (both orders), or is None when
    ``return_sep_sets`` is False. ``final_level`` follows the CPU backend: the first level at which no test
    can run or remove an edge, or ``max_level`` when the search stops there.
    """
    p = int(C.shape[0])
    sep = {} if return_sep_sets else None
    with _Session(lib, C) as s:
        s.level0(fisher_threshold(N, 0, alpha))
        if return_sep_sets:
            G0 = s.get_G()
            off = ~np.eye(p, dtype=bool)
            I, J = np.nonzero((G0 == 0) & off)
            sep.update(dict.fromkeys(zip(I.tolist(), J.tolist()), ()))
        if verbose:
            print(f"[pc_skeleton_cupc] level=0 done", flush=True)
        level = 1
        while True:
            if max_level is not None and level > max_level:
                final_level = int(max_level)
                break
            if N - level - 3 <= 0:
                final_level = level
                break
            nprime, rows = s.scan()
            if nprime - 1 < level:
                final_level = level
                break
            removed, st = s.level_new(level, fisher_threshold(N, level, alpha), rows)
            if return_sep_sets:
                for x, y, S in removed:
                    sep[(x, y)] = S
                    sep[(y, x)] = S
            if verbose:
                print(f"[pc_skeleton_cupc] level={level} edges_before={int(rows.sum() // 2)} "
                      f"removed={len(removed)} tests={st.get('tests', 0)}", flush=True)
            level += 1
        G = s.get_G()
    G = ((G != 0) | (G.T != 0)).astype(np.int32)
    np.fill_diagonal(G, 0)
    return G, sep, final_level
