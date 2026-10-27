"""
cits._cupc_wrapper (internal)

Python ctypes wrapper for cuPC (Zarebavani et al. 2020, TPDS), in the CITS fork
(https://github.com/abbasilab/cupc).

Shared library location
------------------------
The compiled cuPC skeleton library ``Skeleton.so`` is a REQUIRED external
dependency for the GPU and contemporaneous modes (base CITS does not need it). It
is a GPU/CUDA build, not on PyPI, and is licensed separately under cuPC's own
GPL-3.0 license, so it is NOT bundled with this package.

``_locate_cupc_dir()`` resolves the directory containing ``Skeleton.so``,
cached once per process, using the first location that actually contains it:

  1. ``$CUPC_DIR`` environment variable.
  2. Persisted config file
     (``${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir``).
  3. Candidate defaults: ``~/repos/cupc``, ``<package_dir>/external/cupc``,
     ``./cupc``, ``./repos/cupc``.
  4. If still not found AND running interactively (``sys.stdin.isatty()``):
     prompt for the path (up to 3 attempts), validate, and persist it to the
     config file so future runs auto-find it.
  5. Otherwise raise a clear, actionable ``FileNotFoundError`` (candidates
     tried, ``export CUPC_DIR=...``, the nvcc build command, the source URL,
     and a pointer to ``cits.set_cupc_dir(path)``).

Notebook / non-interactive users can set the path programmatically with
``cits.set_cupc_dir('/path/to/cupc')`` (validates and persists it).

Discovery only runs when a GPU/contemporaneous function is actually called, never
at ``import cits`` time.

Build cuPC (the CITS fork)::

    git clone https://github.com/abbasilab/cupc
    cd cupc && nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu

Library versions
----------------
The fork exports ``cupc_cits_version()``. With version >= 1 the skeleton runs
every conditioning level on the GPU with no cap on the conditioning-set size
(``cits._cupc_ext``), and every kernel launch is checked.

A library without that function (the original cuPC) stops at level 14 and has
32-bit combination counters that can skip tests for variables with many
neighbours. CITS then explains how to rebuild. In an interactive session it
asks once whether to continue on the old build. In a script it raises
``OutdatedCupcError``, unless ``CITS_ALLOW_OUTDATED_CUPC=1`` is set.

This wrapper:
  pc_skeleton_cupc(X, alpha=0.05, max_level=None)
      -> (G, sep_sets, inactive_neurons, final_level)

  X : (T, p) data matrix (samples x variables).
"""

from __future__ import annotations
import os
import ctypes
import numpy as np
from scipy.stats import norm

import sys
import warnings

from . import _cupc_ext

_LIB_NAME = "Skeleton.so"
_CUPC_SOURCE_URL = "https://github.com/abbasilab/cupc"  # the CITS fork of LIS-Laboratory/cupc

# Resolved cuPC directory, cached once per process (see _locate_cupc_dir).
_CUPC_DIR_CACHE = None


def _config_file_path():
    """Path to the persisted cuPC-directory config file.

    ${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "cits", "cupc_dir")


def _has_skeleton(cupc_dir):
    """True iff `cupc_dir` is a directory containing Skeleton.so."""
    if not cupc_dir:
        return False
    return os.path.isfile(os.path.join(cupc_dir, _LIB_NAME))


def _normalize_dir(path):
    """Expand ~ and $VARS in a user-supplied path."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path.strip())))


def _read_persisted_dir():
    """Read the cuPC directory from the config file, or None."""
    cfg = _config_file_path()
    try:
        with open(cfg, "r") as fh:
            path = fh.read().strip()
        return _normalize_dir(path) if path else None
    except (OSError, IOError):
        return None


def _persist_dir(cupc_dir):
    """Write `cupc_dir` to the config file, creating parent dirs as needed."""
    cfg = _config_file_path()
    try:
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        with open(cfg, "w") as fh:
            fh.write(cupc_dir + "\n")
    except (OSError, IOError):
        # Persistence is best-effort; failure to write the config file must
        # not break an otherwise-valid resolution.
        pass


def _candidate_dirs():
    """Ordered candidate default directories to probe for Skeleton.so."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.expanduser("~/repos/cupc"),
        os.path.join(pkg_dir, "external", "cupc"),
        os.path.abspath("./cupc"),
        os.path.abspath("./repos/cupc"),
    ]


def _not_found_error():
    """Build the actionable FileNotFoundError for non-interactive failure."""
    tried = []
    env_dir = os.environ.get("CUPC_DIR")
    if env_dir:
        tried.append(f"    $CUPC_DIR       -> {_normalize_dir(env_dir)}")
    persisted = _read_persisted_dir()
    if persisted:
        tried.append(f"    config file     -> {persisted}")
    for c in _candidate_dirs():
        tried.append(f"    candidate       -> {c}")
    tried_str = "\n".join(tried) if tried else "    (none)"
    return FileNotFoundError(
        f"cuPC shared library ('{_LIB_NAME}') not found.\n"
        f"cuPC is required for cits_gpu and for backend='cupc' "
        f"(base CITS, cits_rcit and cits_contemporaneous(backend='cpu') do not need it). It is a compiled GPU/CUDA "
        f"artifact, not on PyPI, so it is not bundled with the cits package.\n"
        f"Directories tried:\n{tried_str}\n"
        f"Fix it in any of these ways:\n"
        f"  - export CUPC_DIR=/path/to/cupc\n"
        f"  - call cits.set_cupc_dir('/path/to/cupc') (persists for future runs)\n"
        f"Build the library once with nvcc (about 30 s):\n"
        f"    git clone {_CUPC_SOURCE_URL}\n"
        f"    cd cupc && nvcc -O3 --shared -Xcompiler -fPIC "
        f"-o {_LIB_NAME} cuPC-S.cu")


def _prompt_for_dir(max_attempts=3):
    """Interactively prompt for the cuPC directory, validate, and persist.

    Returns the resolved directory on success, or None if all attempts fail.
    """
    for _ in range(max_attempts):
        try:
            raw = input(
                f"Enter path to your cuPC directory (must contain "
                f"{_LIB_NAME}): ")
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw.strip():
            continue
        cupc_dir = _normalize_dir(raw)
        if _has_skeleton(cupc_dir):
            _persist_dir(cupc_dir)
            return cupc_dir
        print(f"  No {_LIB_NAME} found in {cupc_dir!r}. Try again.",
              file=sys.stderr)
    return None


def _locate_cupc_dir(force=False, interactive=None):
    """Resolve the directory containing Skeleton.so, cached per process.

    Discovery order (first that actually contains Skeleton.so wins):
      1. $CUPC_DIR environment variable.
      2. Persisted config file (${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir).
      3. Candidate defaults: ~/repos/cupc, <package_dir>/external/cupc,
         ./cupc, ./repos/cupc.
      4. If still not found AND interactive: prompt the user (up to 3
         attempts); persist the answer on success.
      5. Otherwise raise an actionable FileNotFoundError.

    `interactive` : None (default) auto-detects a tty via sys.stdin.isatty();
    pass False to skip the prompt entirely (used by non-interactive backend
    auto-detection, which must never block on input).
    """
    global _CUPC_DIR_CACHE
    if _CUPC_DIR_CACHE is not None and not force:
        return _CUPC_DIR_CACHE

    # 1. Environment variable
    env_dir = os.environ.get("CUPC_DIR")
    if env_dir:
        cand = _normalize_dir(env_dir)
        if _has_skeleton(cand):
            _CUPC_DIR_CACHE = cand
            return cand

    # 2. Persisted config file
    persisted = _read_persisted_dir()
    if persisted and _has_skeleton(persisted):
        _CUPC_DIR_CACHE = persisted
        return persisted

    # 3. Candidate defaults
    for cand in _candidate_dirs():
        if _has_skeleton(cand):
            _CUPC_DIR_CACHE = cand
            return cand

    # 4. Interactive prompt
    if interactive is None:
        try:
            interactive = bool(sys.stdin) and sys.stdin.isatty()
        except (AttributeError, ValueError):
            interactive = False
    if interactive:
        prompted = _prompt_for_dir()
        if prompted is not None:
            _CUPC_DIR_CACHE = prompted
            return prompted

    # 5. Give up with an actionable error
    raise _not_found_error()


def is_cupc_available():
    """Return True iff a working cuPC Skeleton.so can be located and loaded
    WITHOUT prompting. Used for backend='auto' selection.

    Resolution uses only the non-interactive discovery steps (env var,
    config file, candidate defaults); it never prompts and never raises.
    """
    try:
        _locate_cupc_dir(interactive=False)
    except Exception:
        return False
    try:
        _get_lib()
        return True
    except Exception:
        return False


def set_cupc_dir(path):
    """Set and persist the cuPC directory (containing Skeleton.so).

    For notebook / non-interactive use: validates that `path` contains
    Skeleton.so, persists it to the config file
    (${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir) so future runs auto-find
    it, and updates the in-process cache.

    Parameters
    ----------
    path : str
        Directory containing the compiled cuPC Skeleton.so.

    Returns
    -------
    str
        The normalized, validated cuPC directory.

    Raises
    ------
    FileNotFoundError
        If `path` does not contain Skeleton.so.
    """
    global _CUPC_DIR_CACHE, _LIB, _LIB_INFO, _OUTDATED_OK
    cupc_dir = _normalize_dir(str(path))
    if not _has_skeleton(cupc_dir):
        raise FileNotFoundError(
            f"No {_LIB_NAME} found in {cupc_dir!r}. Build it with:\n"
            f"    git clone {_CUPC_SOURCE_URL}\n"
            f"    cd cupc && nvcc -O3 --shared -Xcompiler -fPIC "
            f"-o {_LIB_NAME} cuPC-S.cu")
    _persist_dir(cupc_dir)
    _CUPC_DIR_CACHE = cupc_dir
    # Load the library from the new directory on next use. A library rebuilt in place under the same path
    # needs a fresh Python process, because the operating system keeps the loaded copy.
    _LIB, _LIB_INFO, _OUTDATED_OK = None, None, None
    return cupc_dir


_ML = 14  # level cap of the original cuPC kernels (used only on an outdated library)
_REQUIRED_VERSION = 1  # cupc_cits_version() of the CITS fork that CITS needs
_ALLOW_ENV = "CITS_ALLOW_OUTDATED_CUPC"


class OutdatedCupcError(RuntimeError):
    """The cuPC library is not the CITS fork, or is older than this CITS needs."""


def _rebuild_steps():
    return ("Rebuild it from the CITS fork of cuPC (about 30 s):\n"
            f"    git clone {_CUPC_SOURCE_URL}\n"
            f"    cd cupc && nvcc -O3 --shared -Xcompiler -fPIC -o {_LIB_NAME} cuPC-S.cu\n"
            "Then point CITS to it with cits.set_cupc_dir('/path/to/cupc') or "
            "export CUPC_DIR=/path/to/cupc.")


def _outdated_message(info):
    if info.get("dev_build"):
        what = ("is a development build of the extended kernels without a version. Its Skeleton() "
                "function takes a different argument list, so CITS cannot use it")
    elif info["version"] is None:
        what = ("comes from the original cuPC. It stops the PC search at conditioning level 14, and its "
                "32-bit counters can skip tests for variables with many neighbours, so edges that "
                "should be removed can survive")
    else:
        what = (f"is version {info['version']} of the CITS fork, older than the version "
                f"{_REQUIRED_VERSION} this CITS needs")
    return f"The cuPC library {info['path']} {what}.\n" + _rebuild_steps()


def _load_lib():
    """Load Skeleton.so, read its CITS version and bind the functions it offers."""
    cupc_dir = _locate_cupc_dir()
    lib_path = os.path.join(cupc_dir, _LIB_NAME)
    lib = ctypes.CDLL(lib_path)
    lib.Skeleton.restype = None
    lib.Skeleton.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # C
        ctypes.POINTER(ctypes.c_int),     # P
        ctypes.POINTER(ctypes.c_int),     # G
        ctypes.POINTER(ctypes.c_double),  # Th
        ctypes.POINTER(ctypes.c_int),     # l
        ctypes.POINTER(ctypes.c_int),     # maxlevel
        ctypes.POINTER(ctypes.c_double),  # pMax
        ctypes.POINTER(ctypes.c_int),     # SepSet
    ]
    try:
        version_fn = lib.cupc_cits_version
    except AttributeError:
        version = None
    else:
        version_fn.restype = ctypes.c_int
        version_fn.argtypes = []
        version = int(version_fn())
    dev_build = version is None and hasattr(lib, "ext_create")
    uncapped = version is not None and version >= _REQUIRED_VERSION
    if uncapped:
        _cupc_ext.bind(lib)
    return lib, dict(path=lib_path, version=version, uncapped=uncapped, dev_build=dev_build)


_LIB = None
_LIB_INFO = None
_OUTDATED_OK = None  # per-process answer about an outdated library (None = not asked yet)


def _get_lib():
    global _LIB, _LIB_INFO
    if _LIB is None:
        _LIB, _LIB_INFO = _load_lib()
    return _LIB


def cupc_info():
    """Describe the cuPC library that CITS loads.

    Returns a dict with ``path``, ``version`` (``cupc_cits_version()``, or
    None for the original cuPC) and ``uncapped`` (True when the PC search has
    no cap on the conditioning-set size). Raises FileNotFoundError when no
    ``Skeleton.so`` is found. It never prompts about an outdated build.
    """
    _get_lib()
    return {k: _LIB_INFO[k] for k in ("path", "version", "uncapped")}


def _interactive():
    """True in a terminal or a Jupyter kernel, where a prompt can be answered."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return True
    except (AttributeError, ValueError):
        pass
    ipython = sys.modules.get("IPython")
    if ipython is not None and "ipykernel" in sys.modules:
        try:
            shell = ipython.get_ipython()
        except Exception:
            shell = None
        return shell is not None and getattr(shell, "kernel", None) is not None
    return False


def _confirm_outdated():
    """Decide, once per process, whether to run on an outdated library. Returns True or raises
    OutdatedCupcError."""
    global _OUTDATED_OK
    info = _LIB_INFO
    msg = _outdated_message(info)
    if info.get("dev_build"):
        raise OutdatedCupcError(msg)
    if _OUTDATED_OK:
        return True
    if os.environ.get(_ALLOW_ENV, "").strip().lower() in ("1", "true", "yes"):
        warnings.warn(f"{msg}\nContinuing because {_ALLOW_ENV} is set. The PC search stops at "
                      "conditioning level 14.", UserWarning, stacklevel=3)
        _OUTDATED_OK = True
        return True
    if _OUTDATED_OK is False:
        raise OutdatedCupcError(msg)
    if _interactive():
        print(f"cits warning: {msg}", file=sys.stderr, flush=True)
        try:
            answer = input("Continue on this outdated build for this session? The PC search then stops "
                           "at conditioning level 14. [y/N] ")
        except (EOFError, OSError, RuntimeError, NotImplementedError):
            answer = ""
        if answer.strip().lower() in ("y", "yes"):
            _OUTDATED_OK = True
            warnings.warn("cits: running on an outdated cuPC build. The PC search stops at conditioning "
                          "level 14.", UserWarning, stacklevel=3)
            return True
        _OUTDATED_OK = False
        raise OutdatedCupcError(msg)
    raise OutdatedCupcError(f"{msg}\nTo run on this build anyway, set {_ALLOW_ENV}=1. The PC search then "
                            "stops at conditioning level 14.")


def _cupc_thresholds(N, alpha):
    """Thresholds passed to the original cuPC Skeleton(). It runs conditioning
    levels 0 to ML (ML = 14) and reads Th[l] at level l, so ML + 1 values are
    needed. (The original R wrapper allocates 50.)"""
    return _fisher_thresholds(N, alpha, n_levels=_ML + 1)


def _fisher_thresholds(N, alpha, n_levels=_ML):
    """Per-level Fisher-z thresholds, Th[l] = |z_{alpha/2}| / sqrt(N - l - 3)
    (0 when N - l - 3 <= 0). Matches the R wrapper (cuPC.R line 91)."""
    return np.array([_cupc_ext.fisher_threshold(N, l, alpha) for l in range(n_levels)], dtype=np.float64)


def _legacy_skeleton(lib, C, T, alpha, max_level, return_sep_sets):
    """Original cuPC Skeleton(), capped at level 14 (only on an outdated library the user accepted)."""
    p = C.shape[0]
    G = np.ones((p, p), dtype=np.int32)
    np.fill_diagonal(G, 0)
    G = np.ascontiguousarray(G)
    Th = np.ascontiguousarray(_cupc_thresholds(T, alpha), dtype=np.float64)
    p_int = np.array([p], dtype=np.int32)
    l_int = np.array([0], dtype=np.int32)
    cap = _ML if max_level is None else min(int(max_level), _ML)
    maxlevel_int = np.array([cap], dtype=np.int32)
    pMax = np.ascontiguousarray(np.zeros((p, p), dtype=np.float64))
    SepSet = np.ascontiguousarray(np.full((p * p, _ML), -1, dtype=np.int32))
    lib.Skeleton(
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        p_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        G.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        Th.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        l_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        maxlevel_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        pMax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        SepSet.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    sep_sets = None
    if return_sep_sets:
        sep_sets = {}
        rows = SepSet.reshape(p * p, _ML)
        I, J = np.nonzero((G == 0) & ~np.eye(p, dtype=bool))
        for i, j in zip(I.tolist(), J.tolist()):
            sep_sets[(i, j)] = tuple(int(v) for v in rows[i * p + j] if v != -1)
    return G, sep_sets, int(l_int[0])


def pc_skeleton_cupc(X, alpha: float = 0.05, max_level=None,
                     zero_var_tol: float = 1e-12, verbose: bool = False,
                     return_sep_sets: bool = True):
    """Run the cuPC PC-stable skeleton on data matrix X.

    Parameters
    ----------
    X : array_like, shape (T, p)
        Data matrix. Treated as N=T independent samples.
    alpha : float
        Significance level for the Fisher-z partial-correlation test.
    max_level : int or None
        Max conditioning-set size. None (default) runs until no variable has
        enough neighbours, with no cap. An int stops after that level.
    zero_var_tol : float
        Columns with variance below this are flagged inactive; their
        edges are deterministically removed.
    verbose : bool
    return_sep_sets : bool
        Return the separating sets (default). With False, ``sep_sets`` is
        None, which saves memory for large graphs.

    Returns
    -------
    G : np.ndarray, shape (p, p), int
        Symmetric undirected skeleton. G[i, j] = 1 if edge retained.
    sep_sets : dict[(i, j) -> tuple[int]] or None
        Separating set used to remove each removed edge (both orderings).
        When several sets separate a pair, the GPU may record any of them.
    inactive_neurons : np.ndarray
        Indices of inactive (zero-variance) neurons.
    final_level : int
        The first level at which no test could run or remove an edge, or
        ``max_level`` when the search stopped there.
    """
    lib = _get_lib()

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, p); got {X.shape}")
    T, p = X.shape
    if T < 4:
        raise ValueError(f"T={T} too small (need T >= 4 for Fisher-z)")
    if p < 2:
        raise ValueError(f"p={p} too small")

    # Zero-variance guard
    col_var = X.var(axis=0)
    inactive_mask = col_var < zero_var_tol
    inactive_neurons = np.flatnonzero(inactive_mask)
    if inactive_neurons.size > 0:
        if verbose:
            print(f"[pc_skeleton_cupc] {inactive_neurons.size} inactive "
                  f"neurons; replacing with tiny noise to keep correlation "
                  f"matrix well-defined", flush=True)
        rng = np.random.default_rng(0)
        X = X.copy()
        for j in inactive_neurons:
            X[:, j] = rng.standard_normal(T) * 1e-10

    # Correlation matrix (row-major flat)
    C = np.ascontiguousarray(np.corrcoef(X.T), dtype=np.float64)

    if _LIB_INFO["uncapped"]:
        G, sep_sets, final_level = _cupc_ext.skeleton(
            lib, C, T, alpha=alpha, max_level=max_level,
            return_sep_sets=return_sep_sets, verbose=verbose)
    else:
        _confirm_outdated()
        G, sep_sets, final_level = _legacy_skeleton(lib, C, T, alpha, max_level, return_sep_sets)

    # Force-remove edges incident on inactive neurons (defensive: should
    # already be zero from the noise, but be explicit).
    if inactive_neurons.size > 0:
        G[inactive_neurons, :] = 0
        G[:, inactive_neurons] = 0

    return G, sep_sets, inactive_neurons, final_level
