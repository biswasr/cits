"""
CITS algorithm for Causal Inference in Time Series.

Every function runs the same CITS algorithm. They differ in two choices:

  Conditional-independence test (match it to your data)
    - partial correlation (Fisher z): linear / Gaussian data
    - RCIT kernel test: nonlinear or non-Gaussian data, e.g. spike counts

  Search (match it to graph size and sampling rate)
    - exhaustive conditioning-set search, small to moderate graphs:
        cits.methods.cits_full (partial correlation)
        cits.cits_rcit (RCIT; GPU if available, else CPU)
    - neighbor-restricted search, large graphs (~100 to thousands of variables):
        cits.cits_gpu (partial correlation, GPU via cuPC)
    - lagged plus contemporaneous (same-time) edges, for slow sampling:
        cits.cits_contemporaneous (partial correlation; CPU or GPU)

Or dispatch by name with ``cits.run(X, method)``, where method is one of
'base', 'gpu', 'rcit', 'contemporaneous'.

``import cits`` always succeeds on a CPU-only machine. ``cits_gpu`` (and
``cits_contemporaneous(backend='cupc')``) need a compiled cuPC ``Skeleton.so``
from the CITS fork (https://github.com/abbasilab/cupc) and a CUDA GPU, and
raise a clear error at call time when it is unavailable (see the README "GPU
setup (cuPC)" section). ``cits.cupc_info()`` reports which library is loaded.
The neighbor-restricted search has no cap on the conditioning-set size.

Extras:
  - ``cits.plot_graph`` / ``cits.plot_matrix`` -- publication-style figures
    (needs matplotlib: ``pip install cits[viz]``).
  - ``cits.run`` -- dispatch by method name.
  - ``cits.cite`` -- print how to cite the paper.

Logging: messages go through ``logging.getLogger('cits')``. Set its level to
control verbosity; ``CITS_QUIET=1`` silences informational notices.
"""

__version__ = "1.10.0"

import warnings

from . import methods, simulate_timeseries

# cuPC directory helper. Pure-Python (no GPU needed to import); lets
# notebook / non-interactive users set and persist the cuPC location
# programmatically, bypassing the interactive prompt.
try:
    from ._cupc_wrapper import set_cupc_dir, cupc_info, OutdatedCupcError
except Exception as _set_cupc_err:  # pragma: no cover - env dependent
    _set_cupc_dir_err = _set_cupc_err

    def set_cupc_dir(path):
        raise ImportError(
            "cits.set_cupc_dir is unavailable because cits._cupc_wrapper "
            f"failed to import. Original import error: {_set_cupc_dir_err!r}"
        ) from _set_cupc_dir_err

    def cupc_info():
        raise ImportError(
            "cits.cupc_info is unavailable because cits._cupc_wrapper "
            f"failed to import. Original import error: {_set_cupc_dir_err!r}"
        ) from _set_cupc_dir_err

    class OutdatedCupcError(RuntimeError):
        """Placeholder when cits._cupc_wrapper failed to import."""

# GPU and contemporaneous entry points. Import failures (e.g. missing optional
# dependencies) must not break `import cits` or the base CPU algorithm, so
# each is wrapped: if the import fails, the public name is replaced by a
# shim that raises a clear ImportError only when the function is called.
try:
    from .gpu import cits_gpu
except Exception as _gpu_import_error:  # pragma: no cover - env dependent
    _cits_gpu_err = _gpu_import_error

    def cits_gpu(*args, **kwargs):
        raise ImportError(
            "cits.cits_gpu is unavailable because its dependencies failed to "
            "import. It requires a compiled cuPC 'Skeleton.so' and a "
            "CUDA-capable GPU; set the CUPC_DIR environment variable to the "
            "directory containing Skeleton.so (see the README 'GPU setup "
            f"(cuPC)' section). Original import error: {_cits_gpu_err!r}"
        ) from _cits_gpu_err

try:
    from .contemporaneous import cits_contemporaneous
except Exception as _contemp_import_error:  # pragma: no cover - env dependent
    _cits_contemp_err = _contemp_import_error

    def cits_contemporaneous(*args, **kwargs):
        raise ImportError(
            "cits.cits_contemporaneous is unavailable because its dependencies "
            "failed to import (see the original error below). It runs on the "
            "CPU with backend='cpu'; backend='cupc' additionally needs a compiled "
            "cuPC 'Skeleton.so' (README 'GPU setup (cuPC)'). "
            f"Original import error: {_cits_contemp_err!r}"
        ) from _cits_contemp_err

def cits_versionb(*args, **kwargs):
    """Deprecated name of :func:`cits_contemporaneous` (v1.9.0). Use that instead."""
    warnings.warn("cits.cits_versionb is deprecated; use cits.cits_contemporaneous.",
                  DeprecationWarning, stacklevel=2)
    return cits_contemporaneous(*args, **kwargs)


# RCIT (kernel CI test) variant. Pure Python at import; torch is imported only
# when cits_rcit is called with null='gamma' (pip install cits[rcit]).
from .rcit import cits_rcit

# Plotting helpers. matplotlib is an optional dependency (extras 'viz');
# plot.py imports it lazily, so this import itself never needs matplotlib.
# Wrapped for safety so `import cits` never fails on a plotting-dep issue.
try:
    from .plot import plot_graph, plot_matrix
except Exception as _plot_import_error:  # pragma: no cover - env dependent
    _cits_plot_err = _plot_import_error

    def plot_graph(*args, **kwargs):
        raise ImportError(
            "cits.plot_graph is unavailable; plotting requires matplotlib "
            f"(pip install cits[viz]). Original import error: {_cits_plot_err!r}"
        ) from _cits_plot_err

    def plot_matrix(*args, **kwargs):
        raise ImportError(
            "cits.plot_matrix is unavailable; plotting requires matplotlib "
            f"(pip install cits[viz]). Original import error: {_cits_plot_err!r}"
        ) from _cits_plot_err


_METHODS = ("base", "gpu", "contemporaneous", "rcit")


def run(X, method, **kwargs):
    """Unified entry point dispatching to the three CITS variants.

    Parameters
    ----------
    X : array_like, shape (p, T)
        Time series, p variables by T timepoints.
    method : str
        One of 'base' (-> cits.methods.cits_full), 'gpu' (-> cits.cits_gpu),
        'contemporaneous' (-> cits.cits_contemporaneous), or 'rcit' (-> cits.cits_rcit).
    **kwargs
        Passed through to the dispatched function (e.g. tau, alpha, backend).

    Returns
    -------
    The return value of the dispatched function.
    """
    if method == "base":
        return methods.cits_full(X, **kwargs)
    if method == "gpu":
        return cits_gpu(X, **kwargs)
    if method == "versionb":  # pre-1.9.1 name
        warnings.warn("method='versionb' is deprecated; use method='contemporaneous'.",
                      DeprecationWarning, stacklevel=2)
        method = "contemporaneous"
    if method == "contemporaneous":
        return cits_contemporaneous(X, **kwargs)
    if method == "rcit":
        return cits_rcit(X, **kwargs)
    raise ValueError(
        f"unknown method {method!r}; valid methods are {list(_METHODS)} "
        f"('base' -> cits_full, 'gpu' -> cits_gpu, "
        f"'contemporaneous' -> cits_contemporaneous, 'rcit' -> cits_rcit).")


_CITE = (
    "Please cite the CITS paper:\n"
    "  Biswas, R., Sripada, S., Mukherjee, S. & Abbasi-Asl, R. CITS: "
    "Nonparametric Statistical Causal Modeling for High-Resolution Neural "
    "Time Series. arXiv:2508.01920. https://arxiv.org/abs/2508.01920\n"
    "The citation will be finalized upon publication; see the README "
    "'Citation' section for the current reference.")


_BIBTEX = """@article{biswas2025cits,
  title   = {CITS: Nonparametric Statistical Causal Modeling for High-Resolution Neural Time Series},
  author  = {Biswas, Rahul and Sripada, SuryaNarayana and Mukherjee, Somabha and Abbasi-Asl, Reza},
  journal = {arXiv preprint arXiv:2508.01920},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.01920}
}"""


def cite(bibtex=False):
    """Print how to cite CITS (arXiv pointer; finalized on publication).

    With ``bibtex=True``, print a BibTeX entry instead.
    """
    print(_BIBTEX if bibtex else _CITE)


__all__ = [
    "methods",
    "simulate_timeseries",
    "cits_gpu",
    "cits_contemporaneous",
    "cits_rcit",
    "run",
    "set_cupc_dir",
    "cupc_info",
    "OutdatedCupcError",
    "plot_graph",
    "plot_matrix",
    "cite",
]
