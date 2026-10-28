"""
cits._common (internal)

Shared utilities: the package logger, info/warning routing, and friendly
input validation. Depends only on numpy + the standard library, so it is safe
to import from the base (CPU) algorithm.

Logging
-------
A single logger ``logging.getLogger('cits')`` carries all package messages.
It gets a ``NullHandler`` (so importing the library never forces output) and a
default level of INFO. Informational notices (the one-time backend notice,
verbose progress markers) go through ``log_info``; runtime problems go through
``log_warning``.

By default a user has not configured logging, so the ``cits`` logger has only
the NullHandler and messages would be swallowed. To keep the default
experience informative, ``log_info``/``log_warning`` additionally print to
stderr WHEN (and only when) no real handler is attached to the ``cits`` logger
or its ancestors. Once the user configures logging (adds a handler anywhere up
the chain), routing switches to their handlers and the stderr fallback stops,
so messages are never duplicated.

Users control verbosity the standard way::

    import logging
    logging.getLogger('cits').setLevel(logging.WARNING)   # silence info
    logging.getLogger('cits').setLevel(logging.ERROR)     # silence warnings too

The env var ``CITS_QUIET=1`` silences informational notices (warnings still
surface).
"""

from __future__ import annotations
import os
import sys
import logging
import numpy as np

logger = logging.getLogger("cits")
logger.addHandler(logging.NullHandler())
# Default level so INFO notices are enabled unless the user raises the level.
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)


def _quiet():
    """True if informational notices are silenced (env CITS_QUIET truthy)."""
    return os.environ.get("CITS_QUIET", "0") not in ("0", "", "false", "False")


def _has_real_handler():
    """True if a non-Null handler is attached to `cits` or an ancestor
    (i.e., the user has configured logging output)."""
    cur = logger
    while cur:
        for h in cur.handlers:
            if not isinstance(h, logging.NullHandler):
                return True
        if not cur.propagate:
            break
        cur = cur.parent
    return False


def log_info(msg):
    """Informational message. Suppressed by CITS_QUIET or a raised log level.
    Falls back to stderr when the user has not configured logging so the
    default experience stays informative (never duplicated)."""
    if _quiet() or not logger.isEnabledFor(logging.INFO):
        return
    logger.info(msg)
    if not _has_real_handler():
        print(msg, file=sys.stderr, flush=True)


def log_warning(msg):
    """Warning message. Always surfaces (not silenced by CITS_QUIET). Falls
    back to stderr when no logging handler is configured so warnings are never
    lost to the NullHandler."""
    logger.warning(msg)
    if not _has_real_handler() and logger.isEnabledFor(logging.WARNING):
        print(msg, file=sys.stderr, flush=True)


def _fmt_idx(idx, limit=10):
    idx = list(idx)
    if len(idx) <= limit:
        return str(idx)
    return f"{idx[:limit]} ... (+{len(idx) - limit} more)"


def validate_X(X, context="cits", warn_transpose=True):
    """Friendly validation of the input time series X.

    Expected shape is (p variables, T timepoints). Returns X as a numpy array.

    Raises ValueError (with an actionable message) when X is not 2D or
    contains non-finite values. Emits warnings (via the package logger) when X
    looks transposed (p > T) or has constant / all-zero variable rows
    (partial correlation is undefined on constant series).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            f"{context}: X must be a 2D array of shape (p variables, "
            f"T timepoints); got shape {X.shape} with ndim={X.ndim}.")
    if not np.isfinite(X).all():
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        raise ValueError(
            f"{context}: X contains non-finite values ({n_nan} NaN, "
            f"{n_inf} inf). Remove or impute them before running "
            f"(e.g. interpolate gaps, drop bad timepoints).")

    p, T = X.shape
    if warn_transpose and p > T:
        log_warning(
            f"{context}: X has shape (p={p}, T={T}) with more variables than "
            f"timepoints. The expected layout is (p variables, T timepoints); "
            f"if your array is (T, p) pass X.T instead.")

    # Constant / all-zero variable rows (variance exactly 0).
    dead = np.flatnonzero(X.var(axis=1) == 0)
    if dead.size:
        log_warning(
            f"{context}: {dead.size} constant/all-zero variable row(s) at "
            f"indices {_fmt_idx(dead.tolist())}. Partial correlation is "
            f"undefined on constant series; edges for these variables are "
            f"dropped and their results are unreliable. Consider removing "
            f"them from X.")
    return X
