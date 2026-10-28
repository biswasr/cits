# Changelog

## v1.10.0

- **No cap on the conditioning-set size.** The neighbour-restricted PC search
  used by `cits_gpu`, `cits_contemporaneous` and `pc_skeleton_raw` stopped at
  conditioning level 14, on the GPU (cuPC's compile-time limit) and on the CPU
  (for parity). It now runs until no variable has enough neighbours, as the PC
  algorithm specifies. `max_level` defaults to None, and an int still caps the
  search on request. On 192 skeletons of a calcium-imaging analysis, 179
  searches went past level 14, and 135 of them removed edges there (1,172 in
  total).
- **cuPC from the CITS fork.** The GPU backend now uses a `Skeleton.so` built
  from https://github.com/biswasr/cupc. The fork adds a generic kernel for
  any conditioning level, checks every kernel launch, and fixes a 32-bit
  overflow in the original combination counters. The overflow could skip tests
  for variables with many neighbours (from degree 1,627 at level 3 down to 31
  at level 14), so some edges that should have been removed survived.
- **Older cuPC builds.** CITS reads `cupc_cits_version()` from the library. On
  a build of the original cuPC it shows how to rebuild. An interactive session
  then asks once whether to continue on the old build, which stops at level
  14. A script raises `cits.OutdatedCupcError` unless
  `CITS_ALLOW_OUTDATED_CUPC=1` is set. With `backend='auto'`,
  `cits_contemporaneous` continues on the CPU backend instead.
  `cits.cupc_info()` reports the loaded library.
- `cits.set_cupc_dir()` now loads the library from the new folder on the next
  call.
- Both backends stop at the first level where no test can run or remove an
  edge (including N - level - 3 <= 0) and report the same final level.
- `pc_skeleton_cupc` and `pc_skeleton_cpu` take `return_sep_sets`. `cits_gpu`
  no longer builds separating sets that it does not use.

## v1.9.3

- **cuPC thresholds.** The cuPC wrapper (used by `cits_gpu`, by
  `cits_contemporaneous` and by the cuPC backend of `pc_skeleton_raw`) passed
  14 Fisher-z thresholds, for conditioning levels 0 to 13. cuPC runs levels 0
  to 14 and read the level-14 threshold from past the end of the array. The
  wrapper now passes 15. Results change only when some node keeps 15 or more
  neighbours up to level 14. On a calcium-imaging field with 1,152 neurons and
  1,185 samples, 1 of 6,464 edges changed. The CPU and cuPC backends now agree
  at level 14.

## v1.9.2

- **Simulator.** `simulate_timeseries.simulate('lingauss1', ...)` now generates
  X3 = 2 X1 - X2 + noise, matching Linear Gaussian 1 in the paper. It
  previously used +X2. The ground-truth weight for 1 -> 2 is now -1.
- **Quickstart notebook** `examples/quickstart.ipynb` with an Open in Colab
  link, replacing the old Colab link.

## v1.9.1

- **Renamed `cits.cits_versionb` to `cits.cits_contemporaneous`**, and
  `cits.run(..., method='versionb')` to `method='contemporaneous'`. The old
  names still work and emit a `DeprecationWarning`. Behaviour and output are
  unchanged.
- README: "How to run CITS" now organizes the functions by the two choices
  that distinguish them (conditional-independence test x search) in one table,
  so it is clear that every function runs CITS. Shorter cuPC setup section.
- Docstrings: corrected the RCIT conditioning-set cap used by the paper (no
  cap); corrected the import-error message of `cits_contemporaneous`, which
  runs on the CPU without cuPC.

## v1.9.0

- **New: `cits.cits_rcit`**, CITS with the randomized conditional-independence
  test (RCIT), for nonlinear and non-Gaussian data. Batched gamma-null test via
  PyTorch on CUDA or CPU (same graph on both), plus a NumPy permutation
  reference (`null='perm'`). Windows of 2*tau+1 time points. Ported from the
  code behind the paper's nonlinear and spiking benchmarks and verified to give
  identical graphs (30/30 GPU, 3/3 permutation reference). Default
  `max_cond_size=None` (no cap), as in the paper's Table 1. New extra: `pip install cits[rcit]`. Available through
  `cits.run(X, 'rcit')`.
- Includes all v1.8.3 changes (license, citation, packaging and NumPy 2 fixes).

## v1.8.3

- **`plot_matrix` side strips are opt-in** (`side_strips=False` by default).
  Group separators and side strips are drawn only when the nodes are ordered
  by group, so the default matrix stays clean.
- **Fix: `plot_graph` on NumPy 2.** It used `ndarray.ptp`, which NumPy 2.0
  removed, and raised `AttributeError` on current NumPy. It now uses `np.ptp`.
- **Packaging fix.** Stale `build/` and `cits.egg-info/` artifacts from v1.3
  were tracked in git. Building from a fresh clone reused them and installed
  an old `__init__.py` mixed with the new modules. They are no longer tracked
  and are now ignored. The PyPI v1.4 wheel was checked and is unaffected.
- **`python_requires` corrected to `>=3.7`**, because
  `from __future__ import annotations` needs Python 3.7.
- **License and citation.** Released under the PolyForm Noncommercial License
  1.0.0; for commercial use, contact the authors. Added `CITATION.cff`, a
  "How to cite" notice in `LICENSE`, and `cits.cite(bibtex=True)`.

## v1.8.2

- **`plot_matrix` matches the paper's Neuropixels adjacency style.** Now uses
  `matshow` with a diverging colormap (default `'bwr'`) and a ROBUST SYMMETRIC
  color scale by default: `vmax` is a high percentile (default 98th) of the
  |nonzero off-diagonal weights| and `vmin = -vmax`, so faint edges stay
  visible while a few strong outliers saturate (the old raw min/max autoscale
  washed everything out). New kwargs `vmin`, `vmax`, `cmap`,
  `robust_percentile`, `group_colors`; passing `vmin=-0.1, vmax=0.1`
  reproduces the paper's fixed scale. When `groups` is given, colored group
  side-strips are drawn along the top and left edges (same per-group colors as
  `plot_graph`) plus subtle separator lines. NaN cells are masked to neutral
  gray and excluded from the scale.
- **`plot_graph` handles an arbitrary number of groups.** Up to 6 groups keep
  the per-group hue-family shading; beyond that each group gets a distinct
  solid color from an extended qualitative colorblind-safe palette (no hard
  cap). Single-member groups get a solid color. Documented the many-groups
  guideline (~ up to 10-12 for legibility; pass `group_colors` for full
  control).

## v1.8.1

- **`plot_graph` rewritten to match the paper's causal-graph aesthetic**
  (ports the layout/arc logic of `regen_cfc_stimtypes.py`, kept general for
  arbitrary group labels):
  - Grouped-cluster layout when `groups` is given: group centroids evenly
    spaced around a ring, members in a small sub-cluster around each centroid
    (falls back to circular/spring with no groups).
  - One colorblind-safe hue family per group (Oranges, a teal gradient,
    Purples, Greens, Blues, Greys), shaded 0.3->0.9 within each group; new
    `group_colors` kwarg overrides a group's color.
  - Curved quadratic-Bezier arcs with arrowheads placed partway along the arc;
    bidirectional pairs drawn as a single arc with one head near each end;
    small self-loops. Edge width interpolated over `edge_width_range`; neutral
    dark-gray edges by default with an opt-in `sign_color`.
  - Legend placed OUTSIDE the axes (no overlap with nodes); equal aspect, axis
    off. Save with `bbox_inches='tight'`.
  - New/changed kwargs: `layout` gains `'auto'` (default) and `'grouped'`;
    added `group_colors`, `sign_color`, `edge_color`, `edge_width_range`,
    `node_size`, `min_spacing`, `base_cluster_radius`.
- **`plot_matrix`**: colorbar given its own space; group separators made
  subtle (thin light-gray lines).

## v1.8

Usability pass: plotting, input validation, logging, a unified dispatch
facade, and a citation helper.

### Added

- **Plotting helpers (`cits.plot_graph`, `cits.plot_matrix`)** in a new module
  `cits.plot`, in the style of the paper's causal-graph figures.
  `plot_graph` renders a directed node-link graph (reciprocal pairs as a
  single double-headed edge, width proportional to |weight|, nodes colored by
  optional `groups` with a colorblind-safe Okabe-Ito palette). `plot_matrix`
  renders an adjacency heatmap (diverging colormap for signed weights,
  optional group separators). matplotlib is an OPTIONAL dependency
  (`extras_require['viz']`, `pip install cits[viz]`), imported lazily; calling
  a plot function without it raises a clear ImportError. `import cits` never
  requires matplotlib.
- **Unified dispatch facade `cits.run(X, method, **kwargs)`** with method in
  {'base', 'gpu', 'contemporaneous'} dispatching to `cits_full` / `cits_gpu` /
  `cits_contemporaneous`. Unknown methods raise a clear error listing the valid ones.
- **Friendly input validation** at the top of `cits_full`,
  `cits_full_weighted`, `cits_gpu`, and `cits_contemporaneous`: requires a 2D array,
  raises an actionable error on NaN/inf, warns when X looks transposed
  (p > T, expected `(p variables, T timepoints)`), and warns (with offending
  indices) on constant/all-zero variable rows (partial correlation is
  undefined on constant series).
- **`cits.cite()`** prints how to cite the paper (arXiv pointer; no fabricated
  BibTeX; citation finalized on publication).

### Changed

- **Logging.** Informational notices (the one-time backend notice, verbose
  progress markers) and the runtime GPU-fallback warning now route through
  `logging.getLogger('cits')` (`log_info` / `log_warning`) instead of bare
  prints. The logger gets a `NullHandler` and a default INFO level; when the
  user has not configured logging, notices/warnings still appear on stderr
  (no duplication once a handler is attached). `CITS_QUIET=1` still silences
  informational notices; raising the `cits` logger level
  (`logging.getLogger('cits').setLevel(...)`) is the standard way to control
  verbosity. Warnings/errors still surface.
- **Large-p CPU heads-up is now data-informed.** The notice cites the paper's
  scaling benchmark (cuPC inferred p=1000-variable graphs in ~33 s) and keeps
  ~100 variables as an approximate guideline. No CPU-CITS runtime is claimed.

## v1.7

A correctness fix for exact paper reproduction, plus a user-experience pass so
users always know which backend is running and what to do when something fails.

### Changed

- **`cits_contemporaneous` now reproduces the paper's montage pipeline exactly by
  default.** New parameter `weight_lagged_only: bool = False`. When False
  (default), the pipeline stops after the union LSCM refit, leaving
  lagged-only edges (present in the lagged graph but not the contemporaneous
  skeleton) unweighted -- matching `fig5_neuropixels/_neuropixels_contemp_pooled.py (contemp_directed) in cits-paper`
  and the Fig 5A magnitudes. When True, it also OLS-weights the lagged-only
  edges (the previous, fuller behavior). Note: this changes the default
  numeric output of `cits_contemporaneous` relative to v1.5/v1.6 (e.g. on gabors,
  41 nonzeros instead of 47); the union skeleton and edge-type outputs are
  unchanged.

### Added (user experience)

- **One-time backend notice** (stderr, once per process) when a contemporaneous run
  starts: `cits: using cuPC GPU backend.` (auto found cuPC),
  `cits: cuPC not found; using the CPU backend ...` (auto fell back), or the
  existing >100-variable recommendation for explicit `backend='cpu'`.
  Silence with the env var `CITS_QUIET=1`.
- **Runtime GPU-failure handling.** If a cuPC/CUDA call fails at run time:
  under `backend='auto'` a warning is emitted and the run falls back to the
  CPU backend; under explicit `backend='cupc'` it re-raises with actionable
  guidance (fix cuPC/CUDA or use `backend='cpu'`) rather than silently
  falling back.
- **Actionable `tau>1` error** for `cits_contemporaneous` pointing to `cits_gpu` for
  higher-lag lagged-only inference. The invalid-`backend` `ValueError` is
  unchanged.
- **Progress markers** to stderr when `verbose=True`
  (`[cits contemporaneous] 1/4 lagged skeleton (p=NN)`, `2/4 contemporaneous PC`,
  `3/4 union`, `4/4 LSCM refit`). Silent by default.
- **Equivalence test** (`test/test_paper_repro.py`, runs only when cuPC is
  available) asserting `cits_contemporaneous(X, backend='cupc', weight_lagged_only=False)`
  reproduces the original montage pipeline bit-for-bit on the gabors input.

### Docs

- README: `weight_lagged_only` documented (default reproduces the paper);
  new "What you'll see / troubleshooting" subsection covering the backend
  notice, CPU/GPU guidance, and cuPC-failure guidance.

## v1.6

Makes the package usable on any machine with no author involvement: the GPU is
now recommended (not required) for large graphs, and every mode has a
pure-Python path.

### Added

- **CPU PC-stable skeleton (`cits._pc_skeleton_cpu.pc_skeleton_cpu`).** A
  pure-numpy, neighbor-restricted, increasing-conditioning-size PC-stable
  skeleton with a Fisher-z partial-correlation CI test (reuses the v1.4-fixed
  `cits.methods.partial_corr`). It implements the SAME algorithm as cuPC
  (NOT the exponential powerset conditioning of `cits_full`), so it recovers
  the same skeleton as cuPC. Validated: on a toy graph the CPU and cuPC
  skeletons are identical. Used for both the lagged unrolled-window skeleton
  and the contemporaneous slice.
- **`backend` argument on `cits_contemporaneous`.** `'auto'` (new default) uses cuPC
  if a working `Skeleton.so` is found, else the CPU skeleton; `'cpu'` forces
  CPU; `'cupc'` forces GPU (clear error if unavailable). On the CPU path with
  more than ~100 variables, a one-time recommendation to use the cuPC backend
  is printed. `alpha=0.05`, `tau=1` defaults unchanged; `tau=1` remains the
  only supported union order.
- **`_pc_raw.pc_skeleton_raw` `backend='python'`/`'cpu'`** now runs the new
  CPU skeleton (previously a `NotImplementedError` stub).

### Changed

- **cuPC is optional, recommended for large graphs (> ~100 variables), not
  required.** `import cits`, base CITS, and `cits_contemporaneous(backend='cpu')`
  need nothing native. `cits_gpu` stays cuPC-only (it is the GPU accelerator).
- **R / `kpcalg` is now an optional extra.** `rpy2` moved out of core
  `install_requires` into `extras_require['hsic']` (`pip install cits[hsic]`);
  R is imported lazily inside the HSIC test only, so the Gaussian
  partial-correlation path works with just Python + numpy. (methods.py already
  imported rpy2 lazily; no code change was needed there.)

### Docs

- README: runnable Quickstart via `cits.simulate_timeseries`; "Which method
  should I use?" and "Choosing parameters" decision guides; cuPC reframed as
  recommended-for-large-graphs; optional-R note; Zenodo DOI placeholder for
  the tagged release.

## v1.5

Two additional ways to run CITS, alongside the existing base (CPU) algorithm.
The repository now offers three entry points.

### Added

- **GPU-accelerated CITS (`cits.cits_gpu`).** The faithful scalable CITS-lag
  skeleton. It matches the exact base CITS algorithm
  (`cits.methods.cits_full` with `cond_dep='cond_dep_pcorr'`) on every axis
  except the conditioning-set search, which is replaced by the
  neighbor-restricted, increasing-size PC search (cuPC on GPU). Returns the
  rolled lagged binary adjacency. Requires a compiled cuPC `Skeleton.so` and
  a CUDA-capable GPU.

- **CITS with contemporaneous edges (`cits.cits_contemporaneous`).** Augments the
  lagged CITS skeleton with a contemporaneous PC step, unions the lagged and
  contemporaneous parents, and refits one coherent set of signed structural
  (LSCM) coefficients per child. Pipeline: lagged skeleton (`cits_gpu`) +
  contemporaneous PC skeleton (chi-subsampled lag-0 slice) + v-structure
  orientation (Meek optional) + union + signed LSCM refit with local IDA for
  undirected edges. Returns the signed weighted adjacency (beta / 0 / NaN
  convention). Requires cuPC as above.

- Internal helper modules `cits._cupc_wrapper`, `cits._pc_raw`,
  `cits._pc_orientation`, `cits._union_cpdag`, `cits._lscm_refit`.

### Notes

- The cuPC shared library `Skeleton.so` is not bundled (separate GPL-3.0
  GPU/CUDA build). Its location is configurable via the `CUPC_DIR`
  environment variable, and a clear, actionable error is raised if it is not
  found. See the README "GPU setup (cuPC)" section and cite
  `zarebavani2020cupc`.
- The GPU/contemporaneous imports degrade gracefully: `import cits` and the base
  CPU algorithm work on a CPU-only machine; the GPU/contemporaneous functions raise
  a clear error only when called without cuPC available.

## v1.4

Two correctness fixes to the partial-correlation conditional independence test.

### Fixed

- **`partial_corr` now fits an intercept.** Previously the regression of A and B
  on the conditioning set was forced through the origin (`linalg.lstsq` on
  `C[idx, :].T` without an intercept column). When the data was not centered,
  this biased the residuals and the resulting partial correlation. The fix
  prepends a column of ones to the conditioning matrix before the least-squares
  solve, making the partial correlation shift-invariant.

- **`cits_unrolled` conditioning set spans the full unrolled graph and excludes
  the two variables being tested.** Previously the conditioning powerset
  iterated over `range(2*(tau+1))`, which only covers `2*(tau+1)` nodes — far
  fewer than the `2*p*(tau+1)` nodes of the unrolled graph. The two variables
  being tested (`i = t*p + v` and `j = t1*p + v1`) could also appear in the
  conditioning set, which is incorrect by the definition of conditional
  independence. The fix changes the powerset iterable to
  `(i for i in range(2*p*(tau+1)) if i != t*p+v and i != t1*p+v1)`.

### Tests

- Added a shift-invariance regression test in `test/test_cits.py` that compares
  graph recovery before and after applying per-neuron offsets. With the fix the
  adjacency matrix and weighted effects are identical to numerical precision.

## v1.3

Earlier release. See git history for details.
