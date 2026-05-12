# Changelog

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
