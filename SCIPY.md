# SciPy Ingestion Audit: Valuable New Targets

This list identifies high-value algorithmic targets in SciPy that represent actual implementations (not simple wrappers or basic statistical functions) and are currently missing from the `ageoa` catalog.

---

## 1. Linear Algebra (`scipy.linalg`)
*Beyond standard solve/inv.*

| Target | Description |
|---|---|
| `scipy.linalg.cholesky` | Cholesky decomposition for Hermitian, positive-definite matrices. |
| `scipy.linalg.qr` | QR decomposition (Orthogonal, Upper Triangular). |
| `scipy.linalg.svd` | Singular Value Decomposition. |
| `scipy.linalg.schur` | Schur decomposition of a square matrix. |
| `scipy.linalg.hessenberg` | Hessenberg form of a square matrix. |
| `scipy.linalg.solve_banded` | Optimized solver for banded matrices. |
| `scipy.linalg.solve_toeplitz` | Optimized solver for Toeplitz matrices. |
| `scipy.linalg.solve_riccati` | Solves the continuous algebraic Riccati equation (CARE). |
| `scipy.linalg.solve_sylvester` | Solves the Sylvester equation $AX + XB = C$. |

## 2. Signal Processing (`scipy.signal`)
*Beyond standard IIR/FIR filters.*

| Target | Description |
|---|---|
| `scipy.signal.wiener` | Adaptive Wiener filter for noise reduction. |
| `scipy.signal.medfilt` | Median filter for non-linear smoothing. |
| `scipy.signal.savgol_filter` | Savitzky-Golay filter for polynomial smoothing. |
| `scipy.signal.welch` | Power Spectral Density (PSD) estimation via Welch’s method. |
| `scipy.signal.lombscargle` | Lomb-Scargle periodogram for unevenly sampled data. |
| `scipy.signal.cwt` | Continuous Wavelet Transform for time-frequency analysis. |
| `scipy.signal.hilbert` | Compute the analytic signal using the Hilbert transform. |
| `scipy.signal.remez` | Optimal FIR filter design using the Parks-McClellan algorithm. |

## 3. Optimization (`scipy.optimize`)
*Beyond basic minimize/root.*

| Target | Description |
|---|---|
| `scipy.optimize.shgo` | Simplicial Homology Global Optimization. |
| `scipy.optimize.dual_annealing` | Dual annealing global optimization. |
| `scipy.optimize.differential_evolution` | Stochastic population-based method (genetic algorithm). |
| `scipy.optimize.basinhopping` | Iterative global stepping with local minimization. |
| `scipy.optimize.milp` | Mixed-integer linear programming. |

## 4. Integration (`scipy.integrate`)
*Beyond basic quad/simpson.*

| Target | Description |
|---|---|
| `scipy.integrate.nquad` | Integration over $N$ variables (recursive adaptive quadrature). |
| `scipy.integrate.romberg` | Romberg integration (Richardson extrapolation). |
| `scipy.integrate.solve_bvp` | Solves boundary value problems for ODEs. |
| `scipy.integrate.quad_vec` | Adaptive integration for vector-valued functions. |

## 5. Spatial Algorithms (`scipy.spatial`)
*Geometry and nearest-neighbor search.*

| Target | Description |
|---|---|
| `scipy.spatial.cKDTree` | Fast nearest-neighbor lookup in N-dimensions. |
| `scipy.spatial.Delaunay` | Delaunay triangulation in N dimensions. |
| `scipy.spatial.Voronoi` | Voronoi diagram generation. |
| `scipy.spatial.ConvexHull` | Compute the convex hull of a set of points. |
| `scipy.spatial.HalfspaceIntersection` | Intersection of halfspaces in N dimensions. |

## 6. Interpolation (`scipy.interpolate`)
*Curve fitting and smoothing.*

| Target | Description |
|---|---|
| `scipy.interpolate.CubicSpline` | Standard cubic spline interpolator (C2 smooth). |
| `scipy.interpolate.RBFInterpolator` | Radial Basis Function interpolation for scattered N-D data. |
| `scipy.interpolate.PchipInterpolator` | Shape-preserving, monotonic cubic interpolation. |
| `scipy.interpolate.Akima1DInterpolator` | Akima "visually pleasing" piecewise cubic interpolator. |

## 7. Sparse Graph Algorithms (`scipy.sparse.csgraph`)
*Highly efficient graph logic on sparse matrices.*

| Target | Description |
|---|---|
| `scipy.sparse.csgraph.dijkstra` | Shortest path for non-negative weights. |
| `scipy.sparse.csgraph.bellman_ford` | Shortest path supporting negative weights and cycle detection. |
| `scipy.sparse.csgraph.floyd_warshall` | All-pairs shortest path algorithm. |
| `scipy.sparse.csgraph.minimum_spanning_tree` | MST using Kruskal's algorithm. |
| `scipy.sparse.csgraph.maximum_flow` | Maximum flow between source and sink. |
| `scipy.sparse.csgraph.reverse_cuthill_mckee` | Bandwidth reduction ordering for sparse matrices. |
