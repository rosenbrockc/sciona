"""Heuristic funnel for matching empirical datasets to symbolic physics atoms.

This package implements a multi-stage deterministic filter that avoids expensive
iterative curve fitting by cascading through increasingly selective stages:

1. Boundary Triage (O(1) per candidate)
2. Exponent Extraction via Log-Space SVD (O(N) once)
3. Invariant Variance (O(N) per surviving candidate)
4. Graph-Directed Constraint Propagation (conditional)
5. Multi-Fidelity RANSAC (final survivors only)
"""
