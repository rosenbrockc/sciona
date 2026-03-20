---
name: Expansion family priority analysis
description: Ranked analysis of which skeleton paradigms should get expansion rules next, by richness of established expansion patterns
type: project
---

## Expansion Rule Set Priority by Paradigm

| Paradigm | Expansion richness | Key patterns | Status |
|---|---|---|---|
| Signal Event Rate | Very high | SQI, jump removal, outlier rejection | Done (4 rules, 3 diagnostics) |
| Sequential Filter (Kalman/Particle) | Very high | Observability checks, innovation monitoring, adaptive noise estimation, filter divergence detection | Done (4 rules, 3 diagnostics) |
| MCMC/HMC | Very high | Warmup adaptation, convergence diagnostics (R-hat, ESS), divergence detection, mass matrix tuning | Done (4 rules, 4 diagnostics) |
| Graph Traversal | High | Cycle detection, connectivity pre-check, visited-set compaction, frontier overflow detection | Done (4 rules, 4 diagnostics) |
| Dynamic Programming | High | Sparsification, constraint pruning, space-time tradeoff selection | Not started |
| Signal Filter | Moderate | Stability validation, pre-warping, cascading | Not started |
| Signal Transform | Moderate | Windowing, zero-padding, spectral leakage correction | Not started |
| Graph Optimization | Moderate | Negative cycle detection, heuristic guidance, relaxation ordering | Not started |
| Greedy | Moderate | Feasibility pre-check, solution quality bounds, matroid verification | Not started |
| VI/ADVI | Moderate | ELBO convergence checks, posterior predictive checks | Not started |
| Divide and Conquer | Moderate | Partition balance, base case optimization, result validation | Not started |
| Sorting | Low | Already well-optimized paradigm | Not started |
| Searching | Low | Already well-optimized paradigm | Not started |
| String Matching | Low | Preprocessing is already built in | Not started |
| Geometry | Low | Point degeneracy handling | Not started |
| Number Theory | Low | Minimal expansion needed | Not started |

Total skeleton paradigms: 16. Only SignalEventRateVariantFamily exists as a concrete VariantFamily (plus universal LedgerVariantFamily). Expansion rules are the priority over curated variant swaps for new domains.
