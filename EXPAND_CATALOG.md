# Expand the Primitive Catalog

The architect decomposes algorithms into a Conceptual Dependency Graph (CDG).
Leaf nodes match against a `PrimitiveCatalog` of known algorithmic building
blocks.  Unmatched leaves fall back to generic stubs that downstream rounds
cannot use.  Your job is to expand the catalog so more leaves match.

---

## Choose your approach

| Situation | Approach | Prompt |
|-----------|----------|--------|
| You have **source code** (Python, Rust, C++, Julia) to decompose into atoms automatically | **Ingest** — run `ageom ingest` | `prompts/expand-ingest.md` |
| You want to **hand-author** atoms with contracts, witnesses, and tests in the `ageo-atoms` package | **Source registry** — follow INGESTION.md | `prompts/expand-source.md` |
| You need **domain-general** primitives with no backing implementation (catalog-only entries) | **Built-in** — add to `catalog.py` | `prompts/expand-builtin.md` |

**Most of the time, use Ingest.** It auto-generates atoms, witnesses, CDGs,
and contracts from existing code.  Use the other options only when ingestion
doesn't apply.

### Decision guide

1. **Do you have existing source code that implements the algorithm?**
   - Yes -> Read `prompts/expand-ingest.md` and run `ageom ingest`.
   - No -> Continue to question 2.

2. **Should the primitive have a real implementation with contracts and tests?**
   - Yes -> Read `prompts/expand-source.md` and author atoms in `ageo-atoms`.
   - No -> Read `prompts/expand-builtin.md` and add entries to `catalog.py`.

---

## What already exists

Before adding anything, check for duplicates:

### Built-in primitives (`ageom/architect/catalog.py`)

- `_BAYESIAN_PRIMITIVES` — HMC leapfrog, NUTS u-turn, Kalman gain, sum-product
- `_SIGNAL_FILTER_PRIMITIVES` — IIR/FIR design, stability, coefficient ops
- `_SIGNAL_TRANSFORM_PRIMITIVES` — windowing, FFT, spectral processing, IFFT

### Source registry atoms (`../ageo-atoms/ageoa/`)

```
numpy/    scipy/    algorithms/    biosppy/    e2e_ppg/
kalman_filters/    particle_filters/    mcmc_foundational/
conjugate_priors/    belief_propagation/    quantfin/
institutional_quant_engine/    pulsar/    pulsar_folding/
rust_robotics/    molecular_docking/    mint/    tempo_jl/
pronto/    advancedvi/    bayes_rs/    datadriven/
jax_advi/    pasqal/
```

### Pending targets (`../ageo-atoms/`)

- `INTEREST.md` — curated list of interesting algorithms by source repo
- `PENDING.md` — algorithms identified for future ingestion

---

## De-duplication

The catalog has three dedup layers — understand them to avoid wasted work:

1. **Exact name** — same name or alias -> duplicate.
2. **Embedding similarity** — cosine > 0.85 + matching category and IO
   arity -> merged.  Richer metadata wins, loser becomes alias.
3. **Structural (topo_hash)** — identical CDG subtree topology across
   sources -> second source's leaves skipped.

Write one rich primitive with many aliases rather than multiple thin ones.

---

## Priority gaps

High-value domains not yet well covered (verify against current catalog):

1. **Linear algebra** — LU, Cholesky, SVD, QR, eigendecomposition, least squares
2. **Optimization** — gradient descent, line search, conjugate gradient, L-BFGS
3. **Graph algorithms** — Dijkstra, Bellman-Ford, MST, topological sort, max-flow
4. **Dynamic programming** — LCS, edit distance, knapsack, Viterbi, CTC
5. **Statistical estimators** — MLE, MAP, KDE, bootstrap CI
6. **Interpolation** — polynomial, spline, Chebyshev, RBF

---

## Verify

After any expansion:

```bash
python -m pytest tests/test_catalog.py tests/test_source_catalog.py -v
python -m ageom catalog-gaps --cdg path/to/some_cdg.json
```
