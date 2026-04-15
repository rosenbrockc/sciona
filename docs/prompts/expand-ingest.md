# Expand Catalog via Ingestion

Use `sciona ingest` to automatically decompose existing source code into
atoms and register them in the catalog.  This is the preferred method when
you have a concrete implementation to work from.

---

## Full reference

Read these files in `../sciona-atoms/` before proceeding:

| File | What it covers |
|------|----------------|
| `INGEST_PROMPT.md` | Complete `sciona ingest` command reference, all languages, recursive decomposition, output validation, verification checklist |
| `INGESTION.md` | Atom authoring spec: signatures, contracts, witnesses, CDG schema, tests |
| `INTEREST.md` | Curated interesting algorithms to ingest, organized by source repo |
| `PENDING.md` | Algorithms already identified for future ingestion |

## Ingestion targets

Source repos are cloned in `../sciona-atoms/third_party/`.  The tables below
summarize high-value targets from `../sciona-atoms/INTEREST.md` (detailed
rationale there).  Targets already ingested have a package in
`../sciona-atoms/sciona/atoms/` — check before starting.

### Biosignal processing

| Repo | Target | Source path | Why |
|------|--------|-------------|-----|
| BioSPPy | `christov_segmenter` | `third_party/BioSPPy/biosppy/signals/ecg.py` | Adaptive threshold state machine for QRS detection |
| BioSPPy | `engzee_segmenter` | `third_party/BioSPPy/biosppy/signals/ecg.py` | Threshold-intersection peak detection |
| BioSPPy | `gamboa_segmenter` | `third_party/BioSPPy/biosppy/signals/ecg.py` | Histogram CDF-based QRS detection |
| BioSPPy | `hamilton_segmenter` | `third_party/BioSPPy/biosppy/signals/ecg.py` | Multi-stage rule-based R-peak detector |
| BioSPPy | `ASI_segmenter` | `third_party/BioSPPy/biosppy/signals/ecg.py` | FSM on double-derivative-squared signals |
| BioSPPy | `ZZ2018` | `third_party/BioSPPy/biosppy/signals/ecg.py` | Fuzzy-logic signal quality assessment |
| BioSPPy | `solnik_onset_detector` | `third_party/BioSPPy/biosppy/signals/emg.py` | Teager-Kaiser energy operator for EMG onset |
| BioSPPy | `homomorphic_filter` | `third_party/BioSPPy/biosppy/signals/pcg.py` | Log-domain filtering for heart sound envelopes |
| E2E-PPG | `gan_rec` | `third_party/E2E-PPG/ppg_reconstruction/` | Peak-aligned GAN noise reconstruction |
| E2E-PPG | `heart_cycle_detection` | `third_party/E2E-PPG/ppg_sqa/` | Adaptive beat segmentation |
| E2E-PPG | `Wrapper_function` | `third_party/E2E-PPG/kazemi_peak_detection/` | Windowed refinement of NN peak predictions |

### Quantitative finance

| Repo | Target | Source path | Why |
|------|--------|-------------|-----|
| Institutional-Quant-Engine | `AlmgrenChriss` | `third_party/Institutional-Quant-Engine/execution_hft/` | Optimal execution trajectories |
| Institutional-Quant-Engine | `AvellanedaStoikov` | `third_party/Institutional-Quant-Engine/execution_hft/` | Dynamic market-making spreads |
| Institutional-Quant-Engine | `PINModel` | `third_party/Institutional-Quant-Engine/execution_hft/` | Informed trading detection via MLE |
| Institutional-Quant-Engine | `HRP` | `third_party/Institutional-Quant-Engine/risk_portfolio/` | Hierarchical risk parity |
| Institutional-Quant-Engine | `HestonModel` | `third_party/Institutional-Quant-Engine/derivatives_pricing/` | Correlated stochastic volatility |
| Institutional-Quant-Engine | `HawkesProcess` | `third_party/Institutional-Quant-Engine/research_math/` | Self-exciting point process |
| quantfin | `charFuncOption` | `third_party/quantfin/` | Characteristic function option pricing (Haskell) |
| quantfin | `MonteCarlo` | `third_party/quantfin/` | Antithetic variate variance reduction (Haskell) |

### Astronomy / physics

| Repo | Target | Source path | Why |
|------|--------|-------------|-----|
| Tempo.jl | `offset_tt2tdb` | `third_party/Tempo.jl/src/` | Newton iteration for TDB via nested sine (Julia) |
| Tempo.jl | `tai2utc` / `utc2tai` | `third_party/Tempo.jl/src/` | Fixed-point iteration for leap-second ambiguity (Julia) |
| Pulsar_Folding | `DM_can` | `third_party/Pulsar_Folding/` | Brute-force dedispersion with SNR maximization |

### Robotics / control

| Repo | Target | Source path | Why |
|------|--------|-------------|-----|
| pronto | `DynamicStanceEstimator` | `third_party/pronto/` | Exhaustive contact config search (C++) |
| pronto | `FootContactClassifier` | `third_party/pronto/` | Multi-modal force/torque classifier (C++) |
| pronto | `EKFSmoothBackwardsPass` | `third_party/pronto/` | RTS smoother with INS association (C++) |
| rust_robotics | `n_joint_arm2_d` | `third_party/rust_robotics/` | Geometric Jacobian inverse kinematics (Rust) |

### Quantum / molecular

| Repo | Target | Source path | Why |
|------|--------|-------------|-----|
| Molecular-Docking | `q_solver` | `third_party/Molecular-Docking/src/solver/` | Adiabatic quantum MWIS via Rydberg blockade |
| Molecular-Docking | `GreedyMapping` | `third_party/Molecular-Docking/src/solver/` | Graph-to-lattice embedding heuristic |

### ML / protein

| Repo | Target | Source path | Why |
|------|--------|-------------|-----|
| mint | `RotaryEmbedding` | `third_party/mint/mint/` | RoPE positional encoding |
| mint | `RowSelfAttention` | `third_party/mint/mint/` | Factorized MSA axial attention |

All paths are relative to `../sciona-atoms/`.

---

## Workflow

### 1. Pick a target

Choose from the tables above, or check `../sciona-atoms/INTEREST.md` for
full details on why each target is interesting.  Check
`../sciona-atoms/PENDING.md` for additional candidates.  Verify the target
hasn't already been ingested by checking `../sciona-atoms/sciona/atoms/`.

### 2. Run the ingester

```bash
# LLM-assisted (default)
sciona ingest path/to/source.py --class ClassName \
    --output ../sciona-atoms/sciona/atoms/mydomain

# Deterministic (no LLM)
sciona ingest path/to/source.py --class ClassName \
    --procedural --output ../sciona-atoms/sciona/atoms/mydomain

# With monitoring for large classes
sciona ingest path/to/source.py --class ClassName \
    --output ../sciona-atoms/sciona/atoms/mydomain --monitor --trace
```

Supported languages: `.py`, `.rs`, `.jl`, `.cpp`, `.h`, `.hpp` (auto-detected).

Output directly into `../sciona-atoms/sciona/atoms/` so atoms are available via the
existing `sources.yml` entry without additional configuration.

### 3. Validate

Both `mypy passed` and `Ghost sim passed` must be `True`.  If either fails,
follow Task 2 in `../sciona-atoms/INGEST_PROMPT.md`.

### 4. Write tests

5 categories per atom (see `../sciona-atoms/INGESTION.md` section 12):
positive path, precondition violations, postcondition verification, edge
cases, upstream parity.

### 5. Export

Ensure all atoms are imported in `__init__.py` and reachable from
`sciona/atoms/__init__.py`.

## Recursive decomposition

For complex classes:

```bash
export SCIONA_INGESTER_MAX_DEPTH=3
sciona ingest path/to/source.py --class LargeClass \
    --output ../sciona-atoms/sciona/atoms/mydomain
```

## Batch ingestion

```bash
for cls in ClassA ClassB ClassC; do
    sciona ingest path/to/source.py --class "$cls" \
        --output "../sciona-atoms/sciona/atoms/${cls,,}"
done
```

## Bulk from curated repos

```bash
sciona skill ingest --source clrs --path /tmp/clrs
sciona skill ingest --source coq100 --path /tmp/coq100
```

Writes `catalog_*.json` to the skill index directory.

## How ingested atoms reach the catalog

`sources.yml` declares `sciona-atoms` pointing at `../sciona-atoms` with
package `sciona.atoms`.  `seed_catalog_from_sources` imports the package
(triggering `@register_atom`), scans `**/*cdg*.json`, and derives
`AlgorithmicPrimitive` entries with full de-duplication.
