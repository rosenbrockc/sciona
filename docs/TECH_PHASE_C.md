# Phase C: Verification & Bounty Settlement -- Implementation Plan

## Overview

Phase C is entirely net-new code. No bounty, settlement, or sandbox infrastructure exists in the codebase today. The existing `ExecutionSandbox` in `/Users/conrad/personal/ageo-matcher/ageom/principal/evaluator.py` runs artifacts locally as subprocesses -- it is the local analog of what the cloud sandbox must do, and its `_parse_trace` / `_compute_loss` patterns are directly reusable.

The plan assumes Phase A (Shapley engine, provenance schema, receipt format) and Phase B (FastAPI API, bounty state machine, PostgreSQL schema, S3 buckets, auth) are already implemented or being developed in parallel. Where Phase C depends on Phase B deliverables, I call out the interface contract.

---

## 1. LLM Pre-Screen Gate

### Purpose
Cheap, non-deterministic filter that rejects obviously invalid or malicious CDG submissions before burning sandbox compute. Especially critical for Heavy/GPU tiers.

### Design

**New file: `ageom/clearinghouse/prescreen.py`**

A stateless function (deployable as Lambda or callable inline from the FastAPI submission endpoint) that takes a CDG submission payload and returns a pass/reject verdict with reasons.

**Three checks, run sequentially (fail-fast):**

1. **Suspicious pattern scan** -- AST-walk the CDG's atom source code looking for:
   - `import socket`, `import urllib`, `import requests`, `import http` (network)
   - `exec(`, `eval(`, `compile(` (dynamic code execution)
   - `subprocess`, `os.system`, `os.popen` (shell escape)
   - `ctypes`, `cffi` (FFI bypass)
   - `importlib`, `__import__` (dynamic import)
   - `open(` with write modes, `pathlib.Path.write_*` (filesystem writes)
   
   Implementation: Parse each atom source with `ast.parse`, walk the tree checking `ast.Import`, `ast.ImportFrom`, `ast.Call` nodes. This is the same pattern used in `graph_store.py`'s `extract_contract_metadata` (lines 102-155) which already walks AST trees for function analysis.

2. **Structural validity** -- Verify:
   - Every atom FQDN in the CDG's `DEPENDS_ON` edges exists in the global registry (PostgreSQL lookup)
   - I/O type signatures match the bounty's `target_spec` from `ageom.yml`
   - DAG acyclicity (reuse `ageom/synthesizer/toposort.py` which already implements topological sort)

3. **Resource estimation** -- Flag CDGs with:
   - More than N nodes (configurable, default 100)
   - Maximum DAG depth exceeding M (configurable, default 20)
   - Known expensive atom combinations (e.g., multiple GPU-requiring atoms on a Standard tier bounty)

**Return type:**

```python
@dataclass(frozen=True)
class PreScreenResult:
    passed: bool
    rejection_reasons: list[str]  # empty if passed
    estimated_tier: str  # "standard" | "heavy" | "gpu"
    estimated_memory_gb: float
    estimated_runtime_minutes: float
```

**Deployment:** Initially inline in the FastAPI endpoint (`POST /bounties/{bounty_id}/submissions`). Extract to Lambda only if latency becomes a concern (pre-screen should complete in under 2 seconds for any reasonable CDG).

---

## 2. Tiered Sandbox Execution

### Architecture

**New file: `ageom/clearinghouse/sandbox.py`**

Abstract `SandboxExecutor` protocol with three implementations:

```python
class SandboxExecutor(Protocol):
    async def execute(self, payload: SandboxPayload) -> SandboxResult: ...
```

**`SandboxPayload`** bundles:
- CDG source code (all atoms, pinned by content hash)
- `ageom.yml` (data schema)
- Dataset split reference (S3 key for blind or public split)
- `config.yml` parameters
- Bounty ID (for receipt binding)
- Python lockfile hash (pinned runtime)

**`SandboxResult`** contains:
- `metric_values: dict[str, float]` (all declared objectives)
- `output_hash: str` (SHA-256 of full output)
- `execution_time_s: float`
- `peak_memory_bytes: int`
- `determinism_check: bool` (did a second run produce identical output_hash)
- `trace: dict[str, NodeTelemetry]` (reuse existing `NodeTelemetry` model from `ageom/principal/models.py`)

### Standard Tier (Lambda)

**New file: `ageom/clearinghouse/sandbox_lambda.py`**

- Package the CDG + atoms + pinned dependencies as a Lambda container image (built from a base image with pinned Python + numpy + scipy + pandas)
- Lambda configuration: 15 min timeout, 4-10 GB memory (set based on pre-screen estimate)
- VPC with no internet egress, no NAT gateway
- `/tmp` is read-only (Lambda natively restricts filesystem; mount data from S3 via pre-signed URL read at init)
- Environment variables: `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, numpy/torch seed pinning

The implementation pattern mirrors the existing `ExecutionSandbox.evaluate()` method in `ageom/principal/evaluator.py` (lines 41-123) -- construct a command, run it with a timeout, parse the trace output. The cloud version replaces `asyncio.create_subprocess_exec` with a Lambda invocation via boto3.

**Lambda invocation flow:**
1. FastAPI endpoint calls `sandbox_lambda.execute(payload)`
2. Function uploads payload to `s3://datasets/{bounty_id}/sandbox_payloads/{submission_id}.tar.gz`
3. Invokes Lambda synchronously via `lambda_client.invoke()` with the S3 reference
4. Lambda handler: download payload, download blind split from S3, execute CDG, upload results to S3
5. Function reads results from S3, constructs `SandboxResult`

### Heavy Tier (SageMaker Processing Job)

**New file: `ageom/clearinghouse/sandbox_sagemaker.py`**

- SageMaker Processing Job with a custom container (same base as Lambda but larger)
- Configuration: 2 hr max runtime, `ml.m5.4xlarge` (16 vCPU, 64 GB) or `ml.m5.16xlarge` (64 vCPU, 256 GB) based on pre-screen estimate
- Network isolation: `NetworkConfig.EnableNetworkIsolation = True`
- Input: S3 URI for payload + blind split. Output: S3 URI for results
- Polling: SageMaker jobs are async. The FastAPI endpoint starts the job and returns a job ID. A separate endpoint (`GET /submissions/{id}/status`) polls SageMaker

### GPU Tier (SageMaker/EC2)

**New file: `ageom/clearinghouse/sandbox_gpu.py`**

- SageMaker Processing Job on `ml.g5.xlarge` (1x A10G GPU, 24 GB VRAM, 16 GB RAM) or `ml.p3.2xlarge` (1x V100)
- 4 hr max runtime
- Same isolation as Heavy tier
- Additional: `CUDA_DETERMINISTIC=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` for GPU determinism

### Determinism Verification

All tiers run the CDG twice on a small random subset (10 samples from the blind split). If `output_hash` differs between runs, the submission is flagged as non-deterministic and rejected. This enforces the Sandbox Determinism invariant from TECH_GAP Section 1.

### CDG Packaging

**New file: `ageom/clearinghouse/packager.py`**

Responsible for assembling the sandbox payload:
1. Fetch all atoms by content hash from S3 (`atoms/` prefix)
2. Resolve the dependency DAG (reuse `synthesizer/toposort.py`)
3. Generate a single executable Python module that chains the atoms (reuse the pattern from `synthesizer/compiler.py` which already compiles CDGs into executable artifacts)
4. Include the pinned `requirements.txt` lockfile
5. Package as a tar.gz, upload to S3

---

## 3. Blind Data Splitting Pipeline

### Design

**New file: `ageom/clearinghouse/data_splitter.py`**

The data splitting pipeline runs once when a bounty transitions from `draft` to `open` (bounty funded). It is triggered by the FastAPI endpoint that processes bounty funding.

**Pipeline steps:**

1. **Parse ageom.yml** -- Use the existing `ageom/principal/datasets/_parser.py` and `core.py` infrastructure to understand the dataset structure. The `DataSetCollection` class and `get_attr_groups` function already parse `ageom.yml`-style configs.

2. **LLM stratification analysis** -- Send a structured prompt to the LLM (via `ageom/llm_router.py` which already handles multi-provider routing) that:
   - Receives: group structure, property names/types, `meta.user`/`meta.folder` fields, sample statistics (mean, std, count per group)
   - Returns: a JSON object specifying the stratification axis (e.g., `"stratify_by": "meta.user"`) and any additional split constraints
   - The prompt is deterministic given the same `ageom.yml` (temperature=0, seed fixed)

   **Prompt template** (new file: `ageom/clearinghouse/prompts/split_strategy.py`):
   ```
   You are analyzing a dataset for stratified train/test splitting.
   
   Dataset schema (ageom.yml):
   {ageom_yml_content}
   
   Dataset statistics:
   {statistics_json}
   
   Identify the best stratification axis for splitting this dataset into
   a 20% public / 80% blind partition. The split must ensure:
   1. No data leakage between partitions (e.g., same subject in both)
   2. Representative distribution of key characteristics
   3. Deterministic assignment from the axis values
   
   Return JSON: {"stratify_by": "<field>", "method": "hash|cluster", "reason": "..."}
   ```

3. **Deterministic split assignment** -- Hash each data unit's stratification key with a bounty-specific salt:
   ```python
   partition = "public" if int(sha256(f"{bounty_id}:{unit_key}".encode()).hexdigest(), 16) % 5 == 0 else "blind"
   ```
   This gives exactly 20/80 split, is deterministic, and the salt prevents replay across bounties (per TECH_GAP 4.17).

4. **Validation** -- Reject datasets that:
   - Have fewer than 50 samples (TECH_GAP 4.11)
   - Have all-constant columns
   - Have extreme class imbalance without declared handling
   - Result in fewer than 10 samples in the public split

5. **Storage** -- Write split assignments to PostgreSQL (`dataset_splits` table) and upload the actual split data files to S3:
   - `s3://datasets/{bounty_id}/public/` -- shared with Architects
   - `s3://datasets/{bounty_id}/blind/` -- accessible only by sandbox IAM role

### Split Hash

Generate a `split_hash` for the bounty:
```python
split_hash = sha256(canonical_json(sorted_split_assignments) + bounty_salt).hexdigest()
```
This is included in execution receipts to bind them to a specific bounty's data split.

---

## 4. Verification Budget Tracking

### PostgreSQL Schema Additions

These tables extend the Phase B bounty schema. The schema follows the pattern established in TECH_GAP Section 3.4.

```sql
-- Verification budget and slot tracking
CREATE TABLE verification_budgets (
    bounty_id       UUID PRIMARY KEY REFERENCES bounties(id),
    tier            TEXT NOT NULL CHECK (tier IN ('standard', 'heavy', 'gpu')),
    total_slots     INT NOT NULL,           -- from tier default + purchased
    used_slots      INT NOT NULL DEFAULT 0,
    cost_per_extra  NUMERIC(10,2) NOT NULL, -- $10/$50/$200 per tier
    overhead_deposit NUMERIC(10,2) NOT NULL DEFAULT 0,
    overhead_used   NUMERIC(10,2) NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Individual verification runs
CREATE TABLE verification_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bounty_id       UUID NOT NULL REFERENCES bounties(id),
    submission_id   UUID NOT NULL REFERENCES submissions(id),
    split_type      TEXT NOT NULL CHECK (split_type IN ('public', 'blind')),
    status          TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    metric_values   JSONB,          -- {"rmse": 8.31, "latency_ms": 42.0}
    output_hash     TEXT,
    execution_time_s FLOAT,
    peak_memory_bytes BIGINT,
    is_deterministic BOOLEAN,
    sandbox_job_id  TEXT,           -- Lambda request ID or SageMaker job ARN
    slot_consumed   BOOLEAN NOT NULL DEFAULT false,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Best-to-date tracking per bounty
CREATE TABLE bounty_best_scores (
    bounty_id       UUID NOT NULL REFERENCES bounties(id),
    metric_name     TEXT NOT NULL,
    best_value      FLOAT NOT NULL,
    best_submission_id UUID REFERENCES submissions(id),
    is_baseline     BOOLEAN NOT NULL DEFAULT false,  -- true for Phase 1 baseline
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bounty_id, metric_name)
);

-- Principal target adjustments (between verifications)
CREATE TABLE principal_targets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bounty_id       UUID NOT NULL REFERENCES bounties(id),
    metric_name     TEXT NOT NULL,
    target_value    FLOAT NOT NULL,
    set_by          UUID NOT NULL REFERENCES users(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Execution receipts (from Architects)
CREATE TABLE execution_receipts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id   UUID NOT NULL REFERENCES submissions(id),
    bounty_id       UUID NOT NULL,
    cdg_hash        TEXT NOT NULL,
    atom_versions   JSONB NOT NULL,    -- {fqdn: content_hash}
    split_hash      TEXT NOT NULL,
    output_hash     TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    FLOAT NOT NULL,
    ageom_version   TEXT NOT NULL,
    ssh_signature   TEXT NOT NULL,
    ssh_public_key  TEXT NOT NULL,
    verified        BOOLEAN DEFAULT false,
    receipt_timestamp TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### Verification Flow Implementation

**New file: `ageom/clearinghouse/verification.py`**

The `VerificationEngine` class orchestrates the four-phase flow from TECH_GAP 4.14:

**Phase 1 -- Baseline establishment** (on bounty open):
1. Retrieve the Principal's best CDG from the Flare's `best_structure` in `config.yml`
2. Package and execute against the blind split
3. Record the result in `bounty_best_scores` with `is_baseline = true`
4. Consume one verification slot

**Phase 2 -- Community submission evaluation:**
1. Verify receipt signature (`ssh-keygen -Y verify` against GitHub public keys stored at registration)
2. Re-run CDG on public split (cheap validation -- does the claimed output_hash match?)
3. Compare claimed metric against current best-to-date and any `principal_targets`
4. If improvement exceeds the `minimum_improvement_threshold` from `config.yml` AND verification slots remain: trigger blind-split sandbox execution
5. If blind-split result confirms improvement: update `bounty_best_scores`, notify Principal

**Improvement threshold logic:**

```python
def should_trigger_verification(
    claimed_value: float,
    best_to_date: float,
    threshold_pct: float,  # from config.yml, default 5.0
    direction: str,  # "minimize" or "maximize"
) -> bool:
    if direction == "minimize":
        improvement = (best_to_date - claimed_value) / abs(best_to_date)
    else:
        improvement = (claimed_value - best_to_date) / abs(best_to_date)
    return improvement >= threshold_pct / 100.0
```

---

## 5. Bounty Settlement Engine

### Design

**New file: `ageom/clearinghouse/settlement.py`**

Triggered when a bounty reaches its deadline. The settlement engine must:

1. **Determine winner(s):**
   - Single-objective: CDG with best verified blind-split metric
   - Multi-objective (Pareto): Compute Pareto front, select up to `pareto_winners` CDGs

2. **Compute payouts using exact arithmetic** (TECH_GAP Invariant: Payout Conservation):

```python
from fractions import Fraction
from decimal import Decimal

def compute_payout_split(escrow_amount: Decimal, winning_cdgs: list[WinningCDG]) -> PayoutPlan:
    foundation_share = Fraction(5, 100)
    architect_share = Fraction(65, 100)
    originator_share = Fraction(30, 100)
    
    escrow_frac = Fraction(escrow_amount)
    
    # For Pareto bounties: split architect share by principal-defined weights
    # For single-objective: one winner gets full architect share
    
    assert foundation_share + architect_share + originator_share == Fraction(1)
    # ... distribute originator_share via Shapley values from Phase A
```

3. **Shapley integration** -- Call the Phase A Shapley engine to compute per-atom attribution for each winning CDG. Sum across winners for multi-objective bounties. Validate `sum(shapley_values) == Fraction(1)` per TECH_GAP.

4. **Stripe Connect payouts:**

**New file: `ageom/clearinghouse/stripe_payout.py`**

```python
class StripePayout:
    def __init__(self, stripe_secret_key: str):
        self._stripe = stripe
        stripe.api_key = stripe_secret_key
    
    async def execute_payout(self, plan: PayoutPlan) -> PayoutResult:
        transfers = []
        for recipient in plan.recipients:
            # Stripe Transfer to Connected Account
            transfer = stripe.Transfer.create(
                amount=int(recipient.amount_cents),
                currency="usd",
                destination=recipient.stripe_account_id,
                transfer_group=f"bounty_{plan.bounty_id}",
                metadata={
                    "bounty_id": str(plan.bounty_id),
                    "role": recipient.role,  # "architect" | "originator" | "foundation"
                    "cdg_hash": recipient.cdg_hash,
                },
            )
            transfers.append(transfer)
        return PayoutResult(transfers=transfers, bounty_id=plan.bounty_id)
```

5. **Cancellation handling** (TECH_GAP 4.13):
   - Before any submissions: 10% fee to Foundation, 90% refund
   - After submissions received: 25% fee, 75% refund
   - After deadline: no cancellation

6. **Expiry handling** (TECH_GAP 4.15):
   - Refund escrow minus compute costs (baseline run + verification slots consumed)
   - `overhead_used` from `verification_budgets` tracks actual compute costs

### Settlement State Machine

```
[deadline_reached] 
  -> compute_winners()
  -> compute_shapley_for_each_winner()
  -> compute_payout_plan()
  -> verify_payout_conservation()  # assert sum == escrow
  -> execute_stripe_payouts()
  -> update_bounty_status("settled")
  -> record_settlement_in_provenance_graph()
```

---

## 6. Multi-Objective / Pareto Bounties

### Design

**New file: `ageom/clearinghouse/pareto.py`**

Pareto front computation for multi-objective bounties:

```python
def compute_pareto_front(
    submissions: list[VerifiedSubmission],
    objectives: list[ObjectiveSpec],  # from config.yml
    max_winners: int,
) -> list[VerifiedSubmission]:
    """Return up to max_winners Pareto-optimal submissions."""
    # Standard Pareto dominance: submission A dominates B if A is at least
    # as good on all objectives and strictly better on at least one.
    # Flip sign for maximize objectives.
    ...
```

**Payout splitting for Pareto winners:**
- Architect share (65%) divided among winners by `weight` from `config.yml`
- Originator share (30%) computed per winning CDG via Shapley, then summed
- Foundation share (5%) is a single payment

**config.yml extension:**

```yaml
objectives:
  - metric: rmse
    direction: minimize
    weight: 0.7
  - metric: latency_ms
    direction: minimize
    weight: 0.3
pareto_winners: 3
```

The `weight` field serves double duty: (a) as the payout weight when splitting among Pareto winners, and (b) as the scalarization weight if the Principal wants a single-scalar comparison for verification triggering.

---

## 7. API Endpoints

All endpoints live in the FastAPI service (Phase B infrastructure). Phase C adds these routes:

### Submission

- `POST /bounties/{bounty_id}/submissions` -- Submit a CDG with execution receipt
  - Body: `{cdg: {...}, receipt: {...}, receipt_signature: "..."}`
  - Pre-screen gate runs inline
  - Returns: `{submission_id, prescreen_result, verification_scheduled: bool}`

### Verification Status

- `GET /submissions/{submission_id}/status` -- Poll verification progress
  - Returns: `{status, metric_values, position_on_leaderboard}`

- `GET /bounties/{bounty_id}/leaderboard` -- Current ranking of verified submissions
  - Returns: `[{submission_id, architect_id, metric_values, verified_at}]`

### Principal Actions

- `PUT /bounties/{bounty_id}/target` -- Set new improvement target
  - Body: `{metric_name, target_value}`
  - Auth: must be bounty creator

- `POST /bounties/{bounty_id}/data-round` -- Request new data round
  - Body: `{additional_data_s3_key?, restratify: bool}`
  - Triggers re-split and resets public data

- `POST /bounties/{bounty_id}/cancel` -- Cancel bounty
  - Enforces fee schedule from TECH_GAP 4.13

### Settlement

- `POST /bounties/{bounty_id}/settle` -- Trigger settlement (called by scheduled job at deadline)
  - Internal endpoint, not publicly exposed
  - Returns: `{payout_plan, winners}`

- `GET /bounties/{bounty_id}/settlement` -- Retrieve settlement details
  - Returns: `{winners, payout_breakdown, shapley_allocations}`

---

## 8. Test Strategy

### Unit Tests

Follow the existing pattern in `/Users/conrad/personal/ageo-matcher/tests/` (pytest + pytest-asyncio, `asyncio_mode = "auto"`).

- **`test_prescreen.py`** -- Test each check independently:
  - Feed known-malicious CDGs (network calls, exec/eval) and verify rejection
  - Feed valid CDGs and verify pass
  - Test resource estimation accuracy

- **`test_data_splitter.py`** -- Test split determinism:
  - Same `ageom.yml` + same data + same bounty salt = same split
  - Different bounty salt = different split
  - Minimum size enforcement
  - Mock the LLM call, return a fixed stratification axis

- **`test_verification.py`** -- Test the improvement threshold logic, slot accounting, receipt verification

- **`test_settlement.py`** -- Test payout conservation:
  - `Fraction`-based arithmetic: assert `sum(all_payouts) == escrow`
  - Cancellation fee calculations at each timing tier
  - Edge case: no valid submissions at deadline (expiry refund)

- **`test_pareto.py`** -- Test Pareto front computation:
  - Known Pareto fronts with 2-3 objectives
  - Edge case: all submissions are Pareto-dominated by one
  - Edge case: all submissions are Pareto-optimal (cap at `pareto_winners`)

### Integration Tests

- **`test_sandbox_integration.py`** -- Mock Lambda/SageMaker with `moto` or `localstack`:
  - Submit a simple CDG, verify execution and trace parsing
  - Verify determinism check (same CDG produces identical output_hash)
  - Verify timeout handling

- **`test_settlement_e2e.py`** -- Full settlement flow with mock Stripe:
  - Create bounty, submit CDGs, reach deadline, settle, verify payouts
  - Use `stripe-mock` for Stripe API simulation

### Important: Test Speed

Per the user's feedback, the test suite must stay under 1 minute. All cloud service calls (Lambda, SageMaker, Stripe, LLM) must be mocked. The existing `ExecutionSandbox` test pattern (subprocess-based with short timeouts) should be followed for any local execution tests.

---

## 9. File Structure Summary

New package: `ageom/clearinghouse/`

```
ageom/clearinghouse/
    __init__.py
    prescreen.py          # LLM pre-screen gate
    sandbox.py            # SandboxExecutor protocol + SandboxPayload/Result models
    sandbox_lambda.py     # Standard tier Lambda executor
    sandbox_sagemaker.py  # Heavy/GPU tier SageMaker executor
    sandbox_gpu.py        # GPU-specific SageMaker configuration
    packager.py           # CDG packaging for sandbox execution
    data_splitter.py      # Blind data splitting pipeline
    verification.py       # Verification budget tracking + flow orchestration
    settlement.py         # Payout computation + Shapley integration
    pareto.py             # Multi-objective Pareto front computation
    stripe_payout.py      # Stripe Connect payout execution
    models.py             # Pydantic models for all Phase C data types
    prompts/
        __init__.py
        split_strategy.py # LLM prompt for data stratification
```

New SQL migration: `migrations/003_phase_c_verification.sql`

New tests:
```
tests/
    test_prescreen.py
    test_data_splitter.py
    test_verification.py
    test_settlement.py
    test_pareto.py
    test_sandbox_integration.py
```

---

## 10. Dependencies and Sequencing

| Step | Deliverable | Depends On | Estimated Effort |
|------|------------|------------|-----------------|
| C1 | `prescreen.py` + tests | Phase B: submission endpoint exists | 2 days |
| C2 | `data_splitter.py` + tests | Phase B: bounty creation endpoint, S3 buckets | 3 days |
| C3 | `packager.py` | Phase A: atom content-hash storage in S3 | 2 days |
| C4 | `sandbox_lambda.py` + Lambda infra (Terraform/CDK) | C1, C3 | 4 days |
| C5 | `sandbox_sagemaker.py` + `sandbox_gpu.py` | C4 (same pattern, different backend) | 3 days |
| C6 | `verification.py` + receipt verification | C4, Phase A: receipt format | 3 days |
| C7 | `settlement.py` + `stripe_payout.py` | C6, Phase A: Shapley engine | 4 days |
| C8 | `pareto.py` | C7 | 2 days |
| C9 | API endpoints wiring | All above + Phase B FastAPI | 2 days |
| C10 | Integration tests | All above | 3 days |

**Total: ~4 weeks** (aligns with TECH_GAP Section 4 estimate of 2-3 weeks for sandbox + 1 week for receipt verification)

---

## 11. Risks and Open Questions

1. **Lambda cold start for Standard tier**: Lambda container images can have 5-10 second cold starts with large dependency sets. Mitigate with provisioned concurrency for active bounties, or switch to Lambda SnapStart if Python support arrives.

2. **Floating-point determinism across architectures**: Lambda runs on x86 (Intel/AMD). SageMaker instances may be ARM (Graviton). IEEE 754 behavior can differ subtly. Decision: pin all tiers to x86 (`amd64` container images) for bit-identical guarantees.

3. **SageMaker Processing Job startup time**: SageMaker jobs can take 3-5 minutes to spin up the container. This is acceptable for Heavy/GPU tiers (hours of runtime) but should be communicated to Architects in the UI.

4. **LLM splitting non-determinism**: Even at temperature=0, LLM outputs can vary across model versions. Mitigate by pinning the model version in the split prompt and storing the split assignment deterministically. The LLM only influences the *strategy* (which axis to stratify on); the actual partition assignment uses a deterministic hash.

5. **Stripe Connect onboarding friction**: Recipients must complete Stripe Connect onboarding before receiving payouts. Payouts should be held in platform escrow until onboarding completes, with a 90-day expiry (after which unclaimed funds return to the Foundation pool).

6. **Receipt forgery via public split overfitting**: An Architect could overfit to the public split, claim excellent metrics via a forged receipt, and trigger a verification slot that fails on the blind split. This wastes a verification slot. Mitigate: rate-limit submissions per Architect per bounty (e.g., 3 submissions max), and charge a small submission fee for Heavy/GPU bounties.

7. **Open question -- blind submission window**: TECH_GAP 4.5 describes a hash-commitment / reveal protocol. The current plan implements continuous submission (submit and verify as they come in). The hash-commitment approach prevents front-running but adds complexity (commit phase, reveal phase, batch verification). Recommend deferring the commit/reveal protocol to v2 and implementing continuous submission for v1.

---

### Critical Files for Implementation

- `/Users/conrad/personal/ageo-matcher/ageom/principal/evaluator.py` -- Pattern to follow: the existing `ExecutionSandbox` class demonstrates how to execute artifacts, parse traces, and compute losses. The cloud sandbox mirrors this structure.
- `/Users/conrad/personal/ageo-matcher/ageom/graph_store.py` -- AST walking patterns (lines 39-155) for the pre-screen gate, and Memgraph integration patterns for recording settlement results in the provenance graph.
- `/Users/conrad/personal/ageo-matcher/ageom/principal/datasets/core.py` -- Dataset parsing infrastructure that the data splitter will reuse for understanding `ageom.yml` structure.
- `/Users/conrad/personal/ageo-matcher/docs/TECH_GAP.md` -- The authoritative spec: sections 4.7 (sandbox tiers), 4.10-4.11 (submission format, blind split), 4.13-4.17 (cancellation, verification budget, expiry, Pareto, principal params, receipt replay).
- `/Users/conrad/personal/ageo-matcher/ageom/principal/atom_ledger.py` -- Pattern for `fractions.Fraction`-style exact arithmetic and the UCB1 bandit as a model for how the verification budget's "minimum improvement threshold" trigger should work.
