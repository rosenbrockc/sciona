# Algorithmic Commons Foundation — Technical Gap Analysis

First-pass architectural review of the six components described in
`algo_commons.pdf` (Part 2), evaluated against what sciona already
implements and what must be built.

---

## 1. System Invariants

These properties must hold globally across all components.

| Invariant | Description |
|-----------|-------------|
| **Payout conservation** | For every settled bounty: `platform_fee + architect_share + originator_yield = escrow_amount`. Concretely 5% + 65% + 30% = 100%. Must be enforced in the Settlement Engine with exact arithmetic, never floats. |
| **Shapley budget balance** | The Originator's Yield (30%) is distributed via Shapley values across atom authors. The sum of all Shapley allocations must equal exactly the 30% pool. Use `fractions.Fraction` or fixed-point decimals. |
| **DAG acyclicity** | The `DEPENDS_ON` subgraph in the Provenance Graph must be a DAG. Cycles in atom dependencies make Shapley computation undefined. Enforce on every registry merge. |
| **Fingerprint uniqueness** | Every distinct atom (modulo alpha-renaming) must map to a unique hash. Collisions in the plagiarism index corrupt attribution. |
| **Sandbox determinism** | Given the same CDG and the same inputs, the Verification Sandbox must produce bit-identical outputs across runs. Non-determinism breaks the trust model for payouts. |
| **Flare privacy** | A Dead-End Flare must never leak the Principal's private data, internal gradient history, or `sources.yml` contents. Only the frozen state spec is published. |

---

## 2. Component-by-Component Analysis

### 2.1 sciona Client — Dead-End Flare Protocol

**What exists:** `sciona optimize` in `sciona/commands/optimize_cmds.py`, Principal
HPO framework in `sciona/principal/hpo.py`, variant mutation, atom ledger with
UCB1 bandit ranking.

**What's missing:** The Dead-End Flare — the mechanism that bundles frozen
optimization state when all families/variants are exhausted and prompts the user
to post a bounty.

**Minimal sufficient state bundle:**

```
FlarePayload:
  problem_id:        str           # unique identifier
  target_spec:       IOSpec[]      # desired inputs/outputs with types
  constraints:       str           # domain constraints (e.g. "real-time", "< 10ms")
  concept_types:     ConceptType[] # algorithmic categories attempted
  atoms_tried:       str[]         # list of atom fqdns that were exhausted
  best_metric_value: float         # best-to-date value of the optimised metric
  metric_name:       str           # which metric (e.g. "loss", "latency_ms")
  max_graph_nodes:   int           # node count of largest CDG variant tried
  max_graph_edges:   int           # edge count of largest CDG variant tried
  domain_tags:       str[]         # e.g. "crystallography", "signal-processing"
```

**What must NOT be included:**

- Raw gradient scores from `AtomObservation` (atom_ledger.py:32) — reveals
  optimization trajectory
- Private `sources.yml` contents or file paths
- Internal embedding vectors or FAISS indices
- Trial-level performance data (slot signatures, UCB scores)

**Open question:** How much of the loss landscape to reveal. Even `best_loss`
tells Architects something about difficulty. Consider publishing only a
categorical difficulty tier (easy/medium/hard) derived from the loss.

---

### 2.2 Global Registry — Catalog API

**What exists:** `PrimitiveCatalog` (in-memory, local), `SkillIndex` with FAISS
embeddings, `atom_similarity.fingerprint_function()` for AST hashing.

**What's missing:** A hosted service with a REST/gRPC API, versioning, domain
tags, and merge-time plagiarism checks.

**Fingerprint collision risk (CRITICAL):**

`fingerprint_function()` currently truncates SHA-256 to 16 hex chars (64 bits).
Birthday collision probability reaches 50% at ~2^32 entries (~4 billion atoms).
While 500 atoms are safe today, a global registry must plan for scale.

**Recommendation:** Use full SHA-256 (64 hex chars) for the registry index.
Keep the 16-char truncation only for display/logging. This is a one-line change
in `atom_similarity.py`.

**Plagiarism detection gaps:**

| Clone Type | Detection | Coverage |
|------------|-----------|----------|
| Type-1 (exact copy) | AST fingerprint | Complete |
| Type-2 (renamed variables) | AST fingerprint (alpha-renaming) | Complete |
| Type-3 (near-miss, minor edits) | Embedding similarity via SkillIndex | Partial — needs threshold tuning |
| Type-4 (semantic, different structure) | Not covered by fingerprint | Gap — requires behavioral equivalence testing |

**Mitigation for Type-4:** Use the Fuzzing Cluster (component 4) to run
behavioral equivalence: if two atoms produce identical outputs on a large random
input corpus, flag for human review regardless of structural difference.

---

### 2.3 Provenance Graph & Shapley Engine

**What exists:** Memgraph-backed `graph_store.py` with `_topo_hash()`, 3-layer
retrieval. `atom_ledger.py` tracks per-slot atom performance.

**What's missing:** The Shapley Value computation, the Settlement Engine, and
the full provenance data model.

#### Proposed Provenance Graph Schema

**Node types:**

| Node | Key Properties |
|------|---------------|
| `Atom` | `fqdn`, `fingerprint`, `version`, `domain_tags[]`, `created_at` |
| `CDG` | `cdg_id`, `topo_hash`, `created_at`, `verified: bool` |
| `Bounty` | `bounty_id`, `escrow_amount`, `status` (open/submitted/verified/settled/expired), `created_at`, `deadline` |
| `Principal` | `principal_id`, `org` |
| `Architect` | `architect_id` |
| `Originator` | `originator_id`, `affiliation` |

**Edge types:**

| Edge | From → To | Properties |
|------|-----------|------------|
| `DEPENDS_ON` | CDG → Atom | `depth` (distance from root) |
| `AUTHORED_BY` | Atom → Originator | `contribution_share` (for multi-author atoms) |
| `SOLVED_BY` | Bounty → CDG | `submitted_at`, `verified_at` |
| `POSTED_BY` | Bounty → Principal | `funded_at` |
| `BUILT_BY` | CDG → Architect | |
| `DERIVES_FROM` | Atom → Atom | version lineage |

**Key constraint:** Once a bounty reaches `settled` status, its subgraph
(SOLVED_BY → CDG → DEPENDS_ON → Atoms → AUTHORED_BY → Originators) is
**immutable**. No retroactive edits to attribution.

#### Shapley Value Tractability

**Naive complexity:** O(2^n) for n atoms — enumerate all coalitions. Intractable
for CDGs with more than ~25 atoms.

**DAG shortcut:** For DAG-structured dependency games where the characteristic
function is binary coverage (a coalition "works" if and only if it contains all
atoms in the transitive closure), the Shapley value for each atom can be
computed in **O(n * m)** where n = atoms, m = edges.

**Algorithm sketch:**

```
For each atom a in the CDG's dependency closure:
  shapley[a] = sum over all topological orderings where a is pivotal
             = 1 / (number of atoms at same "depth tier")
             (simplified; real formula uses path-counting on the DAG)
```

**Diamond dependency handling:** Atom A used by both B and C (which are both in
the CDG). A's Shapley value correctly accounts for the fact that removing A
breaks both paths. The DAG algorithm handles this natively — it's the naive
enumeration that would double-count.

**Implementation recommendation:**

- Use `fractions.Fraction` for all intermediate computations
- Convert to `decimal.Decimal` only at payout time
- Store exact rational Shapley values in the Provenance Graph
- Validate `sum(shapley_values) == Fraction(1)` as a post-condition

---

### 2.4 Asymmetric Fuzzing Cluster

**What exists:** Hyperparameter optimization via `sciona/principal/hpo.py`,
parameter smoothing in variant mutation.

**What's missing:** A serverless compute pool that runs on public atoms only.

**Architecture requirements:**

1. **Job queue**: Atom fqdn → fuzz job. Triggered on every registry merge.
2. **Fuzz strategies**: Property-based testing (hypothesis-style), boundary
   value analysis, random input generation typed by `IOSpec`.
3. **Parameter smoothing**: Run the existing HPO framework on the atom's
   `tunable_params` from the manifest. Store optimised defaults back to the
   registry.
4. **Asymmetry enforcement**: The queue must reject jobs for atoms not in the
   public registry. Private `sources.yml` atoms never enter this pipeline.

**Compute budget:** The 5% scrape funds this. At early scale (few bounties),
the cluster should be serverless/spot-instance to keep costs near zero when
idle.

---

### 2.5 Bounty Clearinghouse & Verification Sandbox

**What exists:** Nothing — this is entirely net-new.

#### Bounty Lifecycle State Machine

```
                 fund
  [draft] ──────────────► [open]
                             │
                    submit   │  expire (deadline)
                    ┌────────┤──────────────────► [expired] ──► refund
                    ▼        │
               [submitted]   │
                    │        │
           verify   │        │
                    ▼        │
               [verified]    │
                    │        │
           settle   │        │
                    ▼        │
               [settled] ────┘
                    │
              payout split
              5% → Foundation
             65% → Architect
             30% → Originators (via Shapley)
```

**Concurrent submission race condition:**

Without a submission window, first-to-submit wins. This is unfair and
vulnerable to front-running (an adversary monitors the mempool/queue and
submits a copied solution faster).

**Proposed fix — Blind Submission Window:**

1. Bounty enters `open` state with a deadline (e.g., 7 days).
2. During the window, Architects submit **sealed** solutions (hash commitment).
3. After the deadline, a reveal phase opens (e.g., 24 hours).
4. All revealed CDGs are verified in batch.
5. Best-performing CDG wins (lowest loss on blind test data).
6. Ties broken by submission timestamp of the hash commitment.

This prevents front-running and encourages quality over speed.

#### Verification Sandbox Formal Properties

| Property | Requirement |
|----------|-------------|
| **Soundness** | If the sandbox accepts a CDG, it genuinely solves the problem (produces outputs matching the target spec within tolerance). |
| **Isolation** | Submitted CDGs cannot exfiltrate test data, make network calls, access the filesystem, or side-channel the Principal's constraints. |
| **Determinism** | Same CDG + same inputs = bit-identical outputs across runs. No floating-point non-determinism, no random seeds. |
| **Resource bounds** | CDGs must terminate within a time/memory budget. Prevents DoS via infinite loops or memory bombs. |

**Threat model:**

| Threat | Mitigation |
|--------|------------|
| CDG overfits to blind test set | Multi-round verification with held-out data; require generalisation margin |
| CDG exfiltrates test data via side channels | Run in isolated microVM (Firecracker/gVisor), no network, no filesystem writes |
| CDG contains malicious code | Static analysis + sandboxed execution; CDG must be composed only of registered atoms |
| Architect submits someone else's CDG | Plagiarism check via AST fingerprint + embedding similarity against all prior submissions |
| Sybil attack (same person as Principal and Architect) | KYC/identity verification; anomaly detection on bounty patterns |

---

### 2.6 Metrics & ESG Dashboard

**What exists:** Nothing — net-new frontend.

**Data sources (all from the Provenance Graph):**

- **Algorithmic Impact Factor**: Count of bounties where an Originator's atoms
  were used, weighted by bounty value. Analogous to h-index.
- **Cross-Disciplinary Multipliers**: Bonus for atoms used outside their
  original domain tag (e.g., a crystallography atom used in logistics).
- **Auto-BibTeX**: Generate `.bib` entries from `AUTHORED_BY` edges with
  version, fqdn, and registry DOI.
- **Compute Preserved**: Estimate LLM tokens/energy saved by using a
  deterministic CDG instead of agentic LLM loops. Metric:
  `(estimated_llm_tokens_for_equivalent_task - actual_tokens_used) * cost_per_token`.

---

## 3. Infrastructure Architecture

### 3.1 System Topology

```
┌──────────────────────────────────────────────────────────┐
│  API Gateway                                             │
│  (Cognito auth, rate limiting, WAF, routing)             │
└──────┬──────────────────────────┬────────────────────────┘
       │                          │
       ▼                          ▼
  ECS Fargate                 Lambda functions
  (FastAPI service)           ├─ Sandbox execution (Standard/Heavy/GPU tiers)
  ├─ Registry CRUD            ├─ Webhook sync (discipline repo → global registry)
  ├─ Catalog search           ├─ Receipt signature verification
  ├─ Bounty lifecycle         └─ LLM pre-screen dispatch
  ├─ Shapley computation
  └─ Metrics queries
       │
       ├──► PostgreSQL (users, bounties, submissions, atom metadata)
       ├──► EC2 r6g.medium ─ Memgraph (provenance graph, Bolt protocol)
       ├──► S3 (atom source archives, dataset splits, receipts, backups)
       └──► manifest.sqlite (local CLI snapshots, generated from PostgreSQL)
```

### 3.2 Why This Split

**ECS Fargate for the API (not Lambda):**

The Registry API needs persistent connections to Memgraph (Bolt
protocol). Lambda functions are ephemeral — each cold start creates a
new TCP connection with no pooling across invocations. Under load,
this exhausts Memgraph's connection limit and adds 50–200ms handshake
latency per cold start. Fargate maintains a persistent connection pool
(~10 connections shared across requests) with zero cold-start penalty.

At early scale, a single Fargate task (0.25 vCPU, 512 MB) costs
~$15/month and handles all API traffic. FastAPI gives Pydantic models
(already used throughout sciona), auto-generated OpenAPI docs
(which becomes the public Registry API spec), and async support for
graph queries.

**Lambda for stateless, ephemeral workloads:**

Sandbox execution, webhook handlers, receipt verification, and LLM
pre-screening are all stateless and bursty — perfect for Lambda.
Sandbox execution in particular benefits from Lambda's isolation model
(each invocation is a fresh container with no shared state).

For Heavy/GPU sandbox tiers that exceed Lambda's limits, execution
dispatches to SageMaker Processing Jobs or EC2 GPU instances.

### 3.3 Memgraph on AWS

Memgraph is an in-memory graph database. It needs a dedicated host
with predictable RAM and fast local disk for WAL (write-ahead log)
persistence.

**Hosting: EC2 instance (dedicated, private subnet)**

| Setting | Value | Rationale |
|---------|-------|-----------|
| Instance type | `r6g.medium` (ARM, 1 vCPU, 8 GB RAM) | Memory-optimised; 8 GB covers ~500 atoms + growing provenance graph with headroom |
| Storage | EBS gp3, 20 GB | WAL snapshots + Memgraph data directory |
| Network | Private subnet, security group allows Bolt (port 7687) from Fargate only | No public exposure |
| Cost | ~$35/month on-demand, ~$15/month reserved | Scales to `r6g.large` (16 GB) when needed |

**Why EC2 over Fargate for Memgraph:**

Memgraph is stateful — it needs persistent storage for WAL durability.
Fargate doesn't natively support persistent volumes (EFS is possible
but adds latency to every WAL write). EC2 with EBS gives low-latency
local storage and straightforward volume management.

**Why EC2 over Memgraph Cloud:**

Lower cost at small scale, no vendor lock-in, full control over
configuration. Can migrate to Memgraph Cloud later if operational
burden grows.

**Durability & backup strategy:**

```
Memgraph (in-memory)
    │
    ├──► EBS gp3 volume (WAL + periodic snapshots via SNAPSHOT command)
    │
    ├──► Daily: EBS snapshot → retained 7 days
    │
    └──► Weekly: Full Memgraph export (EXPORT DATABASE) → S3 (Glacier after 30 days)
```

On instance failure: launch new EC2 from latest AMI, attach EBS
volume, Memgraph recovers from WAL automatically. RTO < 10 minutes.

### 3.4 PostgreSQL (Relational Data)

PostgreSQL is the source of truth for all structured relational data.
Memgraph handles only graph-shaped provenance queries.

**Data split between PostgreSQL and Memgraph:**

| Data | Store | Why |
|------|-------|-----|
| Users, auth, reputation scores | PostgreSQL | Relational, transactional |
| Bounties, submissions, verification records | PostgreSQL | State machine with constraints, concurrent writes |
| Payout ledger, cancellation fees | PostgreSQL | Financial data needs ACID |
| Atom metadata (fqdn, versions, benchmarks, fingerprints) | PostgreSQL | Source of truth for the catalog; generates SQLite snapshots |
| Provenance graph (DEPENDS_ON, AUTHORED_BY, etc.) | Memgraph | Path traversals, Shapley computation, DAG validation |

**Hosting: Supabase (prototype) → RDS (production)**

Start with Supabase for speed — managed Postgres with built-in
PgBouncer connection pooling (important for both Fargate and Lambda
clients), dashboard, and a generous free tier. The platform uses
standard SQL and psycopg/asyncpg — no Supabase-specific features
beyond hosting.

When the platform outgrows Supabase's pro tier, migrate to RDS
PostgreSQL or Aurora Serverless v2 via `pg_dump`/`pg_restore`. No
application code changes needed.

| Phase | Host | Cost | Connection Pooling |
|-------|------|------|--------------------|
| Prototype | Supabase Free/Pro | $0–$25/month | Built-in PgBouncer |
| Growth | RDS `db.t4g.micro` | ~$15/month | RDS Proxy or self-hosted PgBouncer |
| Scale | Aurora Serverless v2 | Pay-per-ACU | Built-in |

**SQLite snapshot generation:**

The local `sciona` CLI still consumes `manifest.sqlite` — the existing
loader (`load_hyperparams_manifest_sqlite()`) is unchanged. The
platform generates SQLite snapshots from PostgreSQL on every atom
publish:

```
Atom published → PostgreSQL write
  → Async job: query PostgreSQL → generate manifest.sqlite
  → Upload to S3 (manifests/ prefix)
  → SNS notification → Fargate refreshes local copy
  → CLI: `sciona catalog sync` downloads latest snapshot
```

This keeps the local CLI fully offline-capable while PostgreSQL
handles concurrent writes on the platform side.

### 3.5 Data Storage (S3)

All durable artifacts live in S3:

| Bucket/Prefix | Contents | Access Pattern |
|---------------|----------|---------------|
| `atoms/` | Atom source archives (versioned by content hash) | Write-once on publish, read on CDG assembly |
| `datasets/` | Bounty data folders + platform-generated splits | Write on bounty creation, read by sandbox |
| `receipts/` | Signed execution receipts | Write on submission, read on verification |
| `manifests/` | `manifest.sqlite` snapshots (generated from PostgreSQL) | Write on atom publish, downloaded by CLI |
| `backups/` | Memgraph exports, PostgreSQL dumps | Write on schedule, read on disaster recovery |

### 3.5 Auth & Identity

Direct GitHub OAuth — no Cognito. GitHub is the sole identity
provider, and the platform issues its own JWTs to avoid per-MAU
costs.

**Login flow:**

```
User runs `sciona login`
  → CLI opens browser to GitHub OAuth authorize URL
  → User grants access
  → GitHub redirects with auth code
  → FastAPI backend exchanges code for GitHub access token
  → Backend fetches profile from api.github.com/user
  → Backend upserts user record in PostgreSQL
  → Backend issues a self-signed JWT (KMS-backed signing key)
  → CLI stores JWT in ~/.sciona/config
```

**Subsequent API calls:** Bearer token (the platform JWT) in the
`Authorization` header. Validated in FastAPI middleware with zero
external calls — just a local signature check against the KMS
public key.

**Token lifecycle:**

| Token | Lifetime | Refresh |
|-------|----------|---------|
| Platform JWT | 30 days | `sciona login` re-authenticates via GitHub |
| GitHub access token | Not stored | Used once at login to fetch profile, then discarded |

**Identity tiers (unchanged from 4.8):**

| Tier | Requirement | Unlocks |
|------|-------------|---------|
| Contributor | GitHub OAuth login | Browse, submit atoms/CDGs, build reputation |
| Payee | Stripe Connect onboarding (triggered on first bounty attribution) | Receive payouts |

This approach costs $0 regardless of user count (GitHub OAuth Apps
are free, JWTs are self-issued). The only external dependency is
GitHub's OAuth endpoint at login time.

---

## 4. Implementation Priority

Ordered by dependency (must build earlier things first):

| Priority | Component | Effort | Depends On |
|----------|-----------|--------|------------|
| P0 | Fix fingerprint to full SHA-256 | 1 hour | Nothing |
| P1 | Dead-End Flare protocol + CLI command | 1-2 days | Existing optimize pipeline |
| P1 | Provenance Graph schema + Shapley engine | 1 week | Existing graph_store.py |
| P2 | Global Registry API (FastAPI on Fargate) | 1-2 weeks | Provenance Graph, Memgraph on EC2 |
| P2 | Bounty state machine + escrow integration | 1-2 weeks | Registry API |
| P2 | Infrastructure (EC2 Memgraph, Fargate, S3, API Gateway) | 1 week | — |
| P3 | Verification Sandbox (Lambda + SageMaker tiers) | 2-3 weeks | Bounty state machine |
| P3 | Blind submission window + receipt verification | 1 week | Bounty state machine |
| P4 | Fuzzing Cluster | 2-3 weeks | Registry API, HPO framework |
| P5 | Metrics & ESG Dashboard | 1-2 weeks | Provenance Graph |

---

## 5. Design Decisions (Resolved)

### 4.1 Flare Disclosure & Bounty Pricing

The Dead-End Flare discloses:

- **Highest-complexity graph attempted**: node count and edge count of the
  largest CDG variant tried during optimisation. This tells Architects the
  problem's structural scale without revealing the Principal's private data.
- **Best-to-date metric value**: the scalar value of whichever metric is being
  optimised (e.g., loss, accuracy, latency). Architects can judge difficulty
  and decide whether the bounty price is worth the effort.

Bounty pricing is set by the Principal, informed by these disclosed metrics.
No fixed default — the Flare gives Architects enough signal to self-select.

### 4.2 Settled Bounty Immutability

Settled bounties are **immutable**. Shapley allocations are never recomputed
retroactively. When an atom is updated (new version), historical bounties
that used the old version are unaffected. The `DEPENDS_ON` edges reference a
specific atom version, not `latest`.

### 4.3 Multi-Author Atoms

When an atom is submitted, authorship is either:

- **Self-declared**: the submitter specifies co-authors and their shares, or
- **Equal split**: if no shares are declared, all named authors receive equal
  `contribution_share` on their `AUTHORED_BY` edges (i.e., `1/n`).

### 4.4 Single Global Registry with Discipline Repos

One global registry/catalog for cross-disciplinary discovery. Individual
disciplines (crystallography, signal processing, operations research, etc.)
may maintain their own atoms repos following the canonical format and
architecture (same schema, same manifest.sqlite structure). These
discipline repos sync into the global registry — they are upstream sources,
not independent silos.

### 4.5 Post-Settlement Plagiarism

When an atom in a settled bounty's dependency tree is discovered to be
plagiarised:

**Approach: strict immutability + forward-only correction + blacklisting.**

1. **Past payouts stand.** Settled bounties are immutable. The plagiarist
   keeps royalties already paid. Clawback and insurance mechanisms are out
   of scope for v1.

2. **Forward-only alias.** The registry creates an alias redirecting the
   plagiarised atom to the original. All future CDGs that depend on this
   atom automatically attribute the original author. The plagiarist's
   income stream is cut off immediately.

3. **Registered identity & blacklisting.** All participants (Originators,
   Architects, Principals) must register with verified identity — required
   for tax reporting (1099/W-9 for US entities) regardless. Proven
   plagiarism results in:
   - Public flag on the user's profile (reputation damage).
   - Blacklist from future Originator payouts.
   - Optionally, blacklist from submitting new atoms entirely.
   - The plagiarism finding and evidence are recorded as an immutable
     `PLAGIARISM_DISPUTE` edge in the Provenance Graph for transparency.

4. **Detection.** The Fuzzing Cluster runs behavioral equivalence testing
   (same inputs → same outputs) across all public atoms as a background
   job. This catches Type-4 semantic clones that AST fingerprinting misses.
   Community reports are also accepted and reviewed by Foundation
   maintainers.

### 4.6 Submission Window Duration

Principal-configurable, with a **minimum of 7 days**. The Principal sets
the deadline when funding the bounty. Longer windows attract more
Architects and higher-quality submissions; shorter windows (at minimum)
suit urgent production issues.

### 4.7 Verification Sandbox — Two-Stage Architecture

CDG verification uses a **two-stage pipeline**: an LLM pre-screen gate
followed by a deterministic Lambda execution sandbox.

**Stage 1 — LLM Pre-Screen (non-deterministic, cheap):**

An LLM agent reviews the submitted CDG before it enters the sandbox. It
checks for:

- Suspicious patterns: network calls, file I/O, `exec`/`eval`,
  `subprocess`, `ctypes`, `importlib` dynamic imports.
- Structural validity: does the CDG only reference registered atoms? Do
  the I/O types match the bounty's `target_spec`?
- Resource estimation: flag CDGs with obviously unreasonable node counts
  or deeply nested recursion.

Submissions that fail pre-screening are rejected with a reason. This
filters the 80% of bad/malformed submissions before burning sandbox
compute. Because it is non-deterministic, it is advisory only — it
cannot be the verification mechanism itself.

**Stage 2 — Tiered Sandbox Execution (deterministic, isolated):**

The CDG and its atom dependencies are packaged as a single function
payload and executed against the Principal's blind test data. The
execution environment is selected by **bounty tier**:

| Tier | Environment | Timeout | Memory | GPU | Compute Overhead |
|------|-------------|---------|--------|-----|------------------|
| **Standard** | AWS Lambda | 15 min | 4–10 GB | No | Included in 5% platform fee |
| **Heavy** | SageMaker Processing Job | 2 hours | 16–64 GB | No | Principal pays from escrow |
| **GPU** | SageMaker / EC2 GPU instance | 4 hours | 64 GB + GPU | Yes | Principal pays from escrow |

All tiers share these isolation constraints:

| Constraint | Setting |
|------------|---------|
| Network | Disabled (VPC with no egress) |
| Filesystem | Read-only `/tmp` |
| Runtime | Pinned Python version + pinned dependency lockfile |
| Non-determinism | `PYTHONHASHSEED=0`, numpy/torch seeds fixed, no `random` without seed |

For Heavy and GPU tiers, the bounty escrow must include a **compute
overhead deposit** on top of the bounty amount. This covers the cost of
spinning up SageMaker/EC2 for each submission verification. Unused
overhead is refunded to the Principal when the bounty settles. The LLM
pre-screen (Stage 1) is especially important for these tiers to avoid
burning expensive compute on invalid submissions.

**The flow:**

```
PR submitted
  → Stage 1: LLM pre-screen (reject / pass)
  → Stage 2: Tiered sandbox (Lambda / SageMaker / EC2 per bounty tier)
  → Result: pass / fail + metric value
  → Record to Provenance Graph
  → If pass: enter blind submission pool for batch comparison at deadline
```

### 4.8 Identity & Onboarding

**Two-tier identity model** to minimise onboarding friction while
meeting payout compliance requirements:

| Tier | Requirement | Unlocks |
|------|-------------|---------|
| **Contributor** | GitHub account | Browse registry, submit atoms, submit CDGs, build reputation |
| **Payee** | Full KYC (tax ID, address, W-9/W-8BEN) | Receive bounty payouts (Architect share or Originator yield) |

Users onboard with just a GitHub ID. When they first earn bounty
attribution (either as Architect or Originator), the platform prompts
KYC before releasing funds. Payout is held in escrow until KYC clears.

This means:
- Zero friction for researchers contributing atoms (no KYC needed to
  publish).
- KYC only triggers when real money moves — standard for Stripe Connect
  marketplaces.
- GitHub identity is sufficient for reputation tracking and blacklisting
  on plagiarism.

### 4.9 Atom Versioning — Hybrid Content-Addressed + Semver

**Internally content-addressed; externally labeled with semver.**

Every atom version is immutable and stored by its AST fingerprint hash
(full SHA-256, per P0 fix). The Provenance Graph always references
content hashes — a settled bounty's `DEPENDS_ON` edges point to exact
hashes, never mutable labels.

**Semver labels** are human-friendly aliases managed by the author:

```
bandpass_filter@1.2.3  →  bandpass_filter#a3f8c91d4e...  (content hash)
bandpass_filter@1.2.4  →  bandpass_filter#7b02ef1a88...  (content hash)
```

**`latest` pointer**: Each atom has a mutable `latest` alias. CDGs in
development can reference `latest` for convenience. On bounty
submission, all `latest` references are **pinned** to their current
content hashes — the submitted CDG is fully reproducible.

**`DERIVES_FROM` edges** link content hashes to form a version lineage
DAG:

```
#7b02ef1a88 ──DERIVES_FROM──► #a3f8c91d4e ──DERIVES_FROM──► #0019bbca32
  (v1.2.4)                      (v1.2.3)                      (v1.2.2)
```

**Trade-offs accepted:**

- Version proliferation (a docstring fix creates a new content hash).
  Mitigated by the semver labels — humans interact with `@1.2.x`, the
  hashes are infrastructure plumbing.
- Authors must tag semver manually. The CLI can suggest bump level by
  diffing the normalized AST (same hash = no release needed; structural
  diff = suggest minor/major).

### 4.10 Bounty Submission Format

A bounty submission consists of two files and a data folder:

**`sciona.yml` — Problem definition (data schema):**

Describes the data groups, readers, properties, and metadata. This is
the same format already used by `sciona optimize` locally. Example
structure (from the NIGHTCAP dataset):

```yaml
name: NIGHTCAP
groups:
  capnostream:
    reader:
      fqn: sciona.principal.adapters.parsers.parse_capnostream_folder
      kwargs: { meta: true }
    source: capnostream
    time: t
    properties:
      t:   { source: etime, description: "Seconds since epoch" }
      rr:  { source: RR, description: "Respiration rate in BPM" }
      co2: { source: "CO2 Wave", description: "CO2 waveform" }
      # ...
  h10_ecg:
    reader:
      fqn: sciona.principal.adapters.parsers.polar_h10_to_pandas
    # ...
meta:
  source: tracker_$(tracker).csv
  reader: { fqn: pandas.read_csv }
  user: { source: study_id }
  folder: { source: trial_name }
```

**`config.yml` — Optimization context and objective:**

Captures the problem metadata that would otherwise live in the
telemetry run record. Modeled after the `metadata.optimize` block in
`output/telemetry_runs/run_*.json`:

```yaml
goal: "Detect heart rate from raw ECG signal"
objective: rmse               # loss function
execution_metric: precision   # secondary metric
domain_tags: [biomedical, signal-processing]
max_trials: 50                # how many trials the Principal ran locally

# Flare disclosure (auto-populated by CLI)
best_loss: 10.524
best_structure:
  node_count: 4
  edge_count: 2
  topo_hash: "05c468d81ba2b726"
  primitive_signature: "0aeed7f9da7e392b"
  atomic_primitives:
    - filter_signal_for_detection
    - detect_peaks_in_signal
    - compute_event_rate

# Atoms already exhausted by local optimization
atoms_tried:
  - filter_signal_for_detection
  - detect_peaks_in_signal
  - compute_event_rate
  - compute_event_rate_smoothed
```

**Data folder:** The raw dataset files referenced by `sciona.yml`.

### 4.11 Blind Test Data Validation

The Foundation does **not** trust the Principal's data split. The
platform performs automatic validation and splitting.

**Splitting strategy — LLM-assisted at submission time:**

Rather than hardcoding a clustering method, the sciona CLI ships with a
**splitting prompt** that an LLM agent runs at bounty submission time.
The agent:

1. Reads the `sciona.yml` to understand group structure, property types,
   and time series characteristics.
2. Identifies natural stratification axes: user/subject IDs (from
   `meta.user`), session/folder structure (from `meta.folder`), domain
   categories, or time-based segments.
3. For datasets without explicit strata, runs unsupervised clustering
   on feature summaries (statistics per group/session) to discover
   natural groupings.
4. Produces a deterministic split assignment mapping each data
   unit (session, subject, file) to a partition.

This is flexible enough to handle diverse dataset shapes (time series,
tabular, multi-modal) without a one-size-fits-all algorithm.

**Split allocation:**

- **Public split** (~20%): shared with Architects in the Flare.
  Representative of the full distribution. Architects use this to
  develop and locally test their CDGs.
- **Blind split** (~80%): held by the platform, used for sandbox
  verification. Never exposed to Architects.

**Minimum requirements:** The platform rejects datasets that are too
small for meaningful splitting (< 50 samples) or that fail basic
quality checks (all-constant columns, extreme class imbalance without
declared handling).

### 4.12 Atom Benchmarks & Soft Deprecation

Atoms are **never fully deprecated**. There may be context-specific
cases where an older version outperforms its successor (different data
distributions, different constraint regimes, numerical stability edge
cases).

Instead, the global catalog manifest includes **per-atom benchmark
scores**:

```
Benchmark record (stored in manifest.sqlite):
  atom_fqdn:      str
  content_hash:   str       # specific version
  benchmark_id:   str       # e.g. "signal_denoising_v2", "sorting_10k"
  metric_name:    str       # e.g. "loss", "latency_ms", "memory_mb"
  metric_value:   float
  dataset_tag:    str       # which dataset/domain
  measured_at:    datetime
```

These benchmarks are:

- **Populated by the Fuzzing Cluster** as a background job whenever a
  new atom version is published or a new benchmark suite is added.
- **Stored in the global catalog manifest** alongside the atom metadata
  so they are available locally.
- **Integrated into `sciona optimize`**: The Principal's optimization
  loop queries benchmark scores when ranking candidate atoms for a
  slot. The existing `AtomLedger` UCB1 bandit ranking is augmented
  with benchmark priors — an atom with strong benchmark scores on a
  relevant domain tag starts with a higher prior, requiring fewer
  trials to surface.

**Benchmark suite curation** is a three-source process:

1. **Foundation maintainers** define canonical benchmarks for core
   domains (signal processing, sorting, linear algebra, etc.).
2. **Community experts** (based on reputation / Algorithmic Impact
   Factor) can propose and vote on new benchmarks for their domains.
3. **Automated discovery from bounty history**: When a bounty settles,
   the blind test data (with Principal consent) can be anonymised and
   added as a new benchmark. This creates a growing, real-world
   benchmark corpus derived from actual production problems.

**Soft deprecation** works as follows: when a newer version
consistently outperforms an older one across all benchmarks, the
registry marks the old version as `superseded` (not `deprecated`). The
`sciona optimize` pipeline deprioritises `superseded` atoms in its
candidate ranking but does not exclude them. If the optimizer finds
that a `superseded` atom actually performs better in a specific
context, it can still select it.

### 4.13 Bounty Cancellation

Principals may cancel a funded bounty before the submission window
closes, subject to a **cancellation fee**:

| Timing | Fee | Refund |
|--------|-----|--------|
| Before any submissions received | 10% of escrow → Foundation | 90% refunded to Principal |
| After submissions received but before deadline | 25% of escrow → Foundation | 75% refunded to Principal |
| After deadline (verification in progress) | No cancellation allowed | — |

The cancellation fee serves two purposes:

1. **Discourages frivolous bounties**: Principals shouldn't post bounties
   speculatively and cancel when they find an internal solution.
2. **Buffer for unintended consequences**: The fee absorbs costs when
   Architects have already invested effort on a cancelled bounty.
   Foundation may use this pool to compensate Architects who submitted
   work-in-progress on cancelled bounties (at Foundation discretion).

Cancellation fees are recorded in the Provenance Graph as
`CANCELLED` status on the Bounty node, with the fee amount and timing
preserved for audit.

### 4.14 Compute Verification & Proof-of-Work

Community-submitted CDG solutions need verified execution results that
are difficult to spoof without actually running the computation. The
platform uses a **fixed verification budget** model where sandbox
re-execution is triggered selectively, and the Principal stays in the
loop.

#### Execution Receipts — SSH-Signed Proof of Local Execution

When an Architect runs `sciona verify --public-split`, the CLI produces
a signed execution receipt:

**Step 1 — Receipt generation (local):**

```
receipt.json:
  cdg_hash:      str    # content-addressed AST fingerprint of the CDG
  atom_versions:  dict   # {fqdn: content_hash} for all pinned atoms
  split_hash:    str    # SHA-256 of the public split files
  output_hash:   str    # SHA-256 of the full output array (all samples)
  metric_name:   str    # e.g. "rmse"
  metric_value:  float  # e.g. 8.31
  timestamp:     str    # ISO 8601
  sciona_version: str    # CLI version that produced the receipt
```

The CLI serialises this as canonical JSON (sorted keys, no whitespace)
and signs it with `ssh-keygen -Y sign` using the Architect's SSH key
(discovered from `~/.ssh/id_*`, configurable via `~/.sciona/config`).
The output is a `.receipt` file containing the JSON blob + detached
SSH signature.

**Step 2 — Signature verification (platform):**

At registration, the platform fetches the user's public SSH keys from
`https://github.com/{username}.keys`. On submission, the platform
verifies the receipt signature with `ssh-keygen -Y verify` against the
stored public keys.

**What this proves:** The person who controls the GitHub account
actually ran this specific CDG (by content hash) against this specific
data split (by split hash) and produced this specific output (by
output hash). Forging the output hash requires producing the correct
output for every sample in the public split — effectively requiring
you to run the computation.

**What it doesn't prove:** That they ran it on their own hardware.
This is fine — we care that the claimed result is real, not where it
was computed.

#### Verification Budget & Principal-in-the-Loop

The bounty includes a **fixed number of sandbox verifications** paid
for by the Principal as part of the escrow. The verification flow is:

**Phase 1 — Baseline establishment:**

When a bounty opens, the platform runs the Principal's best CDG (from
the Flare's `best_structure`) against the blind split. This
establishes the **state-of-the-art baseline**, measures execution
time/memory (for compute tier pricing), and consumes the first
verification slot.

**Phase 2 — Community submissions:**

Architects submit CDGs with signed execution receipts. The platform:

1. Verifies the receipt signature (cheap, no compute).
2. Validates the receipt against the public split (re-run on public
   split only — small, cheap).
3. If the claimed metric beats the current best-to-date by a
   **minimum improvement threshold** (e.g., 5% relative improvement),
   a sandbox verification is triggered against the blind split.
   This consumes one verification slot.
4. If the blind-split verification confirms the improvement, the
   submission becomes the new best-to-date. The Principal is notified.

**Phase 3 — Principal interaction:**

Between verifications, the Principal has agency:

- **Set a new target**: After seeing a verified improvement, the
  Principal can raise the bar — setting a new minimum metric target
  before the next verification fires. This prevents burning
  verification slots on marginal improvements.
- **Independent evaluation**: Since the bounty is in escrow and the
  CDG code is available, the Principal can pull the latest/best
  verified CDG and run it independently against their full private
  dataset to assess whether it scales beyond the blind split.
- **Request new data round**: If the community isn't producing good
  alternatives (submissions plateau), the Principal can fund a new
  round of data splitting — providing additional data or requesting
  re-stratification. This refreshes the problem and may attract new
  Architects.

**Phase 4 — Settlement:**

At the submission deadline, the CDG with the best verified blind-split
metric wins. If multiple submissions were verified, the best one wins
regardless of submission order.

**Verification budget sizing:**

| Bounty Tier | Included Verifications | Additional (from escrow) |
|-------------|----------------------|--------------------------|
| Standard    | 5                    | $10 per additional       |
| Heavy       | 3                    | $50 per additional       |
| GPU         | 2                    | $200 per additional      |

Unused verification slots are refunded to the Principal as part of
settlement.

### 4.15 Bounty Expiry, Re-Posting & Multi-Objective

**Expiry and re-posting:**

When a bounty expires with no valid submissions (or no submission that
beats the baseline):

- The escrow is refunded to the Principal, **minus** the compute costs
  already incurred (baseline run + any verification slots consumed).
- The Principal may **re-post** the bounty with adjusted parameters:
  new deadline, new escrow amount, revised `config.yml`, additional
  data. Re-posted bounties link to the original via a `REPOSTED_FROM`
  edge in the Provenance Graph, preserving history.

**Multi-objective bounties:**

Some problems have competing metrics (e.g., accuracy vs. latency,
precision vs. memory). The platform supports **Pareto-front bounties**:

- The Principal declares multiple metrics and their directions
  (minimise/maximise) in `config.yml`.
- Submissions are evaluated on all declared metrics.
- The bounty may have **multiple winners** — one per Pareto-optimal
  point. The payout is split among winners (Architect share divided
  equally, or weighted by a Principal-defined priority ordering of
  the metrics).
- Shapley allocations for the Originator yield are computed per
  winning CDG and summed across winners.

Example `config.yml` for a multi-objective bounty:

```yaml
objectives:
  - metric: rmse
    direction: minimize
    weight: 0.7          # Primary
  - metric: latency_ms
    direction: minimize
    weight: 0.3          # Secondary
pareto_winners: 3        # Up to 3 Pareto-optimal CDGs can win
```

### 4.16 Principal-Configured Parameters

Several parameters are delegated to the Principal at bounty creation
time, since they are domain-specific and the Principal bears the
liability for misconfiguration:

| Parameter | Default | Principal Sets | Notes |
|-----------|---------|---------------|-------|
| Minimum improvement threshold | 5% relative | Yes | Triggers a verification slot. Too low burns verifications on noise; too high discourages submissions. |
| Public split percentage | 20% | Yes | Larger splits give Architects better signal but increase overfit risk. Principal bears the liability. |
| Pareto metric weights | Equal | Yes | Weights determine how the Architect payout is split across Pareto winners. |
| Submission window duration | 7 days | Yes (min 7 days) | See 4.6. |
| Verification budget | Tier default | Yes (can buy more) | See 4.14. |
| New target between verifications | — | Yes | Principal can raise the bar after a verified improvement. |

These are declared in `config.yml` at bounty creation and immutable
once the bounty enters `open` status (except for `new_target`, which
the Principal can adjust between verifications as described in 4.14).

### 4.17 Receipt Replay Prevention

Execution receipts are bound to a specific bounty to prevent replay
attacks. The receipt JSON includes:

```
bounty_id:     str    # unique bounty identifier
split_hash:    str    # SHA-256 of the specific split for this bounty
```

The platform rejects any receipt whose `bounty_id` does not match the
submission target. Since `split_hash` is derived from the platform's
per-bounty split (which includes a bounty-specific salt), a receipt
from bounty A cannot be replayed on bounty B even if the underlying
dataset is the same.

---

## 6. Summary of All Design Decisions

All open questions from the initial gap analysis have been resolved.
The architecture is fully specified across 17 design decisions (4.1
through 4.17). The next step is implementation, starting with the P0
and P1 items from the priority table in Section 3:

1. Fix fingerprint to full SHA-256
2. Dead-End Flare protocol + `sciona bounty generate` CLI command
3. Provenance Graph schema + Shapley engine
4. Global Registry API
