# Business Plan Tooling Gaps

Architect review of `sciona_business_plan.pdf` against the current repository
and the likely operating needs of the Algorithmic Commons Foundation.

Date: 2026-03-31

## Why This Exists

The business plan gets the core marketplace loop right:

- Principals fund bounties
- Architects solve them
- Originators receive passive yield
- the Foundation funds the public infrastructure with a fractional scrape

What it does not fully specify is the operating system around that loop.

This memo focuses on:

- tooling that is missing from the business plan
- tooling that the repo does not yet fully implement
- tooling that will become mandatory once money, attribution, governance, and
  enterprise users are involved
- devil's-advocate failure modes that are easy to miss when the product story
  sounds elegant

## Executive Summary

The biggest gap is not the registry or the graph. It is the operational layer
that turns a good protocol into a real institution.

The business plan currently under-specifies:

- money movement and compliance
- trust and safety operations
- durable workflow orchestration
- auditability and forensic replay
- enterprise onboarding and support
- ESG and impact measurement rigor
- governance and policy enforcement
- incident response and public reliability posture
- contributor identity and citation infrastructure
- platform operations around queues, retries, rollbacks, and human review

If the Foundation wants to be taken seriously by:

- researchers
- enterprise R&D labs
- security reviewers
- finance/compliance stakeholders
- board/governance stakeholders

then these layers need explicit tooling decisions, not just product intent.

## Design Reality Check

The business plan implicitly assumes:

1. verification is the hard part
2. payout math is the hard part
3. public-good governance naturally creates trust

In practice, the hard parts are:

1. operating the workflows reliably
2. proving who did what and when
3. surviving disputes, abuse, sanctions, payout failures, and policy edge cases
4. producing audit-ready evidence for every consequential action

The core lesson:

- deterministic compute is necessary
- deterministic operations are also necessary

## Missing Capability Areas

### 1. Payments, Escrow, and Financial Operations

The business plan mentions escrow and payout splits, but not the tooling needed
to run a platform that moves money.

Missing capabilities:

- connected-account onboarding
- KYC / KYB / sanctions screening
- W-8 / W-9 collection
- 1099 / tax reporting workflows
- payout holds and reserves
- failed transfer retries
- refunds and clawbacks
- reconciliation between internal ledger and payment processor
- handling disputes between attribution settlement and actual payout execution
- multi-country payout constraints
- bookkeeping and treasury reporting for a nonprofit

Tooling needed:

- `Stripe Connect` for marketplace payouts
- Stripe tax reporting / hosted onboarding flows
- accounting sync into a general ledger system
- internal reconciliation jobs and finance dashboards
- case-management workflows for payout exceptions

Recommendation:

- Use `Stripe Connect` first.
- Do not build smart-contract escrow first.
- Treat payout state as a workflow problem plus a compliance problem, not just a
  ledger problem.

### 2. Durable Workflow Orchestration

The business plan describes lifecycle states, but not the system that owns those
states.

Examples that should be workflow-managed:

- bounty creation -> funding -> open -> submit -> verify -> settle
- dispute raised -> evidence collected -> review assigned -> payout hold ->
  resolution
- atom published -> audit jobs -> publication gates -> search indexing ->
  embedding refresh
- sandbox job failure -> retry policy -> escalation -> human review
- payout failure -> retry -> support case -> finance review

Missing if not explicitly adopted:

- durable retries
- timeout handling
- idempotent retries after partial failures
- recovery from worker restarts
- human approval steps
- replayable execution history

Recommended tooling:

- `Temporal`

Why:

- the platform has many long-lived, stateful, failure-prone workflows
- bounty clearing and review are not simple queue jobs
- the audit trail needs to be inspectable and replayable

### 3. Sandbox Isolation and Evidence Integrity

The plan mentions isolated microVM verification, which is directionally right.
What is still missing is the tooling layer around it.

Missing capabilities:

- actual microVM isolation management
- signed execution receipts
- artifact hashing and attestation
- reproducible environment manifests
- immutable storage of verification artifacts
- forensic replay for disputes
- policy enforcement for allowed runtimes, datasets, and network behavior

Recommended tooling:

- `Firecracker` for microVM isolation
- `Sigstore/Cosign` for signing artifacts and attestations
- object storage with immutable retention for receipts, artifacts, and evidence

Devil's-advocate note:

- a sandbox that cannot produce replayable, signed evidence is only "secure"
  until the first serious payout dispute.

### 4. Trust, Safety, and Abuse Operations

The business plan assumes aligned incentives. It does not address bad actors.

Missing capability areas:

- plagiarism review queue
- copyright / license infringement review
- DMCA and takedown handling
- export-control review
- sanctions screening
- fraud detection on bounty patterns
- sybil and collusion detection
- moderation for abusive submissions or spam atoms
- staff override logging
- policy documentation and appeal workflows

Recommended tooling:

- internal review/admin console
- case-management queue for disputes and moderation
- policy engine for publishability and enforcement rules
- immutable staff action audit logs

Recommended core technology:

- `OPA` for policy-as-code

Use cases:

- publication gates
- payout holds
- role-based staff actions
- moderation policies
- sandbox allow/deny rules

### 5. Governance and Foundation Operations

The nonprofit framing is a strength, but the plan treats governance as a brand
attribute rather than an operating function.

Missing capabilities:

- board decision tracking
- conflict-of-interest logging
- grant / donor relationship management
- budget planning and approval workflows
- procurement controls
- legal document retention
- policy versioning
- annual reporting / audit support

Tooling likely needed:

- lightweight CRM for donors, grant partners, and institutions
- internal docs/wiki and records system
- accounting / ERP-lite stack
- governance calendar and decision log

Devil's-advocate note:

- a nonprofit with weak governance tooling loses the trust advantage it is
  counting on.

### 6. Enterprise Security and Customer Readiness

The plan targets Principals in enterprises and R&D labs, but it does not account
for the tooling burden of becoming a serious vendor.

Missing capabilities:

- SSO / SCIM for enterprise customers
- audit logs for customer actions
- status page
- incident management
- support and SLA workflows
- security review packet / questionnaire support
- environment segregation and secrets hygiene
- DPA / data retention enforcement

Recommended tooling:

- SSO platform or Supabase-compatible enterprise auth strategy
- public status page
- incident response paging/on-call
- help center + support desk
- environment-specific config management

Minimum stack:

- `Statuspage`
- `Zendesk` or equivalent support tooling

### 7. Product Analytics, ESG Claims, and Metrics Governance

The business plan leans heavily on:

- Algorithmic Impact Factor
- Compute Preserved
- ESG positioning
- badges and leaderboards

That is a measurement problem, not just a dashboard problem.

Missing capabilities:

- event schema governance
- metrics definitions with versioning
- warehouse modeling
- historical backfills
- lineage and provenance of derived metrics
- methodology review and change control
- reproducible benchmark baselines

Recommended tooling:

- warehouse-backed analytics
- `dbt` for modeled metrics
- BI/dashboard layer for stakeholder views

Devil's-advocate note:

- if `Compute Preserved` is not methodology-controlled and auditable, it turns
  into marketing copy rather than a credible metric.

### 8. Research Identity, Citation, and Attribution Infrastructure

The business plan is right to emphasize reputational rewards, but the tooling is
underspecified.

Missing capabilities:

- canonical researcher identity linking
- ORCID integration
- affiliation normalization
- DOI minting or DOI linkage where appropriate
- provenance record export formats
- citation correction workflows
- attribution dispute workflows

Recommended integrations:

- `ORCID`
- `Crossref`
- `DataCite` if minted artifacts become part of the public record

Devil's-advocate note:

- Auto-BibTeX is not enough if identity resolution and attribution provenance are
  weak.

### 9. Release Governance and Platform Operations

The business plan assumes cloud infrastructure but does not specify how changes
to that infrastructure are governed.

Missing capabilities:

- infrastructure-as-code
- secrets management
- change management
- feature flags
- canary / staged rollout controls
- backup and restore validation
- DR testing
- environment diffing

Recommended tooling:

- `Pulumi` or `Terraform`
- feature flag platform
- managed secrets tooling
- backup validation jobs

Devil's-advocate note:

- once money and public trust are involved, "we changed it in the dashboard" is
  not an acceptable operating model.

### 10. Search, Support, and Community Operations

The business plan assumes a community flywheel, but communities create support
load.

Missing capabilities:

- searchable public docs / help center
- contributor onboarding workflows
- moderation of comments / discussions / PR disputes
- ticket routing by severity
- community operations tooling
- changelog and announcement tooling

Recommended tooling:

- help center / knowledge base
- issue triage workflow
- public roadmap / release notes discipline

## Current Repo vs. Business-Plan Needs

Things the repo already has meaningful foundations for:

- Supabase-backed registry and auth
- provenance- and attribution-oriented data model
- publishability and entitlement concepts
- search / embeddings database primitives
- snapshot generation
- some dashboard metrics primitives
- verification and settlement schema foundations

Things the repo does not yet represent as full operating systems:

- workflow orchestration
- payout operations
- customer support tooling
- incident tooling
- finance tooling
- case management
- artifact attestation
- policy-as-code
- enterprise account operations
- ESG analytics governance

## Recommended Tooling Stack

This is a pragmatic initial stack, not a maximal one.

| Capability | Recommendation | Why |
|---|---|---|
| Workflow orchestration | `Temporal` | Bounty, verification, settlement, dispute, and review flows are long-lived and failure-prone. |
| Sandbox isolation | `Firecracker` | Stronger isolation story than plain containers for payout-critical verification. |
| Artifact signing | `Sigstore/Cosign` | Signed receipts, manifests, and verification artifacts reduce dispute ambiguity. |
| Payments | `Stripe Connect` | Best fit for marketplace onboarding and payout operations. |
| Policy engine | `OPA` | Encodes publishability, moderation, payout holds, and admin controls. |
| Infra management | `Pulumi` or `Terraform` | Makes the cloud footprint reviewable and reproducible. |
| Feature flags | `LaunchDarkly` or simpler alternative | Needed for rollout control on auth, search, payouts, and visibility. |
| Observability | `OpenTelemetry` + tracing/error backend | Correlated traces across API, workers, sandbox, and settlement. |
| Error monitoring | `Sentry` | Useful for API, worker, and frontend exception visibility. |
| Metrics / dashboards | warehouse + `dbt` + BI | Required for ESG metrics, leaderboards, and finance-grade reporting. |
| Support | `Zendesk` or equivalent | Necessary for payout issues, disputes, and onboarding. |
| Status communication | `Statuspage` | Public reliability communication for enterprise-facing operations. |
| Research identity | `ORCID`, `Crossref`, `DataCite` | Needed for attribution credibility and institutional legitimacy. |

## Buy vs. Build Guidance

### Strong buy recommendations

Do not build these first:

- workflow engine
- payout processor
- support desk
- status page
- feature flag system
- observability substrate
- DOI / identity network

### Build around, not instead of

These are good internal build areas:

- provenance-aware payout logic
- publication policy definitions
- plagiarism review workflow
- attribution dispute console
- verification evidence viewer
- metrics methodology layer specific to the Foundation

### Avoid early overbuilding

Do not build:

- custom smart-contract platform before fiat payouts work
- custom full-text/vector platform before the current Supabase search surface is
  fully exploited
- custom donor/board ERP before basic governance operations exist

## Devil's-Advocate Risks

### 1. "Trust through nonprofit status" is not enough

Enterprises will still ask:

- who can see my data
- how do you isolate sandbox workloads
- what happens during an incident
- what is your audit trail
- how do you resolve disputed payouts

### 2. "Deterministic verification" does not remove legal disputes

Disputes can still arise over:

- IP ownership
- plagiarism
- licensing contamination
- payout withholding
- identity fraud
- authorship splits

### 3. "Open-source clause" creates moderation and legal burden

The moment the platform requires public merge for reward, it becomes responsible
for:

- reviewing code provenance
- handling takedowns
- enforcing quality and policy standards
- rejecting low-quality bounty farming submissions

### 4. "ESG savings" can backfire if the metric is weak

If the methodology is not robust, customers may treat it as:

- inflated marketing
- unverifiable carbon accounting
- dubious benchmark theater

### 5. "Passive royalties" create support and dispute load

Once real money arrives, users will ask:

- why did I get this amount
- why was I excluded
- why was this atom counted
- why is my payout delayed
- how can I appeal

Without workflow, case management, and signed evidence, those questions become
organizational debt.

## Phased Tooling Rollout

### Phase A: Mandatory before external money movement

- `Temporal`
- `Stripe Connect`
- support desk
- status page
- observability stack
- signed verification artifacts
- policy-as-code for critical gates

### Phase B: Mandatory before enterprise Principals

- stronger sandbox isolation
- audit logs and admin action trails
- SSO / enterprise auth strategy
- DR and backup validation
- formal incident runbooks
- warehouse-backed metrics layer

### Phase C: Mandatory before serious academic reputation positioning

- ORCID integration
- citation export and correction workflows
- affiliation normalization
- attribution dispute console
- methodology governance for impact metrics

## Priority Recommendations

If only a few things are funded next, they should be:

1. workflow orchestration
2. payout/compliance operations
3. signed verification evidence
4. policy-as-code
5. observability + incident tooling
6. analytics governance for ESG / impact metrics

## Immediate Action Items

1. Create an architectural decision record for workflow orchestration.
   Compare `Temporal` against "keep using ad hoc jobs + SQL state machines".

2. Create a payout-ops design memo.
   Include onboarding, holds, reserves, tax forms, reversals, and reconciliation.

3. Define the verification evidence model.
   What exactly gets signed, stored, retained, and replayed?

4. Define the policy domains that should move into policy-as-code.
   At minimum:
   - publishability
   - moderation
   - payout holds
   - admin privileges
   - sandbox execution policy

5. Stand up a minimal operations stack.
   Include:
   - tracing
   - error monitoring
   - support queue
   - status page

6. Define a metrics governance spec.
   Especially for:
   - Algorithmic Impact Factor
   - Compute Preserved
   - ESG badges
   - leaderboard ranking logic

## Bottom Line

The business plan is compelling because the incentive loop is elegant.

The missing pieces are the institutional tools that keep elegant loops alive
under real-world stress:

- disputes
- payment failures
- abuse
- enterprise due diligence
- governance scrutiny
- metric skepticism
- operational outages

The Foundation should think of itself not just as a registry or marketplace, but
as:

- a payment platform
- a verification authority
- a policy engine
- a citation and identity system
- a support organization
- and a public institution

Each of those roles carries tooling requirements that are mostly absent from the
current business plan, and they should be planned explicitly now rather than
discovered under pressure later.
