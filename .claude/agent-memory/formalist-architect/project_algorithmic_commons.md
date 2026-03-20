---
name: Algorithmic Commons Foundation Architecture Analysis
description: Formal analysis of the 6-component non-profit registry/marketplace architecture built on sciona. Covers Shapley computation, sandbox verification, plagiarism detection, and provenance graph.
type: project
---

## Algorithmic Commons Foundation (2026-03-19)

Six components: sciona Client, Global Registry, Provenance Graph + Shapley Engine, Fuzzing Cluster, Bounty Clearinghouse + Sandbox, Metrics Dashboard.

Revenue model: 5% platform / 65% architect / 30% originator Shapley pool.

### Key findings from formal analysis:

1. **AST fingerprint too short for global registry**: `fingerprint_function` in `atom_similarity.py:73` truncates SHA-256 to 64 bits (16 hex chars). Birthday bound gives ~3% collision probability at 10^6 atoms. Must increase to 128+ bits.

2. **Shapley computation is tractable**: CDG dependency graphs are DAGs. DAG-structured binary coverage games have O(n*m) closed-form Shapley. No need for exponential general algorithm. Monte Carlo fallback is O(n^2 * log(n) / epsilon^2).

3. **Gradient score leakage in Dead-End Flare**: Raw `gradient_score` from `AtomObservation` reveals test data distribution properties. Must redact to ordinal ranks only.

4. **Bounty race condition**: Need submission window state (open -> accepting -> verifying -> cleared), not first-come-first-served, to handle concurrent Architect submissions fairly.

5. **Version pinning required**: Solution submissions must include dependency lockfiles; sandbox verifies against pinned atom versions.

6. **Payout conservation requires integer accounting**: Floating-point Shapley values won't sum exactly to pool. Use integer-cent accounting with deterministic residual assignment.

7. **Semantic plagiarism (Type-4) has no automated solution**: Current AST fingerprinting catches Type-1/2 clones. Type-4 requires embedding similarity + output equivalence as flagging mechanism for human review.
