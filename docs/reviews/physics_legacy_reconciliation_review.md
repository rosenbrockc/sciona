# Physics graph representation reconciliation

The live read-only audit finds 89 draft source graphs: 44 source-step projections
and 45 older representations. It compares complete source-inference contract
sets (rule, assumptions, dimensions, variable bindings and relationship kind)
and exact bound expression-version sets. Grouping pairwise edges into source
steps is permitted; additional edges, changed contracts and changed expression
hashes are reported rather than silently equated.

Of the older representations, 42 exactly match their source-step counterpart.
Two differ:

- Derivation 539398 (interference) adds the previously missing relation
  `pdg_remote:539398:6988426:7607271250:4192519596` and its feed record.
  The approved interference review explicitly identifies that feed as incorrectly
  bound in place of the polar equation and reconstructs the needed substitution.
- Derivation 282755 (orbit) preserves all relation identities and expression
  versions, but changes rule 111777 to 111236 at step1306821. The projected
  signature retains `source_inference_rule_id=111777`, making the reconciliation
  explicit. This is a changed contract, not exact source equality.

Forty projected graphs have served realizations tied to their exact current
source hash. Four instead have served realizations tied to retained derived
source versions: quadratic (000011), Euler identity (000017), Schwarzschild
(142831), and period/frequency (884319). For each, the bound source version's
`derives_from` points to the current projected source version. These links are
valid retained lineage, not missing implementation links. The audit reports
exact-hash links and derived-version links separately; it does not rewrite them.

The remaining first-wave graph is independent of the remote numeric derivation
IDs. Its source equations are F=ma, a=F/m and F(t)=m*d²x/dt² under constant mass.
It has two nodes and four bindings. That graph needs a separate source and
implementation-scope review; it is not included among the 44 remote counterparts.

No catalog state was changed. Older malformed conceptual graphs are not granted
execution approval simply because their corrected counterparts serve. The
catalog currently serves 69 CDGs. The 45 older representations are not evidence
of 45 additional unimplemented physical derivations. Competition graphs remain
separate unfinished work. The retained JSON contains version IDs, hashes,
comparison results and served lineage for reproducible reconciliation.
