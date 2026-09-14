# Repaired source graph: expression selection and inference diagnostics

The read-only selector revalidated 19 unique expressions covering all 31
bindings of graph alternative `76274cb1-ebf6-533d-a777-b505190f3d17`.
Source-ID preservation and legacy source-label correspondence remain distinct
checks. No catalog rebinding, approval or publication was applied.

Versioned replay V4 adds two-premise structural substitutions, rational unity
feeds with retained denominator conditions, and rational integer powers beyond
V3’s forward real-root squaring. Existing V1–V3 implementations are unchanged.
For the new two-premise rules, “substitute RHS” inserts the premise RHS in place
of its LHS; “substitute LHS” inserts the LHS in place of its RHS. These meanings
follow the pinned natural-language rule descriptions. Inherited domain
conditions are also transformed. Unity is proved algebraically before use.

Composed replay stops at step 3. Independent diagnostics find eight of thirteen
local steps pass, but this does not establish a composed proof:

- Steps 3, 4 and 13 match insertion of the premise RHS, despite their LHS labels.
- Step 5 cancels the common factor `pdg0004851` (the second mass). This needs
  explicit division and a nonzero condition; sidewise simplification is
  insufficient. At zero mass the input equality can hold while its proposed
  conclusion fails.
- Step 9 is not a structural substitution in either direction. The stored
  preceding expression already canceled the unity feed; its desired conclusion
  needs an explicit algebraic rewrite using the separation premise. This is a
  validator/source-representation limitation, not a demonstrated false equation.

The terminal equation has the algebraic form T² = 4 p² r³ / [G (m1 + m2)].
Here p remains the exact source symbol; identifying it with mathematical pi,
assigning physical meanings, and bounding the circular two-body model are
separate semantic checks still required for an executable realization.

Validation: 24 relevant tests pass, including both substitution directions,
transformed nonzero conditions, canceled unity denominators, invalid unity,
reciprocal singularities, unsupported roots, and rejection of unstated equation
factor cancellation. One existing event-loop deprecation warning was emitted.

Evidence: `repaired_source_graph_selection.json` and
`repaired_source_rule_diagnostics.json`. Next: create an explicitly corrected,
composed proof with retained original provenance and domain conditions before
building and validating a Tier 3 execution realization. The original source
must retain its distinct status and discrepancies.
