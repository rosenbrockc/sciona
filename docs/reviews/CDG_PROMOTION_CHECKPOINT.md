# CDG promotion work checkpoint

This is a work-in-progress checkpoint, not completion of the promotion goal.
Code, source reviews, qualification scripts and reports include both served
implementations and candidates that remain unapproved. Consult each candidate's
review and live catalog state before selecting it for a pipeline.

Association trust tiers are version-specific. Automated Community publication
does not claim human certification. Tier 1 still requires human review; Tier 2
requires its own rigorous validation and usage evidence.

At the latest read-only competition audit, 39 of 142 intakes had direct provenance
links to served realizations. Such a realization can be partial. Original intake
bindings are not proof of executable implementations.

Recent execution state at checkpoint:

- Bengali's synthetic classifier-guided CycleGAN completed 40 epochs; its full
  seen/OOD/two-font workflow and publication qualification remain incomplete.
- Global Wheat's Faster R-CNN synthetic full-fit process is running. EfficientDet
  requires a CUDA/Apex O1 runner; local CPU qualifications do not satisfy that gate.
- The larger-population salt pilot is running. The earlier complete-workflow
  attempt stopped after 30 fits because its pseudo-label selection was empty.

Checkpoint verification: 207 targeted tests passed and four were skipped.
All staged Python and JSON files parsed. Staged bytes matched the reviewed
file manifest. Credential and dataset-metadata scans were reviewed. This is not a
claim that every implementation's full test suite or promotion gates passed.
Local build products, runtime benchmark/catalog snapshots, model weights and
private runtime evidence are excluded from this checkpoint. Historical reference
and license formatting is preserved to retain its content identity.

The corresponding provider revisions are:

| Repository | Branch | Revision |
| --- | --- | --- |
| sciona-atoms | `kaggle-ingest-batch-1` | `deb79dd7d1d46e4b7509c1c8cb39b6636080ce7c` |
| sciona-atoms-ml | `main` | `fc8077bce6ea89d91830493b3c97fc4117312195` |
| sciona-atoms-dl | `main` | `4c8a56260aa4133f467b5c73bd8b01ef9274c08e` |
| sciona-atoms-signal | `kaggle-ingest-batch-1` | `3cec57587fe07aac8b5cadb37849f0863fb46095` |
| sciona-atoms-physics | `kaggle-ingest-batch-1` | `6d95997c5e00d3d369db68479e83a6e47e7c0872` |
