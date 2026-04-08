# Signal Phase 2: Expansion Asset Library

## Status

Drafted on April 7, 2026 as Phase 2 of the signal-processing expansion
implementation plan in
[SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md).

## Purpose

This phase turns signal-family enrichment knowledge into a real asset library
instead of leaving it in local rule code and diagnostics.

## Problem

The framework currently knows some signal-family refinements implicitly, but it
does not yet have a disciplined inventory of sanctioned enrichments such as:

- preprocessing insertion
- correction insertion
- validation insertion
- branch-and-compare structures
- safer stage replacements

Without an explicit inventory, expansion stays ad hoc and hard to review.

## Goals

1. Define the first signal-family enrichment inventory.
2. Express each enrichment as an auditable asset.
3. Make the assets general across signal modalities where possible.
4. Leave room for modality-specific specializations without polluting the
   framework core.

## Asset Categories

The initial signal-family asset library should cover at least:

- conditioning enrichments
- correction enrichments
- validation enrichments
- robustness enrichments
- branch-and-compare enrichments
- replacement enrichments

## Candidate Initial Asset Families

The first library should likely include assets for patterns such as:

- add pre-detection cleanup before a detector
- insert correction after a detector
- insert quality gate before downstream estimation
- insert outlier rejection between event sequence and measure stage
- replace a brittle detector family with a more context-aware detector stage
- branch to compare multiple detectors or estimators
- insert measure smoothing only after support checks pass

These should be described generically enough to apply beyond ECG.

## Deliverables

1. A signal-family expansion asset schema specialization or profile.
2. An initial signal-family asset set.
3. Applicability metadata for each asset.
4. Risk and uncertainty metadata for each asset.
5. Dejargonized documentation for each asset.
6. A compatibility loader in `ageo-matcher`.

## Implementation Work

### Workstream A: Asset inventory design

- define the initial asset categories
- decide which enrichments are family-wide versus modality-specific

### Workstream B: Asset authoring

- write the first library of signal-family expansion assets
- include before/after graph forms and applicability conditions

### Workstream C: Runtime loader integration

- add support for loading and validating the signal-family expansion asset set
- expose asset metadata through diagnostics and telemetry

### Workstream D: Governance and review model

- define what makes an asset family-ready
- define which assets remain experimental

## Testing Strategy

- asset-schema validation tests
- asset-loading tests
- runtime applicability tests
- explainability tests showing which asset was selected and why

## Risks

### Risk: assets become too modality-specific too early

Mitigation:

- define a family-wide layer first
- push modality-specific specializations into a narrower secondary layer

### Risk: asset library grows without discipline

Mitigation:

- require rationale, uncertainty notes, and explicit applicability for each
  asset

## Exit Criteria

- signal-family enrichment knowledge exists as a real asset inventory
- the runtime can load and inspect that inventory
- at least a core set of useful enrichments is represented outside private
  framework logic
