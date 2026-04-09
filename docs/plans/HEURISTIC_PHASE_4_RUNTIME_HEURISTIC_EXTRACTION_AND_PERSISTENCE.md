# Phase 4: Runtime Heuristic Extraction And Persistence

## Purpose

Make heuristic outputs a real runtime artifact with provenance, confidence, and
compatibility with the existing evaluation and telemetry pipeline.

## Problem Statement

Without runtime persistence, heuristics remain conceptual. Proposal selection
and benchmark analysis need stable artifacts that show:

- which heuristics were produced
- what evidence supported them
- how confident the system was
- which producer generated them

The framework currently persists runtime summaries and trace artifacts, but not
yet a dedicated first-class heuristic layer.

## Scope

This phase covers:

- runtime artifact schema extensions
- heuristic extraction hooks
- persistence in runtime evidence and traces
- provenance and confidence handling
- compatibility with trimmed or summary-only artifacts

## Non-Goals

This phase does not:

- define family registries
- finalize proposal ranking policy
- implement long-term outcome memory

## Deliverables

1. Runtime schema support for heuristic outputs.
2. Extraction hooks for heuristic-producing atoms and support transforms.
3. Persistence of heuristic outputs into runtime evidence and related artifacts.
4. Provenance links from heuristics back to producing atoms, summaries, or
   transforms.
5. Reporting surfaces that make heuristic outputs inspectable in debugging and
   benchmark workflows.

## Required Design Decisions

### Artifact shape

The runtime artifacts should preserve:

- heuristic identifier
- value
- confidence
- uncertainty notes
- producer provenance
- source summaries or intermediate references
- family or skeleton interpretation context when relevant

### Persistence locations

The phase must decide where heuristics live, at minimum across:

- runtime evidence
- trace summaries
- proposal-selection inputs
- benchmark reporting artifacts

### Summary-only compatibility

The design must support heuristic reconstruction when full arrays or rich
intermediates are unavailable.

## Implementation Considerations

- Runtime persistence should remain compact enough for e2e and optimize runs.
- The schema should work whether heuristics come directly from atoms or from
  deterministic support transforms over runtime summaries.
- The reporting surfaces should make debugging practical without flooding logs.

## Testing Strategy

- runtime artifact round-trip tests
- evaluator and trace integration tests
- summary-only reconstruction tests
- provenance validation tests

## Risks

- Runtime artifacts may grow too large.
- Confidence and uncertainty may not stay coherent across producers.
- Provenance may be incomplete when heuristics are reconstructed from summaries.

## Exit Criteria

Phase 4 is complete when:

- heuristic outputs are persisted in runtime artifacts
- proposal selection can consume them without family-specific hacks
- debugging surfaces can explain where the heuristic evidence came from
