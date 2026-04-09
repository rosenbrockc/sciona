# Phase 2: Heuristic-Producing Atom Metadata

## Purpose

Allow audited atoms to declare that some of their outputs are usable
heuristics, with the same rigor and auditability expected of executable atoms.

## Problem Statement

Today, heuristic-like information is often produced implicitly:

- through runtime summaries
- through family-local code
- through expansion diagnostics

That makes it difficult to audit where the evidence came from and whether it
should be trusted. If heuristic production is going to be a first-class part of
the framework, audited atoms need a way to expose heuristic outputs explicitly.

## Scope

This phase covers:

- atom metadata extensions for heuristic outputs
- output-level heuristic typing
- auditability requirements for heuristic producers
- compatibility rules for atoms that produce both executable outputs and
  heuristic outputs

## Non-Goals

This phase does not:

- define family registries
- implement runtime persistence
- define heuristic-driven ranking policy

## Deliverables

1. A metadata extension that allows atoms to mark outputs as heuristic-capable.
2. Documentation for the audit requirements of heuristic-producing outputs.
3. Validation rules for heuristic metadata.
4. A migration guide for local heuristic logic that should move into atom
   metadata or audited support assets.

## Required Design Decisions

### Output contract

Each heuristic-producing output should be able to declare:

- `heuristic_id`
- output path or field
- semantic kind
- expected value shape
- confidence semantics
- whether the output is advisory, gating, or structural
- provenance and uncertainty notes

### Producer categories

The system should support at least:

- executable atoms that emit heuristic outputs as side outputs
- diagnostic atoms whose main purpose is heuristic production
- support atoms that transform raw summaries into reusable heuristic signals

### Audit requirements

Heuristic-producing outputs should carry:

- de-jargonized explanation
- rationale
- known failure modes
- references when relevant
- example interpretation notes

## Implementation Considerations

- The metadata model should avoid forcing every atom to become a heuristic
  producer.
- Existing audited-atom review workflows should remain usable with minimal
  extension.
- The design should support eventual ownership in `../ageo-atoms`.

## Testing Strategy

- metadata validation tests
- fixture atoms with heuristic-producing outputs
- compatibility tests for atoms without heuristic metadata
- audit linting for required heuristic fields

## Risks

- Heuristic metadata may become an informal dumping ground.
- Producers may encode family assumptions in shared metadata.
- Too many heuristic outputs may create noisy proposal selection.

## Exit Criteria

Phase 2 is complete when:

- atom metadata can explicitly represent heuristic outputs
- validation enforces the new fields
- representative audited atoms demonstrate the pattern
- the schema stays generic enough for non-signal families
