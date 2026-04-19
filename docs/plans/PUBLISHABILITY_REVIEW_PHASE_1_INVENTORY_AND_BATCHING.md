# Publishability Review Phase 1: Inventory And Batching

## Goal

Turn the unpublished backlog into execution batches that workers can own
without stepping on each other.

## Inputs

- [UNPUBLISHED_ATOM_AUDIT_STATUS.md](/Users/conrad/personal/sciona-atoms/docs/audit/UNPUBLISHED_ATOM_AUDIT_STATUS.md)
- [unpublished_atom_audit_status.json](/Users/conrad/personal/sciona-atoms/docs/audit/unpublished_atom_audit_status.json)
- current license metadata replay summaries

## Tasks

1. Classify every unpublished atom by exact blocker set.
2. Split the backlog into:
   - references-only
   - audit-rollup-only
   - full-metadata-missing
   - license/provenance-missing
   - mixed hard cases
3. Group atoms into provider-family batches sized for one worker.
4. Mark high-leverage early batches:
   - references-only approved atoms
   - audit-rollup-only atoms with complete metadata
   - families with strong shared provenance
5. Produce a family queue with:
   - repo owner
   - atom count
   - blocker pattern
   - expected evidence sources
   - likely need for web browsing
   - likely need for human signoff

## Output

A canonical execution inventory that later phases use as their source of truth.

Current generated outputs:

- [PUBLISHABILITY_REVIEW_BATCH_QUEUE.md](/Users/conrad/personal/sciona-atoms/docs/audit/PUBLISHABILITY_REVIEW_BATCH_QUEUE.md)
- [publishability_review_batch_queue.json](/Users/conrad/personal/sciona-atoms/docs/audit/publishability_review_batch_queue.json)

## Parallelization

Do not parallelize this phase. One integrator should freeze the batch map
before workers start generating review bundles.

## Exit Criteria

- every unpublished atom belongs to exactly one worker-owned batch
- every batch has a named primary blocker pattern
- easy-win batches are identified explicitly
