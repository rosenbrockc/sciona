---
name: CDG Acyclicity Constraints Analysis
description: Formal analysis of 7 invariants requiring DAG acyclicity and recommendations for supporting recursive algorithms via FixedPoint combinator
type: project
---

## Seven Acyclicity-Dependent Invariants (ageo-matcher)

1. **INV-1 Topological Sort Termination** -- `toposort_nodes()` (synthesizer/toposort.py) uses Kahn's algorithm, crashes on cycles
2. **INV-2 Well-Founded Code Emission** -- assembler emits definitions in topological order; forward references are compile errors in all 3 target languages
3. **INV-3 Precision Gradient Propagation** -- `_compute_precision_gradients()` (ghost_sim.py) single-pass interval propagation assumes predecessors already computed
4. **INV-4 Ghost Simulation State Coherence** -- `_simulate_with_bayesian_checks()` sequential state enrichment
5. **INV-5 Shapley Value Computability** -- provenance/shapley.py explicit acyclicity precondition
6. **INV-6 Credit Assignment Separability** -- backprop.py assumes per-node cost is separable (no cyclic feedback)
7. **INV-7 Clearinghouse Prescreen** -- clearinghouse/prescreen.py rejects cyclic submissions

## MESSAGE_PASSING Exception (already implemented)

- Ghost simulator handles cycles via `_simulate_message_passing_iterative()` with max_iterations=100
- DeterministicCycleBreaker patches witness code with iteration caps, convergence checks, damping
- **Gap**: Assembler has NO cycle support -- MESSAGE_PASSING CDGs crash at `toposort_nodes` in assembler

## Recursive Algorithm Categories

- **Cat A (iterative-equivalent)**: Already handled by DP/filter/optimization skeletons as DAGs
- **Cat B (D&C with opaque recursive step)**: Handled via opaque atomic nodes matched to library functions
- **Cat C (structural recursion / self-reference)**: NOT supported; requires new architecture

## Recommendation: FixedPoint Combinator Node (Option C2)

New `ConceptType.FIXED_POINT` wrapping a cyclic sub-CDG body with `max_iterations` bound.
Assembler emits while-loop. Ghost sim already has 80% of machinery.
Preserves all invariants by keeping body as DAG, expressing cycle in combinator.
Avoids C3 (recursive code emission) which requires termination proofs in Lean/Coq.
