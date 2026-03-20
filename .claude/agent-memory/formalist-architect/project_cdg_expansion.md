---
name: CDG Expansion Architecture Analysis
description: Formal analysis of topology expansion mechanism for CDGs - graph rewriting beyond node swapping, cross-domain generalizability
type: project
---

## CDG Expansion Architecture (2026-03-20)

### Current State
- Variant mutation (`variant_mutation.py`) only does in-place node swaps (same IOSpec arity required)
- DPO graph rewriter (`graph_rewriter.py`) has correct algebraic structure but `_find_match` is a stub returning None
- `_remove_lhs_minus_k` and `_add_rhs_minus_k` are also identity stubs
- `signal_detect_measure` skeleton is a 3-node linear chain: filter -> detect -> compute
- ECG HR error ~8 BPM, target <3 BPM; requires topology expansion (SQI, jump removal, etc.)

### Key Design Decisions
- Expansion must be deterministic-tool-driven, not unconstrained LLM generation
- Must compose with existing VariantFamily protocol
- DPO rewriter is the correct algebraic framework but needs: match implementation, expansion rules catalog, trigger diagnostics

### Formal Model
- Expansion = DPO rewrite where |R| > |L| (RHS has more nodes than LHS)
- Variant swap = DPO rewrite where |R| = |L| and interface K preserves all ports
- Both are instances of the same algebraic framework

### Cross-Domain Generalizability (analyzed 2026-03-20)
- DPO engine is fully domain-agnostic by construction (operates on CDGExport/AlgorithmicNode/DependencyEdge)
- Data model is parametric over domain via ConceptType enum (17 paradigms, 17 skeletons)
- VariantFamily protocol provides proven registry pattern for ExpansionFamily
- Three critical stubs must be implemented before any expansion rules work
- No RuntimeTrace protocol exists yet -- needed for diagnostic triggering
- VARIANT_FAMILIES is a hardcoded singleton tuple (signal-only); needs configurable registry
- Recommendation: define ExpansionDiagnostic/RuntimeTrace/ExpansionTrigger protocols first, then implement DPO core, then write domain-specific rules as data not classes
- Add `applicable_domains: frozenset[ConceptType]` to RewriteRule for multi-domain rules

### Key Files
- `/Users/conrad/personal/ageo-matcher/sciona/architect/graph_rewriter.py` - DPO engine (stubs)
- `/Users/conrad/personal/ageo-matcher/sciona/principal/variant_mutation.py` - current swap-only system
- `/Users/conrad/personal/ageo-matcher/sciona/architect/skeletons.py` - 17 skeleton templates
- `/Users/conrad/personal/ageo-matcher/sciona/architect/models.py` - ConceptType enum, AlgorithmicNode
- `/Users/conrad/personal/ageo-matcher/sciona/architect/strategy_classifier.py` - phrase_rules.json registry pattern
- `/Users/conrad/personal/ageo-matcher/sciona/runtime_signal_event_rate.py` - ECG HR runtime atoms
