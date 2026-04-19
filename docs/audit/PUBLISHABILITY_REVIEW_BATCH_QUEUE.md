# Publishability Review Batch Queue

Generated from `docs/audit/unpublished_atom_audit_status.json` on 2026-04-19T20:13:16.590994+00:00.

- Remaining unpublished atoms: `21`
- Remediation-excluded atoms: `48`
- Remaining worker batches: `1`

## Remediation Exclusions

- `mcmc_foundational.mini_mcmc`: excluded `5` unpublished atoms via `prefix` match
- `sciona.atoms.expansion.signal_event_rate`: excluded `2` unpublished atoms via `prefix` match
- `e2e_ppg.kazemi_wrapper.wrapperpredictionsignalcomputation`: excluded `1` unpublished atoms via `exact` match
- `biosppy.svm_proc`: excluded `8` unpublished atoms via `prefix` match
- `hpdb`: excluded `2` unpublished atoms via `prefix` match
- `sciona.atoms.bio.mint.axial_attention`: excluded `2` unpublished atoms via `prefix` match
- `molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.greedy_subgraph.greedy_maximum_subgraph`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.map_to_udg.graphtoudgmapping`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.quantum_solver.adiabaticquantumsampler`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.quantum_solver.quantumproblemdefinition`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.quantum_solver.solutionextraction`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.quantum_solver_d12`: excluded `5` unpublished atoms via `prefix` match
- `quantfin.tdma_solver_d12`: excluded `2` unpublished atoms via `prefix` match
- `institutional_quant_engine.fractional_diff.fractional_differentiator`: excluded `1` unpublished atoms via `exact` match
- `institutional_quant_engine.pin_model.pinlikelihoodevaluation`: excluded `1` unpublished atoms via `exact` match
- `institutional_quant_engine.pin_model.pinlikelihoodevaluator`: excluded `1` unpublished atoms via `exact` match
- `institutional_quant_engine.wash_trade.detect_wash_trade_rings`: excluded `1` unpublished atoms via `exact` match
- `pronto.torque_adjustment`: excluded `1` unpublished atoms via `prefix` match
- `physics.pasqal.docking.quantum_mwis_solver`: excluded `1` unpublished atoms via `exact` match
- `scipy.sparse_graph`: excluded `7` unpublished atoms via `prefix` match
- `scipy.stats.norm`: excluded `1` unpublished atoms via `exact` match

## Batches

### pubrev-014

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `9`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel`, `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees.targetlogkerneloracle`, `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel`

- The canonical machine-readable queue is [publishability_review_batch_queue.json](/Users/conrad/personal/sciona-matcher/docs/audit/publishability_review_batch_queue.json).
