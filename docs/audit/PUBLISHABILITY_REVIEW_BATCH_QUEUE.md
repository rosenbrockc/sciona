# Publishability Review Batch Queue

Generated from `docs/audit/unpublished_atom_audit_status.json` on 2026-04-19T19:35:15.794049+00:00.

- Remaining unpublished atoms: `29`
- Remediation-excluded atoms: `48`
- Remaining worker batches: `9`

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

### pubrev-045

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `2`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.expansion.signal_detect_measure.analyze_peak_threshold_sensitivity`, `sciona.atoms.expansion.signal_detect_measure.check_event_rate_stationarity`, `sciona.atoms.expansion.signal_detect_measure.estimate_false_positive_rate`

### pubrev-046

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `2`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.expansion.signal_filter.analyze_group_delay_variation`, `sciona.atoms.expansion.signal_filter.analyze_pole_stability`, `sciona.atoms.expansion.signal_filter.detect_transient_response`

### pubrev-047

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `2`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.expansion.signal_transform.analyze_window_leakage`, `sciona.atoms.expansion.signal_transform.check_inverse_reconstruction`, `sciona.atoms.expansion.signal_transform.detect_spectral_aliasing`

### pubrev-049

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `4`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator.hamiltonianphasepointtransition`, `sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator.temperingfactorcomputation`, `sciona.atoms.inference.mcmc_foundational.advancedhmc.trajectory.buildnutstree`

### pubrev-058

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `3`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.advancedvi.core.evaluate_log_probability_density`, `sciona.atoms.inference.advancedvi.core.gradient_oracle_evaluation`, `sciona.atoms.inference.advancedvi.core.optimizationlooporchestration`

### pubrev-066

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `2`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.conjugate_priors.beta_binom.posterior_randmodel`, `sciona.atoms.inference.conjugate_priors.beta_binom.posterior_randmodel_weighted`

### pubrev-067

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `2`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.jax_advi.optimize_advi.meanfieldvariationalfit`, `sciona.atoms.inference.jax_advi.optimize_advi.posteriordrawsampling`

### pubrev-075

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `1`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.bayes_rs.bernoulli.bernoulli_probabilistic_oracle`

- The canonical machine-readable queue is [publishability_review_batch_queue.json](/Users/conrad/personal/sciona-matcher/docs/audit/publishability_review_batch_queue.json).
