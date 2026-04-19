# Publishability Review Batch Queue

Generated from `docs/audit/unpublished_atom_audit_status.json` on 2026-04-19T16:18:39.371698+00:00.

- Remaining unpublished atoms: `107`
- Remediation-excluded atoms: `30`
- Remaining worker batches: `26`

## Remediation Exclusions

- `e2e_ppg.kazemi_wrapper.wrapperpredictionsignalcomputation`: excluded `1` unpublished atoms via `exact` match
- `biosppy.svm_proc`: excluded `8` unpublished atoms via `prefix` match
- `molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.greedy_subgraph.greedy_maximum_subgraph`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.map_to_udg.graphtoudgmapping`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.quantum_solver.adiabaticquantumsampler`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.quantum_solver.quantumproblemdefinition`: excluded `1` unpublished atoms via `exact` match
- `molecular_docking.quantum_solver.solutionextraction`: excluded `1` unpublished atoms via `exact` match
- `institutional_quant_engine.fractional_diff.fractional_differentiator`: excluded `1` unpublished atoms via `exact` match
- `institutional_quant_engine.pin_model.pinlikelihoodevaluation`: excluded `1` unpublished atoms via `exact` match
- `institutional_quant_engine.pin_model.pinlikelihoodevaluator`: excluded `1` unpublished atoms via `exact` match
- `institutional_quant_engine.wash_trade.detect_wash_trade_rings`: excluded `1` unpublished atoms via `exact` match
- `pronto.torque_adjustment`: excluded `1` unpublished atoms via `prefix` match
- `physics.pasqal.docking.quantum_mwis_solver`: excluded `1` unpublished atoms via `exact` match
- `scipy.sparse_graph`: excluded `7` unpublished atoms via `prefix` match
- `scipy.stats.norm`: excluded `1` unpublished atoms via `exact` match

## Batches

### pubrev-008

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `5`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.initializehmcstate`, `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.leapfrogproposalkernel`, `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.collectposteriorchain`

### pubrev-010

- Repo: `sciona-atoms-fintech`
- Wave: `wave_1_audit_completion`
- Atoms: `5`
- Blocker class: `audit_rollup_only`
- Primary blocker pattern: `['publishable_rollup']`
- Representative atoms: `sciona.atoms.fintech.quantfin.char_func_option_d12.cf`, `sciona.atoms.fintech.quantfin.char_func_option_d12.charfuncoption`, `sciona.atoms.fintech.quantfin.char_func_option_d12.f`

### pubrev-013

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `7`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.expansion.signal_event_rate.assess_signal_quality`, `sciona.atoms.expansion.signal_event_rate.compute_event_rate`, `sciona.atoms.expansion.signal_event_rate.compute_event_rate_median_smoothed`

### pubrev-014

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `9`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel`, `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees.targetlogkerneloracle`, `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel`

### pubrev-015

- Repo: `sciona-atoms-physics`
- Wave: `wave_1_audit_completion`
- Atoms: `9`
- Blocker class: `audit_rollup_only`
- Primary blocker pattern: `['publishable_rollup']`
- Representative atoms: `sciona.atoms.numpy.fft.fft`, `sciona.atoms.numpy.fft.fftfreq`, `sciona.atoms.numpy.fft.fftn`

### pubrev-019

- Repo: `sciona-atoms-bio`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `7`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.bio.mint.apc_module.apccoreevaluation`, `sciona.atoms.bio.mint.axial_attention.row_self_attention`, `sciona.atoms.bio.mint.axial_attention.rowselfattention`

### pubrev-024

- Repo: `sciona-atoms-physics`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `6`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.numpy.random.combinatorics_sampler`, `sciona.atoms.numpy.random.continuous_multivariate_sampler`, `sciona.atoms.numpy.random.default_rng`

### pubrev-026

- Repo: `sciona-atoms-physics`
- Wave: `wave_1_audit_completion`
- Atoms: `2`
- Blocker class: `audit_rollup_only`
- Primary blocker pattern: `['publishable_rollup']`
- Representative atoms: `sciona.atoms.scipy.optimize.curve_fit`, `sciona.atoms.scipy.optimize.differential_evolution`, `sciona.atoms.scipy.optimize.linprog`

### pubrev-028

- Repo: `sciona-atoms-bio`
- Wave: `wave_1_audit_completion`
- Atoms: `5`
- Blocker class: `audit_rollup_only`
- Primary blocker pattern: `['publishable_rollup']`
- Representative atoms: `sciona.atoms.bio.molecular_docking.quantum_solver_d12.adiabaticpulseassembler`, `sciona.atoms.bio.molecular_docking.quantum_solver_d12.interactionboundscomputer`, `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumcircuitsampler`

### pubrev-030

- Repo: `sciona-atoms-physics`
- Wave: `wave_1_audit_completion`
- Atoms: `5`
- Blocker class: `audit_rollup_only`
- Primary blocker pattern: `['publishable_rollup']`
- Representative atoms: `sciona.atoms.numpy.arrays.array`, `sciona.atoms.numpy.arrays.dot`, `sciona.atoms.numpy.arrays.reshape`

### pubrev-031

- Repo: `sciona-atoms-physics`
- Wave: `wave_1_audit_completion`
- Atoms: `5`
- Blocker class: `audit_rollup_only`
- Primary blocker pattern: `['publishable_rollup']`
- Representative atoms: `sciona.atoms.numpy.emath.log`, `sciona.atoms.numpy.emath.log10`, `sciona.atoms.numpy.emath.logn`

### pubrev-037

- Repo: `sciona-atoms-bio`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `4`
- Blocker class: `metadata_plus_rollup`
- Primary blocker pattern: `['publishable_rollup', 'parameters', 'description']`
- Representative atoms: `sciona.atoms.bio.mint.fasta_dataset.dataset_item_retrieval`, `sciona.atoms.bio.mint.fasta_dataset.dataset_length_query`, `sciona.atoms.bio.mint.fasta_dataset.dataset_state_initialization`

### pubrev-044

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `4`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.expansion.sequential_filter.adapt_process_noise`, `sciona.atoms.expansion.sequential_filter.check_observability`, `sciona.atoms.expansion.sequential_filter.detect_filter_divergence`

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

### pubrev-050

- Repo: `sciona-atoms-physics`
- Wave: `wave_1_audit_completion`
- Atoms: `4`
- Blocker class: `audit_rollup_only`
- Primary blocker pattern: `['publishable_rollup']`
- Representative atoms: `sciona.atoms.numpy.linalg.det`, `sciona.atoms.numpy.linalg.inv`, `sciona.atoms.numpy.linalg.norm`

### pubrev-055

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `4`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.state_estimation.particle_filters.basic.filter_step_preparation_and_dispatch`, `sciona.atoms.state_estimation.particle_filters.basic.hypothesis_propagation_kernel`, `sciona.atoms.state_estimation.particle_filters.basic.likelihood_reweight_kernel`

### pubrev-058

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `3`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.advancedvi.core.evaluate_log_probability_density`, `sciona.atoms.inference.advancedvi.core.gradient_oracle_evaluation`, `sciona.atoms.inference.advancedvi.core.optimizationlooporchestration`

### pubrev-059

- Repo: `sciona-atoms-physics`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `3`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.numpy.search_sort.binary_search_insertion`, `sciona.atoms.numpy.search_sort.lexicographic_indirect_sort`, `sciona.atoms.numpy.search_sort.partial_sort_partition`

### pubrev-064

- Repo: `sciona-atoms-bio`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `2`
- Blocker class: `metadata_plus_rollup`
- Primary blocker pattern: `['publishable_rollup', 'parameters', 'description']`
- Representative atoms: `sciona.atoms.bio.hpdb.iterate_pdb_atoms`, `sciona.atoms.bio.hpdb.iterate_pdb_residues`

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

### pubrev-074

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `1`
- Blocker class: `metadata_plus_rollup`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description']`
- Representative atoms: `sciona.atoms.dynamic_programming.kadane.max_subarray`

### pubrev-075

- Repo: `sciona-atoms`
- Wave: `wave_2_metadata_and_llm_review`
- Atoms: `1`
- Blocker class: `full_metadata_missing`
- Primary blocker pattern: `['publishable_rollup', 'io_specs', 'parameters', 'description', 'references']`
- Representative atoms: `sciona.atoms.inference.bayes_rs.bernoulli.bernoulli_probabilistic_oracle`

- The canonical machine-readable queue is [publishability_review_batch_queue.json](/Users/conrad/personal/sciona-matcher/docs/audit/publishability_review_batch_queue.json).
