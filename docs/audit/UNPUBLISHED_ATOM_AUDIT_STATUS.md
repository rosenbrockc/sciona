# Unpublished Atom Audit Status

Generated from the live local Supabase replay on 2026-04-19T17:03:34.765377+00:00.

This document is a working debt register for every currently unpublished atom.

## Summary

- Total atoms in local catalog: `522`
- Publishable atoms: `402`
- Total non-publishable atoms in local catalog: `120`
- Remediation-excluded non-publishable atoms: `37`
- Non-publishable atoms remaining in matcher backlog: `83`


### Remediation Exclusions

- Source: `/Users/conrad/personal/sciona-atoms/REMEDIATION.md`
- `e2e_ppg.kazemi_wrapper.wrapperpredictionsignalcomputation`: excluded `1` unpublished atoms via `exact` match
- `biosppy.svm_proc`: excluded `8` unpublished atoms via `prefix` match
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

### Marginal Blocker Counts

- `description`: `66`
- `io_specs`: `61`
- `parameters`: `66`
- `publishable_rollup`: `83`
- `references`: `54`

### Top Exact Blocker Combinations

- `publishable_rollup,io_specs,parameters,description,references`: `54`
- `publishable_rollup`: `17`
- `publishable_rollup,io_specs,parameters,description`: `7`
- `publishable_rollup,parameters,description`: `5`

### Largest Non-Publishable Domains

- `inference`: `26`
- `numpy`: `23`
- `bio`: `13`
- `expansion`: `12`
- `state_estimation`: `4`
- `medical_imaging_3d`: `2`
- `scipy`: `2`
- `dynamic_programming`: `1`

## Status Legend

- `publishable_rollup`: no approved audit rollup satisfying the current publication rule
- `io_specs`: no atom IO spec rows
- `parameters`: no atom parameter rows
- `description`: no English low-jargon description
- `references`: no atom references rows
- `missing_row`: there is no audit rollup row for the atom yet

## bio

- Non-publishable atoms: `13`
- Missing publishable rollup: `13`
- Missing IO specs: `8`
- Missing parameters: `13`
- Missing description: `13`
- Missing references: `7`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.bio.hpdb.iterate_pdb_atoms` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.bio.hpdb.iterate_pdb_residues` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.bio.mint.apc_module.apccoreevaluation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.mint.axial_attention.row_self_attention` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.mint.axial_attention.rowselfattention` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.mint.encoding_dist_mat.encodedistancematrix` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.mint.fasta_dataset.dataset_item_retrieval` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.bio.mint.fasta_dataset.dataset_length_query` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.bio.mint.fasta_dataset.dataset_state_initialization` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.bio.mint.fasta_dataset.token_budget_batch_planning` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.bio.mint.incremental_attention.enable_incremental_state_configuration` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.mint.rotary_embedding.rotaryembedding_numpy` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.mint.rotary_embedding.rotaryembedding_torch` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## dynamic_programming

- Non-publishable atoms: `1`
- Missing publishable rollup: `1`
- Missing IO specs: `1`
- Missing parameters: `1`
- Missing description: `1`
- Missing references: `0`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.dynamic_programming.kadane.max_subarray` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |

## expansion

- Non-publishable atoms: `12`
- Missing publishable rollup: `12`
- Missing IO specs: `12`
- Missing parameters: `12`
- Missing description: `12`
- Missing references: `12`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.expansion.sequential_filter.adapt_process_noise` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.check_observability` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.detect_filter_divergence` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.validate_innovation_whiteness` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_detect_measure.analyze_peak_threshold_sensitivity` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_detect_measure.check_event_rate_stationarity` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.reject_outlier_intervals` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.remove_signal_jumps` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_filter.analyze_group_delay_variation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_filter.measure_passband_ripple` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_transform.analyze_window_leakage` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_transform.detect_spectral_aliasing` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## inference

- Non-publishable atoms: `26`
- Missing publishable rollup: `26`
- Missing IO specs: `26`
- Missing parameters: `26`
- Missing description: `26`
- Missing references: `21`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.inference.advancedvi.core.evaluate_log_probability_density` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.advancedvi.core.gradient_oracle_evaluation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.advancedvi.core.optimizationlooporchestration` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.bayes_rs.bernoulli.bernoulli_probabilistic_oracle` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.conjugate_priors.beta_binom.posterior_randmodel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.conjugate_priors.beta_binom.posterior_randmodel_weighted` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.jax_advi.optimize_advi.meanfieldvariationalfit` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.jax_advi.optimize_advi.posteriordrawsampling` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator.hamiltonianphasepointtransition` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.advancedhmc.integrator.temperingfactorcomputation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.advancedhmc.trajectory.buildnutstree` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.advancedhmc.trajectory.nutstransitionkernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees.metropolishastingstransitionkernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.aees.targetlogkerneloracle` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.de.build_de_transition_kernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.hmc.buildhmckernelfromlogdensityoracle` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.mala.mala_proposal_adjustment` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.mcmc_algos.dispatch_mcmc_algorithm` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.nuts.nuts_recursive_tree_build` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.rmhmc.buildrmhmctransitionkernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc.rwmh.constructrandomwalkmetropoliskernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.metropolishmctransition` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.runsamplingloop` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.collectposteriorchain` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts.nuts_recursive_tree_build` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts_llm.runnutstransitions` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |

## medical_imaging_3d

- Non-publishable atoms: `2`
- Missing publishable rollup: `2`
- Missing IO specs: `2`
- Missing parameters: `2`
- Missing description: `2`
- Missing references: `2`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.medical_imaging_3d.aggregation.casenet` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.medical_imaging_3d.aggregation.debug_atoms.casenet` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## numpy

- Non-publishable atoms: `23`
- Missing publishable rollup: `23`
- Missing IO specs: `6`
- Missing parameters: `6`
- Missing description: `6`
- Missing references: `6`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.numpy.arrays.array` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.arrays.dot` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.arrays.reshape` | `missing` | `not_reviewed` | `unknown` | `unknown` | `misleading` | `publishable_rollup` |
| `sciona.atoms.numpy.arrays.vstack` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.arrays.zeros` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.emath.log` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.emath.log10` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.emath.logn` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.emath.power` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.emath.sqrt` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.linalg.det` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.linalg.inv` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.linalg.norm` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.linalg.solve` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.random.combinatorics_sampler` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.random.continuous_multivariate_sampler` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.random.default_rng` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.random.discrete_event_sampler` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.random.rand` | `missing` | `not_reviewed` | `unknown` | `unknown` | `misleading` | `publishable_rollup` |
| `sciona.atoms.numpy.random.uniform` | `missing` | `not_reviewed` | `unknown` | `unknown` | `misleading` | `publishable_rollup` |
| `sciona.atoms.numpy.search_sort.binary_search_insertion` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.search_sort.lexicographic_indirect_sort` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.search_sort.partial_sort_partition` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## scipy

- Non-publishable atoms: `2`
- Missing publishable rollup: `2`
- Missing IO specs: `2`
- Missing parameters: `2`
- Missing description: `2`
- Missing references: `2`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.scipy.optimize.differential_evolution` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.scipy.optimize.shgo` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## state_estimation

- Non-publishable atoms: `4`
- Missing publishable rollup: `4`
- Missing IO specs: `4`
- Missing parameters: `4`
- Missing description: `4`
- Missing references: `4`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.state_estimation.particle_filters.basic.filter_step_preparation_and_dispatch` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.state_estimation.particle_filters.basic.hypothesis_propagation_kernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.state_estimation.particle_filters.basic.likelihood_reweight_kernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.state_estimation.particle_filters.basic.resample_and_hypothesis_distribution_projection` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
