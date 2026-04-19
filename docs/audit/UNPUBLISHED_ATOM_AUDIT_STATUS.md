# Unpublished Atom Audit Status

Generated from the live local Supabase replay on 2026-04-19T18:54:40.299870+00:00.

This document is a working debt register for every currently unpublished atom.

## Summary

- Total atoms in local catalog: `527`
- Publishable atoms: `428`
- Total non-publishable atoms in local catalog: `99`
- Remediation-excluded non-publishable atoms: `48`
- Non-publishable atoms remaining in matcher backlog: `51`


### Remediation Exclusions

- Source: `/Users/conrad/personal/sciona-atoms/REMEDIATION.md`
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

### Marginal Blocker Counts

- `description`: `43`
- `io_specs`: `43`
- `parameters`: `43`
- `publishable_rollup`: `46`
- `references`: `48`

### Top Exact Blocker Combinations

- `publishable_rollup,io_specs,parameters,description,references`: `43`
- `references`: `5`
- `publishable_rollup`: `3`

### Largest Non-Publishable Domains

- `inference`: `21`
- `expansion`: `10`
- `numpy`: `9`
- `particle_tracking`: `5`
- `state_estimation`: `4`
- `medical_imaging_3d`: `2`

## Status Legend

- `publishable_rollup`: no approved audit rollup satisfying the current publication rule
- `io_specs`: no atom IO spec rows
- `parameters`: no atom parameter rows
- `description`: no English low-jargon description
- `references`: no atom references rows
- `missing_row`: there is no audit rollup row for the atom yet

## expansion

- Non-publishable atoms: `10`
- Missing publishable rollup: `10`
- Missing IO specs: `10`
- Missing parameters: `10`
- Missing description: `10`
- Missing references: `10`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.expansion.sequential_filter.adapt_process_noise` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.check_observability` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.detect_filter_divergence` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.validate_innovation_whiteness` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_detect_measure.analyze_peak_threshold_sensitivity` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_detect_measure.check_event_rate_stationarity` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_filter.analyze_group_delay_variation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_filter.measure_passband_ripple` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_transform.analyze_window_leakage` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_transform.detect_spectral_aliasing` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## inference

- Non-publishable atoms: `21`
- Missing publishable rollup: `21`
- Missing IO specs: `21`
- Missing parameters: `21`
- Missing description: `21`
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

- Non-publishable atoms: `9`
- Missing publishable rollup: `9`
- Missing IO specs: `6`
- Missing parameters: `6`
- Missing description: `6`
- Missing references: `6`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.numpy.random.combinatorics_sampler` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.random.continuous_multivariate_sampler` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.random.default_rng` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.random.discrete_event_sampler` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.random.rand` | `missing` | `not_reviewed` | `unknown` | `unknown` | `misleading` | `publishable_rollup` |
| `sciona.atoms.numpy.random.uniform` | `missing` | `not_reviewed` | `unknown` | `unknown` | `misleading` | `publishable_rollup` |
| `sciona.atoms.numpy.search_sort.binary_search_insertion` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.search_sort.lexicographic_indirect_sort` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.search_sort.partial_sort_partition` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## particle_tracking

- Non-publishable atoms: `5`
- Missing publishable rollup: `0`
- Missing IO specs: `0`
- Missing parameters: `0`
- Missing description: `0`
- Missing references: `5`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.particle_tracking.helix_geometry.circle_from_three_points` | `approved` | `reviewed_with_limits` | `pass` | `pass_with_limits` | `acceptable_with_limits` | `references` |
| `sciona.atoms.particle_tracking.helix_geometry.helix_direction_from_two_points` | `approved` | `reviewed_with_limits` | `pass` | `pass_with_limits` | `acceptable_with_limits` | `references` |
| `sciona.atoms.particle_tracking.helix_geometry.helix_nearest_point_distance` | `approved` | `reviewed_with_limits` | `pass` | `pass_with_limits` | `acceptable_with_limits` | `references` |
| `sciona.atoms.particle_tracking.helix_geometry.helix_pitch_from_two_points` | `approved` | `reviewed_with_limits` | `pass` | `pass_with_limits` | `acceptable_with_limits` | `references` |
| `sciona.atoms.particle_tracking.helix_geometry.helix_pitch_least_squares` | `approved` | `reviewed_with_limits` | `pass` | `pass_with_limits` | `acceptable_with_limits` | `references` |

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
