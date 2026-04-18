# Unpublished Atom Audit Status

Generated from the live local Supabase replay on 2026-04-18T01:26:11.747527+00:00.

This document is a working debt register for every currently unpublished atom.

## Summary

- Total atoms in local catalog: `504`
- Publishable atoms: `290`
- Total non-publishable atoms in local catalog: `214`
- Remediation-excluded non-publishable atoms: `17`
- Non-publishable atoms remaining in matcher backlog: `197`


### Remediation Exclusions

- Source: `/Users/conrad/personal/sciona-atoms/REMEDIATION.md`
- `biosppy.svm_proc`: excluded `8` unpublished atoms via `prefix` match
- `pronto.torque_adjustment`: excluded `1` unpublished atoms via `prefix` match
- `scipy.sparse_graph`: excluded `7` unpublished atoms via `prefix` match
- `scipy.stats.norm`: excluded `1` unpublished atoms via `exact` match

### Marginal Blocker Counts

- `description`: `143`
- `io_specs`: `133`
- `parameters`: `143`
- `publishable_rollup`: `197`
- `references`: `118`

### Top Exact Blocker Combinations

- `publishable_rollup,io_specs,parameters,description,references`: `117`
- `publishable_rollup`: `54`
- `publishable_rollup,io_specs,parameters,description`: `16`
- `publishable_rollup,parameters,description`: `9`
- `publishable_rollup,parameters,description,references`: `1`

### Largest Non-Publishable Domains

- `bio`: `46`
- `inference`: `34`
- `numpy`: `32`
- `expansion`: `31`
- `fintech`: `31`
- `physics`: `10`
- `state_estimation`: `6`
- `signal_processing`: `4`
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

- Non-publishable atoms: `46`
- Missing publishable rollup: `46`
- Missing IO specs: `14`
- Missing parameters: `19`
- Missing description: `19`
- Missing references: `11`

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
| `sciona.atoms.bio.molecular_docking.add_quantum_link.addquantumlink` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.molecular_docking.build_complementary.constructcomplementarygraph` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.molecular_docking.build_interaction_graph.networkx_weighted_graph_materialization` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.build_interaction_graph.pair_distance_compatibility_check` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.build_interaction_graph.weighted_interaction_edge_derivation` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.greedy_mapping_d12.construct_mapping_state_via_greedy_expansion` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.greedy_mapping_d12.init_problem_context` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.greedy_mapping_d12.orchestrate_generation_and_validate` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.greedy_subgraph.greedy_maximum_subgraph` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.molecular_docking.map_to_udg.graphtoudgmapping` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.aggregate_maximum_distance_as_bandwidth` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.build_sparse_graph_view` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.build_threshold_search_space` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.compute_absolute_weighted_index_distances` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.compute_symmetric_bandwidth_reducing_order` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.enforce_threshold_sparsity` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.enumerate_threshold_based_permutations` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.extract_final_permutation` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.initialize_reduction_state` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.propose_greedy_permutation_step` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.select_minimum_bandwidth_permutation` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.update_state_with_improvement_criterion` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.validate_square_matrix_shape` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.validate_symmetric_input_dense` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.bio.molecular_docking.minimize_bandwidth.validate_symmetric_input_thresholded` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.bio.molecular_docking.quantum_solver.adiabaticquantumsampler` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.quantum_solver.quantumproblemdefinition` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.quantum_solver.solutionextraction` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.quantum_solver_d12.adiabaticpulseassembler` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.quantum_solver_d12.interactionboundscomputer` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumcircuitsampler` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolutionextractor` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |
| `sciona.atoms.bio.molecular_docking.quantum_solver_d12.quantumsolverorchestrator` | `reviewed_pending` | `blocked_on_uncertainty_backfill` | `unknown` | `unknown` | `unknown` | `publishable_rollup` |

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

- Non-publishable atoms: `31`
- Missing publishable rollup: `31`
- Missing IO specs: `31`
- Missing parameters: `31`
- Missing description: `31`
- Missing references: `31`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.expansion.divide_and_conquer.check_recursion_depth` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.divide_and_conquer.detect_subproblem_overlap` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.divide_and_conquer.measure_split_balance` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.divide_and_conquer.profile_merge_cost` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.graph_signal_processing.analyze_spectral_gap` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.graph_signal_processing.check_laplacian_symmetry` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.kalman_filter.analyze_kalman_gain_magnitude` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.kalman_filter.check_innovation_consistency` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.kalman_filter.check_state_smoothness` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.kalman_filter.validate_covariance_pd` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.particle_filter.analyze_particle_diversity` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.particle_filter.check_resampling_quality` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.particle_filter.monitor_effective_sample_size` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.particle_filter.track_weight_variance` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.adapt_process_noise` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.check_observability` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.detect_filter_divergence` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.sequential_filter.validate_innovation_whiteness` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_detect_measure.analyze_peak_threshold_sensitivity` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_detect_measure.check_event_rate_stationarity` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.assess_signal_quality` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.compute_event_rate_median_smoothed` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.compute_event_rate_smoothed` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.detect_peaks_in_signal` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.estimate_event_rate_from_signal` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.reject_outlier_intervals` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_event_rate.remove_signal_jumps` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_filter.analyze_group_delay_variation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_filter.measure_passband_ripple` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_transform.analyze_window_leakage` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.expansion.signal_transform.detect_spectral_aliasing` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## fintech

- Non-publishable atoms: `31`
- Missing publishable rollup: `31`
- Missing IO specs: `27`
- Missing parameters: `27`
- Missing description: `27`
- Missing references: `19`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.fintech.institutional_quant_engine.copula_dependence.simulate_copula_dependence` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.dynamic_hedge.kalman_hedge_ratio` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.evt_model.fit_gpd_tail` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.fractional_diff.fractional_differentiator` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.hawkes_process.hawkesprocesssimulator` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.hawkes_process.sample_hawkes_event_trajectory` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.heston_model.hestonpathsampler` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.heston_model.simulate_heston_paths` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.hierarchical_risk_parity.compute_hrp_weights` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.hierarchical_risk_parity.hrppipelinerun` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.order_flow_imbalance.orderflowimbalanceevaluation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.pin_model.pinlikelihoodevaluation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.pin_model.pinlikelihoodevaluator` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.queue_estimator.initializeorderstate` | `reviewed_pending` | `conditional` | `unknown` | `pass` | `unknown` | `publishable_rollup` |
| `sciona.atoms.fintech.institutional_quant_engine.queue_estimator.updatequeueontrade` | `reviewed_pending` | `conditional` | `unknown` | `pass` | `unknown` | `publishable_rollup` |
| `sciona.atoms.fintech.institutional_quant_engine.supply_chain.propagate_supply_shock` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.triangular_arbitrage.detect_triangular_arbitrage` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.institutional_quant_engine.wash_trade.detect_wash_trade_rings` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12.insertcf_recursive` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12.insertcf_singleton` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12.insertcflist_fold` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12.insertcflist_fold_alt` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12.process_base_case` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12.process_with_cashflows_only` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12.process_with_observation_only` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12.process_with_pending_cashflows` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.fintech.quantfin.montecarlo.quick_sim_anti` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.quantfin.montecarlo.run_simulation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.quantfin.montecarlo.run_simulation_anti` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.fintech.quantfin.tdma_solver_d12.cotraversevec` | `reviewed_pending` | `conditional` | `unknown` | `pass` | `unknown` | `publishable_rollup` |
| `sciona.atoms.fintech.quantfin.tdma_solver_d12.tdmasolver` | `reviewed_pending` | `conditional` | `unknown` | `pass` | `unknown` | `publishable_rollup` |

## inference

- Non-publishable atoms: `34`
- Missing publishable rollup: `34`
- Missing IO specs: `33`
- Missing parameters: `34`
- Missing description: `34`
- Missing references: `34`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.inference.advancedvi.core.evaluate_log_probability_density` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.advancedvi.core.gradient_oracle_evaluation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.advancedvi.core.optimizationlooporchestration` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.bayes_rs.bernoulli.bernoulli_probabilistic_oracle` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.belief_propagation.loopy_bp.initialize_message_passing_state` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description, references` |
| `sciona.atoms.inference.belief_propagation.loopy_bp.run_loopy_message_passing_and_belief_query` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
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
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.initializehmcstate` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.leapfrogproposalkernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.metropolishmctransition` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc.runsamplingloop` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.collectposteriorchain` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.hamiltoniantransitionkernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.initializehmckernelstate` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.hmc_llm.initializesamplerrng` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts.nuts_recursive_tree_build` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts_llm.initializenutsstate` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.inference.mcmc_foundational.mini_mcmc.nuts_llm.runnutstransitions` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## numpy

- Non-publishable atoms: `32`
- Missing publishable rollup: `32`
- Missing IO specs: `9`
- Missing parameters: `9`
- Missing description: `9`
- Missing references: `9`

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
| `sciona.atoms.numpy.fft.fft` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.fft.fftfreq` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.fft.fftn` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.fft.fftshift` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.fft.hfft` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.fft.ifft` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.fft.ifftn` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.numpy.fft.irfft` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
| `sciona.atoms.numpy.fft.rfft` | `missing` | `not_reviewed` | `unknown` | `unknown` | `acceptable_with_limits` | `publishable_rollup` |
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

## physics

- Non-publishable atoms: `10`
- Missing publishable rollup: `10`
- Missing IO specs: `6`
- Missing parameters: `10`
- Missing description: `10`
- Missing references: `2`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.physics.astroflow.dedispersionkernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.physics.pasqal.docking.graph_transformer` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.physics.pasqal.docking.quantum_mwis_solver` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.physics.pasqal.docking.sub_graph_embedder` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description` |
| `sciona.atoms.physics.pulsar_folding.dm_can.dm_candidate_filter` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.physics.pulsar_folding.dm_can_brute_force` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.physics.pulsar_folding.spline_bandpass_correction` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.physics.tempo_jl.apply_offsets._zero_offset` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.physics.tempo_jl.graph_time_scale_management` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.physics.tempo_jl.high_precision_duration` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |

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

## signal_processing

- Non-publishable atoms: `4`
- Missing publishable rollup: `4`
- Missing IO specs: `4`
- Missing parameters: `4`
- Missing description: `4`
- Missing references: `4`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.signal_processing.e2e_ppg.heart_cycle.detect_heart_cycles` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.signal_processing.e2e_ppg.heart_cycle.heart_cycle_detection` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.signalarraynormalization` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.signal_processing.e2e_ppg.kazemi_wrapper.wrapperpredictionsignalcomputation` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |

## state_estimation

- Non-publishable atoms: `6`
- Missing publishable rollup: `6`
- Missing IO specs: `6`
- Missing parameters: `6`
- Missing description: `6`
- Missing references: `6`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.state_estimation.kalman_filters.track_linear_gaussian_state` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.state_estimation.particle_filters.basic.filter_step_preparation_and_dispatch` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.state_estimation.particle_filters.basic.hypothesis_propagation_kernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.state_estimation.particle_filters.basic.likelihood_reweight_kernel` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.state_estimation.particle_filters.basic.resample_and_hypothesis_distribution_projection` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
| `sciona.atoms.state_estimation.particle_filters.track_particle_hidden_state` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, io_specs, parameters, description, references` |
