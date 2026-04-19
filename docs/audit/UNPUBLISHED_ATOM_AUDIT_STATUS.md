# Unpublished Atom Audit Status

Generated from the live local Supabase replay on 2026-04-19T20:13:16.590994+00:00.

This document is a working debt register for every currently unpublished atom.

## Summary

- Total atoms in local catalog: `537`
- Publishable atoms: `468`
- Total non-publishable atoms in local catalog: `69`
- Remediation-excluded non-publishable atoms: `48`
- Non-publishable atoms remaining in matcher backlog: `21`


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

- `description`: `21`
- `io_specs`: `11`
- `parameters`: `21`
- `publishable_rollup`: `21`
- `references`: `11`

### Top Exact Blocker Combinations

- `publishable_rollup,io_specs,parameters,description,references`: `11`
- `publishable_rollup,parameters,description`: `10`

### Largest Non-Publishable Domains

- `ml`: `10`
- `inference`: `9`
- `medical_imaging_3d`: `2`

## Status Legend

- `publishable_rollup`: no approved audit rollup satisfying the current publication rule
- `io_specs`: no atom IO spec rows
- `parameters`: no atom parameter rows
- `description`: no English low-jargon description
- `references`: no atom references rows
- `missing_row`: there is no audit rollup row for the atom yet

## inference

- Non-publishable atoms: `9`
- Missing publishable rollup: `9`
- Missing IO specs: `9`
- Missing parameters: `9`
- Missing description: `9`
- Missing references: `9`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
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

## ml

- Non-publishable atoms: `10`
- Missing publishable rollup: `10`
- Missing IO specs: `0`
- Missing parameters: `10`
- Missing description: `10`
- Missing references: `0`

| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| `sciona.atoms.ml.sklearn.covariance.empirical_covariance` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.covariance.shrunk_covariance` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.feature_selection.chi2` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.feature_selection.f_classif` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.feature_selection.f_regression` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.feature_selection.r_regression` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.images.extract_patches_2d` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.images.grid_to_graph` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.images.img_to_graph` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
| `sciona.atoms.ml.sklearn.images.reconstruct_from_patches_2d` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `missing_row` | `publishable_rollup, parameters, description` |
