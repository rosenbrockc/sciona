# Publishability Review Batch Queue

Generated from `docs/audit/unpublished_atom_audit_status.json` on 2026-04-16T14:40:18.739760+00:00.

## Headline

- Unpublished atoms: `444`
- Frozen worker batches: `79`
- References-only atom slice: `12` atoms
- Audit-rollup-only atom slice: `214` atoms

## Immediate High-Leverage Slices

1. References-only atoms: `12` atoms already audit-approved and blocked only on references.
2. Audit-rollup-only atoms: `214` atoms that already have the rest of the metadata surface and mainly need provider review completion.
3. License/provenance focus batches: external-wrapper families in physics/numpy/scipy and finance where public publication may still need provenance or policy confirmation.

### References-Only Atom Slice

- `sciona.atoms.expansion.graph_signal_processing.validate_filter_response`
- `sciona.atoms.expansion.graph_signal_processing.validate_graph_connectivity`
- `sciona.atoms.expansion.signal_detect_measure.estimate_false_positive_rate`
- `sciona.atoms.expansion.signal_detect_measure.estimate_snr`
- `sciona.atoms.expansion.signal_event_rate.compute_event_rate`
- `sciona.atoms.expansion.signal_event_rate.filter_signal_for_detection`
- `sciona.atoms.expansion.signal_filter.analyze_pole_stability`
- `sciona.atoms.expansion.signal_filter.detect_transient_response`
- `sciona.atoms.expansion.signal_transform.check_inverse_reconstruction`
- `sciona.atoms.expansion.signal_transform.validate_parseval_energy`
- `sciona.atoms.signal_processing.e2e_ppg.reconstruction.gan_patch_reconstruction`
- `sciona.atoms.signal_processing.e2e_ppg.reconstruction.windowed_signal_reconstruction`

## Recommended Worker Waves

- `wave_1_easy_win`: `0` batches
- `wave_1_audit_completion`: `33` batches
- `wave_2_metadata_and_llm_review`: `45` batches
- `wave_3_residual_policy_or_provenance`: `1` batches

## Priority Queue

| Batch | Repo | Atoms | Class | Wave | Web | Human | Scope |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `pubrev-001` | `sciona-atoms-fintech` | `27` | `full_metadata_missing` | `wave_2_metadata_and_llm_review` | `high` | `medium` | `sciona.atoms.fintech.institutional_quant_engine.__remainder__` |
| `pubrev-002` | `sciona-atoms-fintech` | `18` | `audit_rollup_only` | `wave_1_audit_completion` | `medium` | `medium` | `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12` |
| `pubrev-003` | `sciona-atoms-bio` | `15` | `audit_rollup_only` | `wave_1_audit_completion` | `medium` | `low` | `sciona.atoms.bio.molecular_docking.minimize_bandwidth` |
| `pubrev-004` | `sciona-atoms-robotics` | `14` | `audit_rollup_only` | `wave_1_audit_completion` | `high` | `low` | `sciona.atoms.robotics.pronto.__remainder__` |
| `pubrev-005` | `sciona-atoms-bio` | `13` | `audit_rollup_only` | `wave_1_audit_completion` | `high` | `medium` | `sciona.atoms.bio.molecular_docking.__remainder__` |
| `pubrev-006` | `sciona-atoms-fintech` | `13` | `audit_rollup_only` | `wave_1_audit_completion` | `medium` | `medium` | `sciona.atoms.fintech.quantfin.rng_skip_d12` |
| `pubrev-007` | `sciona-atoms-signal` | `12` | `audit_rollup_only` | `wave_1_audit_completion` | `medium` | `low` | `sciona.atoms.signal_processing.biosppy.__remainder__` |
| `pubrev-008` | `sciona-atoms` | `11` | `full_metadata_missing` | `wave_2_metadata_and_llm_review` | `high` | `low` | `sciona.atoms.inference.mcmc_foundational.mini_mcmc` |
| `pubrev-009` | `sciona-atoms-signal` | `11` | `full_metadata_missing` | `wave_2_metadata_and_llm_review` | `high` | `low` | `sciona.atoms.signal_processing.e2e_ppg.__remainder__` |
| `pubrev-010` | `sciona-atoms-fintech` | `10` | `audit_rollup_only` | `wave_1_audit_completion` | `high` | `medium` | `sciona.atoms.fintech.quantfin.__remainder__` |
| `pubrev-011` | `sciona-atoms-physics` | `10` | `metadata_plus_rollup` | `wave_2_metadata_and_llm_review` | `medium` | `low` | `sciona.atoms.physics.tempo_jl.find_month` |
| `pubrev-012` | `sciona-atoms-physics` | `10` | `metadata_plus_rollup` | `wave_2_metadata_and_llm_review` | `medium` | `low` | `sciona.atoms.physics.tempo_jl.jd2cal` |
| `pubrev-013` | `sciona-atoms` | `9` | `full_metadata_missing` | `wave_2_metadata_and_llm_review` | `high` | `low` | `sciona.atoms.expansion.signal_event_rate.__remainder__` |
| `pubrev-014` | `sciona-atoms` | `9` | `full_metadata_missing` | `wave_2_metadata_and_llm_review` | `high` | `low` | `sciona.atoms.inference.mcmc_foundational.kthohr_mcmc` |
| `pubrev-015` | `sciona-atoms-physics` | `9` | `audit_rollup_only` | `wave_1_audit_completion` | `high` | `medium` | `sciona.atoms.numpy.fft.__remainder__` |
| `pubrev-016` | `sciona-atoms-robotics` | `8` | `audit_rollup_only` | `wave_1_audit_completion` | `medium` | `low` | `sciona.atoms.robotics.rust_robotics.longitudinal_dynamics` |
| `pubrev-017` | `sciona-atoms-signal` | `8` | `audit_rollup_only` | `wave_1_audit_completion` | `medium` | `low` | `sciona.atoms.signal_processing.biosppy.ecg_detectors` |
| `pubrev-018` | `sciona-atoms-signal` | `8` | `audit_rollup_only` | `wave_1_audit_completion` | `medium` | `low` | `sciona.atoms.signal_processing.biosppy.svm_proc` |
| `pubrev-019` | `sciona-atoms-bio` | `7` | `full_metadata_missing` | `wave_2_metadata_and_llm_review` | `high` | `low` | `sciona.atoms.bio.mint.__remainder__` |
| `pubrev-020` | `sciona-atoms-physics` | `7` | `audit_rollup_only` | `wave_1_audit_completion` | `high` | `medium` | `sciona.atoms.numpy.polynomial.__remainder__` |

## License/Provenance Focus Batches

- `pubrev-001` `sciona.atoms.fintech.institutional_quant_engine.__remainder__` (`sciona-atoms-fintech`, `27` atoms, human signoff `medium`)
- `pubrev-002` `sciona.atoms.fintech.quantfin.monte_carlo_anti_d12` (`sciona-atoms-fintech`, `18` atoms, human signoff `medium`)
- `pubrev-005` `sciona.atoms.bio.molecular_docking.__remainder__` (`sciona-atoms-bio`, `13` atoms, human signoff `medium`)
- `pubrev-006` `sciona.atoms.fintech.quantfin.rng_skip_d12` (`sciona-atoms-fintech`, `13` atoms, human signoff `medium`)
- `pubrev-010` `sciona.atoms.fintech.quantfin.__remainder__` (`sciona-atoms-fintech`, `10` atoms, human signoff `medium`)
- `pubrev-011` `sciona.atoms.physics.tempo_jl.find_month` (`sciona-atoms-physics`, `10` atoms, human signoff `low`)
- `pubrev-012` `sciona.atoms.physics.tempo_jl.jd2cal` (`sciona-atoms-physics`, `10` atoms, human signoff `low`)
- `pubrev-015` `sciona.atoms.numpy.fft.__remainder__` (`sciona-atoms-physics`, `9` atoms, human signoff `medium`)
- `pubrev-020` `sciona.atoms.numpy.polynomial.__remainder__` (`sciona-atoms-physics`, `7` atoms, human signoff `medium`)
- `pubrev-022` `sciona.atoms.scipy.signal.__remainder__` (`sciona-atoms-physics`, `7` atoms, human signoff `medium`)
- `pubrev-023` `sciona.atoms.scipy.sparse_graph.__remainder__` (`sciona-atoms-physics`, `7` atoms, human signoff `medium`)
- `pubrev-024` `sciona.atoms.numpy.random.__remainder__` (`sciona-atoms-physics`, `6` atoms, human signoff `medium`)
- `pubrev-026` `sciona.atoms.scipy.optimize.__remainder__` (`sciona-atoms-physics`, `6` atoms, human signoff `medium`)
- `pubrev-028` `sciona.atoms.bio.molecular_docking.quantum_solver_d12` (`sciona-atoms-bio`, `5` atoms, human signoff `medium`)
- `pubrev-029` `sciona.atoms.fintech.quantfin.local_vol_d12` (`sciona-atoms-fintech`, `5` atoms, human signoff `medium`)

## Notes

- The canonical machine-readable queue is [publishability_review_batch_queue.json](/Users/conrad/personal/sciona-matcher/docs/audit/publishability_review_batch_queue.json).
- Family batches are frozen for worker ownership, but the references-only slice should be handled as a cross-batch early win because those atoms are already approved on audit semantics.
- Batches with `likely_human_signoff = low` should be attempted end to end with deterministic checks, LLM review, and targeted browsing before asking a human anything.
- `numpy` and `scipy` batches are explicitly called out as provenance-sensitive even when the semantic review itself is easy.
