# OpenVaccine worker contract

The published Tier 3 execution requires a provisioned macOS arm64 Python 3.13 worker matching `requirements/openvaccine-execution.txt`. The environment validator checks the declared dependency closure, imported NumPy/pandas/TensorFlow/tf-keras versions, legacy Keras selection, source identities and exact folding executable/coefficients. A clean installation and unrelated packages in the surrounding environment are outside this qualification.

Set `TF_USE_LEGACY_KERAS=1` before starting the process. Provision the reviewed dependencies through the worker environment; when using an installation target, put that target first on `PYTHONPATH` before startup. Keep this overlay in a dedicated worker process because its package versions and global random seed may differ from other pipelines.

Provide these software locations through environment configuration:

- `SCIONA_OPENVACCINE_SOURCE_DIR`: pinned DasLab source with its license.
- `SCIONA_OPENVACCINE_FOLD_BINARY`: the exact reviewed CONTRAfold-SE executable.
- `SCIONA_OPENVACCINE_FOLD_PARAMETERS`: pinned separately licensed EternaFold coefficients.

Run `scripts/validate_openvaccine_environment.py` with the repository virtual environment before execution. The recorded build flags and compiler identity are in `competition_openvaccine_folding_engine.json`; exact executable identity is enforced, so rebuilding on another platform does not automatically qualify that binary.

Pass private populations and training controls as runtime payloads. The generated example builder in `scripts/openvaccine_synthetic.py` demonstrates the contract without using real records. Never store actual payloads, targets, predictions or checkpoints in repository evidence. The graph validator suppresses intermediate-value persistence and uses a temporary run directory; production callers must configure private run storage appropriately.

The scope and exclusions in `competition_openvaccine_semantic_review.json` are part of the approval. The original intake CDGs remain drafts. Their provenance links do not certify the original historical recipe, final blend or accuracy.
