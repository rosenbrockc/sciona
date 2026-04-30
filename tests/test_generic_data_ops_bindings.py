import json
from pathlib import Path


SOLUTION_CDG_DIR = Path.home() / "personal" / "sciona-atoms" / "data" / "solution_cdgs"


EXPECTED_BINDINGS = {
    ("commonlit_readability_4th", "concatenation"): ("gap", "orchestration", None),
    ("cdiscount_image_classification_1st", "bson_chunking"): (
        "gap",
        "external_knowledge",
        None,
    ),
    ("amex_default_1st", "memory_optimization"): ("gap", "trivial_inline", None),
    (
        "amex_default_1st",
        "difference_features",
    ): (
        "active",
        "replace_stage",
        "sciona.atoms.ml.tabular.gradient_boosting.temporal_difference",
    ),
    (
        "jane_street_market_prediction_1st",
        "streaming_imputation",
    ): ("active", "replace_stage", "sciona.atoms.time_series_features.forward_fill"),
    (
        "rsna_pe_1st",
        "dicom_windowing",
    ): (
        "active",
        "replace_stage",
        "sciona.atoms.medical_imaging_3d.preprocessing.dicom_window",
    ),
    (
        "moa_prediction_1st",
        "label_smoothing",
    ): ("active", "replace_stage", "sciona.atoms.dl.loss.label_smoothing_ce"),
}


def _binding(solution_id: str, stage_id: str) -> dict:
    data = json.loads(
        (SOLUTION_CDG_DIR / f"{solution_id}_bindings.json").read_text()
    )
    bindings = {item["stage_id"]: item for item in data["bindings"]}
    return bindings[stage_id]


def test_generic_data_ops_bindings_are_reclassified_or_bound() -> None:
    for (solution_id, stage_id), (status, action_class, fqdn) in EXPECTED_BINDINGS.items():
        binding = _binding(solution_id, stage_id)

        assert binding["status"] == status
        assert binding["action_class"] == action_class
        assert binding.get("bound_artifact_fqdn") == fqdn


def test_target_scaling_uses_multi_atom_chain() -> None:
    binding = _binding("champs_molecular_properties_1st", "target_scaling")

    assert binding["status"] == "active"
    assert binding["action_class"] == "replace_stage"
    assert binding["bound_artifact_fqdns"] == [
        "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_fit",
        "sciona.atoms.ml.sklearn.preprocessing.standard_scaler_transform",
    ]
