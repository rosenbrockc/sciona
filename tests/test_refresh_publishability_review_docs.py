from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "refresh_publishability_review_docs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "refresh_publishability_review_docs",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(
    fqdn: str,
    *,
    is_publishable: bool = False,
    review_status: str = "missing_row",
    trust_readiness: str = "missing_row",
    review_semantic_verdict: str = "missing_row",
    review_developer_semantics_verdict: str = "missing_row",
    overall_verdict: str = "missing_row",
    has_io_specs: bool = False,
    has_parameters: bool = False,
    has_description: bool = False,
    has_references: bool = False,
) -> dict[str, object]:
    return {
        "fqdn": fqdn,
        "is_publishable": is_publishable,
        "review_status": review_status,
        "trust_readiness": trust_readiness,
        "review_semantic_verdict": review_semantic_verdict,
        "review_developer_semantics_verdict": review_developer_semantics_verdict,
        "overall_verdict": overall_verdict,
        "has_io_specs": has_io_specs,
        "has_parameters": has_parameters,
        "has_description": has_description,
        "has_references": has_references,
    }


def test_load_remediation_exclusions_handles_prefix_and_exact_matches(tmp_path):
    module = _load_module()
    remediation_path = tmp_path / "REMEDIATION.md"
    remediation_path.write_text(
        "\n".join(
            [
                "# REMEDIATION",
                "",
                "## Signal Processing",
                "",
                "### `biosppy.svm_proc`",
                "",
                "## SciPy",
                "",
                "### `scipy.stats.norm`",
                "",
            ]
        )
        + "\n"
    )
    rows = [
        _row("sciona.atoms.signal_processing.biosppy.svm_proc.get_id_rates"),
        _row(
            "sciona.atoms.signal_processing.biosppy.svm_proc.cross_validation",
            is_publishable=True,
        ),
        _row("sciona.atoms.scipy.stats.norm"),
        _row("sciona.atoms.bio.mint.fasta_dataset.dataset_length_query"),
    ]

    exclusions = module._load_remediation_exclusions(rows, remediation_path)

    assert exclusions["excluded_unpublished_fqdns"] == [
        "sciona.atoms.scipy.stats.norm",
        "sciona.atoms.signal_processing.biosppy.svm_proc.get_id_rates",
    ]
    targets = {item["label"]: item for item in exclusions["matched_targets"]}
    assert targets["biosppy.svm_proc"]["match_type"] == "prefix"
    assert targets["biosppy.svm_proc"]["matched_atom_count"] == 2
    assert targets["biosppy.svm_proc"]["matched_unpublished_atom_count"] == 1
    assert targets["scipy.stats.norm"]["match_type"] == "exact"
    assert targets["scipy.stats.norm"]["matched_unpublished_atom_count"] == 1


def test_status_and_queue_drop_remediation_atoms(tmp_path, monkeypatch):
    module = _load_module()
    remediation_path = tmp_path / "REMEDIATION.md"
    remediation_path.write_text(
        "\n".join(
            [
                "# REMEDIATION",
                "",
                "## Signal Processing",
                "",
                "### `biosppy.svm_proc`",
                "",
            ]
        )
        + "\n"
    )
    queue_path = tmp_path / "publishability_review_batch_queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_id": "pubrev-biosppy",
                        "repo_owner": "sciona-atoms-signal",
                        "recommended_wave": "wave-a",
                        "blocker_class": "audit_rollup_only",
                        "primary_blocker_pattern": "publishable_rollup",
                        "representative_atoms": [
                            "sciona.atoms.signal_processing.biosppy.svm_proc.get_id_rates"
                        ],
                        "atoms": [
                            {
                                "fqdn": "sciona.atoms.signal_processing.biosppy.svm_proc.get_id_rates"
                            }
                        ],
                    },
                    {
                        "batch_id": "pubrev-bio",
                        "repo_owner": "sciona-atoms-bio",
                        "recommended_wave": "wave-b",
                        "blocker_class": "missing_metadata",
                        "primary_blocker_pattern": "publishable_rollup,parameters,description",
                        "representative_atoms": [
                            "sciona.atoms.bio.mint.fasta_dataset.dataset_length_query"
                        ],
                        "atoms": [
                            {
                                "fqdn": "sciona.atoms.bio.mint.fasta_dataset.dataset_length_query"
                            }
                        ],
                    },
                ],
                "special_slices": {
                    "largest_batches": [
                        {
                            "batch_id": "pubrev-biosppy",
                            "batch_key": "sciona.atoms.signal_processing.biosppy.svm_proc",
                        },
                        {
                            "batch_id": "pubrev-bio",
                            "batch_key": "sciona.atoms.bio.mint.fasta_dataset",
                        },
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )
    monkeypatch.setattr(module, "QUEUE_JSON_PATH", queue_path)

    rows = [
        _row("sciona.atoms.signal_processing.biosppy.svm_proc.get_id_rates"),
        _row("sciona.atoms.bio.mint.fasta_dataset.dataset_length_query"),
    ]
    totals = {
        "atoms": 2,
        "publishable_atoms": 0,
        "non_publishable_atoms": 2,
    }
    exclusions = module._load_remediation_exclusions(rows, remediation_path)
    payload = module._build_status_payload(
        rows,
        totals,
        "2026-04-17T00:00:00+00:00",
        exclusions,
    )

    assert payload["backlog_totals"]["review_backlog_non_publishable_atoms"] == 1
    assert payload["backlog_totals"]["remediation_excluded_non_publishable_atoms"] == 1
    assert payload["domains"] == [
        {
            "domain": "bio",
            "atom_count": 1,
            "missing_publishable_rollup": 1,
            "missing_io_specs": 1,
            "missing_parameters": 1,
            "missing_description": 1,
            "missing_references": 1,
            "atoms": [
                {
                    "fqdn": "sciona.atoms.bio.mint.fasta_dataset.dataset_length_query",
                    "domain": "bio",
                    "review_status": "missing_row",
                    "trust_readiness": "missing_row",
                    "semantic_verdict": "missing_row",
                    "developer_semantic_verdict": "missing_row",
                    "overall_verdict": "missing_row",
                    "blockers": [
                        "publishable_rollup",
                        "io_specs",
                        "parameters",
                        "description",
                        "references",
                    ],
                }
            ],
        }
    ]

    unpublished_fqdns = {
        atom["fqdn"]
        for domain in payload["domains"]
        for atom in domain["atoms"]
    }
    refreshed, queue_md = module._refresh_queue(
        unpublished_fqdns,
        "2026-04-17T00:00:00+00:00",
        payload["blocker_counts"],
        exclusions,
    )

    assert refreshed["batch_count"] == 1
    assert [batch["batch_id"] for batch in refreshed["batches"]] == ["pubrev-bio"]
    assert refreshed["totals"]["remediation_excluded_atoms"] == 1
    assert refreshed["special_slices"]["largest_batches"] == [
        {
            "batch_id": "pubrev-bio",
            "batch_key": "sciona.atoms.bio.mint.fasta_dataset",
        }
    ]
    assert "biosppy.svm_proc" in queue_md
    assert "pubrev-biosppy" not in queue_md
