"""Versioned ARC runtime using a locally provisioned, reviewed solver bundle.

Runtime grids and predictions are private data. No payload paths, downloads or
implicit shallow search. SCIONA_ARC_BUILD_DIR is deployment configuration only.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

from sciona.arc_contract import Task, validate_task
from sciona.arc_schedule import SOURCE_MEMORY_MIB, SOURCE_SECONDS, run_schedule

IDENTITY_SHA256 = "5f011c13990165f5ae8e0a7028dd4e28b20a55f0faec837b18957bea1e014b66"


@dataclass(frozen=True)
class Prepared:
    task: Task
    seconds: float
    memory_mib: float


def _keys(value, fields):
    if type(value) is not dict or value.keys() != set(fields):
        raise ValueError("Exact ARC runtime fields required")


def _limits(seconds, memory):
    for value, maximum in [(seconds, SOURCE_SECONDS), (memory, SOURCE_MEMORY_MIB)]:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("ARC budget must be positive, finite, and within source defaults")


def prepare(payload) -> Prepared:
    _keys(payload, ("version", "task", "budget"))
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise ValueError("Unsupported ARC runtime version")
    budget = payload["budget"]
    _keys(budget, ("seconds", "memory_mib"))
    _limits(budget["seconds"], budget["memory_mib"])
    return Prepared(validate_task(payload["task"]), float(budget["seconds"]), float(budget["memory_mib"]))


def reviewed_binary() -> tuple[Path, str]:
    raw = Path(__file__).with_name("arc_build_identity.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != IDENTITY_SHA256:
        raise ValueError("ARC reviewed identity mismatch")
    identity = json.loads(raw)
    configured = os.environ.get("SCIONA_ARC_BUILD_DIR")
    if not configured:
        raise RuntimeError("Provision the reviewed ARC bundle through SCIONA_ARC_BUILD_DIR")
    directory = Path(configured)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        license_bytes = (directory / "LICENSE").read_bytes()
        binary = directory / "solver"
        binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        raise RuntimeError("ARC build bundle is unavailable or invalid") from error
    if manifest != identity["manifest"] or binary_hash != manifest["binary_sha256"]:
        raise ValueError("ARC build bundle differs from reviewed identity")
    if hashlib.sha256(license_bytes).hexdigest() != identity["license_sha256"]:
        raise ValueError("ARC build license notice mismatch")
    return binary, binary_hash


def execute(prepared: Prepared) -> dict:
    if type(prepared) is not Prepared:
        raise ValueError("Prepared ARC task required")
    _limits(prepared.seconds, prepared.memory_mib)
    binary, binary_hash = reviewed_binary()
    result = run_schedule(binary, binary_hash, prepared.task,
                          seconds=prepared.seconds, memory_mib=prepared.memory_mib)
    return {"version": 1, "search_arguments": [3, 23, 33, 4],
            "budget": {"seconds": prepared.seconds, "memory_mib": prepared.memory_mib},
            "completed_all_runs": result.completed_all_runs,
            "missing_test_indices": list(result.missing_test_indices),
            "predictions": [[{"grid": [list(row) for row in candidate.grid], "score": candidate.score}
                             for candidate in candidates] for candidates in result.predictions],
            "attempts": [{"test_index": i, "argument": attempt.argument, "status": attempt.status,
                          "seconds": attempt.seconds, "peak_rss_mib": attempt.peak_rss_mib}
                         for i, attempt in result.attempts],
            "build_identity_sha256": IDENTITY_SHA256}


def run(payload):
    return execute(prepare(payload))
