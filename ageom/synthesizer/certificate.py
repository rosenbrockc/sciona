"""Verification certificate generation and validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ageom.synthesizer.models import SkeletonFile, VerificationCertificate


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_certificate(
    source_path: Path,
    artifact_path: Path | None,
    skeleton: SkeletonFile,
    prover_version: str,
    goal: str = "",
) -> VerificationCertificate:
    """Generate a verification certificate for the export bundle."""
    source_hash = _sha256(source_path)
    artifact_hash = _sha256(artifact_path) if artifact_path and artifact_path.exists() else ""

    return VerificationCertificate(
        source_hash=source_hash,
        artifact_hash=artifact_hash,
        prover=skeleton.prover,
        prover_version=prover_version,
        goal=goal,
        node_count=len(skeleton.units),
        sorry_count=skeleton.sorry_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def save_certificate(cert: VerificationCertificate, path: Path) -> None:
    """Write certificate to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cert.model_dump(), indent=2) + "\n")


def load_certificate(path: Path) -> VerificationCertificate:
    """Read and validate a certificate from a JSON file."""
    data = json.loads(path.read_text())
    return VerificationCertificate(**data)


def verify_certificate(
    cert: VerificationCertificate,
    source_path: Path,
    artifact_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """Re-hash source and artifact, compare against certificate.

    Returns (valid, list_of_issues).
    """
    issues: list[str] = []

    if not source_path.exists():
        issues.append(f"source file not found: {source_path}")
    else:
        actual_hash = _sha256(source_path)
        if actual_hash != cert.source_hash:
            issues.append("source hash mismatch")

    if cert.artifact_hash and artifact_path is not None:
        if not artifact_path.exists():
            issues.append(f"artifact file not found: {artifact_path}")
        else:
            actual_hash = _sha256(artifact_path)
            if actual_hash != cert.artifact_hash:
                issues.append("artifact hash mismatch")

    return (len(issues) == 0, issues)
