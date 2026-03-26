"""Execution receipt generation, signing, and verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ExecutionReceipt(BaseModel):
    """Unsigned execution receipt binding a bounty solution to its outputs."""

    bounty_id: str
    cdg_hash: str
    atom_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of atom FQDN to content hash.",
    )
    split_hash: str
    output_hash: str
    metric_name: str
    metric_value: float
    timestamp: str
    sciona_version: str


class SignedReceipt(BaseModel):
    """An execution receipt with an SSH signature."""

    receipt: ExecutionReceipt
    signature: str = Field(
        default="",
        description="SSH signature block (PEM-like text).",
    )
    namespace: str = "sciona"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def generate_receipt(
    *,
    bounty_id: str,
    cdg_path: Path,
    split_path: Path,
    output_path: Path,
    atom_versions: dict[str, str] | None = None,
    metric_name: str,
    metric_value: float,
    sciona_version: str = "0.0.0",
) -> ExecutionReceipt:
    """Build an unsigned execution receipt from file artifacts."""
    return ExecutionReceipt(
        bounty_id=bounty_id,
        cdg_hash=_sha256_file(cdg_path),
        atom_versions=atom_versions or {},
        split_hash=_sha256_file(split_path),
        output_hash=_sha256_file(output_path),
        metric_name=metric_name,
        metric_value=metric_value,
        timestamp=datetime.now(timezone.utc).isoformat(),
        sciona_version=sciona_version,
    )


def canonicalize_receipt(receipt: ExecutionReceipt) -> bytes:
    """Canonical JSON with a stable schema-first field order."""
    payload = receipt.model_dump()
    canonical_payload = {"sciona_version": payload.pop("sciona_version")}
    canonical_payload.update(dict(sorted(payload.items())))
    return json.dumps(
        canonical_payload,
        separators=(",", ":"),
    ).encode("utf-8")


def sign_receipt(
    receipt: ExecutionReceipt,
    ssh_key_path: Path,
    namespace: str = "sciona",
) -> SignedReceipt:
    """Sign a receipt using ``ssh-keygen -Y sign``.

    Requires ``ssh-keygen`` to be available on PATH.
    """
    canonical = canonicalize_receipt(receipt)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "receipt.json"
        sig_file = Path(tmpdir) / "receipt.json.sig"
        data_file.write_bytes(canonical)

        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(ssh_key_path),
                "-n",
                namespace,
                str(data_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ssh-keygen sign failed: {result.stderr}")

        signature = sig_file.read_text(encoding="utf-8")

    return SignedReceipt(
        receipt=receipt,
        signature=signature,
        namespace=namespace,
    )


def verify_receipt(
    signed_receipt: SignedReceipt,
    allowed_signers_path: Path,
) -> bool:
    """Verify a signed receipt using ``ssh-keygen -Y verify``.

    The *allowed_signers_path* should contain lines in the format
    ``principal algo base64key`` as expected by ``ssh-keygen``.

    Returns ``True`` if verification succeeds, ``False`` otherwise.
    """
    canonical = canonicalize_receipt(signed_receipt.receipt)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "receipt.json"
        sig_file = Path(tmpdir) / "receipt.json.sig"
        data_file.write_bytes(canonical)
        sig_file.write_text(signed_receipt.signature, encoding="utf-8")

        # We need a principal identity — use a wildcard
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed_signers_path),
                "-I",
                "sciona-receipt",
                "-n",
                signed_receipt.namespace,
                "-s",
                str(sig_file),
            ],
            input=canonical,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0


def save_signed_receipt(signed_receipt: SignedReceipt, path: Path) -> None:
    """Persist a signed receipt to disk as JSON."""
    path = Path(path)
    path.write_text(
        signed_receipt.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def load_signed_receipt(path: Path) -> SignedReceipt:
    """Load a signed receipt from a JSON file."""
    return SignedReceipt.model_validate_json(Path(path).read_bytes())
