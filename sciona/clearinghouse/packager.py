"""CDG packaging for sandbox execution."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from typing import Sequence

from sciona.clearinghouse.models import SandboxPayload


def package_cdg(
    atom_sources: dict[str, str],
    dependency_edges: Sequence[tuple[str, str]],
    sciona_yml: dict,
    config_yml: dict,
    *,
    bounty_id: str,
    submission_id: str,
    lockfile_hash: str = "",
) -> SandboxPayload:
    """Assemble a SandboxPayload from CDG components.

    Parameters
    ----------
    atom_sources
        Mapping of atom FQDN to Python source code.
    dependency_edges
        Directed edges (from_fqdn, to_fqdn) in the CDG.
    sciona_yml
        Parsed sciona.yml dataset specification.
    config_yml
        Parsed config.yml hyperparameters.
    bounty_id
        The bounty this submission targets.
    submission_id
        Unique submission identifier.
    lockfile_hash
        SHA-256 of the Python lockfile for reproducibility.
    """
    atom_versions = {
        fqdn: hashlib.sha256(source.encode("utf-8")).hexdigest()
        for fqdn, source in atom_sources.items()
    }

    return SandboxPayload(
        bounty_id=bounty_id,
        submission_id=submission_id,
        cdg_source=atom_sources,
        sciona_yml=sciona_yml,
        config_yml=config_yml,
        lockfile_hash=lockfile_hash,
        atom_versions=atom_versions,
    )


def create_sandbox_tarball(payload: SandboxPayload) -> bytes:
    """Create a tar.gz archive from a sandbox payload.

    The archive layout:
    - atoms/<fqdn>.py  — one file per atom
    - sciona.yml        — dataset spec (JSON)
    - config.yml       — hyperparameters (JSON)
    - manifest.json    — atom versions + metadata
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fqdn, source in payload.cdg_source.items():
            data = source.encode("utf-8")
            info = tarfile.TarInfo(name=f"atoms/{fqdn}.py")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        _add_json(tar, "sciona.yml", payload.sciona_yml)
        _add_json(tar, "config.yml", payload.config_yml)
        _add_json(tar, "manifest.json", {
            "bounty_id": payload.bounty_id,
            "submission_id": payload.submission_id,
            "atom_versions": payload.atom_versions,
            "lockfile_hash": payload.lockfile_hash,
        })

    return buf.getvalue()


def _add_json(tar: tarfile.TarFile, name: str, obj: dict) -> None:
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
