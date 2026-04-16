#!/usr/bin/env python3
"""Export manifest bundles and optionally upload them to S3."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from sciona.api import snapshot as snapshot_api

MANIFEST_BUNDLE_DIR = "manifests"
DEFAULT_SUPABASE_URL_ENV = ("SCIONA_SUPABASE_URL", "SUPABASE_URL")
DEFAULT_SUPABASE_SERVICE_KEY_ENV = (
    "SCIONA_SUPABASE_SERVICE_KEY",
    "SUPABASE_SERVICE_KEY",
)
DEFAULT_S3_BUCKET_ENV = (
    "SCIONA_MANIFEST_BUCKET",
    "SCIONA_S3_BUCKET",
    "SCIONA_CATALOG_BUCKET",
)
_LEGACY_NAMESPACE_PREFIX = "age" + "oa."
_LEGACY_REPO_LABEL = "ageo" + "-atoms"


def _developer_mode_enabled(explicit: bool = False) -> bool:
    if explicit:
        return True
    return os.environ.get("SCIONA_DEVELOPER_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class ExportSettings:
    supabase_url: str
    service_key: str


@dataclass(frozen=True)
class UploadSettings:
    bucket: str


def _require_first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"Required environment variable {names[0]} is not set")


def _load_export_settings() -> ExportSettings:
    return ExportSettings(
        supabase_url=_require_first_env(DEFAULT_SUPABASE_URL_ENV),
        service_key=_require_first_env(DEFAULT_SUPABASE_SERVICE_KEY_ENV),
    )


def _load_upload_settings() -> UploadSettings:
    return UploadSettings(bucket=_require_first_env(DEFAULT_S3_BUCKET_ENV))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_manifest_sqlite(path: Path) -> dict[str, int]:
    if not path.exists():
        return {
            "total_atoms": 0,
            "publishable_atoms": 0,
            "non_publishable_atoms": 0,
            "benchmark_rows": 0,
            "approved_license_atoms": 0,
            "unknown_license_atoms": 0,
            "restricted_license_atoms": 0,
        }
    try:
        with sqlite3.connect(path) as con:
            con.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            total_atoms = 0
            publishable_atoms = 0
            benchmark_rows = 0
            approved_license_atoms = 0
            unknown_license_atoms = 0
            restricted_license_atoms = 0
            if "atoms" in tables:
                total_atoms = int(
                    con.execute("SELECT COUNT(*) AS count FROM atoms").fetchone()["count"] or 0
                )
                publishable_atoms = int(
                    con.execute(
                        "SELECT COUNT(*) AS count FROM atoms WHERE is_publishable = 1"
                    ).fetchone()["count"]
                    or 0
                )
                atom_columns = {
                    str(row["name"])
                    for row in con.execute("PRAGMA table_info(atoms)").fetchall()
                }
                if "license_status" in atom_columns:
                    approved_license_atoms = int(
                        con.execute(
                            "SELECT COUNT(*) AS count FROM atoms WHERE license_status = 'approved'"
                        ).fetchone()["count"]
                        or 0
                    )
                    unknown_license_atoms = int(
                        con.execute(
                            "SELECT COUNT(*) AS count FROM atoms WHERE license_status = 'unknown' OR license_status = ''"
                        ).fetchone()["count"]
                        or 0
                    )
                    restricted_license_atoms = int(
                        con.execute(
                            "SELECT COUNT(*) AS count FROM atoms WHERE license_status IN ('restricted', 'needs_legal_review')"
                        ).fetchone()["count"]
                        or 0
                    )
            if "benchmarks" in tables:
                benchmark_rows = int(
                    con.execute("SELECT COUNT(*) AS count FROM benchmarks").fetchone()["count"]
                    or 0
                )
    except sqlite3.DatabaseError:
        return {
            "total_atoms": 0,
            "publishable_atoms": 0,
            "non_publishable_atoms": 0,
            "benchmark_rows": 0,
            "approved_license_atoms": 0,
            "unknown_license_atoms": 0,
            "restricted_license_atoms": 0,
        }
    return {
        "total_atoms": total_atoms,
        "publishable_atoms": publishable_atoms,
        "non_publishable_atoms": max(total_atoms - publishable_atoms, 0),
        "benchmark_rows": benchmark_rows,
        "approved_license_atoms": approved_license_atoms,
        "unknown_license_atoms": unknown_license_atoms,
        "restricted_license_atoms": restricted_license_atoms,
    }


def _normalize_tier_paths(value: Any) -> dict[str, Path]:
    if isinstance(value, Mapping):
        return {str(tier): Path(path) for tier, path in value.items()}
    raise TypeError(
        "export_tiered_manifests() must return a mapping of tier names to paths"
    )


def _export_single_manifest_bundle(
    bundle_dir: Path,
    *,
    supabase_url: str,
    service_key: str,
) -> dict[str, Path]:
    data = asyncio.run(snapshot_api.fetch_manifest_data(supabase_url, service_key))
    output_path = bundle_dir / "manifest-all.sqlite"
    con = snapshot_api.generate_manifest_sqlite(data, output_path=output_path)
    con.close()
    _assert_no_legacy_namespace_rows(output_path)
    return {"all": output_path}


def _export_tiered_manifests(
    bundle_dir: Path,
    *,
    supabase_url: str,
    service_key: str,
    developer_mode: bool,
) -> dict[str, Path]:
    helper = getattr(snapshot_api, "export_tiered_manifests", None)
    if callable(helper):
        tier_paths = _normalize_tier_paths(
            asyncio.run(
                helper(
                    supabase_url,
                    service_key,
                    bundle_dir,
                    include_developer_manifest=developer_mode,
                )
            )
        )
        for path in tier_paths.values():
            _assert_no_legacy_namespace_rows(path)
        return tier_paths
    tier_paths = _export_single_manifest_bundle(
        bundle_dir,
        supabase_url=supabase_url,
        service_key=service_key,
    )
    for path in tier_paths.values():
        _assert_no_legacy_namespace_rows(path)
    return tier_paths


def _build_latest_payload(
    bundle_root: Path,
    tier_paths: Mapping[str, Path],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    artifacts: dict[str, dict[str, Any]] = {}
    for tier, path in sorted(tier_paths.items()):
        resolved = path.resolve()
        artifacts[tier] = {
            "path": str(resolved.relative_to(bundle_root.resolve()).as_posix()),
            "sha256": _sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
            "publishability": _summarize_manifest_sqlite(resolved),
        }
    return {"generated_at": generated_at, "artifacts": artifacts}


def _write_latest_json(bundle_root: Path, tier_paths: Mapping[str, Path]) -> Path:
    latest_path = bundle_root / MANIFEST_BUNDLE_DIR / "latest.json"
    payload = _build_latest_payload(bundle_root, tier_paths)
    latest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return latest_path


def _build_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required when --upload is enabled") from exc
    return boto3.client("s3")


def _assert_no_legacy_namespace_rows(path: Path) -> None:
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        checks = {
            "atoms": """
                SELECT COUNT(*) AS count
                FROM atoms
                WHERE fqdn LIKE ?
                   OR atom_id LIKE ?
                   OR source_repo_id LIKE ?
            """,
            "benchmarks": """
                SELECT COUNT(*) AS count
                FROM benchmarks
                WHERE atom_fqdn LIKE ?
                   OR atom_fqdn LIKE ?
            """,
        }
        params = {
            "atoms": (
                f"{_LEGACY_NAMESPACE_PREFIX}%",
                f"{_LEGACY_NAMESPACE_PREFIX}%",
                f"%{_LEGACY_REPO_LABEL}%",
            ),
            "benchmarks": (
                f"{_LEGACY_NAMESPACE_PREFIX}%",
                f"{_LEGACY_REPO_LABEL}%",
            ),
        }
        for table, sql in checks.items():
            count = int(con.execute(sql, params[table]).fetchone()["count"] or 0)
            if count:
                raise RuntimeError(
                    f"Legacy namespace references remain in exported manifest table {table}: {count}"
                )


def _upload_bundle(
    s3_client: Any,
    bucket: str,
    bundle_root: Path,
    tier_paths: Mapping[str, Path],
    latest_path: Path,
) -> None:
    for _, path in sorted(tier_paths.items()):
        key = path.resolve().relative_to(bundle_root.resolve()).as_posix()
        s3_client.upload_file(str(path), bucket, key)
    latest_key = latest_path.resolve().relative_to(bundle_root.resolve()).as_posix()
    s3_client.upload_file(str(latest_path), bucket, latest_key)


def export_manifest_bundle(output_dir: Path, *, upload: bool) -> dict[str, Path]:
    settings = _load_export_settings()
    developer_mode = _developer_mode_enabled()
    bundle_root = output_dir.expanduser().resolve()
    bundle_dir = bundle_root / MANIFEST_BUNDLE_DIR
    bundle_dir.mkdir(parents=True, exist_ok=True)

    tier_paths = _export_tiered_manifests(
        bundle_dir,
        supabase_url=settings.supabase_url,
        service_key=settings.service_key,
        developer_mode=developer_mode,
    )
    latest_path = _write_latest_json(bundle_root, tier_paths)

    if upload:
        upload_settings = _load_upload_settings()
        s3_client = _build_s3_client()
        _upload_bundle(
            s3_client,
            upload_settings.bucket,
            bundle_root,
            tier_paths,
            latest_path,
        )

    return {"latest": latest_path, **tier_paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export tiered manifest bundles and optionally publish them to S3.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the manifest bundle should be written.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        default=False,
        help="Upload the exported bundle to S3 after writing local files.",
    )
    parser.add_argument(
        "--developer-mode",
        action="store_true",
        default=False,
        help="Also export a developer manifest that includes approved but unpublished atoms.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.developer_mode:
        os.environ["SCIONA_DEVELOPER_MODE"] = "1"

    try:
        bundle = export_manifest_bundle(Path(args.output_dir), upload=bool(args.upload))
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    latest_path = bundle["latest"]
    print(f"Manifest bundle written to {latest_path.parent}")
    print(f"Latest metadata written to {latest_path}")
    if args.upload:
        print("Bundle uploaded to S3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
