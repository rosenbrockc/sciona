#!/usr/bin/env python3
"""Export manifest bundles and optionally upload them to S3."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
    return {"all": output_path}


def _export_tiered_manifests(
    bundle_dir: Path,
    *,
    supabase_url: str,
    service_key: str,
) -> dict[str, Path]:
    helper = getattr(snapshot_api, "export_tiered_manifests", None)
    if callable(helper):
        return _normalize_tier_paths(
            asyncio.run(helper(supabase_url, service_key, bundle_dir))
        )
    return _export_single_manifest_bundle(
        bundle_dir,
        supabase_url=supabase_url,
        service_key=service_key,
    )


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
    bundle_root = output_dir.expanduser().resolve()
    bundle_dir = bundle_root / MANIFEST_BUNDLE_DIR
    bundle_dir.mkdir(parents=True, exist_ok=True)

    tier_paths = _export_tiered_manifests(
        bundle_dir,
        supabase_url=settings.supabase_url,
        service_key=settings.service_key,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
