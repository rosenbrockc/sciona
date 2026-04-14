from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_manifest import build_parser, export_manifest_bundle, main


def _write_manifest(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def test_parser_requires_output_dir() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_reports_missing_supabase_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SCIONA_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SCIONA_SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

    exit_code = main(["--output-dir", str(tmp_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Required environment variable SCIONA_SUPABASE_URL is not set" in captured.err


def test_export_manifest_writes_latest_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SCIONA_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SCIONA_SUPABASE_SERVICE_KEY", "secret")

    def fake_export_tiered_manifests(
        bundle_dir: Path,
        *,
        supabase_url: str,
        service_key: str,
    ) -> dict[str, Path]:
        assert supabase_url == "https://example.supabase.co"
        assert service_key == "secret"
        public_path = bundle_dir / "manifest-public.sqlite"
        internal_path = bundle_dir / "manifest-internal.sqlite"
        _write_manifest(public_path, "public bundle")
        _write_manifest(internal_path, "internal bundle")
        return {"public": public_path, "internal": internal_path}

    monkeypatch.setattr(
        "scripts.export_manifest._export_tiered_manifests",
        fake_export_tiered_manifests,
    )

    result = export_manifest_bundle(tmp_path, upload=False)

    latest_path = result["latest"]
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["generated_at"].endswith("Z")
    assert set(payload["artifacts"]) == {"internal", "public"}
    assert payload["artifacts"]["public"]["path"] == "manifests/manifest-public.sqlite"
    assert payload["artifacts"]["internal"]["path"] == "manifests/manifest-internal.sqlite"
    assert payload["artifacts"]["public"]["sha256"]
    assert payload["artifacts"]["public"]["size_bytes"] == len("public bundle")


def test_export_manifest_uploads_bundle_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SCIONA_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SCIONA_SUPABASE_SERVICE_KEY", "secret")
    monkeypatch.setenv("SCIONA_MANIFEST_BUCKET", "manifest-bucket")

    def fake_export_tiered_manifests(
        bundle_dir: Path,
        *,
        supabase_url: str,
        service_key: str,
    ) -> dict[str, Path]:
        del supabase_url, service_key
        public_path = bundle_dir / "manifest-public.sqlite"
        _write_manifest(public_path, "public bundle")
        return {"public": public_path}

    class FakeS3Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            self.calls.append((filename, bucket, key))

    fake_client = FakeS3Client()

    monkeypatch.setattr(
        "scripts.export_manifest._export_tiered_manifests",
        fake_export_tiered_manifests,
    )
    monkeypatch.setattr(
        "scripts.export_manifest._build_s3_client",
        lambda: fake_client,
    )

    export_manifest_bundle(tmp_path, upload=True)

    assert fake_client.calls == [
        (
            str(tmp_path / "manifests" / "manifest-public.sqlite"),
            "manifest-bucket",
            "manifests/manifest-public.sqlite",
        ),
        (
            str(tmp_path / "manifests" / "latest.json"),
            "manifest-bucket",
            "manifests/latest.json",
        ),
    ]
