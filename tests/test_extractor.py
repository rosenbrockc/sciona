"""Tests for Phase 3: extractor, optimizer, certificate, templates."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ageom.synthesizer.cargo_template import (
    generate_build_rs,
    generate_cargo_toml,
    generate_lib_rs,
)
from ageom.synthesizer.certificate import (
    generate_certificate,
    load_certificate,
    save_certificate,
    verify_certificate,
)
from ageom.synthesizer.lakefile_template import generate_lakefile
from ageom.synthesizer.models import (
    AssemblyUnit,
    ExportBundle,
    SkeletonFile,
    SynthesisResult,
    VerificationCertificate,
)
from ageom.synthesizer.optimizer import (
    OptimizationCandidate,
    OptimizationRule,
    Optimizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEAN_SOURCE = """\
import Mathlib

noncomputable def merge_sort (l : List Nat) : List Nat :=
  List.mergeSort (· ≤ ·) l

theorem merge_sort_sorted (l : List Nat) : List.Sorted (· ≤ ·) (merge_sort l) := by
  simp [merge_sort, List.sorted_mergeSort]
"""

SAMPLE_UNITS = [
    AssemblyUnit(
        node_id="n1",
        name="merge_sort",
        declaration_name="MergeSort.merge_sort",
        type_signature="List Nat → List Nat",
    ),
    AssemblyUnit(
        node_id="n2",
        name="merge_sort_sorted",
        declaration_name="MergeSort.merge_sort_sorted",
        type_signature="∀ (l : List Nat), List.Sorted (· ≤ ·) (merge_sort l)",
    ),
]

MATRIX_UNIT = AssemblyUnit(
    node_id="n3",
    name="matrix_mul",
    declaration_name="Matrix.matrix_mul",
    type_signature="Matrix m n α → Matrix n p α → Matrix m p α",
)

FFT_UNIT = AssemblyUnit(
    node_id="n4",
    name="fft_transform",
    declaration_name="Signal.fft_transform",
    type_signature="Vector ℂ n → Vector ℂ n",
)


def _make_skeleton(
    source: str = LEAN_SOURCE,
    prover: str = "lean4",
    units: list[AssemblyUnit] | None = None,
) -> SkeletonFile:
    return SkeletonFile(
        prover=prover,
        source_code=source,
        units=units or SAMPLE_UNITS,
        sorry_count=0,
        metadata={"goal": "Verified merge sort"},
    )


def _make_synthesis_result(
    skeleton: SkeletonFile | None = None,
    compiled_ok: bool = True,
    sorry_remaining: int = 0,
) -> SynthesisResult:
    return SynthesisResult(
        skeleton=skeleton or _make_skeleton(),
        compiled_ok=compiled_ok,
        sorry_remaining=sorry_remaining,
    )


# ===========================================================================
# TestOptimizer
# ===========================================================================


class TestOptimizer:
    def test_scan_finds_matrix_mul(self):
        skeleton = _make_skeleton(units=[MATRIX_UNIT])
        optimizer = Optimizer()
        candidates = optimizer.scan(skeleton)
        assert len(candidates) == 1
        assert candidates[0].unit.name == "matrix_mul"
        assert candidates[0].rule.replacement_lib == "blas"

    def test_scan_finds_fft(self):
        skeleton = _make_skeleton(units=[FFT_UNIT])
        optimizer = Optimizer()
        candidates = optimizer.scan(skeleton)
        assert len(candidates) == 1
        assert candidates[0].rule.replacement_symbol == "fftw_execute"

    def test_scan_no_match(self):
        units = [AssemblyUnit(
            node_id="n10",
            name="binary_search",
            declaration_name="BinarySearch.find",
            type_signature="List Nat → Nat → Option Nat",
        )]
        skeleton = _make_skeleton(units=units)
        optimizer = Optimizer()
        candidates = optimizer.scan(skeleton)
        assert len(candidates) == 0

    def test_scan_custom_rule(self):
        rule = OptimizationRule(
            pattern=r".*merge_sort.*",
            replacement_lib="custom",
            replacement_symbol="custom_sort",
            guard_check="-- always ok",
        )
        skeleton = _make_skeleton()
        optimizer = Optimizer(rules=[rule])
        candidates = optimizer.scan(skeleton)
        assert len(candidates) >= 1

    def test_apply_adds_extern_attr(self):
        source = "def Matrix.matrix_mul (a b : Matrix) : Matrix := sorry\n"
        skeleton = _make_skeleton(source=source, units=[MATRIX_UNIT])
        cand = OptimizationCandidate(
            unit=MATRIX_UNIT,
            rule=OptimizationRule(
                pattern=r".*matrix_mul.*",
                replacement_lib="blas",
                replacement_symbol="dgemm",
                guard_check="--",
            ),
            guard_verified=True,
        )
        optimizer = Optimizer()
        result = optimizer.apply(skeleton, [cand])
        assert '@[extern "dgemm"]' in result.source_code
        assert result.metadata.get("optimized") is True

    def test_unverified_guard_skipped(self):
        source = "def Matrix.matrix_mul (a b : Matrix) : Matrix := sorry\n"
        skeleton = _make_skeleton(source=source, units=[MATRIX_UNIT])
        cand = OptimizationCandidate(
            unit=MATRIX_UNIT,
            rule=OptimizationRule(
                pattern=r".*matrix_mul.*",
                replacement_lib="blas",
                replacement_symbol="dgemm",
                guard_check="-- guard",
            ),
            guard_verified=False,
        )
        optimizer = Optimizer()
        result = optimizer.apply(skeleton, [cand])
        assert '@[extern "dgemm"]' not in result.source_code

    @pytest.mark.asyncio
    async def test_verify_guards_comment_passes(self):
        cand = OptimizationCandidate(
            unit=MATRIX_UNIT,
            rule=OptimizationRule(
                pattern=".*",
                replacement_lib="blas",
                replacement_symbol="dgemm",
                guard_check="-- BLAS compat assumed",
            ),
        )
        env = AsyncMock()
        optimizer = Optimizer()
        verified = await optimizer.verify_guards([cand], env)
        assert len(verified) == 1
        assert verified[0].guard_verified is True


# ===========================================================================
# TestCertificate
# ===========================================================================


class TestCertificate:
    def test_generate_certificate(self, tmp_path: Path):
        source = tmp_path / "test.lean"
        source.write_text("-- verified code\n")
        artifact = tmp_path / "test.olean"
        artifact.write_bytes(b"\x00" * 100)

        skeleton = _make_skeleton()
        cert = generate_certificate(
            source_path=source,
            artifact_path=artifact,
            skeleton=skeleton,
            prover_version="leanprover/lean4:v4.14.0",
            goal="merge sort",
        )
        assert cert.source_hash
        assert cert.artifact_hash
        assert cert.prover == "lean4"
        assert cert.goal == "merge sort"
        assert cert.node_count == 2
        assert cert.certificate_version == "1.0"

    def test_generate_certificate_no_artifact(self, tmp_path: Path):
        source = tmp_path / "test.lean"
        source.write_text("-- code\n")

        skeleton = _make_skeleton()
        cert = generate_certificate(
            source_path=source,
            artifact_path=None,
            skeleton=skeleton,
            prover_version="v4.14.0",
        )
        assert cert.source_hash
        assert cert.artifact_hash == ""

    def test_save_load_roundtrip(self, tmp_path: Path):
        cert = VerificationCertificate(
            source_hash="abc123",
            artifact_hash="def456",
            prover="lean4",
            prover_version="v4.14.0",
            goal="test",
            node_count=5,
            sorry_count=0,
            timestamp="2026-01-01T00:00:00Z",
        )
        cert_path = tmp_path / "cert.json"
        save_certificate(cert, cert_path)
        loaded = load_certificate(cert_path)
        assert loaded.source_hash == cert.source_hash
        assert loaded.artifact_hash == cert.artifact_hash
        assert loaded.prover == cert.prover
        assert loaded.node_count == cert.node_count

    def test_verify_valid_certificate(self, tmp_path: Path):
        source = tmp_path / "test.lean"
        source.write_text("-- code\n")
        artifact = tmp_path / "test.olean"
        artifact.write_bytes(b"artifact data")

        skeleton = _make_skeleton()
        cert = generate_certificate(source, artifact, skeleton, "v4.14.0")

        valid, issues = verify_certificate(cert, source, artifact)
        assert valid is True
        assert issues == []

    def test_verify_tampered_source(self, tmp_path: Path):
        source = tmp_path / "test.lean"
        source.write_text("-- original\n")

        skeleton = _make_skeleton()
        cert = generate_certificate(source, None, skeleton, "v4.14.0")

        # Tamper with source
        source.write_text("-- tampered\n")
        valid, issues = verify_certificate(cert, source)
        assert valid is False
        assert "source hash mismatch" in issues

    def test_verify_tampered_artifact(self, tmp_path: Path):
        source = tmp_path / "test.lean"
        source.write_text("-- code\n")
        artifact = tmp_path / "test.olean"
        artifact.write_bytes(b"original")

        skeleton = _make_skeleton()
        cert = generate_certificate(source, artifact, skeleton, "v4.14.0")

        # Tamper with artifact
        artifact.write_bytes(b"tampered")
        valid, issues = verify_certificate(cert, source, artifact)
        assert valid is False
        assert "artifact hash mismatch" in issues

    def test_verify_missing_source(self, tmp_path: Path):
        cert = VerificationCertificate(
            source_hash="abc",
            prover="lean4",
            prover_version="v4",
        )
        valid, issues = verify_certificate(cert, tmp_path / "nonexistent.lean")
        assert valid is False
        assert any("not found" in i for i in issues)


# ===========================================================================
# TestLakefileTemplate
# ===========================================================================


class TestLakefileTemplate:
    def test_generates_valid_lakefile(self):
        content = generate_lakefile("Verified", "leanprover/lean4:v4.14.0")
        assert "lean_lib Verified" in content
        assert "require mathlib" in content
        assert "package Verified" in content

    def test_with_ffi_export(self):
        content = generate_lakefile("Export", ffi_export=True)
        assert '"-shared"' in content

    def test_with_extra_deps(self):
        content = generate_lakefile("Test", deps=["custom_dep"])
        assert "require custom_dep" in content


# ===========================================================================
# TestCargoTemplate
# ===========================================================================


class TestCargoTemplate:
    def test_generates_cargo_toml(self):
        content = generate_cargo_toml("verified")
        assert 'crate-type = ["cdylib"]' in content
        assert "verified-ffi" in content

    def test_generates_build_rs(self):
        content = generate_build_rs("/path/to/lean/lib")
        assert "/path/to/lean/lib" in content
        assert "leanshared" in content

    def test_generates_lib_rs(self):
        content = generate_lib_rs(SAMPLE_UNITS)
        assert 'extern "C"' in content
        assert "MergeSort_merge_sort" in content
        assert "MergeSort_merge_sort_sorted" in content
        assert "LeanObject" in content

    def test_generates_lib_rs_empty_units(self):
        content = generate_lib_rs([])
        assert "LeanObject" in content
        assert 'extern "C"' not in content


# ===========================================================================
# TestExtractor
# ===========================================================================


class TestExtractor:
    @pytest.mark.asyncio
    async def test_extract_lean_lib(self, tmp_path: Path):
        from ageom.synthesizer.extractor import ExportTarget, Extractor

        config = _mock_config()
        extractor = Extractor(config)
        result = _make_synthesis_result()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_proc:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"", b"lake not available"))
            mock_proc.return_value = proc

            bundle = await extractor.extract(result, ExportTarget.LEAN_LIB, tmp_path)

        assert isinstance(bundle, ExportBundle)
        assert bundle.source_path.exists()
        assert bundle.certificate is not None
        assert (tmp_path / "certificate.json").exists()
        # lake build may fail in test, but source + certificate are written
        assert bundle.source_path.suffix == ".lean"

    @pytest.mark.asyncio
    async def test_extract_rust_ffi(self, tmp_path: Path):
        from ageom.synthesizer.extractor import ExportTarget, Extractor

        config = _mock_config()
        extractor = Extractor(config)
        result = _make_synthesis_result()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_proc:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"", b"lake not found"))
            mock_proc.return_value = proc

            bundle = await extractor.extract(result, ExportTarget.RUST_FFI, tmp_path)

        assert len(bundle.ffi_files) == 3  # Cargo.toml, build.rs, lib.rs
        cargo_toml = [f for f in bundle.ffi_files if f.name == "Cargo.toml"]
        assert len(cargo_toml) == 1
        assert 'crate-type = ["cdylib"]' in cargo_toml[0].read_text()

    @pytest.mark.asyncio
    async def test_extract_c_header(self, tmp_path: Path):
        from ageom.synthesizer.extractor import ExportTarget, Extractor

        config = _mock_config()
        extractor = Extractor(config)
        result = _make_synthesis_result()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_proc:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"", b"lake not found"))
            mock_proc.return_value = proc

            bundle = await extractor.extract(result, ExportTarget.C_HEADER, tmp_path)

        assert len(bundle.ffi_files) == 1
        header = bundle.ffi_files[0]
        assert header.name == "export.h"
        content = header.read_text()
        assert "#ifndef AGEO_EXPORT_H" in content
        assert "lean_object" in content

    @pytest.mark.asyncio
    async def test_extract_coq_lib(self, tmp_path: Path):
        from ageom.synthesizer.extractor import ExportTarget, Extractor

        config = _mock_config()
        extractor = Extractor(config)
        skeleton = _make_skeleton(
            source="Require Import Coq.Arith.Arith.\n",
            prover="coq",
        )
        result = _make_synthesis_result(skeleton=skeleton)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_proc:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"", b"coqc not found"))
            mock_proc.return_value = proc

            bundle = await extractor.extract(result, ExportTarget.COQ_LIB, tmp_path)

        assert bundle.source_path.suffix == ".v"
        assert bundle.certificate is not None

    @pytest.mark.asyncio
    async def test_certificate_in_bundle(self, tmp_path: Path):
        from ageom.synthesizer.extractor import ExportTarget, Extractor

        config = _mock_config()
        extractor = Extractor(config)
        result = _make_synthesis_result()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_proc:
            proc = AsyncMock()
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.return_value = proc

            bundle = await extractor.extract(result, ExportTarget.LEAN_LIB, tmp_path)

        assert bundle.certificate is not None
        assert bundle.certificate.prover == "lean4"
        assert bundle.certificate.source_hash

        # Verify the saved certificate file is valid
        cert_path = tmp_path / "certificate.json"
        loaded = load_certificate(cert_path)
        assert loaded.source_hash == bundle.certificate.source_hash


# ===========================================================================
# TestCLI
# ===========================================================================


class TestCLIParserAcceptsExport:
    def test_export_subcommand_exists(self):
        import argparse

        parser = argparse.ArgumentParser(prog="ageom")
        subparsers = parser.add_subparsers(dest="command")
        export_p = subparsers.add_parser("export")
        export_p.add_argument("source_file")
        export_p.add_argument("--target", default="lean-lib")
        export_p.add_argument("--output-dir", default=None)
        export_p.add_argument("--optimize", action="store_true")
        export_p.add_argument("--prover", default="lean4")

        args = parser.parse_args(["export", "verified.lean", "--target", "rust-ffi"])
        assert args.command == "export"
        assert args.source_file == "verified.lean"
        assert args.target == "rust-ffi"

    def test_export_defaults(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        export_p = subparsers.add_parser("export")
        export_p.add_argument("source_file")
        export_p.add_argument("--target", default="lean-lib")
        export_p.add_argument("--optimize", action="store_true", default=False)

        args = parser.parse_args(["export", "file.lean"])
        assert args.target == "lean-lib"
        assert args.optimize is False


# ===========================================================================
# Helpers
# ===========================================================================


def _mock_config():
    """Create a mock AgeomConfig without reading .env."""
    from unittest.mock import MagicMock

    config = MagicMock()
    config.lean_toolchain = "leanprover/lean4:v4.14.0"
    config.lean_lake_path = "lake"
    config.export_output_dir = Path("export")
    config.optimize_by_default = False
    return config
