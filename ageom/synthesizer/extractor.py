"""Extractor: export verified source to compiled artifacts and FFI bindings."""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path

from ageom.config import AgeomConfig
from ageom.synthesizer.cargo_template import (
    generate_build_rs,
    generate_cargo_toml,
    generate_lib_rs,
)
from ageom.synthesizer.certificate import generate_certificate, save_certificate
from ageom.synthesizer.lakefile_template import generate_lakefile
from ageom.synthesizer.models import ExportBundle, SkeletonFile, SynthesisResult


class ExportTarget(str, Enum):
    """Supported export targets."""

    LEAN_LIB = "lean-lib"
    COQ_LIB = "coq-lib"
    RUST_FFI = "rust-ffi"
    C_HEADER = "c-header"
    PYTHON_PKG = "python-pkg"


class Extractor:
    """Exports verified source into compiled artifacts and FFI bindings."""

    def __init__(self, config: AgeomConfig) -> None:
        self._config = config

    async def extract(
        self,
        synthesis_result: SynthesisResult,
        target: ExportTarget,
        output_dir: Path,
    ) -> ExportBundle:
        """Build the export bundle for the given target."""
        output_dir.mkdir(parents=True, exist_ok=True)
        skeleton = synthesis_result.skeleton
        errors: list[str] = []

        # Write verified source
        src_dir = output_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        if skeleton.prover == "lean4":
            ext = ".lean"
        elif skeleton.prover == "python":
            ext = ".py"
        else:
            ext = ".v"
        source_path = src_dir / f"Verified{ext}"
        source_path.write_text(skeleton.source_code)

        compiled_artifact: Path | None = None
        ffi_files: list[Path] = []

        if target == ExportTarget.LEAN_LIB:
            compiled_artifact, build_errors = await self._build_lean(
                skeleton, source_path, output_dir
            )
            errors.extend(build_errors)

        elif target == ExportTarget.COQ_LIB:
            compiled_artifact, build_errors = await self._build_coq(
                source_path, output_dir
            )
            errors.extend(build_errors)

        elif target == ExportTarget.RUST_FFI:
            compiled_artifact, build_errors = await self._build_lean(
                skeleton, source_path, output_dir
            )
            errors.extend(build_errors)
            ffi_files = self._generate_rust_ffi(skeleton, output_dir)

        elif target == ExportTarget.C_HEADER:
            compiled_artifact, build_errors = await self._build_lean(
                skeleton, source_path, output_dir
            )
            errors.extend(build_errors)
            ffi_files = self._generate_c_header(skeleton, output_dir)

        elif target == ExportTarget.PYTHON_PKG:
            compiled_artifact, build_errors = await self._build_python(
                skeleton, source_path, output_dir
            )
            errors.extend(build_errors)

        # Generate certificate
        goal = skeleton.metadata.get("goal", "")
        cert = generate_certificate(
            source_path=source_path,
            artifact_path=compiled_artifact,
            skeleton=skeleton,
            prover_version=(
                self._config.lean_toolchain
                if skeleton.prover == "lean4"
                else "python" if skeleton.prover == "python" else "coq"
            ),
            goal=goal,
        )
        cert_path = output_dir / "certificate.json"
        save_certificate(cert, cert_path)

        return ExportBundle(
            target=target.value,
            output_dir=output_dir,
            source_path=source_path,
            compiled_artifact=compiled_artifact,
            ffi_files=ffi_files,
            certificate=cert,
            errors=errors,
        )

    async def _build_lean(
        self,
        skeleton: SkeletonFile,
        source_path: Path,
        output_dir: Path,
    ) -> tuple[Path | None, list[str]]:
        """Generate lakefile and run lake build."""
        errors: list[str] = []

        # Determine library name from source filename
        lib_name = source_path.stem

        # Write lakefile.lean
        lakefile_content = generate_lakefile(
            name=lib_name,
            lean_version=self._config.lean_toolchain,
        )
        lakefile_path = output_dir / "lakefile.lean"
        lakefile_path.write_text(lakefile_content)

        # Write lean-toolchain
        toolchain_path = output_dir / "lean-toolchain"
        toolchain_path.write_text(self._config.lean_toolchain + "\n")

        # Run lake build
        lake_bin = self._config.lean_lake_path
        try:
            proc = await asyncio.create_subprocess_exec(
                lake_bin,
                "build",
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                errors.append(
                    f"lake build failed (exit {proc.returncode}): {stderr.decode()[:500]}"
                )
                return None, errors
        except FileNotFoundError:
            errors.append(
                f"lake binary not found at '{lake_bin}' — skipping compilation"
            )
            return None, errors

        # Find .olean artifact
        build_dir = output_dir / ".lake" / "build" / "lib"
        olean_candidates = (
            list(build_dir.rglob("*.olean")) if build_dir.exists() else []
        )
        artifact = olean_candidates[0] if olean_candidates else None

        return artifact, errors

    async def _build_coq(
        self,
        source_path: Path,
        output_dir: Path,
    ) -> tuple[Path | None, list[str]]:
        """Run coqc to compile the source."""
        errors: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "coqc",
                str(source_path),
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                errors.append(
                    f"coqc failed (exit {proc.returncode}): {stderr.decode()[:500]}"
                )
                return None, errors
        except FileNotFoundError:
            errors.append("coqc not found — skipping compilation")
            return None, errors

        vo_path = source_path.with_suffix(".vo")
        artifact = vo_path if vo_path.exists() else None

        return artifact, errors

    async def _build_python(
        self,
        skeleton: SkeletonFile,
        source_path: Path,
        output_dir: Path,
    ) -> tuple[Path | None, list[str]]:
        """Generate Python package and run mypy to validate."""
        from ageom.synthesizer.python_template import (
            generate_init_py,
            generate_pipeline_py,
            generate_pyproject_toml,
        )

        errors: list[str] = []
        package_name = skeleton.metadata.get("name", "verified_pkg")

        # Collect dependencies from imports in source
        dependencies = ["numpy", "scipy"]

        # Generate pyproject.toml
        pyproject_content = generate_pyproject_toml(package_name, dependencies)
        pyproject_path = output_dir / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)

        # Generate package structure
        pkg_dir = output_dir / "src" / package_name
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # __init__.py
        exports = [u.name for u in skeleton.units]
        init_content = generate_init_py(package_name, exports)
        (pkg_dir / "__init__.py").write_text(init_content)

        # atoms.py — the verified source
        (pkg_dir / "atoms.py").write_text(skeleton.source_code)

        # pipeline.py
        pipeline_content = generate_pipeline_py([])
        (pkg_dir / "pipeline.py").write_text(pipeline_content)

        # Run mypy --strict on the package
        mypy_bin = getattr(self._config, "python_mypy_path", "mypy")
        try:
            proc = await asyncio.create_subprocess_exec(
                mypy_bin,
                "--strict",
                str(pkg_dir),
                cwd=str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raw = stdout.decode() + stderr.decode()
                errors.append(
                    f"mypy validation failed (exit {proc.returncode}): {raw[:500]}"
                )
        except FileNotFoundError:
            errors.append(f"mypy not found at '{mypy_bin}' — skipping validation")

        artifact = pkg_dir if pkg_dir.exists() else None
        return artifact, errors

    def _generate_rust_ffi(
        self,
        skeleton: SkeletonFile,
        output_dir: Path,
    ) -> list[Path]:
        """Generate Rust FFI crate files."""
        ffi_dir = output_dir / "ffi"
        src_dir = ffi_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        name = skeleton.metadata.get("name", "verified")
        files: list[Path] = []

        # Cargo.toml
        cargo_path = ffi_dir / "Cargo.toml"
        cargo_path.write_text(generate_cargo_toml(name))
        files.append(cargo_path)

        # build.rs
        build_rs_path = ffi_dir / "build.rs"
        lean_lib_path = str(output_dir / ".lake" / "build" / "lib")
        build_rs_path.write_text(generate_build_rs(lean_lib_path))
        files.append(build_rs_path)

        # src/lib.rs
        lib_rs_path = src_dir / "lib.rs"
        lib_rs_path.write_text(generate_lib_rs(skeleton.units))
        files.append(lib_rs_path)

        return files

    def _generate_c_header(
        self,
        skeleton: SkeletonFile,
        output_dir: Path,
    ) -> list[Path]:
        """Generate a C header file for FFI."""
        ffi_dir = output_dir / "ffi"
        ffi_dir.mkdir(parents=True, exist_ok=True)

        header_lines: list[str] = [
            "/* Auto-generated by AGEO-Matcher extractor. */",
            "#ifndef AGEO_EXPORT_H",
            "#define AGEO_EXPORT_H",
            "",
            "#include <stdint.h>",
            "",
            "/* Opaque handle to a Lean/Coq managed object. */",
            "typedef struct lean_object lean_object;",
            "",
        ]

        for unit in skeleton.units:
            c_name = unit.declaration_name.replace(".", "_")
            header_lines.append(f"/* Verified: {unit.name} */")
            header_lines.append(f"/* Type: {unit.type_signature} */")
            header_lines.append(f"lean_object* {c_name}(lean_object* arg);")
            header_lines.append("")

        header_lines.append("#endif /* AGEO_EXPORT_H */")
        header_lines.append("")

        header_path = ffi_dir / "export.h"
        header_path.write_text("\n".join(header_lines))

        return [header_path]
