# Phase 3: The Extractor

## Context

Phases 1–2 produce a **verified Lean 4 or Coq source file** where every definition type-checks and every proof compiles. Phase 3 exports this into something that actually runs: a Rust crate with FFI bindings to the compiled proof artifact, plus a verification certificate that ties the binary back to its proof.

**Existing infrastructure used**: `ProofEnvironment` from `ageom/judge/`, `SkeletonFile` + `SynthesisResult` from `ageom/synthesizer/models.py`, `AgeomConfig` from `ageom/config.py`.

---

## New Files (5 source + 1 test)

### 1. `ageom/synthesizer/extractor.py`

The main extraction logic: takes a verified source file and produces an export bundle.

**`class ExportTarget(str, Enum)`**:
- `LEAN_LIB` — just the `.olean` compiled artifact (default for Lean 4)
- `COQ_LIB` — just the `.vo` compiled artifact (default for Coq)
- `RUST_FFI` — Rust crate with C FFI bindings to the compiled proof
- `C_HEADER` — standalone C header + shared library

**`@dataclass class ExportBundle`** (add to `models.py`):
- `target: ExportTarget`
- `output_dir: Path`
- `source_path: Path` — the verified `.lean` or `.v` file
- `compiled_artifact: Path | None` — `.olean` / `.vo` if compilation succeeded
- `ffi_files: list[Path]` — generated FFI files (Rust `.rs`, C `.h`, etc.)
- `certificate: VerificationCertificate | None`
- `errors: list[str]`

**`@dataclass class VerificationCertificate`** (add to `models.py`):
- `source_hash: str` — SHA-256 of the verified source file
- `artifact_hash: str` — SHA-256 of the compiled artifact
- `prover: str` — "lean4" or "coq"
- `prover_version: str` — toolchain version string
- `goal: str` — from CDG metadata
- `node_count: int` — number of verified definitions
- `sorry_count: int` — 0 if fully verified, >0 if partial
- `timestamp: str` — ISO 8601
- `certificate_version: str = "1.0"`

**`class Extractor`**:

Constructor:
- `__init__(self, config: AgeomConfig)`

**`async extract(synthesis_result: SynthesisResult, target: ExportTarget, output_dir: Path) -> ExportBundle`**:
1. Write the final verified source to `output_dir/src/`
2. Branch on `target`:
   - `LEAN_LIB` → `_build_lean()`
   - `COQ_LIB` → `_build_coq()`
   - `RUST_FFI` → `_build_lean()` then `_generate_rust_ffi()`
   - `C_HEADER` → `_build_lean()` then `_generate_c_header()`
3. Generate `VerificationCertificate`
4. Write certificate to `output_dir/certificate.json`
5. Return `ExportBundle`

**`async _build_lean(source_path, output_dir) -> Path`**:
- Generate a `lakefile.lean` in `output_dir` with the source as a library
- Run `lake build` via `asyncio.create_subprocess_exec`
- Return path to `.olean` artifact
- Capture stderr for error reporting

**`async _build_coq(source_path, output_dir) -> Path`**:
- Run `coqc {source_path}` via `asyncio.create_subprocess_exec`
- Return path to `.vo` artifact

**`_generate_rust_ffi(skeleton, olean_path, output_dir) -> list[Path]`**:
- Generate `output_dir/ffi/src/lib.rs` with:
  - `extern "C"` function signatures matching each `AssemblyUnit`'s type signature
  - Opaque type wrappers for Lean types (`LeanObject`, etc.)
  - Documentation comments linking back to the proof source
- Generate `output_dir/ffi/Cargo.toml` with `crate-type = ["cdylib"]` and `links = "lean_export"`
- Generate `output_dir/ffi/build.rs` that links to the Lean shared library
- Return list of generated file paths

**`_generate_c_header(skeleton, olean_path, output_dir) -> list[Path]`**:
- Generate `output_dir/ffi/export.h` with:
  - C function declarations for each exported definition
  - Opaque struct types
  - Include guard
- Return list of generated file paths

**`_compute_certificate(source_path, artifact_path, skeleton, config) -> VerificationCertificate`**:
- SHA-256 hash source and artifact
- Pull prover version from config
- Count definitions and sorry placeholders

### 2. `ageom/synthesizer/optimizer.py`

The "Shell Game" — for hot-path operations, swap the verified (slow) implementation with a high-performance library that matches the same interface.

**`@dataclass class OptimizationRule`**:
- `pattern: str` — regex matching the declaration name or type signature (e.g., `".*matrix_mul.*"`, `".*fft.*"`)
- `replacement_lib: str` — the native library to link (e.g., `"blas"`, `"fftw3"`)
- `replacement_symbol: str` — the symbol name in the native library
- `guard_check: str` — a Lean/Coq statement that the optimizer verifies before swapping (e.g., "dimensions match")

**`DEFAULT_RULES: list[OptimizationRule]`** — built-in rules for:
- Matrix multiplication → BLAS `dgemm`
- FFT → FFTW `fftw_execute`
- Sorting (large arrays) → system `qsort`

**`class Optimizer`**:

Constructor:
- `__init__(self, rules: list[OptimizationRule] | None = None)` — defaults to `DEFAULT_RULES`

**`scan(skeleton: SkeletonFile) -> list[OptimizationCandidate]`**:
- Matches each `AssemblyUnit` against rules
- Returns candidates with the matched rule and unit

**`@dataclass class OptimizationCandidate`**:
- `unit: AssemblyUnit`
- `rule: OptimizationRule`
- `guard_verified: bool = False`

**`async verify_guards(candidates, env: ProofEnvironment) -> list[OptimizationCandidate]`**:
- For each candidate, compile the `guard_check` via `env.check_term()`
- Sets `guard_verified = True` if the guard passes
- Returns only candidates whose guards passed

**`apply(skeleton: SkeletonFile, candidates: list[OptimizationCandidate]) -> SkeletonFile`**:
- For each verified candidate, replace the definition body with an `@[extern "{replacement_symbol}"]` attribute (Lean) or `Extract Constant` directive (Coq)
- Returns modified skeleton with the swaps applied

### 3. `ageom/synthesizer/lakefile_template.py`

Template for generating `lakefile.lean` for Lean 4 builds.

**`generate_lakefile(name: str, lean_version: str, deps: list[str]) -> str`**:
- Returns a `lakefile.lean` string that:
  - Declares the package
  - Requires Mathlib
  - Defines a `lean_lib` target for the verified source
  - Optionally adds FFI export configuration

### 4. `ageom/synthesizer/cargo_template.py`

Template for generating Rust FFI crate files.

**`generate_cargo_toml(name: str) -> str`**:
- Returns a `Cargo.toml` for a `cdylib` crate

**`generate_build_rs(lean_lib_path: str) -> str`**:
- Returns a `build.rs` that tells cargo to link against the Lean shared library

**`generate_lib_rs(units: list[AssemblyUnit]) -> str`**:
- Returns a `lib.rs` with:
  - `extern "C"` bindings for each unit
  - Rust-idiomatic wrapper functions with proper types
  - Safety documentation

### 5. `ageom/synthesizer/certificate.py`

Certificate generation and verification.

**`generate_certificate(source_path, artifact_path, skeleton, config) -> VerificationCertificate`**:
- Computes SHA-256 hashes
- Populates all certificate fields

**`save_certificate(cert: VerificationCertificate, path: Path) -> None`**:
- Writes JSON

**`load_certificate(path: Path) -> VerificationCertificate`**:
- Reads JSON, validates schema

**`verify_certificate(cert: VerificationCertificate, source_path: Path, artifact_path: Path) -> tuple[bool, list[str]]`**:
- Re-hashes source and artifact
- Compares against certificate hashes
- Returns `(valid, list_of_issues)`

### 6. `tests/test_extractor.py`

**Fixtures**:
- `verified_skeleton()` — a `SynthesisResult` with `compiled_ok=True`, `sorry_remaining=0`
- `partial_skeleton()` — `compiled_ok=True` but `sorry_remaining=2`
- `sample_units()` — list of `AssemblyUnit` with realistic type signatures

**Test classes**:

`TestOptimizer`:
- `test_scan_finds_matrix_mul` — unit with "matrix_mul" in name matches BLAS rule
- `test_scan_no_match` — unit with "binary_search" matches nothing
- `test_apply_adds_extern_attr` — verified candidate → `@[extern]` in source
- `test_unverified_guard_skipped` — candidate with failed guard not applied

`TestCertificate`:
- `test_generate_certificate` — produces valid certificate with hashes
- `test_save_load_roundtrip` — JSON serialization roundtrip
- `test_verify_valid_certificate` — matching hashes → `(True, [])`
- `test_verify_tampered_source` — modified source → `(False, ["source hash mismatch"])`
- `test_verify_tampered_artifact` — modified artifact → `(False, ["artifact hash mismatch"])`

`TestLakefileTemplate`:
- `test_generates_valid_lakefile` — contains `lean_lib`, `require mathlib`

`TestCargoTemplate`:
- `test_generates_cargo_toml` — contains `crate-type = ["cdylib"]`
- `test_generates_lib_rs` — contains `extern "C"` for each unit

`TestExtractor`:
- `test_extract_lean_lib` — mock `lake build` → produces bundle with `.olean` path
- `test_extract_rust_ffi` — produces bundle with Rust FFI files
- `test_extract_c_header` — produces bundle with `.h` file
- `test_extract_with_optimization` — optimizer swaps hot-path, certificate reflects it
- `test_certificate_in_bundle` — bundle always includes certificate

---

## Modified Files (3)

### 7. `ageom/cli.py` — Add `export` subcommand

**New subparser** `export` (after `synthesize`):
- `source_file` positional arg — path to verified `.lean` or `.v` file (or path to `SynthesisResult` JSON)
- `--target` (choices: `lean-lib`, `coq-lib`, `rust-ffi`, `c-header`, default: `lean-lib`)
- `--output-dir` (str, default: `./export/`)
- `--optimize` flag — run the optimizer before export
- `--prover` (choices: `lean4`, `coq`, default: `lean4`)

**New handler** `async _cmd_export(args)`:
1. Load source file or `SynthesisResult`
2. If `--optimize`: run `Optimizer.scan()` + `verify_guards()` + `apply()`
3. Create `Extractor(config)`
4. Call `extractor.extract(result, target, output_dir)` → `ExportBundle`
5. Print summary: target, output paths, certificate hash, optimization swaps

### 8. `ageom/config.py` — Add extractor config fields

```python
# Extractor (Round 3 Phase 3)
export_output_dir: Path = Field(default=Path("export"))
lean_lake_path: str = "lake"  # path to lake binary
optimize_by_default: bool = False
```

### 9. `pyproject.toml` — Update dependency groups

```toml
synthesizer = [
    "ageo-matcher[hunter]",
    "jinja2>=3.1",  # only if templates grow complex later
]
```

Update `all` to include `synthesizer`.

---

## Key Design Decisions

1. **`lake build` for Lean compilation** — Lean 4's build system (`lake`) handles dependency resolution, caching, and `.olean` generation. We generate a `lakefile.lean` and shell out to `lake` rather than trying to invoke `lean` directly. This is the standard approach for Lean 4 projects.

2. **Verification certificate as JSON** — the certificate is a static JSON file that can be checked independently of the AGEO-Matcher toolchain. Any tool can re-hash the source and artifact to verify integrity.

3. **Optimizer is opt-in** — the `--optimize` flag must be explicitly passed. By default, the export uses the verified (potentially slower) implementation. This prevents accidental correctness regressions from optimization swaps.

4. **Guard checks before optimization** — before swapping a verified implementation for a native library, the optimizer compiles a guard statement through the proof environment. For matrix multiplication, this might verify dimension constraints. If the guard fails, the swap is skipped and the verified implementation is kept.

5. **Rust FFI via `cdylib`** — the generated Rust crate compiles to a C-compatible shared library. This is the standard Lean 4 FFI approach and works with any language that can call C functions.

6. **No transpilation** — we do NOT transpile Lean/Coq code to Rust/C++. Instead, we use FFI to call the compiled proof artifact. The verified code runs as-is in its native runtime. Transpilation would break the verification guarantee.

7. **ExportBundle is a directory** — the output is a self-contained directory with source, artifacts, FFI files, and certificate. This can be versioned, shipped, or embedded in a larger build system.

---

## Verification

1. `pytest tests/test_extractor.py -v` — all unit tests pass
2. Manual: `ageom synthesize cdg.json matches.json --output verified.lean` → `ageom export verified.lean --target rust-ffi --output-dir ./export/` → inspect generated Rust crate
3. Manual: `cd export/ffi && cargo build` → verify the crate compiles (requires Lean 4 toolchain)
4. Manual: `ageom export verified.lean --target lean-lib` → verify `.olean` produced and certificate valid
5. `pytest tests/ -v` — full suite still green
