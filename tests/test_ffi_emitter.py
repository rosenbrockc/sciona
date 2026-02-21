"""Tests for FFI binding generation (ageom.ingester.ffi_emitter)."""

from __future__ import annotations


from ageom.architect.models import IOSpec
from ageom.ingester.ffi_emitter import (
    generate_ffi_bindings,
    generate_ffi_imports,
    generate_ffi_stub,
)
from ageom.ingester.models import MacroAtomSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atom(
    name: str = "Test Atom",
    inputs: list[tuple[str, str]] | None = None,
    outputs: list[tuple[str, str]] | None = None,
) -> MacroAtomSpec:
    if inputs is None:
        inputs = [("x", "np.ndarray")]
    if outputs is None:
        outputs = [("result", "np.ndarray")]
    return MacroAtomSpec(
        name=name,
        inputs=[IOSpec(name=n, type_desc=t) for n, t in inputs],
        outputs=[IOSpec(name=n, type_desc=t) for n, t in outputs],
    )


# ---------------------------------------------------------------------------
# Tests: FFI imports
# ---------------------------------------------------------------------------


class TestFFIImports:
    def test_cpp_imports(self):
        result = generate_ffi_imports("cpp")
        assert "ctypes" in result
        assert "Path" in result

    def test_julia_imports(self):
        result = generate_ffi_imports("julia")
        assert "juliacall" in result
        assert "Main" in result

    def test_rust_imports(self):
        result = generate_ffi_imports("rust")
        assert "ctypes" in result

    def test_unknown_returns_empty(self):
        result = generate_ffi_imports("fortran")
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: FFI stubs
# ---------------------------------------------------------------------------


class TestFFIStubs:
    def test_cpp_stub_structure(self):
        atom = _make_atom()
        result = generate_ffi_stub(atom, "cpp")
        assert "def test_atom_ffi(x):" in result
        assert "ctypes.CDLL" in result
        assert "argtypes" in result
        assert "restype" in result

    def test_julia_stub_structure(self):
        atom = _make_atom()
        result = generate_ffi_stub(atom, "julia")
        assert "def test_atom_ffi(x):" in result
        assert "jl.eval" in result

    def test_cpp_stub_multiple_params(self):
        atom = _make_atom(inputs=[("a", "float"), ("b", "float")])
        result = generate_ffi_stub(atom, "cpp")
        assert "def test_atom_ffi(a, b):" in result

    def test_julia_stub_no_params(self):
        atom = _make_atom(inputs=[])
        result = generate_ffi_stub(atom, "julia")
        assert "def test_atom_ffi():" in result

    def test_rust_stub(self):
        atom = _make_atom()
        result = generate_ffi_stub(atom, "rust")
        assert "def test_atom_ffi(x):" in result
        assert "ctypes.CDLL" in result

    def test_unknown_language_returns_empty(self):
        atom = _make_atom()
        result = generate_ffi_stub(atom, "fortran")
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: full bindings
# ---------------------------------------------------------------------------


class TestFFIBindings:
    def test_cpp_full_bindings(self):
        atoms = [
            _make_atom("Signal Filter", inputs=[("signal", "np.ndarray")]),
            _make_atom("Normalizer", inputs=[("data", "np.ndarray")]),
        ]
        result = generate_ffi_bindings(atoms, "cpp")
        assert "ctypes" in result
        assert "signal_filter_ffi" in result
        assert "normalizer_ffi" in result
        assert "Auto-generated FFI bindings for cpp" in result

    def test_julia_full_bindings(self):
        atoms = [
            _make_atom("Epoch Offset", inputs=[("epoch", "Epoch")]),
        ]
        result = generate_ffi_bindings(atoms, "julia")
        assert "juliacall" in result
        assert "epoch_offset_ffi" in result

    def test_bindings_are_valid_python(self):
        atoms = [_make_atom()]
        for lang in ("cpp", "julia"):
            result = generate_ffi_bindings(atoms, lang)
            # Should be syntactically valid Python
            compile(result, "<test>", "exec")

    def test_empty_atoms(self):
        result = generate_ffi_bindings([], "cpp")
        assert "Auto-generated" in result
        # Should still be valid Python
        compile(result, "<test>", "exec")
