"""FFI binding generation for C++ (ctypes) and Julia (juliacall).

Generates Python wrappers that call into foreign-language implementations,
so CDG output remains format-identical to pure Python outputs.
"""

from __future__ import annotations

from ageom.ingester.models import MacroAtomSpec


def _snake_case(name: str) -> str:
    """Convert a name like 'Signal Conditioner' to 'signal_conditioner'."""
    return name.lower().replace(" ", "_").replace("-", "_")


def generate_ffi_imports(language: str) -> str:
    """Generate the import block for FFI bindings.

    Args:
        language: ``"cpp"``, ``"julia"``, ``"rust"``, or ``"haskell"``.

    Returns:
        Python source code for the import section.
    """
    if language == "cpp":
        return "import ctypes\n" "import ctypes.util\n" "from pathlib import Path\n"
    elif language == "julia":
        return "from juliacall import Main as jl\n"
    elif language == "rust":
        return "import ctypes\n" "import ctypes.util\n" "from pathlib import Path\n"
    elif language == "haskell":
        return "import ctypes\n" "import ctypes.util\n" "from pathlib import Path\n"
    else:
        return ""


def generate_ffi_stub(atom: MacroAtomSpec, language: str) -> str:
    """Generate an FFI wrapper function for a single atom.

    Args:
        atom: The macro-atom specification.
        language: ``"cpp"``, ``"julia"``, ``"rust"``, or ``"haskell"``.

    Returns:
        Python source code for one FFI wrapper function.
    """
    fn_name = _snake_case(atom.name)

    # Build parameter list
    params = ", ".join(inp.name for inp in atom.inputs)

    # Build return type
    if atom.outputs:
        if len(atom.outputs) == 1:
            ret_type = atom.outputs[0].type_desc
        else:
            ret_type = "tuple[" + ", ".join(o.type_desc for o in atom.outputs) + "]"
    else:
        ret_type = "None"

    if language == "cpp":
        return _cpp_stub(fn_name, atom, params, ret_type)
    elif language == "julia":
        return _julia_stub(fn_name, atom, params, ret_type)
    elif language == "rust":
        return _rust_stub(fn_name, atom, params, ret_type)
    elif language == "haskell":
        return _haskell_stub(fn_name, atom, params, ret_type)
    else:
        return ""


def _cpp_stub(fn_name: str, atom: MacroAtomSpec, params: str, ret_type: str) -> str:
    """Generate a ctypes-based FFI stub for C++."""
    lines = [
        f"def {fn_name}_ffi({params}):",
        f'    """FFI bridge to C++ implementation of {atom.name}."""',
        f'    _lib = ctypes.CDLL("./{fn_name}.so")',
        f"    _func_name = atom.method_names[0] if atom.method_names else '{fn_name}'",
        "    _func = _lib[_func_name]",
    ]

    # Set argtypes
    ctypes_args = []
    for inp in atom.inputs:
        ctypes_args.append("ctypes.c_void_p")
    if ctypes_args:
        lines.append(f"    _func.argtypes = [{', '.join(ctypes_args)}]")

    # Set restype
    lines.append("    _func.restype = ctypes.c_void_p")
    lines.append(f"    return _func({params})")
    lines.append("")
    return "\n".join(lines)


def _julia_stub(fn_name: str, atom: MacroAtomSpec, params: str, ret_type: str) -> str:
    """Generate a juliacall-based FFI stub for Julia."""
    lines = [
        f"def {fn_name}_ffi({params}):",
        f'    """FFI bridge to Julia implementation of {atom.name}."""',
        f'    return jl.eval("{fn_name}({params})")',
        "",
    ]
    return "\n".join(lines)


def generate_ffi_bindings(atoms: list[MacroAtomSpec], language: str) -> str:
    """Generate a complete FFI binding module.

    Args:
        atoms: List of macro-atom specifications.
        language: ``"cpp"``, ``"julia"``, ``"rust"``, or ``"haskell"``.

    Returns:
        Complete Python source code for the FFI binding module.
    """
    lines = [
        f'"""Auto-generated FFI bindings for {language} implementations."""',
        "",
        "from __future__ import annotations",
        "",
        generate_ffi_imports(language),
        "",
    ]

    for atom in atoms:
        lines.append(generate_ffi_stub(atom, language))

    return "\n".join(lines)


def _rust_stub(fn_name: str, atom: MacroAtomSpec, params: str, ret_type: str) -> str:
    """Generate a ctypes-based FFI stub for Rust."""
    lines = [
        f"def {fn_name}_ffi({params}):",
        f'    """FFI bridge to Rust implementation of {atom.name}."""',
        '    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"',
        '    _lib = ctypes.CDLL("./target/release/librust_robotics.so")',
        f"    _func_name = atom.method_names[0] if atom.method_names else '{fn_name}'",
        "    _func = _lib[_func_name]",
    ]

    # Set argtypes
    ctypes_args = []
    for inp in atom.inputs:
        ctypes_args.append("ctypes.c_void_p")
    if ctypes_args:
        lines.append(f"    _func.argtypes = [{', '.join(ctypes_args)}]")

    # Set restype
    lines.append("    _func.restype = ctypes.c_void_p")
    lines.append(f"    return _func({params})")
    lines.append("")
    return "\n".join(lines)


def _haskell_stub(fn_name: str, atom: MacroAtomSpec, params: str, ret_type: str) -> str:
    """Generate a ctypes-based FFI stub for Haskell."""
    lines = [
        f"def {fn_name}_ffi({params}):",
        f'    """FFI bridge to Haskell implementation of {atom.name}."""',
        f'    # Ensure Haskell is compiled with -dynamic -fPIC and has hs_init()',
        f'    _lib = ctypes.CDLL("./{fn_name}.so")',
        f"    _func_name = atom.method_names[0] if atom.method_names else '{fn_name}'",
        "    _func = _lib[_func_name]",
    ]

    # Set argtypes
    ctypes_args = []
    for inp in atom.inputs:
        ctypes_args.append("ctypes.c_void_p")
    if ctypes_args:
        lines.append(f"    _func.argtypes = [{', '.join(ctypes_args)}]")

    # Set restype
    lines.append("    _func.restype = ctypes.c_void_p")
    lines.append(f"    return _func({params})")
    lines.append("")
    return "\n".join(lines)
