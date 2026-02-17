"""Template for generating lakefile.lean for Lean 4 builds."""

from __future__ import annotations


def generate_lakefile(
    name: str,
    lean_version: str = "leanprover/lean4:v4.14.0",
    deps: list[str] | None = None,
    ffi_export: bool = False,
) -> str:
    """Generate a lakefile.lean string for building the verified source.

    Args:
        name: Package/library name.
        lean_version: Lean toolchain version string.
        deps: Additional lake dependencies beyond Mathlib.
        ffi_export: If True, add moreLinkArgs for shared library export.
    """
    dep_lines = ['require mathlib from git\n  "https://github.com/leanprover-community/mathlib4"']
    for dep in deps or []:
        dep_lines.append(f'require {dep}')

    deps_block = "\n\n".join(dep_lines)

    lib_block = f'lean_lib {name} where\n  srcDir := "src"'
    if ffi_export:
        lib_block += '\n  moreLinkArgs := #["-shared"]'

    return f"""\
import Lake
open Lake DSL

package {name} where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

{deps_block}

{lib_block}
"""
