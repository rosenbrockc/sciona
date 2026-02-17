"""Python library indexer: extracts Declaration objects from Python packages."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field

from ageom.types import Declaration, Prover


@dataclass
class PythonFunctionInfo:
    """Extracted information about a Python function."""

    module: str
    qualname: str
    signature: str
    docstring: str
    constraints: list[str] = field(default_factory=list)
    return_type: str = ""
    parameter_types: dict[str, str] = field(default_factory=dict)
    source_code: str = ""


class PythonDeclarationSource:
    """Extracts Declaration objects from Python packages using ast + inspect."""

    def get_declarations_from_package(self, package_name: str) -> list[Declaration]:
        """Extract declarations from all modules in a Python package."""
        declarations: list[Declaration] = []
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return declarations

        package_path = getattr(package, "__path__", None)
        if package_path is None:
            # Single module, not a package
            return self.get_declarations_from_module(package_name)

        for importer, modname, ispkg in pkgutil.walk_packages(
            package_path, prefix=package_name + "."
        ):
            try:
                declarations.extend(self.get_declarations_from_module(modname))
            except Exception:
                continue

        return declarations

    def get_declarations_from_module(self, module_name: str) -> list[Declaration]:
        """Extract declarations from a single Python module."""
        declarations: list[Declaration] = []
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return declarations

        source: str | None = None
        try:
            source = inspect.getsource(module)
        except (OSError, TypeError):
            pass

        if source is None:
            # Fall back to runtime inspection
            return self._inspect_module_runtime(module, module_name)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return declarations

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                info = self._ast_extract_function(node, module_name)
                if info is not None:
                    type_sig = self._build_type_signature(info)
                    declarations.append(
                        Declaration(
                            name=info.qualname,
                            type_signature=type_sig,
                            docstring=info.docstring,
                            source_lib=module_name,
                            prover=Prover.PYTHON,
                            raw_code=info.source_code,
                        )
                    )

        return declarations

    def _inspect_module_runtime(
        self, module: object, module_name: str
    ) -> list[Declaration]:
        """Fallback: extract declarations from a module via runtime inspection."""
        declarations: list[Declaration] = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_"):
                continue
            try:
                sig = inspect.signature(obj)
            except (ValueError, TypeError):
                continue
            qualname = f"{module_name}.{name}"
            docstring = inspect.getdoc(obj) or ""
            type_sig = self._signature_to_type_string(sig)
            declarations.append(
                Declaration(
                    name=qualname,
                    type_signature=type_sig,
                    docstring=docstring,
                    source_lib=module_name,
                    prover=Prover.PYTHON,
                )
            )
        return declarations

    def _ast_extract_function(
        self, node: ast.FunctionDef, module: str
    ) -> PythonFunctionInfo | None:
        """Extract function info from an AST FunctionDef node."""
        if node.name.startswith("_"):
            return None

        # Extract parameter types from annotations
        parameter_types: dict[str, str] = {}
        params: list[str] = []
        for arg in node.args.args:
            arg_name = arg.arg
            if arg.annotation is not None:
                type_str = ast.unparse(arg.annotation)
                parameter_types[arg_name] = type_str
                params.append(f"{arg_name}: {type_str}")
            else:
                params.append(arg_name)

        # Extract return type
        return_type = ""
        if node.returns is not None:
            return_type = ast.unparse(node.returns)

        # Build signature string
        sig_parts = ", ".join(params)
        signature = f"({sig_parts})"
        if return_type:
            signature += f" -> {return_type}"

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Extract icontract decorators
        preconditions, postconditions = self._extract_icontract_decorators(node)
        constraints = preconditions + postconditions

        # Source code
        try:
            source_code = ast.unparse(node)
        except Exception:
            source_code = ""

        qualname = f"{module}.{node.name}"

        return PythonFunctionInfo(
            module=module,
            qualname=qualname,
            signature=signature,
            docstring=docstring,
            constraints=constraints,
            return_type=return_type,
            parameter_types=parameter_types,
            source_code=source_code,
        )

    def _extract_icontract_decorators(
        self, node: ast.FunctionDef
    ) -> tuple[list[str], list[str]]:
        """Extract icontract @require and @ensure decorator expressions."""
        preconditions: list[str] = []
        postconditions: list[str] = []

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            # Match icontract.require or icontract.ensure
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id

            if name == "require":
                if decorator.args:
                    preconditions.append(ast.unparse(decorator.args[0]))
            elif name == "ensure":
                if decorator.args:
                    postconditions.append(ast.unparse(decorator.args[0]))

        return preconditions, postconditions

    def _build_type_signature(self, info: PythonFunctionInfo) -> str:
        """Build a normalized type signature string."""
        parts: list[str] = []
        for name, type_str in info.parameter_types.items():
            parts.append(f"{name}: {type_str}")
        sig = f"({', '.join(parts)})"
        if info.return_type:
            sig += f" -> {info.return_type}"
        return sig

    def _signature_to_type_string(self, sig: inspect.Signature) -> str:
        """Convert an inspect.Signature to a type signature string."""
        parts: list[str] = []
        for name, param in sig.parameters.items():
            if param.annotation is not inspect.Parameter.empty:
                ann = (
                    param.annotation.__name__
                    if hasattr(param.annotation, "__name__")
                    else str(param.annotation)
                )
                parts.append(f"{name}: {ann}")
            else:
                parts.append(name)
        ret = ""
        if sig.return_annotation is not inspect.Signature.empty:
            ret_ann = sig.return_annotation
            ret = (
                ret_ann.__name__
                if hasattr(ret_ann, "__name__")
                else str(ret_ann)
            )
        result = f"({', '.join(parts)})"
        if ret:
            result += f" -> {ret}"
        return result
