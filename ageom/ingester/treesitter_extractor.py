"""Tree-sitter based extractor for C++ and Julia source files.

Parses foreign-language classes/structs via tree-sitter and produces the
same ``RawDataFlowGraph`` schema as the Python AST extractor, enabling
downstream phases (chunker, emitter) to work unchanged.
"""

from __future__ import annotations

from pathlib import Path

from tree_sitter import Node as TSNode
from tree_sitter_language_pack import get_parser

from ageom.ingester.base_extractor import SourceLanguage
from ageom.ingester.models import MethodFact, RawDataFlowGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_text(node: TSNode) -> str:
    """Decode a tree-sitter node's text to a Python string."""
    return node.text.decode("utf-8") if node.text else ""


def _find_children(node: TSNode, type_name: str) -> list[TSNode]:
    """Return direct children matching *type_name*."""
    return [c for c in node.children if c.type == type_name]


def _find_descendants(node: TSNode, type_name: str) -> list[TSNode]:
    """Return all descendants (BFS) matching *type_name*."""
    result: list[TSNode] = []
    stack = list(node.children)
    while stack:
        n = stack.pop(0)
        if n.type == type_name:
            result.append(n)
        stack.extend(n.children)
    return result


# ---------------------------------------------------------------------------
# C++ class visitor
# ---------------------------------------------------------------------------


class _CppClassVisitor:
    """Extract class metadata from a ``class_specifier`` tree-sitter node."""

    def __init__(self, class_node: TSNode, source_code: str) -> None:
        self.class_node = class_node
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.class_name = ""
        self.base_classes: list[str] = []
        self.fields: list[tuple[str, str]] = []  # (name, type)
        self.methods: list[MethodFact] = []
        self.known_fields: set[str] = set()

    def visit(self) -> None:
        """Walk the class_specifier and extract all metadata."""
        # Class name
        for child in self.class_node.children:
            if child.type == "type_identifier":
                self.class_name = _node_text(child)
                break

        # Base classes
        for child in self.class_node.children:
            if child.type == "base_class_clause":
                for bc_child in child.children:
                    if bc_child.type == "type_identifier":
                        self.base_classes.append(_node_text(bc_child))

        # Field declaration list
        field_list = None
        for child in self.class_node.children:
            if child.type == "field_declaration_list":
                field_list = child
                break

        if field_list is None:
            return

        # First pass: collect fields
        current_access = "private"  # C++ default
        for child in field_list.children:
            if child.type == "access_specifier":
                current_access = _node_text(child).rstrip(":")
            elif child.type == "field_declaration":
                self._visit_field_declaration(child, current_access)

        self.known_fields = {name for name, _ in self.fields}

        # Second pass: collect methods (need known_fields for bare member detection)
        current_access = "private"
        for child in field_list.children:
            if child.type == "access_specifier":
                current_access = _node_text(child).rstrip(":")
            elif child.type == "function_definition":
                self._visit_method(child, current_access)

    def _visit_field_declaration(self, node: TSNode, access: str) -> None:
        """Extract a member variable declaration."""
        type_parts: list[str] = []
        field_name = ""
        for child in node.children:
            if child.type == "field_identifier":
                field_name = _node_text(child)
            elif child.type in (
                "primitive_type", "type_identifier", "qualified_identifier",
                "sized_type_specifier", "template_type",
            ):
                type_parts.append(_node_text(child))
        if field_name:
            type_str = " ".join(type_parts) if type_parts else "auto"
            self.fields.append((field_name, type_str))

    def _visit_method(self, node: TSNode, access: str) -> None:
        """Extract a method definition and track field accesses in its body."""
        method_name = ""
        params: list[str] = []
        return_type = ""

        # Find the function declarator for name + params
        for child in node.children:
            if child.type == "function_declarator":
                for fc in child.children:
                    if fc.type in ("field_identifier", "identifier"):
                        method_name = _node_text(fc)
                    elif fc.type == "parameter_list":
                        params = self._extract_params(fc)
            elif child.type in (
                "primitive_type", "type_identifier", "qualified_identifier",
                "sized_type_specifier", "template_type",
            ):
                return_type = _node_text(child)

        # Source code
        start_line = node.start_point[0]
        end_line = node.end_point[0] + 1
        method_source = "\n".join(self.source_lines[start_line:end_line])

        # Walk body for field accesses
        reads: set[str] = set()
        writes: set[str] = set()
        calls: set[str] = set()

        body = None
        for child in node.children:
            if child.type == "compound_statement":
                body = child
                break

        # Also check field_initializer_list (constructor)
        init_list = None
        for child in node.children:
            if child.type == "field_initializer_list":
                init_list = child
                break

        if init_list:
            for init_node in _find_children(init_list, "field_initializer"):
                for ic in init_node.children:
                    if ic.type == "field_identifier":
                        writes.add(_node_text(ic))

        if body:
            self._walk_body_accesses(body, reads, writes, calls)

        is_constructor = method_name == self.class_name

        self.methods.append(MethodFact(
            name="__init__" if is_constructor else method_name,
            params=params,
            return_type=return_type,
            reads=sorted(reads),
            writes=sorted(writes),
            calls=sorted(calls),
            source_code=method_source,
        ))

    def _extract_params(self, param_list: TSNode) -> list[str]:
        """Extract parameter names from a parameter_list node."""
        params: list[str] = []
        for child in param_list.children:
            if child.type == "parameter_declaration":
                for pc in child.children:
                    if pc.type == "identifier":
                        params.append(_node_text(pc))
        return params

    def _walk_body_accesses(
        self,
        node: TSNode,
        reads: set[str],
        writes: set[str],
        calls: set[str],
    ) -> None:
        """Recursively walk a method body for this->field accesses and bare member accesses."""
        # Check for assignment targets (writes)
        if node.type in ("assignment_expression",):
            lhs = node.children[0] if node.children else None
            if lhs is not None:
                field = self._extract_member_field(lhs)
                if field:
                    writes.add(field)
                # Walk RHS for reads
                for child in node.children[1:]:
                    self._walk_body_accesses(child, reads, writes, calls)
                return

        # Check for field_expression reads: this->field or bare member
        if node.type == "field_expression":
            field = self._extract_member_field(node)
            if field:
                reads.add(field)
            # Still recurse into children for nested expressions
            for child in node.children:
                self._walk_body_accesses(child, reads, writes, calls)
            return

        # Check for bare member access used as identifier in expressions
        if node.type == "identifier":
            name = _node_text(node)
            if name in self.known_fields and name != "this":
                # Check parent context to determine read vs write
                # (writes are handled at assignment_expression level)
                reads.add(name)
            return

        # Check for this->method() calls
        if node.type == "call_expression":
            func_node = node.children[0] if node.children else None
            if func_node and func_node.type == "field_expression":
                field = self._extract_member_field(func_node)
                if field:
                    calls.add(field)
                    # Don't add to reads — it's a call, not a field read
                    for child in node.children[1:]:
                        self._walk_body_accesses(child, reads, writes, calls)
                    return

        # Recurse
        for child in node.children:
            self._walk_body_accesses(child, reads, writes, calls)

    def _extract_member_field(self, node: TSNode) -> str | None:
        """Extract the field name from this->field or a bare known field."""
        if node.type == "field_expression":
            children = node.children
            if len(children) >= 3:
                obj = children[0]
                field_node = children[-1]
                if field_node.type == "field_identifier":
                    field_name = _node_text(field_node)
                    # this->field
                    if obj.type == "this":
                        return field_name
                    # Could also be obj.field for known patterns
            return None
        if node.type == "identifier":
            name = _node_text(node)
            if name in self.known_fields:
                return name
        return None


# ---------------------------------------------------------------------------
# Julia struct/function visitors
# ---------------------------------------------------------------------------


class _JuliaStructVisitor:
    """Extract struct metadata from a ``struct_definition`` tree-sitter node."""

    def __init__(self, struct_node: TSNode) -> None:
        self.struct_node = struct_node
        self.struct_name = ""
        self.type_params: list[str] = []
        self.fields: list[tuple[str, str]] = []  # (name, type)

    def visit(self) -> None:
        for child in self.struct_node.children:
            if child.type == "type_head":
                self._visit_type_head(child)
            elif child.type == "identifier":
                # Simple struct without type params
                if not self.struct_name:
                    self.struct_name = _node_text(child)
            elif child.type == "typed_expression":
                self._visit_field(child)

    def _visit_type_head(self, node: TSNode) -> None:
        for child in node.children:
            if child.type == "parametrized_type_expression":
                for pc in child.children:
                    if pc.type == "identifier":
                        self.struct_name = _node_text(pc)
                    elif pc.type == "curly_expression":
                        for ce in pc.children:
                            if ce.type == "identifier":
                                self.type_params.append(_node_text(ce))
            elif child.type == "identifier":
                if not self.struct_name:
                    self.struct_name = _node_text(child)

    def _visit_field(self, node: TSNode) -> None:
        """Extract a typed field like ``dur::T``."""
        parts = [c for c in node.children if c.type == "identifier"]
        if len(parts) >= 2:
            field_name = _node_text(parts[0])
            field_type = _node_text(parts[1])
            self.fields.append((field_name, field_type))
        elif len(parts) == 1:
            self.fields.append((_node_text(parts[0]), "Any"))


class _JuliaFunctionVisitor:
    """Extract function metadata and associate methods with structs."""

    def __init__(
        self,
        func_node: TSNode,
        source_code: str,
        struct_names: set[str],
        struct_fields: dict[str, set[str]],
    ) -> None:
        self.func_node = func_node
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.struct_names = struct_names
        self.struct_fields = struct_fields

        self.func_name = ""
        self.params: list[str] = []
        self.return_type = ""
        self.associated_struct: str | None = None
        self.self_param_name: str | None = None
        self.reads: set[str] = set()
        self.writes: set[str] = set()

    def visit(self) -> None:
        # Extract signature
        for child in self.func_node.children:
            if child.type == "signature":
                self._visit_signature(child)

        # Walk body for field accesses
        for child in self.func_node.children:
            if child.type not in ("function", "signature", "end"):
                self._walk_body(child)

    def _visit_signature(self, sig_node: TSNode) -> None:
        """Parse function signature for name, params, return type."""
        # The signature can be:
        #   call_expression (no return type) or
        #   typed_expression wrapping call_expression (with return type)
        for child in sig_node.children:
            if child.type == "call_expression":
                self._visit_call_sig(child)
            elif child.type == "typed_expression":
                self._visit_typed_sig(child)

    def _visit_typed_sig(self, node: TSNode) -> None:
        """Handle ``func_name(params)::ReturnType``."""
        children = [c for c in node.children if c.type != "::"]
        for child in children:
            if child.type == "call_expression":
                self._visit_call_sig(child)
            elif child.type == "identifier":
                self.return_type = _node_text(child)

    def _visit_call_sig(self, node: TSNode) -> None:
        """Handle ``func_name(params)``."""
        for child in node.children:
            if child.type == "identifier":
                self.func_name = _node_text(child)
            elif child.type == "argument_list":
                self._visit_arg_list(child)

    def _visit_arg_list(self, node: TSNode) -> None:
        """Extract parameters and detect struct association."""
        first_param = True
        for child in node.children:
            if child.type == "typed_expression":
                parts = [c for c in child.children if c.type == "identifier"]
                if len(parts) >= 2:
                    param_name = _node_text(parts[0])
                    param_type = _node_text(parts[1])
                    self.params.append(param_name)
                    # Check if first param is typed as a known struct
                    if first_param and param_type in self.struct_names:
                        self.associated_struct = param_type
                        self.self_param_name = param_name
                elif len(parts) == 1:
                    self.params.append(_node_text(parts[0]))
                first_param = False
            elif child.type == "identifier":
                self.params.append(_node_text(child))
                first_param = False

    def _walk_body(self, node: TSNode) -> None:
        """Walk function body tracking field accesses on the struct parameter."""
        if node.type == "field_expression":
            self._check_field_access(node)

        for child in node.children:
            self._walk_body(child)

    def _check_field_access(self, node: TSNode) -> None:
        """Check if a field_expression accesses the struct parameter's fields."""
        if self.self_param_name is None:
            return

        children = [c for c in node.children if c.type not in (".",)]
        if len(children) >= 2:
            obj_node = children[0]
            field_node = children[1]
            if (
                obj_node.type == "identifier"
                and _node_text(obj_node) == self.self_param_name
                and field_node.type == "identifier"
            ):
                field_name = _node_text(field_node)
                # Check if this is an assignment target
                parent = node.parent
                if parent and parent.type in (
                    "assignment_expression", "assignment",
                ):
                    # LHS of assignment → write
                    lhs = parent.children[0] if parent.children else None
                    if lhs == node:
                        self.writes.add(field_name)
                        return
                self.reads.add(field_name)


# ---------------------------------------------------------------------------
# C++ procedural extraction
# ---------------------------------------------------------------------------


def _extract_cpp_functions(root: TSNode, source_code: str) -> list[MethodFact]:
    """Extract top-level function definitions from C++ source."""
    source_lines = source_code.splitlines()
    methods: list[MethodFact] = []

    for child in root.children:
        if child.type == "function_definition":
            func_name = ""
            params: list[str] = []
            return_type = ""

            for fc in child.children:
                if fc.type == "function_declarator":
                    for fcc in fc.children:
                        if fcc.type in ("identifier", "field_identifier"):
                            func_name = _node_text(fcc)
                        elif fcc.type == "parameter_list":
                            for pc in fcc.children:
                                if pc.type == "parameter_declaration":
                                    for pcc in pc.children:
                                        if pcc.type == "identifier":
                                            params.append(_node_text(pcc))
                elif fc.type in (
                    "primitive_type", "type_identifier",
                    "qualified_identifier",
                ):
                    return_type = _node_text(fc)

            start = child.start_point[0]
            end = child.end_point[0] + 1
            source = "\n".join(source_lines[start:end])

            if func_name:
                methods.append(MethodFact(
                    name=func_name,
                    params=params,
                    return_type=return_type,
                    source_code=source,
                ))

    return methods


# ---------------------------------------------------------------------------
# Julia procedural extraction
# ---------------------------------------------------------------------------


def _extract_julia_functions_procedural(
    root: TSNode,
    source_code: str,
) -> list[MethodFact]:
    """Extract top-level functions not associated with any struct."""
    source_lines = source_code.splitlines()

    # Collect struct names
    struct_names: set[str] = set()
    for child in root.children:
        if child.type == "struct_definition":
            v = _JuliaStructVisitor(child)
            v.visit()
            if v.struct_name:
                struct_names.add(v.struct_name)

    methods: list[MethodFact] = []
    for child in root.children:
        if child.type == "function_definition":
            fv = _JuliaFunctionVisitor(child, source_code, struct_names, {})
            fv.visit()
            # Only include functions NOT associated with a struct
            if fv.associated_struct is None and fv.func_name:
                start = child.start_point[0]
                end = child.end_point[0] + 1
                source = "\n".join(source_lines[start:end])
                methods.append(MethodFact(
                    name=fv.func_name,
                    params=fv.params,
                    return_type=fv.return_type,
                    source_code=source,
                ))

    return methods


# ---------------------------------------------------------------------------
# TreeSitterExtractor — public API
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Rust visitors
# ---------------------------------------------------------------------------

class _RustStructVisitor:
    def __init__(self, node: TSNode) -> None:
        self.node = node
        self.struct_name = ""
        self.fields: list[tuple[str, str]] = []

    def visit(self) -> None:
        for child in self.node.children:
            if child.type == "type_identifier":
                self.struct_name = _node_text(child)
            elif child.type == "field_declaration_list":
                for field in _find_children(child, "field_declaration"):
                    fname = ""
                    ftype = ""
                    for fc in field.children:
                        if fc.type == "field_identifier":
                            fname = _node_text(fc)
                        elif fc.type in ("type_identifier", "primitive_type", "generic_type"):
                            ftype = _node_text(fc)
                    if fname:
                        self.fields.append((fname, ftype or "Any"))

class _RustFunctionVisitor:
    def __init__(self, node: TSNode, source_code: str) -> None:
        self.node = node
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.func_name = ""
        self.params: list[str] = []
        self.return_type = ""
        self.reads: set[str] = set()
        self.writes: set[str] = set()

    def visit(self) -> None:
        for child in self.node.children:
            if child.type == "identifier":
                self.func_name = _node_text(child)
            elif child.type == "parameters":
                for param in _find_children(child, "parameter"):
                    for pc in param.children:
                        if pc.type == "identifier":
                            self.params.append(_node_text(pc))
            elif child.type == "type_identifier": # Return type
                self.return_type = _node_text(child)

    def get_fact(self) -> MethodFact:
        start = self.node.start_point[0]
        end = self.node.end_point[0] + 1
        return MethodFact(
            name=self.func_name,
            params=self.params,
            return_type=self.return_type,
            reads=sorted(self.reads),
            writes=sorted(self.writes),
            source_code="\n".join(self.source_lines[start:end]),
        )

class TreeSitterExtractor:
    """Extract data-flow graphs from C++ or Julia source via tree-sitter."""

    def __init__(self, language: SourceLanguage) -> None:
        if language == SourceLanguage.PYTHON:
            raise ValueError("Use PythonASTExtractor for Python sources")
        self.language = language
        if language == SourceLanguage.CPP:
            lang_key = "cpp"
        elif language == SourceLanguage.JULIA:
            lang_key = "julia"
        else:
            lang_key = "rust"
        self._parser = get_parser(lang_key)

    async def extract_class(
        self, source_path: str, class_name: str
    ) -> RawDataFlowGraph:
        """Extract a class (C++) or struct (Julia) data-flow graph."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        source_code = path.read_text()
        tree = self._parser.parse(source_code.encode("utf-8"))
        root = tree.root_node

        if self.language == SourceLanguage.CPP:
            return self._extract_cpp_class(root, source_code, class_name)
        elif self.language == SourceLanguage.JULIA:
            return self._extract_julia_struct(root, source_code, class_name)
        else:
            return self._extract_rust_struct(root, source_code, class_name)

    async def extract_procedural(
        self, source_path: str, pipeline_name: str | None = None
    ) -> RawDataFlowGraph:
        """Extract procedural/top-level functions."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        source_code = path.read_text()
        name = pipeline_name or path.stem
        tree = self._parser.parse(source_code.encode("utf-8"))
        root = tree.root_node

        if self.language == SourceLanguage.CPP:
            methods = _extract_cpp_functions(root, source_code)
        elif self.language == SourceLanguage.JULIA:
            methods = _extract_julia_functions_procedural(root, source_code)
        else:
            methods = self._extract_rust_functions_procedural(root, source_code)

        # Build all_attributes from method reads/writes
        all_attributes: dict[str, list[str]] = {}
        for mf in methods:
            for attr in mf.reads:
                all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
            for attr in mf.writes:
                all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

        return RawDataFlowGraph(
            class_name=name,
            source_code=source_code,
            methods=methods,
            all_attributes=all_attributes,
            source_language=self.language.value,
        )

    # --- C++ extraction ---

    def _extract_cpp_class(
        self, root: TSNode, source_code: str, class_name: str
    ) -> RawDataFlowGraph:
        """Find and extract a C++ class by name."""
        class_nodes = _find_descendants(root, "class_specifier")

        target: TSNode | None = None
        for cn in class_nodes:
            for child in cn.children:
                if child.type == "type_identifier" and _node_text(child) == class_name:
                    target = cn
                    break
            if target:
                break

        if target is None:
            raise ValueError(f"Class '{class_name}' not found in source")

        visitor = _CppClassVisitor(target, source_code)
        visitor.visit()

        # Build all_attributes index
        all_attributes: dict[str, list[str]] = {}
        for mf in visitor.methods:
            for attr in mf.reads:
                all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
            for attr in mf.writes:
                all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

        # Init chain from constructor
        init_method = next(
            (m for m in visitor.methods if m.name == "__init__"), None
        )
        init_chain = list(init_method.writes) if init_method else []

        # Internal call graph
        method_names = {m.name for m in visitor.methods}
        internal_call_graph: dict[str, list[str]] = {}
        for mf in visitor.methods:
            internal_calls = [c for c in mf.calls if c in method_names]
            if internal_calls:
                internal_call_graph[mf.name] = internal_calls

        return RawDataFlowGraph(
            class_name=class_name,
            source_code=source_code,
            methods=visitor.methods,
            all_attributes=all_attributes,
            init_chain=init_chain,
            internal_call_graph=internal_call_graph,
            source_language="cpp",
        )

    # --- Julia extraction ---

    def _extract_julia_struct(
        self, root: TSNode, source_code: str, struct_name: str
    ) -> RawDataFlowGraph:
        """Find and extract a Julia struct by name, associating methods."""
        # Collect all structs
        struct_visitors: list[_JuliaStructVisitor] = []
        for child in root.children:
            if child.type == "struct_definition":
                sv = _JuliaStructVisitor(child)
                sv.visit()
                struct_visitors.append(sv)

        # Find target struct
        target_sv: _JuliaStructVisitor | None = None
        for sv in struct_visitors:
            if sv.struct_name == struct_name:
                target_sv = sv
                break

        if target_sv is None:
            raise ValueError(f"Struct '{struct_name}' not found in source")

        struct_names = {sv.struct_name for sv in struct_visitors}
        struct_fields = {
            sv.struct_name: {f[0] for f in sv.fields}
            for sv in struct_visitors
        }

        # Find functions associated with this struct
        methods: list[MethodFact] = []
        source_lines = source_code.splitlines()
        for child in root.children:
            if child.type == "function_definition":
                fv = _JuliaFunctionVisitor(
                    child, source_code, struct_names, struct_fields
                )
                fv.visit()
                if fv.associated_struct == struct_name:
                    start = child.start_point[0]
                    end = child.end_point[0] + 1
                    source = "\n".join(source_lines[start:end])
                    # Params: skip the self-like first param
                    params = fv.params[1:] if fv.params else []
                    methods.append(MethodFact(
                        name=fv.func_name,
                        params=params,
                        return_type=fv.return_type,
                        reads=sorted(fv.reads),
                        writes=sorted(fv.writes),
                        source_code=source,
                    ))

        # Build all_attributes
        all_attributes: dict[str, list[str]] = {}
        for mf in methods:
            for attr in mf.reads:
                all_attributes.setdefault(attr, []).append(f"read:{mf.name}")
            for attr in mf.writes:
                all_attributes.setdefault(attr, []).append(f"write:{mf.name}")

        return RawDataFlowGraph(
            class_name=struct_name,
            source_code=source_code,
            methods=methods,
            all_attributes=all_attributes,
            source_language="julia",
        )
    def _extract_rust_struct(self, root: TSNode, source_code: str, name: str) -> RawDataFlowGraph:
        struct_nodes = _find_descendants(root, "struct_item")
        target = None
        for sn in struct_nodes:
            v = _RustStructVisitor(sn)
            v.visit()
            if v.struct_name == name:
                target = v
                break
        
        if not target:
            raise ValueError(f"Rust struct {name} not found")

        methods = []
        impl_nodes = _find_descendants(root, "impl_item")
        for im in impl_nodes:
            is_target = False
            for child in im.children:
                if child.type == "type_identifier" and _node_text(child) == name:
                    is_target = True
                    break
            
            if is_target:
                for child in im.children:
                    if child.type == "function_item":
                        fv = _RustFunctionVisitor(child, source_code)
                        fv.visit()
                        methods.append(fv.get_fact())

        return RawDataFlowGraph(
            class_name=name,
            source_code=source_code,
            methods=methods,
            source_language="rust"
        )

    def _extract_rust_functions_procedural(self, root: TSNode, source_code: str) -> list[MethodFact]:
        func_nodes = _find_descendants(root, "function_item")
        methods = []
        for fn in func_nodes:
            if fn.parent and fn.parent.type == "source_file":
                fv = _RustFunctionVisitor(fn, source_code)
                fv.visit()
                methods.append(fv.get_fact())
        return methods
