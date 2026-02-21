"""Safe atom wrapper generator with icontract decorators."""

from __future__ import annotations

from dataclasses import dataclass, field

from ageom.architect.models import IOSpec
from ageom.synthesizer.models import AssemblyUnit
from ageom.types import Declaration


@dataclass
class ContractSpec:
    """A single contract (precondition or postcondition)."""

    kind: str  # "require" or "ensure"
    lambda_expr: str
    description: str = ""


@dataclass
class SafeAtomWrapper:
    """A Python function that wraps a library call with icontract decorators."""

    function_name: str
    original_qualname: str
    imports: list[str] = field(default_factory=list)
    parameters: list[tuple[str, str]] = field(default_factory=list)  # (name, type)
    return_type: str = ""
    preconditions: list[ContractSpec] = field(default_factory=list)
    postconditions: list[ContractSpec] = field(default_factory=list)
    body: str = ""


class ContractGenerator:
    """Generates icontract-decorated wrapper functions."""

    def generate_wrapper(
        self, unit: AssemblyUnit, declaration: Declaration
    ) -> SafeAtomWrapper:
        """Generate a SafeAtomWrapper from an AssemblyUnit and Declaration."""
        # Parse parameters from the declaration's type signature
        parameters: list[tuple[str, str]] = []
        for inp in unit.inputs:
            parameters.append((inp.name, inp.type_desc))

        return_type = ""
        if unit.outputs:
            return_type = unit.outputs[0].type_desc

        # Build contracts from IOSpec constraints
        preconditions: list[ContractSpec] = []
        for inp in unit.inputs:
            contract = self._iospec_to_contract(inp, "require")
            if contract is not None:
                preconditions.append(contract)

        postconditions: list[ContractSpec] = []
        for out in unit.outputs:
            contract = self._iospec_to_contract(out, "ensure")
            if contract is not None:
                postconditions.append(contract)

        # Infer imports from the declaration
        imports = ["import icontract"]
        module = declaration.source_lib
        if module:
            top_level = module.split(".")[0]
            imports.append(f"import {top_level}")

        # Build body
        body = f"return {declaration.name}({', '.join(name for name, _ in parameters)})"

        function_name = (
            unit.name.replace(" ", "_").replace("-", "_").lower() + "_wrapper"
        )

        return SafeAtomWrapper(
            function_name=function_name,
            original_qualname=declaration.name,
            imports=imports,
            parameters=parameters,
            return_type=return_type,
            preconditions=preconditions,
            postconditions=postconditions,
            body=body,
        )

    def render_wrapper(self, wrapper: SafeAtomWrapper) -> str:
        """Render a SafeAtomWrapper to Python source code."""
        lines: list[str] = []

        # Decorators
        for pre in wrapper.preconditions:
            desc = f', "{pre.description}"' if pre.description else ""
            lines.append(f"@icontract.require({pre.lambda_expr}{desc})")

        for post in wrapper.postconditions:
            desc = f', "{post.description}"' if post.description else ""
            lines.append(f"@icontract.ensure({post.lambda_expr}{desc})")

        # Function signature
        params = ", ".join(
            f"{name}: {typ}" if typ else name for name, typ in wrapper.parameters
        )
        ret = f" -> {wrapper.return_type}" if wrapper.return_type else ""
        lines.append(f"def {wrapper.function_name}({params}){ret}:")

        # Body
        if wrapper.body:
            for body_line in wrapper.body.splitlines():
                lines.append(f"    {body_line}")
        else:
            lines.append(
                f'    raise NotImplementedError("TODO: compose {wrapper.original_qualname}")'
            )

        return "\n".join(lines)

    def _iospec_to_contract(self, spec: IOSpec, kind: str) -> ContractSpec | None:
        """Convert an IOSpec constraint to a ContractSpec.

        Recognizes DSP-specific constraint patterns:
        - "round_trip: INVERSE(FORWARD(x))" -> epsilon-metric postcondition
        - "poles inside unit circle" -> stability postcondition
        - "positive semi-definite" -> eigenvalue check
        - "TV(result) <= TV(input)" -> total variation reduction
        """
        if not spec.constraints:
            return None

        constraint = spec.constraints.strip()

        # DSP pattern: round-trip epsilon-metric
        if constraint.startswith("round_trip:"):
            expr = constraint[len("round_trip:") :].strip()
            lambda_expr = (
                f"lambda result, {spec.name}: " f"np.allclose({expr}, atol=1e-10)"
            )
            return ContractSpec(
                kind="ensure",
                lambda_expr=lambda_expr,
                description=f"Round-trip: {expr}",
            )

        # DSP pattern: filter stability
        if constraint == "poles inside unit circle":
            lambda_expr = "lambda result: _poles_inside_unit_circle(result[1])"
            return ContractSpec(
                kind="ensure",
                lambda_expr=lambda_expr,
                description="Filter must be stable (poles inside unit circle)",
            )

        # DSP pattern: positive semi-definite
        if constraint == "positive semi-definite":
            lambda_expr = "lambda result: _eigenvalues_nonneg(result, k=1)"
            return ContractSpec(
                kind="ensure",
                lambda_expr=lambda_expr,
                description="Matrix must be positive semi-definite",
            )

        # DSP pattern: total variation reduction
        if constraint == "TV(result) <= TV(input)":
            lambda_expr = (
                f"lambda result, {spec.name}: "
                f"_total_variation(L, result) <= _total_variation(L, {spec.name}) + 1e-8"
            )
            return ContractSpec(
                kind="ensure",
                lambda_expr=lambda_expr,
                description="Output total variation must not exceed input total variation",
            )

        # Default: generic constraint
        if kind == "require":
            lambda_expr = f"lambda {spec.name}: {constraint}"
        else:
            lambda_expr = f"lambda result: {constraint}"

        return ContractSpec(
            kind=kind,
            lambda_expr=lambda_expr,
            description=f"{spec.name}: {constraint}",
        )
