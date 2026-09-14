"""Pinned PDG scalar definitions for reviewable symbolic correspondence.

These definitions are source claims. They do not certify a formula, its domain,
its dimensional validity, or a mapping inferred only from positional similarity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from sciona.ghost.dimensions import DimensionalSignature


@dataclass(frozen=True)
class PdgScalarDefinition:
    source_id: str
    latex: str
    dimension: DimensionalSignature | None


_FIELDS = (
    'time', 'electric_charge', 'luminous_intensity', 'length',
    'amount_of_substance', 'mass', 'temperature',
)
_PROPERTY = re.compile(r'(\w+)\s*:\s*("(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?|true|false)')
_ROW = re.compile(
    r'^\[\{id:"(\d+)",\s*properties:\{(.*?)\}\}\] AS row\s*'
    r'CREATE \(n:scalar\{id: row.id\}\) SET n \+= row.properties SET n:symbol;', re.S,
)


def load_pinned_pdg_scalars(content: bytes, expected_sha256: str) -> dict[str, PdgScalarDefinition]:
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('symbol definition content does not match the ingestion pin')
    result = {}
    for block in re.split(r'(?m)^UNWIND ', content.decode('utf-8'))[1:]:
        match = _ROW.match(block)
        if not match:
            continue
        source_id, properties = match.groups()
        props = {}
        for item in _PROPERTY.finditer(properties):
            key, value = item.groups()
            if key in props:
                raise ValueError('duplicate scalar property')
            props[key] = value
        if 'pdg' + source_id in result:
            raise ValueError('duplicate scalar identity')
        latex = props.get('latex', '')
        if not latex.startswith('"'):
            raise ValueError('scalar definition is missing its LaTeX representation')
        latex = latex[1:-1].replace(r'\"', '"')
        raw_dimensions = [props.get('dimension_' + field, '') for field in _FIELDS]
        dimension = None
        if all(re.fullmatch(r'-?\d+', value) for value in raw_dimensions):
            time, charge, luminous, length, amount, mass, temperature = map(int, raw_dimensions)
            # PDG uses charge, while the SI signature uses current. Q = I*T.
            dimension = DimensionalSignature(
                M=mass, L=length, T=time + charge, I=charge,
                Theta=temperature, N=amount, J=luminous,
            )
        result['pdg' + source_id] = PdgScalarDefinition(source_id, latex, dimension)
    if not result:
        raise ValueError('no supported scalar definitions found')
    return result


def map_pdg_scalar_names(expression, definitions: dict[str, PdgScalarDefinition]):
    """Map only source-defined atomic symbols; reject collisions and ambiguity."""
    import sympy as sp
    from sympy.parsing.latex import parse_latex

    replacements = {}
    unresolved = []
    targets = {}
    for symbol in expression.free_symbols:
        definition = definitions.get(str(symbol))
        if definition is None:
            unresolved.append(str(symbol))
            continue
        try:
            target = parse_latex(definition.latex, strict=True)
        except Exception:
            unresolved.append(str(symbol))
            continue
        if not isinstance(target, sp.Symbol) or target in targets:
            unresolved.append(str(symbol))
            if target in targets:
                prior = targets[target]
                unresolved.append(str(prior))
                replacements.pop(prior, None)
            continue
        targets[target] = symbol
        replacements[symbol] = target
    # A mapped identity must also not collide with a preserved literal symbol.
    for source, target in list(replacements.items()):
        if target in expression.free_symbols and target != source:
            unresolved.append(str(source))
            replacements.pop(source)
    return replacements, sorted(set(unresolved))


def inspect_pdg_correspondence(source_srepr: str, stored_srepr: str,
                               definitions: dict[str, PdgScalarDefinition]) -> dict:
    """Inspect correspondence and dimensions without asserting physical validity."""
    import sympy as sp
    from sciona.ghost.symbolic import deserialize_expr, serialize_expr, SymbolicExpression

    expression = deserialize_expr(source_srepr)
    replacements, unresolved = map_pdg_scalar_names(expression, definitions)
    result = {
        'runner_version': 'pdg-source-correspondence.v1',
        'scope': 'source-defined scalar mapping and dimensional check only; no publication approval',
        'mapping_status': 'partial' if unresolved else 'complete',
        'correspondence': 'unchecked',
        'dimension_status': 'incomplete',
    }
    if not unresolved:
        with sp.evaluate(False):
            mapped = expression.xreplace(replacements)
        serialized = serialize_expr(mapped)
        if serialize_expr(deserialize_expr(serialized)) != serialized:
            result['correspondence'] = 'mapped_roundtrip_failed'
        else:
            result['correspondence'] = 'exact_ast_match' if serialized == stored_srepr else 'different_ast'
    dimensions = {
        str(symbol): definitions[str(symbol)].dimension
        for symbol in expression.free_symbols
        if str(symbol) in definitions and definitions[str(symbol)].dimension is not None
    }
    if dimensions and len(dimensions) == len(expression.free_symbols):
        errors = SymbolicExpression(srepr_str=source_srepr, dim_map=dimensions).check_dimensional_consistency()
        result['dimension_status'] = 'checker_errors' if errors else 'checker_passed'
    return result
