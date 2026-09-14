"""Lossless source-ID normalization, independent of display-label parsing."""
import sympy as sp
from sciona.ghost.symbolic import deserialize_expr,serialize_expr,SymbolicExpression
from sciona.ghost.symbolic_normalization import normalize_symbolic_candidate


def normalize_source_identities(source_srepr,definitions):
    expression=deserialize_expr(source_srepr)
    if not isinstance(expression,sp.Equality):raise ValueError('Source equation required')
    if serialize_expr(expression)!=source_srepr:raise ValueError('Source AST roundtrip differs')
    names={str(s) for s in expression.free_symbols}
    if not names:raise ValueError('Explicit source quantities required')
    if not names.issubset(definitions):raise ValueError('Source identity missing from pinned definitions')
    dimensions={name:definitions[name].dimension for name in names}
    if any(dim is None or dim.is_unknown for dim in dimensions.values()):raise ValueError('Complete source dimensions required')
    if SymbolicExpression(srepr_str=source_srepr,dim_map=dimensions).check_dimensional_consistency():
        raise ValueError('Source dimensional check failed')
    normalized=normalize_symbolic_candidate({'sympy_expr':expression,
        'variable_hints':{name:{'dim_signature':dim} for name,dim in dimensions.items()}},require_dimensions=True)
    if normalized.srepr_str!=source_srepr or normalized.review_tasks or normalized.parse_status!='parsed':
        raise ValueError('Identity normalization changed source or left review tasks')
    if set(normalized.variables)!=names:raise ValueError('Source variable coverage differs')
    labels={name:definitions[name].latex for name in sorted(names)}
    return normalized,labels
