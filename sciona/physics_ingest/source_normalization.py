"""Prepare source-AST normalization alternatives while retaining the original parse."""
import sympy as sp
from sciona.ghost.symbolic import deserialize_expr,serialize_expr,SymbolicExpression
from sciona.ghost.symbolic_normalization import normalize_symbolic_candidate
from sciona.physics_ingest.pdg_symbols import map_pdg_scalar_names


def normalize_pinned_source_ast(source_srepr,definitions):
    source=deserialize_expr(source_srepr)
    if not isinstance(source,sp.Equality):raise ValueError('source equation required')
    mapping,unresolved=map_pdg_scalar_names(source,definitions)
    if unresolved or not mapping:raise ValueError('unambiguous scalar mapping required')
    dimensions={};source_symbols={}
    for before,after in mapping.items():
        dimension=definitions[str(before)].dimension
        if dimension is None or dimension.is_unknown:raise ValueError('complete source dimensions required')
        dimensions[str(after)]=dimension;source_symbols[str(after)]=str(before)
    with sp.evaluate(False):mapped=source.xreplace(mapping)
    normalized=normalize_symbolic_candidate({'sympy_expr':mapped,'variable_hints':{name:{'dim_signature':dim} for name,dim in dimensions.items()}},require_dimensions=True)
    if normalized.parse_status!='parsed' or normalized.review_tasks or not normalized.srepr_str:raise ValueError('normalization unresolved')
    if serialize_expr(deserialize_expr(normalized.srepr_str))!=normalized.srepr_str:raise ValueError('source normalization roundtrip failed')
    if SymbolicExpression(srepr_str=normalized.srepr_str,dim_map=dimensions).check_dimensional_consistency():raise ValueError('mapped source dimensional check failed')
    return normalized,source_symbols
