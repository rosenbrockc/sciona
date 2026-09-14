"""Catalog contracts for validated equation residuals, without choosing a solve direction."""
import hashlib
import sympy as sp
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.symbolic import deserialize_expr, _infer_dim
from sciona.physics_ingest.equation_runtime import compile_equation_runtime


def build_residual_contract(expression, variables):
    evidence = expression['evidence_json']['numpy_runtime']
    if evidence.get('kind') != 'equation_residual_validator' or evidence.get('tests_passed') is not True:
        raise ValueError('passed equation residual validation is required')
    equation = deserialize_expr(expression['sympy_srepr'])
    if not isinstance(equation, sp.Equality):
        raise ValueError('an equation is required')
    compiled = compile_equation_runtime(sp.Eq(sp.Symbol('_validation_residual'), equation.lhs-equation.rhs, evaluate=False))
    if (compiled.source != evidence.get('source') or compiled.source_sha256 != evidence.get('runtime_source_sha256')
            or list(compiled.argument_symbols) != evidence.get('argument_symbols')):
        raise ValueError('validated runtime no longer matches the expression')
    by_name = {row['symbol_name']:row for row in variables}
    if len(by_name) != len(variables) or set(by_name) != set(compiled.argument_symbols):
        raise ValueError('variables must cover the runtime interface exactly')
    dimensions = {name:DimensionalSignature.from_compact(row['dim_signature']) for name,row in by_name.items()}
    output_dimension = _infer_dim(equation.lhs, dimensions, sp)
    if output_dimension != _infer_dim(equation.rhs, dimensions, sp):
        raise ValueError('equation sides must have matching dimensions')
    ports = []
    for ordinal,name in enumerate(compiled.argument_symbols):
        ports.append({'direction':'input', 'ordinal':ordinal, 'name':f'x{ordinal}',
                      'type_desc':f'Finite real scalar or broadcastable array for equation symbol {name}; cast to float64',
                      'dim_signature':dimensions[name].to_compact(), 'constraints':'finite; real numeric; broadcastable',
                      'required':True, 'default_value_repr':''})
    ports.append({'direction':'output', 'ordinal':0, 'name':'residual',
                  'type_desc':'Float64 array equal to equation LHS minus RHS; zero indicates consistency, not a solved unknown',
                  'dim_signature':output_dimension.to_compact(), 'constraints':'finite; broadcast input shape',
                  'required':True, 'default_value_repr':''})
    return {'runtime_kind':'equation_residual_validator', 'callable':'evaluate',
            'runtime_source_sha256':hashlib.sha256(compiled.source.encode()).hexdigest(),
            'argument_symbols':dict(zip((p['name'] for p in ports[:-1]),compiled.argument_symbols)),
            'ports':ports, 'limitations':['Does not solve for any variable',
            'Physical regime must be satisfied independently; finite arithmetic alone does not establish physical validity']}
