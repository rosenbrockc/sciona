"""Generate a NumPy-only runtime for an explicit rational terminal equation."""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class EquationRuntime:
    source: str
    argument_symbols: tuple[str, ...]
    output_symbol: str
    source_sha256: str


def compile_equation_runtime(equation, conditions=()):
    import sympy as sp
    from sympy.printing.numpy import NumPyPrinter
    if not isinstance(equation,sp.Equality) or not isinstance(equation.lhs,sp.Symbol) or equation.lhs in equation.rhs.free_symbols:
        raise ValueError('terminal equation must explicitly define one output symbol')
    for expression in [equation.rhs,*(side for c in conditions for side in c.args)]:
        for node in sp.preorder_traversal(expression):
            if isinstance(node,(sp.Symbol,sp.Rational,sp.Add,sp.Mul)):continue
            if isinstance(node,sp.Pow) and isinstance(node.exp,sp.Integer):continue
            raise ValueError('only rational arithmetic is supported')
    for condition in conditions:
        if not isinstance(condition,(sp.Unequality,sp.GreaterThan,sp.StrictGreaterThan,sp.LessThan,sp.StrictLessThan)):
            raise ValueError('unsupported runtime condition')
    symbols=set(equation.rhs.free_symbols)
    for condition in conditions:symbols.update(condition.free_symbols)
    if equation.lhs in symbols:raise ValueError('output-dependent guards require a separate postcondition contract')
    ordered=sorted(symbols,key=sp.srepr)
    aliases={symbol:sp.Symbol('x'+str(i)) for i,symbol in enumerate(ordered)}
    printer=NumPyPrinter()
    # Every free symbol is replaced by a generated identifier. Source symbol
    # spelling can never inject Python into the standalone implementation.
    rhs=printer.doprint(equation.rhs.xreplace(aliases))
    lines=['import numpy', '', 'def evaluate('+', '.join(str(aliases[s]) for s in ordered)+'):', '    arrays = []']
    for symbol in ordered:
        name=str(aliases[symbol])
        lines.extend([f'    value = numpy.asarray({name})',
                      "    if value.dtype.kind not in 'iuf': raise ValueError('real numeric inputs required')",
                      "    value = value.astype(numpy.float64)",
                      "    if not numpy.all(numpy.isfinite(value)): raise ValueError('finite inputs required')",
                      '    arrays.append(value)'])
    if ordered:
        lines.append('    arrays = numpy.broadcast_arrays(*arrays)')
        for i,symbol in enumerate(ordered):lines.append(f'    {aliases[symbol]} = arrays[{i}]')
    lines.append("    with numpy.errstate(all='raise'):")
    for condition in conditions:
        predicate=printer.doprint(condition.xreplace(aliases))
        lines.append(f"        if not numpy.all({predicate}): raise ValueError('required domain condition failed')")
    lines.append(f'        result = numpy.asarray({rhs}, dtype=numpy.float64)')
    if ordered:lines.append('        result = numpy.broadcast_to(result, arrays[0].shape)')
    lines.extend(["        if not numpy.all(numpy.isfinite(result)): raise ValueError('nonfinite output')",'        return result'])
    source='\n'.join(lines)+'\n'
    return EquationRuntime(source,tuple(str(s) for s in ordered),str(equation.lhs),hashlib.sha256(source.encode()).hexdigest())
