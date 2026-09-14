"""Exact polynomial-residual equivalence under dimension-preserving symbol renaming."""
from itertools import permutations,product
from math import factorial,prod
import sympy as sp
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.symbolic import _infer_dim


def equivalent_residual_mapping(source, target, source_dimensions, target_dimensions, *, max_mappings=720):
    """Return source-to-target variable names, or None; never infer physical equivalence.

    Equal residuals establish reusable arithmetic only. Each relation retains its
    source provenance and physical validity bounds.
    """
    def prepare(equation, dimensions):
        if not isinstance(equation,sp.Equality):raise ValueError('equations required')
        for node in sp.preorder_traversal(equation):
            if isinstance(node,(sp.Equality,sp.Symbol,sp.Rational,sp.Add,sp.Mul)):continue
            if isinstance(node,sp.Pow) and isinstance(node.exp,sp.Integer) and node.exp >= 0:continue
            raise ValueError('only polynomial expressions supported; denominator domains require separate proof')
        symbols={str(symbol):symbol for symbol in equation.free_symbols}
        if set(symbols)!=set(dimensions):raise ValueError('complete dimensions required')
        dims={name:DimensionalSignature.from_compact(value) for name,value in dimensions.items()}
        output=_infer_dim(equation.lhs,dims,sp)
        if output!=_infer_dim(equation.rhs,dims,sp):raise ValueError('inconsistent equation dimensions')
        groups={}
        for name in sorted(symbols):groups.setdefault(dims[name].to_compact(),[]).append(name)
        return symbols,groups,output
    ss,sg,sd=prepare(source,source_dimensions);ts,tg,td=prepare(target,target_dimensions)
    if sd!=td or {k:len(v) for k,v in sg.items()}!={k:len(v) for k,v in tg.items()}:return None
    keys=sorted(sg)
    if prod(factorial(len(sg[k])) for k in keys)>max_mappings:raise ValueError('symbol-renaming search exceeds conservative limit')
    for choices in product(*(permutations(tg[k]) for k in keys)):
        mapping={a:b for key,choice in zip(keys,choices) for a,b in zip(sg[key],choice)}
        renamed=(source.lhs-source.rhs).xreplace({ss[a]:ts[b] for a,b in mapping.items()})
        if sp.cancel(renamed-(target.lhs-target.rhs))==0:return mapping
    return None
