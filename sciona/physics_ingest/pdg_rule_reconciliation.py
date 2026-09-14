"""Conservative source-rule corrections proved independently of stored normalization."""
import sympy as sp
from sciona.ghost.dimensions import DimensionalSignature


def prove_variable_renaming(source,target,feeds,dimensions):
    """Prove simultaneous, dimension-preserving renaming with unchanged expression structure.

    Structural equality after renaming also preserves denominator/branch structure;
    no cancellation, branch assumption or physical-validity claim is introduced.
    """
    if not isinstance(source,sp.Equality) or not isinstance(target,sp.Equality):raise ValueError('equations required')
    if not feeds or len(feeds)%2 or not all(isinstance(symbol,sp.Symbol) for symbol in feeds):raise ValueError('ordered atomic symbol pairs required')
    old=feeds[::2];new=feeds[1::2]
    if len(set(old))!=len(old):raise ValueError('duplicate source substitutions')
    if len(set(new))!=len(new) or set(new)&(source.free_symbols-set(old)):
        raise ValueError('renaming must not merge distinct variables')
    if not set(old)<=source.free_symbols:raise ValueError('replacement source symbol absent')
    for before,after in zip(old,new):
        a=DimensionalSignature.from_compact(dimensions.get(str(before),'unknown'))
        b=DimensionalSignature.from_compact(dimensions.get(str(after),'unknown'))
        if a.is_unknown or b.is_unknown or a!=b:raise ValueError('renaming dimensions must match')
    actual=source.xreplace(dict(zip(old,new)))
    def structural_key(node):
        args=[structural_key(arg) for arg in node.args]
        if isinstance(node,(sp.Add,sp.Mul)) and node.is_commutative:
            args.sort(key=repr)
        return (node.func.__name__,tuple(args)) if args else sp.srepr(node)
    if structural_key(actual)!=structural_key(target):raise ValueError('renamed expression does not exactly match conclusion')
    return {str(before):str(after) for before,after in zip(old,new)}
