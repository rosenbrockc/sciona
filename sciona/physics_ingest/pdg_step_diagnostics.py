"""Independent arithmetic-step diagnostics; never an approval or branch solver."""
import sympy as sp


def arithmetic_step_differences(rule,inputs,feeds,expected):
    """Compare both sides with the declared elementary operation, preserving sign.

    Unsupported operations return None. A nonzero symbolic difference requires
    domain analysis or an admissible counterexample; it is not itself a proof
    that every possible constrained interpretation is invalid.
    """
    operations={'divide both sides by','multiply both sides by','add X to both sides','subtract X from both sides'}
    if rule.name not in operations:return None
    if (rule.inputs,rule.feeds,rule.outputs)!=(1,1,1) or len(inputs)!=1 or len(feeds)!=1:
        raise ValueError('Invalid arithmetic rule arity')
    if not isinstance(inputs[0],sp.Equality) or not isinstance(expected,sp.Equality):
        raise ValueError('Equations required')
    factor=feeds[0]
    if rule.name=='divide both sides by' and factor==0:raise ValueError('Zero divisor')
    def transform(side):
        if rule.name=='divide both sides by':return side/factor
        if rule.name=='multiply both sides by':return side*factor
        if rule.name=='add X to both sides':return side+factor
        return side-factor
    return tuple(sp.simplify(transform(before)-after) for before,after in zip(inputs[0].args,expected.args))
