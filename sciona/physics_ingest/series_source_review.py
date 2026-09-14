"""Conservative scientific classification for the validated series derivation."""
import sympy as sp
from sciona.ghost.dimensions import DimensionalSignature

RESISTANCE=DimensionalSignature(M=1,L=2,T=-3,I=-2).to_compact()
VOLTAGE=DimensionalSignature(M=1,L=2,T=-3,I=-1).to_compact()
CURRENT=DimensionalSignature(I=1).to_compact()


def classify_series_equation(expression, dimensions):
    if not isinstance(expression,sp.Equality):raise ValueError('equation required')
    symbols=expression.free_symbols
    if set(dimensions)!={str(s) for s in symbols}:raise ValueError('complete source dimensions required')
    groups={kind:sorted([s for s in symbols if dimensions[str(s)]==kind],key=sp.srepr) for kind in [RESISTANCE,VOLTAGE,CURRENT]}
    if sum(map(len,groups.values()))!=len(symbols):raise ValueError('non-circuit quantity present')
    resistors,voltages,currents=(groups[k] for k in [RESISTANCE,VOLTAGE,CURRENT])
    if len(voltages)==1 and len(currents)==1 and len(resistors)==1:
        if expression.lhs==voltages[0] and sp.cancel(expression.rhs-currents[0]*resistors[0])==0:return 'ohmic_voltage_current_resistance'
    if len(voltages)==3 and not resistors and not currents and expression.lhs in voltages:
        others=[s for s in voltages if s!=expression.lhs]
        if sp.cancel(expression.rhs-sum(others))==0:return 'series_voltage_addition'
    if len(resistors)==3 and not voltages and not currents and expression.lhs in resistors:
        others=[s for s in resistors if s!=expression.lhs]
        if sp.cancel(expression.rhs-sum(others))==0:return 'series_equivalent_resistance'
    if len(resistors)==3 and not voltages and len(currents)==1:
        current=currents[0]
        for total in resistors:
            others=[s for s in resistors if s!=total]
            if sp.cancel(expression.lhs-current*total)==0 and sp.cancel(expression.rhs-current*sum(others))==0:return 'ohmic_series_voltage_balance'
    raise ValueError('equation does not match the supported source model')
