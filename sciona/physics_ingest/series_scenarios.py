"""Synthetic circuit scenarios for independent source-expression validation."""
import numpy as np
import sympy as sp
from sciona.physics_ingest.series_source_review import classify_series_equation,CURRENT,RESISTANCE


def build_series_scenarios(expressions,dimensions,*,count=256,seed=20260912):
    classes={key:classify_series_equation(expr,{str(s):dimensions[str(s)] for s in expr.free_symbols}) for key,expr in expressions.items()}
    if sorted(classes.values())!=sorted(['ohmic_voltage_current_resistance']*4+['series_voltage_addition','ohmic_series_voltage_balance','series_equivalent_resistance']):
        raise ValueError('complete supported seven-expression model required')
    terminal=expressions[next(k for k,c in classes.items() if c=='series_equivalent_resistance')]
    total=terminal.lhs;branches=sorted(terminal.rhs.free_symbols,key=sp.srepr)
    symbols=set().union(*(expr.free_symbols for expr in expressions.values()))
    current_symbols=[s for s in symbols if dimensions[str(s)]==CURRENT]
    if len(current_symbols)!=1:raise ValueError('one common branch current required')
    current=current_symbols[0]
    rng=np.random.default_rng(seed)
    values={str(s):10**rng.uniform(-2,2,count) for s in sorted(symbols,key=sp.srepr) if dimensions[str(s)]==RESISTANCE and s!=total}
    values[str(current)]=rng.uniform(.01,2,count)*rng.choice([-1,1],count)
    branch_current=values[str(current)]
    # Solve independent voltage-drop and loop constraints, then infer the total
    # resistance from applied voltage/current. Never evaluate a source expression
    # to manufacture its own expected residual.
    matrix=np.array([[1.,0.,0.],[0.,1.,0.],[-1.,-1.,1.]])
    voltages=np.linalg.solve(matrix,np.stack([branch_current*values[str(branches[0])],branch_current*values[str(branches[1])],np.zeros(count)]))
    values[str(total)]=voltages[2]/branch_current
    for key,kind in classes.items():
        if kind!='ohmic_voltage_current_resistance':continue
        expr=expressions[key]
        resistor=next(s for s in expr.rhs.free_symbols if dimensions[str(s)]==RESISTANCE)
        if resistor==total:voltage=voltages[2]
        elif resistor in branches:voltage=voltages[branches.index(resistor)]
        else:voltage=branch_current*values[str(resistor)]
        if str(expr.lhs) in values:raise ValueError('duplicate voltage assignment')
        values[str(expr.lhs)]=voltage
    if set(values)!={str(s) for s in symbols}:raise ValueError('unassigned circuit quantity')
    return values
