"""Two-resistor parallel network with explicit nonzero-division branch."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION='90400aa6-60ab-58d4-a7cd-295acde66d27'
SOURCE_HASH='ad359f977e5b93e8e30be731ecf615fedd5cd1a010bbac93b11710ab0a891df1'


def symbols():
    V=sp.Symbol('V',real=True,nonzero=True)
    I,I1,I2,It=sp.symbols('I I1 I2 It',real=True)
    R,R1,R2,Rt=sp.symbols('R R1 R2 Rt',positive=True)
    return V,I,I1,I2,It,R,R1,R2,Rt


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class ParallelResistanceProof:
    steps:tuple


def build_proof():
    V,I,I1,I2,It,R,R1,R2,Rt=symbols()
    return ParallelResistanceProof((eq(V,I1*R1),eq(V,I2*R2),eq(V/R2,I2),
        eq(V/R1,I1),eq(V,It*Rt),eq(V/Rt,It),
        eq(V/Rt,V/R1+V/R2),eq(1/Rt,1/R1+1/R2)))


def verify_proof(proof):
    V,I,I1,I2,It,R,R1,R2,Rt=symbols()
    if len(proof.steps)!=8:raise ValueError('Eight steps required')
    base=eq(V,I*R);steps=[]
    steps.append(base.subs({I:I1,R:R1},simultaneous=True))
    steps.append(base.subs({I:I2,R:R2},simultaneous=True))
    steps.append(eq(steps[1].lhs/R2,sp.simplify(steps[1].rhs/R2)))
    steps.append(eq(steps[0].lhs/R1,sp.simplify(steps[0].rhs/R1)))
    steps.append(base.subs({I:It,R:Rt},simultaneous=True))
    steps.append(eq(steps[4].lhs/Rt,sp.simplify(steps[4].rhs/Rt)))
    steps.append(eq(It,I1+I2).subs({It:steps[5].lhs,I1:steps[3].lhs,I2:steps[2].lhs},simultaneous=True))
    steps.append(eq(sp.simplify(steps[6].lhs/V),sp.expand(steps[6].rhs/V)))
    for i,(actual,wanted) in enumerate(zip(proof.steps,steps)):
        if actual!=wanted:raise ValueError(f'Step{i+1} differs')
    equivalent=R1*R2/(R1+R2)
    if sp.simplify(1/equivalent-1/R1-1/R2)!=0:raise ValueError('Conductance law failed')
    # Define resistance by constitutive law across applied voltages; a single
    # zero-voltage observation cannot identify it via the ratio V/It.
    voltage=sp.Symbol('voltage',real=True)
    total=voltage/R1+voltage/R2
    if sp.simplify(total*equivalent-voltage)!=0 or total.subs(voltage,0)!=0:
        raise ValueError('All-voltage constitutive extension failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,steps_verified=8,
        source_division_requires_nonzero_voltage=True,zero_voltage_constitutive_extension=True,
        steps=[sp.srepr(s) for s in steps],equivalent_resistance=sp.srepr(equivalent),
        assumptions=['Two ideal positive finite linear resistors across the same two nodes.',
                     'Common signed voltage, branch currents with consistent orientation, Kirchhoff current sum.',
                     'Resistances voltage independent; source cancellation requires nonzero test voltage.'],
        limitations=['Zero applied voltage gives zero currents; resistance follows constitutive model, not a measured0/0 ratio.',
                     'No ideal short/open circuit, negative resistance, nonlinear device or reactive impedance scope.'])
