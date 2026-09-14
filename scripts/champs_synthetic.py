"""Invented idealized chemistry for local validation; no real records."""
import numpy as np
import pandas as pd


def synthetic_inputs():
    # Idealized synthetic ethane: two carbons and six hydrogens. No real records.
    coordinates = [[-.77,0.,0.],[.77,0.,0.]]
    for x, phase in [(-1.13,0.),(1.13,np.pi/3)]:
        for j in range(3):
            angle = phase + 2*np.pi*j/3
            coordinates.append([x,1.03*np.cos(angle),1.03*np.sin(angle)])
    symbols = ["C","C"] + ["H"]*6
    atom_rows, bond_rows = [], []
    for case,shift in enumerate([0.,2.]):
        key = f"synthetic-{case}"
        for i,(symbol,xyz) in enumerate(zip(symbols,coordinates)):
            atom_rows.append(dict(molecule_name=key,atom_index=i,atom=symbol,x=xyz[0]+shift,y=xyz[1],z=xyz[2]))
        for index,(carbon,hydrogen) in enumerate([(0,2),(0,3),(0,4),(1,5),(1,6),(1,7)]):
            bond_rows.append(dict(id=case*6+index,molecule_name=key,atom_index_0=hydrogen,
                                  atom_index_1=carbon,type="1JHC",scalar_coupling_constant=float(index+case)))
    atoms,bonds = pd.DataFrame(atom_rows),pd.DataFrame(bond_rows)
    return atoms,bonds
