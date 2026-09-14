"""Reusable corrected CHAMPS chemistry, vocabulary, scaling and tensor packing.

Derived from pinned MIT Bosch source and MIT Jensen xyz2mol. Fitted state is
private runtime data; callers must not publish its labels or scaling values.
"""
import ast
import collections
import contextlib
import copy
import io
import itertools
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
import torch

from sciona.champs_ensemble import COUPLING_TYPES
from sciona.champs_source_corrections import correct_triplet_layout
from sciona.champs_source_runtime import ChampsSourceRuntime


def _namespace(runtime, manual_corrections):
    code = correct_triplet_layout(runtime.sources["src/pipeline_pre.py"], runtime.hashes["src/pipeline_pre.py"], kind="packer")
    substitutions = {
        "    import pybel": "    from openbabel import pybel",
        "mol.atoms[i].heavyvalence": "mol.atoms[i].OBAtom.GetHvyDegree()",
        "mol.atoms[i].heterovalence": "mol.atoms[i].OBAtom.GetHeteroDegree()",
        "mol.atoms[i].valence": "mol.atoms[i].OBAtom.GetExplicitDegree()",
        "bond_dataframe.append(new_data,verify_integrity=True,sort=False)":
        "pd.concat([bond_dataframe,new_data],verify_integrity=True,sort=False)",
        'bonds_train.groupby("labeled_type").mean()["scalar_coupling_constant"]':
        'bonds_train.groupby("labeled_type")["scalar_coupling_constant"].mean()',
        'bonds_train.groupby("labeled_type").std()["scalar_coupling_constant"]':
        'bonds_train.groupby("labeled_type")["scalar_coupling_constant"].std()',
    }
    for old,new in substitutions.items():
        if code.count(old) != 1:
            raise ValueError("CHAMPS preprocessing source correction mismatch")
        code = code.replace(old,new)
    code = code.replace('.count().max()[0]', '.count().max().iloc[0]')
    tree = ast.parse(code)
    names = {"make_structure_dict", "enhance_structure_dict", "enhance_atoms", "enhance_bonds",
             "add_all_pairs", "make_triplets", "make_quadruplets", "_create_embedding",
             "add_embedding", "get_scaling", "add_scaling", "create_dataset"}
    definitions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(definitions) != len(names):
        raise ValueError("CHAMPS preprocessing definition mismatch")
    x2m = types.ModuleType("champs_verified_xyz")
    exec(compile(runtime.sources["src/xyz2mol.py"],"<verified-xyz2mol>","exec"),x2m.__dict__)
    ns = {"np":np, "pd":pd, "collections":collections, "itertools":itertools, "x2m":x2m,
          "manual_bond_order_dict": copy.deepcopy(manual_corrections or {}), "atomic_num_dict":{"H":1,"C":6,"N":7,"O":8,"F":9},
          "bond_order_dict":{Chem.BondType.SINGLE:1,Chem.BondType.AROMATIC:1.5,Chem.BondType.DOUBLE:2,Chem.BondType.TRIPLE:3},
          "MAX_ATOM_COUNT":29,"MAX_BOND_COUNT":406,"MAX_TRIPLET_COUNT":54,"MAX_QUAD_COUNT":117}
    # Algorithmic label rules from pinned source, not per-record corrections.
    constants = [n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in {"classification_corrections","small_longtypes"} for t in n.targets)]
    exec(compile(ast.Module(body=constants+definitions,type_ignores=[]),"<source-preprocessing>","exec"),ns)
    return ns


@dataclass
class PreprocessingState:
    vocabulary: dict
    means: dict
    stds: dict


class ChampsPreprocessor:
    def __init__(self, runtime: ChampsSourceRuntime, *, manual_corrections=None):
        self._functions = _namespace(runtime,manual_corrections)

    def prepare(self, atoms: pd.DataFrame, couplings: pd.DataFrame, *,
                state: PreprocessingState | None = None, labeled: bool = True):
        """Fit state or reuse it, returning source tensors and an independent state.

        Inputs are copied. Unknown chemistry labels fail rather than extending
        an existing vocabulary. Per-record chemistry corrections are optional
        caller-owned runtime input; no original correction records are bundled.
        """
        if state is None and not labeled:
            raise ValueError("Inference requires fitted CHAMPS preprocessing state")
        ns = self._functions
        atoms,bonds = atoms.copy(deep=True),couplings.copy(deep=True)
        state = copy.deepcopy(state)
        # Original routines print molecule-specific progress. Keep it local and
        # discard it rather than exposing real input identifiers in run logs.
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            structures = ns["enhance_structure_dict"](ns["make_structure_dict"](atoms))
            atoms = ns["enhance_atoms"](atoms,structures)
            bonds = ns["add_all_pairs"](ns["enhance_bonds"](bonds,structures),structures).reset_index()
            triplets = ns["make_triplets"](list(structures),structures)
            quads = ns["make_quadruplets"](list(structures),structures)
            for frame,keys in [(atoms,["atom_index"]),(bonds,["atom_index_0","atom_index_1"]),
                               (triplets,["atom_index_0","atom_index_1","atom_index_2"]),
                               (quads,["atom_index_0","atom_index_1","atom_index_2","atom_index_3"])]:
                frame.sort_values(["molecule_name"]+keys,inplace=True)
            if state is None:
                vocabulary = ns["add_embedding"](atoms,bonds,triplets,quads)
                vocabulary[("bond",0)] = ns["_create_embedding"](
                    pd.Series(list(bonds["type"].unique())+list(COUPLING_TYPES)))
                means,stds = ns["get_scaling"](bonds)
                state = PreprocessingState(vocabulary,means,stds)
            ns["add_embedding"](atoms,bonds,triplets,quads,state.vocabulary)
            bonds = ns["add_scaling"](bonds,state.means,state.stds)
            packed = ns["create_dataset"](atoms,bonds,triplets,quads,labeled=labeled)
        if not all(torch.isfinite(t).all() for t in packed):
            raise ValueError("CHAMPS preprocessing produced nonfinite tensors")
        return packed,state
