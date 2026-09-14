"""Synthetic raw-geometry probe of pinned CHAMPS chemistry preprocessing."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from openbabel import openbabel as ob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads((repo / "docs/reviews/competition_champs_source_pins.json").read_text())
    hashes = {p["software_path"]: p["sha256"] for p in pins["pins"]}
    for path in ["src/xyz2mol.py", "src/pipeline_pre.py"]:
        assert hashlib.sha256((args.source / path).read_bytes()).hexdigest() == hashes[path]
    spec = importlib.util.spec_from_file_location("champs_xyz2mol_probe", args.source / "src/xyz2mol.py")
    xyz2mol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xyz2mol)
    source = (args.source / "src/pipeline_pre.py").read_text()
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "enhance_structure_dict")
    code = ast.get_source_segment(source, function)
    code = code.replace("    import pybel", "    from openbabel import pybel")
    changes = {"heavyvalence": "GetHvyDegree", "heterovalence": "GetHeteroDegree", "valence": "GetExplicitDegree"}
    for old, new in changes.items():
        token = "mol.atoms[i]." + old
        assert code.count(token) == 1
        code = code.replace(token, "mol.atoms[i].OBAtom." + new + "()")
    ns = {"np": np, "x2m": xyz2mol, "manual_bond_order_dict": {},
          "atomic_num_dict": {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9},
          "bond_order_dict": {Chem.BondType.SINGLE: 1, Chem.BondType.AROMATIC: 1.5,
                              Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}}
    exec(compile(code, "<chemistry-compatibility-probe>", "exec"), ns)
    tetra = .63 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    cases = [("water", ["O", "H", "H"], np.array([[0.,0.,0.], [.96,0.,0.], [-.24,.93,0.]]), 2),
             ("methane", ["C"] + ["H"]*4, np.vstack((np.zeros((1,3)), tetra)), 4)]
    results = []
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for label, symbols, positions, bonds in cases:
        outputs = []
        for coordinates in [positions, positions @ rotation + np.array([3., -2., 7.])]:
            out = ns["enhance_structure_dict"]({"synthetic": {"symbols": symbols, "positions": coordinates}})["synthetic"]
            assert np.count_nonzero(out["bond_orders"]) == bonds * 2
            assert np.isfinite(out["charges"]).all() and np.isfinite(out["angle"]).all()
            assert abs(sum(out["charges"])) < 1e-10
            outputs.append(out)
        for field in ["distances", "angle", "charges", "bond_orders", "bond_ids", "heavyvalences", "heterovalences", "valences"]:
            np.testing.assert_allclose(outputs[0][field], outputs[1][field], atol=1e-12)
        results.append({"synthetic_case": label, "bonds": bonds, "finite_features": True,
                        "rigid_transform_invariant": True})
    args.output.write_text(json.dumps({"results": results, "rdkit": rdkit.__version__,
        "openbabel": ob.OBReleaseVersion(), "compatibility_changes": changes,
        "legacy_semantics_source": "https://github.com/openbabel/openbabel/blob/openbabel-2-4-1/include/openbabel/atom.h",
        "scope": "Two synthetic geometries and rigid transforms; no label vocabulary, scaling, tensor packing or full pipeline parity established.",
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    print("Four synthetic chemistry executions passed")


if __name__ == "__main__":
    main()
