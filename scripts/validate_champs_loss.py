"""Compare pinned source loss and gradients with scalar synthetic references."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads((repo / "docs/reviews/competition_champs_source_pins.json").read_text())
    hashes = {p["software_path"]: p["sha256"] for p in pins["pins"]}
    raw = (args.source / "src/train.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == hashes["src/train.py"]
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "loss"]
    assert len(nodes) == 1
    ns = {"torch": torch, "NUM_BOND_ORIG_TYPES": 8}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<source-loss>", "exec"), ns)
    torch.manual_seed(1729)
    results = []
    # Unequal type frequencies distinguish per-type log MAE from global MAE.
    for log_objective in (False, True):
        prediction = torch.randn(2, 68, 12, dtype=torch.float64, requires_grad=True)
        bonds = torch.zeros(2, 12, 5, dtype=torch.long)
        targets = torch.zeros(2, 12, 4, dtype=torch.float64)
        for m in range(2):
            for b in range(10):
                bonds[m,b,0] = b % 8 + 1
                bonds[m,b,1] = 68 - b
                targets[m,b] = torch.tensor([b + m + .25, b / 3., .5 + b, 1.])
            # A non-supervised pair and padding have huge errors, excluded.
            bonds[m,10,0] = 9
            bonds[m,10,1] = 11
            targets[m,10:,:3] = 1e6
        ns["args"] = SimpleNamespace(champs_loss=log_objective)
        absolute, errors, counts = ns["loss"](prediction, targets, bonds)
        terms = [[] for _ in range(8)]
        for m in range(2):
            for b in range(10):
                p = prediction[m, int(bonds[m,b,1])-1, b]
                terms[b % 8].append((p * targets[m,b,2] + targets[m,b,1] - targets[m,b,0]).abs())
        expected_errors = torch.stack([torch.stack(t).sum() for t in terms])
        expected_counts = torch.tensor([len(t) for t in terms])
        torch.testing.assert_close(absolute, expected_errors.sum())
        torch.testing.assert_close(errors, expected_errors.detach() if not log_objective else expected_errors)
        torch.testing.assert_close(counts, expected_counts)
        objective = (errors/counts + 1e-9).log().mean() if log_objective else absolute/counts.sum()
        expected = (expected_errors/expected_counts + 1e-9).log().mean() if log_objective else expected_errors.sum()/expected_counts.sum()
        actual_grad, = torch.autograd.grad(objective, prediction, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, prediction)
        torch.testing.assert_close(actual_grad, expected_grad)
        assert torch.count_nonzero(actual_grad[:,:,10:]) == 0
        assert torch.count_nonzero(actual_grad) == 20
        results.append({"objective": "mean_log_type_mae" if log_objective else "global_mae",
                        "all_eight_types": True, "unequal_type_counts": True,
                        "scalar_loss_and_gradient_match": True, "auxiliary_and_padding_excluded": True})
    args.output.write_text(json.dumps({"source_commit": pins["commit"], "results": results,
        "scope": "Original loss on synthetic targets with fixed source type semantics; no optimizer/schedule/checkpoint claim.",
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+"\n")
    print("Both source objectives and gradients match scalar references")


if __name__ == "__main__":
    main()
