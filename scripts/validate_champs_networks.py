"""Exercise every configured CHAMPS network on synthetic tensors, without data I/O.

This is a source compatibility probe, not full pipeline validation. Requires the
reviewed public software cache; never downloads weights or competition records.
"""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def definitions(path, expected, namespace):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    tree = ast.parse(raw)
    # Execute definitions only. Imports, notebook/CLI launchers, and file I/O at
    # module level are excluded. Each variant gets its own dependency namespace.
    body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads((repo / "docs/reviews/competition_champs_source_pins.json").read_text())
    hashes = {p["software_path"]: p["sha256"] for p in pins["pins"]}

    def config(path):
        raw = (args.source / path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == hashes[path]
        return ast.literal_eval(raw.decode())

    catalog = config("config/models.json")
    torch.set_num_threads(2)
    results = []
    for label in catalog["names"]:
        folder = "models/" + catalog[label + "_dir"]
        cfg = config(folder + "/config")
        cfg.pop("name", None)
        cfg["dim"] = cfg.pop("d_model")
        cfg.update({k: v for k, v in catalog.items() if k.startswith("num_")})
        ns = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional,
              "np": np, "weight_norm": torch.nn.utils.weight_norm,
              "colored": lambda text, *_: text}
        for file in ["modules/hierarchical_embedding.py", "modules/embeddings.py", "graph_transformer.py"]:
            path = folder + "/" + file
            definitions(args.source / path, hashes[path], ns)
        torch.manual_seed(1729)
        np.random.seed(1729)
        model = ns["GraphTransformer"](**cfg)
        # Four synthetic sites, three links, two triplets, one quadruplet.
        atoms = torch.ones(1, 4, 3, dtype=torch.long)
        positions = torch.tensor([[[0., 0., 0., .1, .2], [1., 0., 0., .2, .3],
                                   [1., 1., 0., .3, .4], [1., 1., 1., .4, .5]]])
        bonds = torch.tensor([[[1, 1, 1, 0, 1], [1, 1, 1, 1, 2], [1, 1, 1, 2, 3]]])
        triplets = torch.tensor([[[1, 1, 0, 2, 0, 1], [1, 2, 1, 3, 1, 2]]])
        quads = torch.tensor([[[1, 0, 1, 2, 3, 0, 1, 2, 0, 1]]])
        inputs = (atoms, positions, bonds, torch.ones(1, 3), triplets,
                  torch.full((1, 2), .5), quads, torch.full((1, 1), .25))
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        pred, _ = model(*inputs)
        assert pred.shape == (1, catalog["num_bond_types"][1], 3)
        loss = pred.square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        changed_parameter = next(p for p in model.parameters()
                                 if p.grad is not None and torch.count_nonzero(p.grad))
        before_step = changed_parameter.detach().clone()
        optimizer.step()
        assert not torch.equal(before_step, changed_parameter.detach())
        model.eval()
        with torch.no_grad():
            after, _ = model(*inputs)
        assert torch.isfinite(after).all() and not torch.equal(pred.detach(), after)
        item = {"variant": folder.split("/")[-1], "parameters": sum(p.numel() for p in model.parameters()),
                "layers": cfg["n_layers"], "dimension": cfg["dim"], "quadruplets": bool(cfg.get("use_quad", False)),
                "synthetic_loss": loss.item(), "finite_backward": True, "optimizer_steps": 1}
        results.append(item)
        print(json.dumps(item), flush=True)
        del model, optimizer, pred, after, loss, grads, ns, before_step, changed_parameter
        gc.collect()
    assert len(results) == 13
    args.output.write_text(json.dumps({"source_commit": pins["commit"], "results": results,
        "scope": "Full configured network topology; one SGD probe step each on synthetic tensors. No chemistry preprocessing, original training schedule, saved state parity, or ensemble execution validated.",
        "source_changes": "None. Module-level imports/launchers excluded; termcolor display replaced with plain text."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
