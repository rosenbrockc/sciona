"""Synthetic probes for current-runtime hazards in the pinned TGS reference.

Independent minimal API experiments; no competition source or records embedded.
These establish API behavior, not historical runtime parity or model accuracy.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


def main():
    torch.manual_seed(183)
    class FunctionalDropout(nn.Module):
        def forward(self, values):
            return F.dropout2d(values, p=0.4)
    model = FunctionalDropout().eval()
    values = torch.ones(8, 16, 4, 4)
    first, second = model(values), model(values)
    assert not torch.equal(first, second)
    layer = nn.Linear(2, 1, bias=False)
    snapshot = layer.state_dict()
    saved = snapshot['weight'].clone()
    with torch.no_grad():
        layer.weight.add_(1)
    assert torch.equal(snapshot['weight'], layer.weight)
    assert not torch.equal(snapshot['weight'], saved)
    population = TensorDataset(torch.arange(12))
    # The source supplies a RandomSampler through shuffle, not sampler.
    loader = DataLoader(population, shuffle=RandomSampler(population), batch_size=4)
    assert isinstance(loader.sampler, RandomSampler)
    report = dict(synthetic_only=True, catalog_mutations=0,
        torch_version=torch.__version__,
        functional_dropout_active_under_parent_eval=True,
        state_dict_snapshot_aliases_live_parameters=True,
        sampler_in_shuffle_is_truthy_current_runtime=True,
        scope='Current torch API behavior; source integration and historical behavior unverified.',
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    root = Path(__file__).resolve().parents[1]
    (root / 'docs/reviews/competition_tgs_source_probe.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
