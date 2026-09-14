"""Qualify installed RPN and ROI neural heads against torchvision 0.5."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor


def main(root, output):
    namespace = dict(torch=torch, nn=nn, F=F)
    sources = {
        'rpn.py': ('62ea420ac8bba44d91d3b27f46c627fcc6b91c15ae706142fcbe079c10a31751', ['RPNHead']),
        'faster_rcnn.py': ('b9df7856bad4aa091e67d04d363ea85e2bbf47ca4d0008f58418d6b8d304ff59', ['TwoMLPHead', 'FastRCNNPredictor'])}
    for file, (digest, names) in sources.items():
        raw = (root / 'models/detection' / file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('historical detection head source drift')
        nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical-heads>', 'exec'), namespace)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(852)
    rows = []
    for cls, args, shapes in [
        (RPNHead, (256, 3), [(1, 256, s, s) for s in (200, 100, 50, 25, 13)]),
        (TwoMLPHead, (256 * 7 * 7, 1024), [(32, 256, 7, 7)]),
        (FastRCNNPredictor, (1024, 91), [(32, 1024)]),
        (FastRCNNPredictor, (1024, 2), [(32, 1024)])]:
        torch.manual_seed(851)
        actual = cls(*args)
        torch.manual_seed(851)
        reference = namespace[cls.__name__](*args)
        def source_key(key):
            return key.replace('conv.0.0.', 'conv.') if cls is RPNHead else key
        actual_state = {source_key(k): v for k, v in actual.state_dict().items()}
        assert set(actual_state) == set(reference.state_dict())
        for key, value in actual_state.items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
        x = [torch.randn(shape, generator=generator).requires_grad_() for shape in shapes]
        y = [v.detach().clone().requires_grad_() for v in x]
        observed = actual(x if cls is RPNHead else x[0])
        expected = reference(y if cls is RPNHead else y[0])
        def flatten(value):
            if isinstance(value, torch.Tensor):
                return [value]
            return [tensor for item in value for tensor in flatten(item)]
        observed, expected = flatten(observed), flatten(expected)
        assert len(observed) == len(expected)
        for a, b in zip(observed, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum(t.square().mean() for t in observed).backward()
        sum(t.square().mean() for t in expected).backward()
        for a, b in zip(x, y):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            assert torch.isfinite(a.grad).all()
        reference_parameters = dict(reference.named_parameters())
        for key, value in actual.named_parameters():
            torch.testing.assert_close(value.grad, reference_parameters[source_key(key)].grad, rtol=0, atol=0)
            assert torch.isfinite(value.grad).all()
        rows.append(dict(head=cls.__name__, arguments=args, input_shapes=shapes,
            initialized_state_exact=True, outputs_and_all_gradients_exact=True))
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256={k: v[0] for k, v in sources.items()}, cases=rows,
        torch_version=torch.__version__, torchvision_version=torchvision.__version__,
        installed_class_source_sha256={c.__name__: hashlib.sha256(inspect.getsource(c).encode()).hexdigest()
            for c in (RPNHead, TwoMLPHead, FastRCNNPredictor)},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Neural heads only; anchor generation, proposal selection, ROI pooling, losses and complete detector integration remain pending.',
                'Both paths use installed Torch kernels; no historical native-kernel equivalence claim.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.output)
