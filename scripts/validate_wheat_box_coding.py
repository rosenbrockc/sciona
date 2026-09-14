"""Compare source box coding values and gradients on valid detector inputs."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import torch
from torchvision.models.detection._utils import BoxCoder


def main(source, output):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'cb80cd35f7f0ba41e5929c1bf37b1096ce6d2f0d1a34954d9dfa8d9ac3e29484':
        raise ValueError('historical box coding source drift')
    nodes = [n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {'BoxCoder', 'encode_boxes'}]
    for node in nodes:
        node.decorator_list = []
    namespace = dict(torch=torch, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical-box-coding>', 'exec'), namespace)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(1051)
    cases = 0
    for weights in ((1., 1., 1., 1.), (10., 10., 5., 5.)):
        actual, reference = BoxCoder(weights), namespace['BoxCoder'](weights)
        for dtype in (torch.float32, torch.float64):
            for seed in range(16):
                def boxes(count):
                    starts = torch.rand(count, 2, generator=generator, dtype=dtype) * 800 - 50
                    return torch.cat([starts, starts + torch.rand(count, 2, generator=generator, dtype=dtype) * 250 + .001], 1)
                counts = [31, 0, 19]
                anchors, targets = [boxes(n) for n in counts], [boxes(n) for n in counts]
                encoded, expected_encoded = actual.encode(targets, anchors), reference.encode(targets, anchors)
                for a, b in zip(encoded, expected_encoded):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    assert torch.isfinite(a).all()
                classes = 1 if weights[0] == 1 else 2
                offsets = (torch.randn(sum(counts), 4 * classes, generator=generator, dtype=dtype) * (50 if seed == 0 else .2)).requires_grad_()
                ref_offsets = offsets.detach().clone().requires_grad_()
                observed, expected = actual.decode(offsets, anchors), reference.decode(ref_offsets, anchors)
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                assert torch.isfinite(observed).all()
                observed.square().mean().backward()
                expected.square().mean().backward()
                torch.testing.assert_close(offsets.grad, ref_offsets.grad, rtol=0, atol=0)
                assert torch.isfinite(offsets.grad).all()
                cases += 1
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=digest, exact_encode_decode_and_decode_gradient_cases=cases,
        rpn_and_roi_weights_exercised=True, exponential_clipping_and_mixed_empty_images_exercised=True,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Positive-area boxes and nonempty total proposals only; historical all-empty decode is unsupported.',
                'Float32/float64 CPU eager execution; historical TorchScript and native-kernel equivalence are not claimed.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
