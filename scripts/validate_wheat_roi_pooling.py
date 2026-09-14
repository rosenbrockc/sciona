"""Compare source multi-scale ROI routing, pooling and feature gradients."""
import argparse
import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torchvision
from torchvision.ops import MultiScaleRoIAlign, roi_align
from torchvision.ops.boxes import box_area


def main(source, output):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'f3f65f56bf3a3e2fe9b0d5e9082b779fd8344e7ed4b5fb2a8fcb5215ca5af1aa':
        raise ValueError('historical ROI pooler source drift')
    names = {'LevelMapper', 'initLevelMapper', 'MultiScaleRoIAlign'}
    nodes = [n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace = dict(torch=torch, torchvision=torchvision, nn=torch.nn, Tensor=torch.Tensor,
                     roi_align=roi_align, box_area=box_area, Optional=Optional, List=List, Dict=Dict, Tuple=Tuple)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical-roi-pooler>', 'exec'), namespace)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(1227)
    cases = []
    for dtype in (torch.float32, torch.float64):
        for height, width in ((800, 800), (832, 1344)):
            for mixed_empty in (False, True):
                actual = MultiScaleRoIAlign(['0', '1', '2', '3'], 7, 2)
                reference = namespace['MultiScaleRoIAlign'](['0', '1', '2', '3'], 7, 2)
                features = OrderedDict((str(i), torch.randn(2, 256, (height+s-1)//s, (width+s-1)//s,
                    generator=generator, dtype=dtype).requires_grad_()) for i, s in enumerate((4, 8, 16, 32)))
                features['pool'] = torch.randn(2, 256, (height+63)//64, (width+63)//64,
                    generator=generator, dtype=dtype).requires_grad_()
                ref_features = OrderedDict((k, v.detach().clone().requires_grad_()) for k, v in features.items())
                sizes = (1., 16., 55.999, 56., 56.001, 111.999, 112., 112.001,
                         223.999, 224., 224.001, 447.999, 448., 448.001, 700.)
                first = torch.tensor([[3.25, 7.5, 3.25+s, 7.5+s] for s in sizes], dtype=dtype)
                second = torch.empty(0, 4, dtype=dtype) if mixed_empty else first + 11
                boxes = [first, second]
                image_shapes = [(height, width), (height-17, width-23)]
                observed, expected = actual(features, boxes, image_shapes), reference(ref_features, boxes, image_shapes)
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                assert actual.scales == reference.scales == [.25, .125, .0625, .03125]
                torch.testing.assert_close(actual.map_levels(boxes), reference.map_levels(boxes), rtol=0, atol=0)
                assert set(actual.map_levels(boxes).tolist()) == {0, 1, 2, 3}
                observed.square().mean().backward()
                expected.square().mean().backward()
                for key in ('0', '1', '2', '3'):
                    torch.testing.assert_close(features[key].grad, ref_features[key].grad, rtol=0, atol=0)
                    assert torch.isfinite(features[key].grad).all()
                assert features['pool'].grad is None and ref_features['pool'].grad is None
                with torch.no_grad():
                    torch.testing.assert_close(actual(features, boxes, image_shapes), reference(ref_features, boxes, image_shapes), rtol=0, atol=0)
                cases.append(dict(dtype=str(dtype), padded_canvas=[height, width], mixed_empty_image=mixed_empty,
                    output_shape=list(observed.shape), all_four_levels_exercised=True,
                    levels_scales_outputs_and_feature_gradients_exact=True, cached_scale_repeat_exact=True))
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=digest, cases=cases, validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Historical pooler dispatch uses installed ROIAlign and box-area kernels; historical native-kernel equivalence is not claimed.',
                'Float32/float64 CPU eager execution only; full detector execution remains pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
