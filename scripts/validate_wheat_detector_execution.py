"""Recovered EfficientDet source with publisher-pinned private dependencies.

Full synthetic loss/backward paths only; not trained-model or historical-kernel
equivalence. The original torch._six helper is loaded only in this process.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def main(source_root, dependency_root, output):
    closure = json.loads(Path('docs/reviews/competition_global_wheat_runtime_source_closure.json').read_text())
    for name, digest in closure['files_sha256'].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('runtime source changed')
    manifests = {}
    for package, expected_version in [('timm', '0.1.28'), ('omegaconf', '2.0.0')]:
        manifest = json.loads((dependency_root / (package + '_manifest.json')).read_text())
        if manifest['version'] != expected_version:
            raise ValueError('historical dependency version differs')
        for name, digest in manifest['files_sha256'].items():
            if hashlib.sha256((dependency_root / package / name).read_bytes()).hexdigest() != digest:
                raise ValueError('dependency wheel payload changed')
        manifests[package] = manifest
    helper = dependency_root / 'torch14_six.py'
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    if helper_sha != '0b8506ccd7eef2424d655e9cb64a0e1c81f3c8481ac665adc6eeded4e57d1489':
        raise ValueError('historical compatibility helper differs')
    spec = importlib.util.spec_from_file_location('torch._six', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['torch._six'] = module
    import models
    import timm
    import omegaconf
    if Path(models.__file__).resolve() != (source_root / 'models.py').resolve():
        raise ValueError('wrong detector factory imported')
    for package, loaded in [('timm', timm), ('omegaconf', omegaconf)]:
        if not Path(loaded.__file__).resolve().is_relative_to((dependency_root / package).resolve()):
            raise ValueError('dependency resolved outside checked overlay')
    torch.set_num_threads(2)
    results = []
    for backbone, size in [('ed5', 512), ('ed6', 640), ('ed7', 768), ('ed7', 1024)]:
        torch.manual_seed(831)
        model = models.get_effdet_train(backbone, img_size=size).eval()
        images = torch.rand(1, 3, size, size).requires_grad_()
        boxes = [torch.tensor([[50., 60., 200., 220.]])]
        labels = [torch.tensor([1.])]
        loss, classification, box = model(images, boxes, labels)
        if not all(torch.isfinite(value).all() for value in [loss, classification, box]):
            raise ValueError('nonfinite detector loss')
        loss.backward()
        if not torch.isfinite(images.grad).all() or not images.grad.abs().sum() > 0:
            raise ValueError('missing finite input gradient')
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise ValueError('missing or nonfinite trainable parameter gradient')
        result = dict(backbone=backbone, image_size=size,
            parameters=sum(p.numel() for p in model.parameters()),
            total_loss=float(loss.detach()), classification_loss=float(classification.detach()),
            box_loss=float(box.detach()), full_backward_passed=True,
            finite_nonzero_input_gradient=True, all_trainable_parameter_gradients_finite=True)
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, images, boxes, labels, loss, classification, box, gradients
        gc.collect()
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        pretrained_weights=False, model_mode='eval_with_gradients', configurations=results,
        publisher_dependencies=manifests, historical_torch_six_sha256=helper_sha,
        installed_torch=torch.__version__, runtime_source_sha256=closure['files_sha256'],
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Full random-initialized detector loss/backward execution, not training or winning accuracy.',
                'Historical timm/OmegaConf and original torch._six run on modern Torch; numerical equivalence to historical kernels is unproven.',
                'Pretrained artifacts, augmentation, Apex/optimizer behavior, full training and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'configurations': len(results), 'approved': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dependency-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root, args.dependency_root, args.output)
