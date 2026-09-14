"""Qualify detector matchers and balanced samplers against historical source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from torchvision.models.detection._utils import Matcher, BalancedPositiveNegativeSampler


def main(source, output):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != 'cb80cd35f7f0ba41e5929c1bf37b1096ce6d2f0d1a34954d9dfa8d9ac3e29484':
        raise ValueError('historical matching source drift')
    names = {'zeros_like', 'Matcher', 'BalancedPositiveNegativeSampler'}
    nodes = [n for n in ast.parse(raw).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    for node in nodes:
        node.decorator_list = []  # Compare eager semantics on the installed runtime.
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical-matching>', 'exec'), namespace)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(987)
    match_cases = sample_cases = 0
    for high, low, allow_low, batch, fraction in ((.7, .3, True, 256, .5), (.5, .5, False, 512, .25)):
        actual_matcher, reference_matcher = Matcher(high, low, allow_low), namespace['Matcher'](high, low, allow_low)
        actual_sampler = BalancedPositiveNegativeSampler(batch, fraction)
        reference_sampler = namespace['BalancedPositiveNegativeSampler'](batch, fraction)
        for seed in range(64):
            quality = torch.randint(0, 11, (7, 701), generator=generator).float() / 10
            if seed == 0:
                quality.zero_()
            torch.testing.assert_close(actual_matcher(quality), reference_matcher(quality), rtol=0, atol=0)
            match_cases += 1
            labels = [torch.randint(-1, 3, (n,), generator=generator) for n in (0, 71, 701)]
            if seed in (0, 1):
                labels = [torch.full_like(value, seed) for value in labels]
            torch.manual_seed(seed)
            observed = actual_sampler(labels)
            actual_rng = torch.get_rng_state()
            torch.manual_seed(seed)
            expected = reference_sampler(labels)
            torch.testing.assert_close(torch.get_rng_state(), actual_rng, rtol=0, atol=0)
            for observed_group, expected_group in zip(observed, expected):
                for a, b in zip(observed_group, expected_group):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            sample_cases += 1
        for shape in ((0, 3), (3, 0)):
            errors = []
            for matcher in (actual_matcher, reference_matcher):
                try:
                    matcher(torch.empty(shape))
                except ValueError:
                    errors.append(True)
            assert errors == [True, True]
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        source_sha256=digest, matcher_exact_cases=match_cases, sampler_mask_and_rng_exact_cases=sample_cases,
        threshold_ties_all_background_all_positive_and_empty_cases_included=True,
        empty_matcher_inputs_rejected_by_both=True,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Matching and sampling utilities only; complete target assignment and detector integration remain separate.',
                'Historical TorchScript decorators omitted for eager comparison; historical runtime equivalence is not claimed.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
