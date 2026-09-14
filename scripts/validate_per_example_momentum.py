#!/usr/bin/env python3
"""Compare new providers to pinned source update statements on synthetic tensors.

Only exact allowlisted assignment ASTs execute, with NumPy TensorFlow-operation
shims. This verifies update formulas, not TF runtime or model-gradient parity.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sciona.atoms.ml.gradient_attacks import per_example as provider


STD = 'noise = noise / tf.reshape(tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1), [FLAGS.batch_size, 1, 1, 1])'
L1 = 'noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)'
ACCUMULATE = 'noise = momentum * grad + noise'


def extract(content, variant):
    expected = [STD, ACCUMULATE, STD] if variant == 'targeted' else [L1, ACCUMULATE]
    expected_ast = [ast.dump(ast.parse(s).body[0], include_attributes=False) for s in expected]
    names = ['graph_large', 'graph_small'] if variant == 'targeted' else ['graph']
    found = {}
    for function in ast.parse(content).body:
        if not isinstance(function, ast.FunctionDef) or function.name not in names:
            continue
        for i, statement in enumerate(function.body):
            if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Subscript):
                continue
            call = statement.value.value
            if not isinstance(call, ast.Call) or ast.unparse(call.func) != 'tf.gradients':
                continue
            statements = function.body[i+1:i+1+len(expected)]
            if [ast.dump(s, include_attributes=False) for s in statements] != expected_ast:
                raise ValueError('Pinned update differs from reviewed formula')
            if function.name in found:
                raise ValueError('Ambiguous source update')
            found[function.name] = compile(ast.Module(body=statements, type_ignores=[]), '<reviewed-update>', 'exec')
    if set(found) != set(names):
        raise ValueError('Source update inventory differs')
    return found


def source_update(code, gradient, previous, momentum):
    tf = SimpleNamespace(reshape=np.reshape, abs=np.abs,
        reduce_mean=lambda x, axis, keep_dims: np.mean(x, axis=tuple(axis), keepdims=keep_dims),
        contrib=SimpleNamespace(keras=SimpleNamespace(backend=SimpleNamespace(std=np.std))))
    env = dict(tf=tf, FLAGS=SimpleNamespace(batch_size=gradient.shape[0]), noise=gradient.copy(), grad=previous.copy(), momentum=momentum)
    exec(code, {'__builtins__': {}}, env)
    return env['noise']


def validate(root, source_dir):
    pins = json.loads((root/'docs/reviews/competition_gradient_source_pins.json').read_text())
    reports = []
    rng = np.random.default_rng(739)
    for variant, fn in [('targeted', provider.per_example_std_momentum), ('untargeted', provider.per_example_l1_momentum)]:
        content = (source_dir/(variant+'.py')).read_bytes()
        if hashlib.sha256(content).hexdigest() != pins[variant]['sha256']:
            raise ValueError('Source file pin mismatch')
        for name, code in extract(content, variant).items():
            cases = 0
            largest = 0.
            for shape in [(1, 2, 3, 1), (3, 4, 2, 3), (2, 1, 4, 2)]:
                actual_state = np.zeros(shape)
                reference_state = np.zeros(shape)
                for _ in range(4):
                    gradient = rng.normal(size=shape)*np.arange(1, shape[0]+1)[:, None, None, None]
                    actual_state = fn(gradient, actual_state, .8)
                    reference_state = source_update(code, gradient, reference_state, .8)
                    np.testing.assert_allclose(actual_state, reference_state, rtol=2e-14, atol=2e-14)
                    largest = max(largest, float(np.max(np.abs(actual_state-reference_state))))
                    cases += 1
            reports.append(dict(variant=variant, source_function=name, update_cases=cases, maximum_absolute_error=largest))
    files = ['scripts/validate_per_example_momentum.py', 'tests/test_per_example_momentum.py']
    return dict(approved=False, synthetic_only=True, source_pins=pins, checks=reports,
                provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
                scope='Float64 NumPy emulation of exact pinned normalization/update ASTs. No TensorFlow runtime, autodiff gradient, trained-model or full attack-loop parity claim.',
                domain_extension='Zero or nonfinite normalization denominators fail explicitly; source may produce nonfinite values there. No denominator epsilon added.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(Path(__file__).resolve().parents[1], args.source_dir)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['checks']))
