"""Replay winner runtime scheduling against the PyTorch1.4 scheduler classes.

Optimizer state handoff uses installed Adam; this checks scheduling and restored
learning-rate metadata, not historical Adam update or mixed-precision parity.
"""
import argparse
import ast
import copy
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import types
import warnings
import weakref

import torch


SCHEDULER_SHA256 = '1efe08d7f86f12bbe2c1ce722646ae4945c6fe19611db5c9058ffbf396bc94c3'
RUNTIME_SHA256 = '36551e0efd6b7d60778d2c4965a7dc4936e7101937bcde1cda4aeef1b9f5418b'


def checked(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('reference source content differs')
    return ast.parse(raw)


def run(scheduler_class, lf_node):
    def schedule(optimizer, epochs):
        namespace = dict(math=math, args=SimpleNamespace(epochs=epochs))
        exec(compile(ast.Module(body=[lf_node], type_ignores=[]), '<pinned-wheat-lambda>', 'exec'), namespace)
        scheduler = scheduler_class(optimizer, lr_lambda=namespace['lf'])
        scheduler.last_epoch = 0
        return scheduler

    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.Adam([parameter], lr=8e-5)
    first = schedule(optimizer, 10)
    snapshots, first_rates = [], []
    for _ in range(10):
        first_rates.append(optimizer.param_groups[0]['lr'])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        first.step()
        snapshots.append(copy.deepcopy(optimizer.state_dict()))
    second_runs = []
    for best_epoch, state in enumerate(snapshots):
        fresh_parameter = torch.nn.Parameter(torch.tensor([1.]))
        fresh = torch.optim.Adam([fresh_parameter], lr=1e-5)
        fresh.load_state_dict(state)
        restored_lr = fresh.param_groups[0]['lr']
        restored_initial_lr = fresh.param_groups[0]['initial_lr']
        second = schedule(fresh, 6)
        rates = []
        for _ in range(6):
            rates.append(fresh.param_groups[0]['lr'])
            fresh_parameter.grad = torch.ones_like(fresh_parameter)
            fresh.step()
            second.step()
        assert restored_initial_lr == 8e-5 and second.base_lrs == [8e-5]
        assert rates[0] == 8e-5 and rates[0] != 1e-5
        second_runs.append(dict(prior_best_epoch=best_epoch, restored_lr_before_scheduler=restored_lr,
            restored_initial_lr=restored_initial_lr, training_rates=rates,
            final_lr=fresh.param_groups[0]['lr']))
    return dict(round1_training_rates=first_rates, round2_cases=second_runs)


def main(scheduler_source, runtime_source, output):
    tree = checked(scheduler_source, SCHEDULER_SHA256)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('_LRScheduler', 'LambdaLR')]
    if len(nodes) != 2:
        raise ValueError('historical scheduler classes absent')
    namespace = dict(Optimizer=torch.optim.Optimizer, wraps=wraps, weakref=weakref,
                     warnings=warnings, types=types, EPOCH_DEPRECATION_WARNING='explicit epochs unused')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pytorch-1.4-scheduler>', 'exec'), namespace)
    runtime = checked(runtime_source, RUNTIME_SHA256)
    function = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == 'train_model')
    lf = next(n for n in function.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'lf' for t in n.targets))
    historical = run(namespace['LambdaLR'], lf)
    current = run(torch.optim.lr_scheduler.LambdaLR, lf)
    if historical != current:
        raise ValueError('installed scheduler differs from pinned historical sequence')
    report = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        historical_scheduler_source='https://raw.githubusercontent.com/pytorch/pytorch/v1.4.0/torch/optim/lr_scheduler.py',
        historical_scheduler_sha256=SCHEDULER_SHA256, runtime_source_sha256=RUNTIME_SHA256,
        installed_torch=torch.__version__, possible_round1_checkpoint_epochs_checked=10,
        exact_historical_and_installed_schedules=True, round2_effective_base_lr=8e-5,
        round2_nominal_cli_lr=1e-5, **historical,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Pinned PyTorch1.4scheduler classes execute over installed Adam and synthetic optimizer state.',
                'No claim of historical Adam numerical updates, Apex state or mixed-precision equivalence.',
                'Full detector training, checkpoint selection, population construction and publication remain pending.'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('round1_training_rates', 'round2_cases')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler-source', type=Path, required=True)
    parser.add_argument('--runtime-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.scheduler_source, args.runtime_source, args.output)
