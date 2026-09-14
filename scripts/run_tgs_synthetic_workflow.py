"""Run the complete reviewed TGS budget using generated, learnable mask images.

This is an execution proof, never competition performance evidence. Runtime
state and checkpoints belong in the caller's private directory outside Git.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy
import torch
import torchvision

from sciona.tgs_checkpoint_store import CheckpointStore
from sciona.tgs_workflow import run_workflow
from sciona.tgs_workflow_state import WorkflowStateStore


def synthetic_populations(seed):
    rng = np.random.default_rng(seed)
    masks = np.zeros((104, 101, 101), dtype=bool)
    for index in range(104):
        if index % 2:
            start = int(rng.integers(10, 50))
            masks[index, :, start:start + 30] = True
    # Nonconstant texture on both empty and striped images. Segmentation is
    # learnable from intensity; no target values come from an external dataset.
    images = (32 + masks.astype(np.uint8) * 180 + rng.integers(0, 16, masks.shape, dtype=np.uint8)).astype(np.uint8)
    base = dict(labeled_masks=masks[:40], folds=np.tile(np.arange(5), 8),
                labeled_nonconstant=np.ones(40, dtype=bool), query_nonconstant=np.ones(64, dtype=bool),
                pseudo_validation=np.array([0, 1, 2, 3, 4], dtype=np.int64))
    keras = dict(base, labeled_images=np.repeat(images[:40, ..., None], 3, axis=-1),
                 query_images=np.repeat(images[40:, ..., None], 3, axis=-1))
    pytorch = dict(base, labeled_images=images[:40].astype(np.float32) / 255,
                   query_images=images[40:].astype(np.float32) / 255)
    return dict(keras=keras, pytorch=pytorch)


def main(args):
    root = Path(__file__).resolve().parents[1]
    runtime = args.runtime_directory.resolve()
    if runtime == root or root in runtime.parents:
        raise ValueError('runtime directory must be outside the source repository')
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = root / 'docs/reviews/competition_tgs_training_plan.json'
        plan = json.loads(plan_path.read_text())
        references = {}
        for name, digest in [('resnet34', '333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef'),
                             ('resnext50', '3bcb9dedb226c5e5cdd3510d25cdc33297f59016a6d7069758024caa13e3172d')]:
            path = args.reference_directory / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('unqualified reference bytes')
            references[name] = dict(path=str(path), sha256=digest)
        sources = list((root / 'sciona').glob('tgs_*.py')) + [root / 'sciona/hubmap_losses.py', Path(__file__), plan_path]
        context = dict(seed=args.seed, threads=args.threads,
                       source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
                       references={k: v['sha256'] for k, v in references.items()},
                       versions={k: v.__version__ for k, v in dict(numpy=np, torch=torch, torchvision=torchvision,
                                                                  opencv=cv2, h5py=h5py, scipy=scipy).items()})
        context_digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        states = WorkflowStateStore(runtime / 'state', context_sha256=context_digest)
        state = states.load() if (runtime / 'state/state.json').exists() else dict(fits={}, rounds={}, seed=args.seed)
        torch.set_num_threads(args.threads)
        populations = synthetic_populations(args.seed)
        def record(value):
            states.save(value)
            print(json.dumps(dict(completed_fits=len(value['fits']), completed_rounds=len(value['rounds']))), flush=True)
        if args.prepare_only:
            states.save(state)
            print(json.dumps(dict(prepared=True, training_executed=False, fit_count=plan['fit_count'], epoch_ceiling=plan['epoch_ceiling'])))
            return
        run_workflow(plan, populations, references, CheckpointStore(runtime / 'checkpoints'), seed=args.seed,
                     record=record, state=state)
        print(json.dumps(dict(execution_complete=True, approved=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7913)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args())
