"""Execute a declared Bengali CycleGAN fit over validated paired populations.

Completing this loop does not qualify the chosen classifier, training budget,
dependencies or full winning pipeline. Epoch generator files are inference
artifacts, not resumable optimizer/replay checkpoints.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from sciona.bengali_gan_training import CycleGANTraining
from sciona.bengali_population import training_batches
from sciona.bengali_sampling import SamplingBudget


def fit_population(trainer,hand,font,budget,*,hand_seed,font_seed,font_sampler,output_directory):
    if (not isinstance(trainer,CycleGANTraining) or not isinstance(budget,SamplingBudget)
            or trainer.completed_steps!=0 or trainer.total_steps!=budget.total_steps):
        raise ValueError('fresh trainer with matching declared population budget required')
    batches=iter(training_batches(hand,font,budget,hand_seed=hand_seed,font_seed=font_seed,font_sampler=font_sampler))
    # Force population/seed checks before creating output or updating networks.
    first=next(batches)
    directory=Path(output_directory)
    directory.mkdir(parents=True,exist_ok=False)
    history=[]
    for epoch in range(budget.epochs):
        totals={}
        for offset in range(budget.steps_per_epoch):
            batch=first if epoch==0 and offset==0 else next(batches)
            expected_step=epoch*budget.steps_per_epoch+offset
            if batch['epoch']!=epoch or batch['step']!=expected_step:
                raise ValueError('population sequence differs from declared epoch budget')
            result=trainer.step(batch['images_a'],batch['images_b'],batch['labels_a'])
            for group in ['generator','discriminator']:
                for name,value in result[group].items():
                    if not math.isfinite(value):
                        raise ValueError('nonfinite fit metric; fit cannot be accepted')
                    key=group+'/'+name
                    totals[key]=totals.get(key,0.)+value
        for model in [trainer.generator_a,trainer.generator_b,trainer.discriminator_a,trainer.discriminator_b]:
            model.eval()
        trainer.guidance.eval()
        name=f'generator_epoch_{epoch+1}.pt'
        partial=directory/(name+'.partial')
        torch.save({k:v.detach().cpu().clone() for k,v in trainer.generator_b.state_dict().items()},partial)
        partial.replace(directory/name)
        entry=dict(epoch=epoch,steps=budget.steps_per_epoch,
            metrics={k:v/budget.steps_per_epoch for k,v in totals.items()},
            generator_file=name,generator_sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest())
        history.append(entry)
        pending=directory/'history.json.partial'
        pending.write_text(json.dumps(history,indent=2)+'\n')
        pending.replace(directory/'history.json')
    if next(batches,None) is not None or trainer.completed_steps!=budget.total_steps:
        raise ValueError('fit did not consume exactly the declared training budget')
    return dict(completed_steps=trainer.completed_steps,completed_epochs=len(history),history=history,
                approved=False,scope='Declared CycleGAN fit only; classifier and full pipeline qualification are separate.')
