"""Source font-classifier lifecycle with same-population deterministic evaluation.

Evaluation is not held-out accuracy. A zero-accuracy run has no source best
checkpoint; callers must not silently substitute the final model for it.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

from sciona.bengali_population import ImagePopulation
from sciona.bengali_preprocessing import crop_resize,normalize_rgb,prepare_handwriting
from sciona.bengali_pretraining_augmentation import augment_pretraining
from sciona.bengali_pretraining_sampling import PretrainingParameterSampler
from sciona.bengali_sampling import SamplingBudget,paired_batches


def fit_font_classifier(model,population,*,epochs,batch_size,train_seed,validation_seed,sampler,output_directory):
    if not isinstance(model,nn.Module) or not isinstance(population,ImagePopulation) or not isinstance(sampler,PretrainingParameterSampler):
        raise ValueError('classifier, validated font population and pretraining sampler required')
    parameters=list(model.parameters())
    if not parameters or any(p.device.type!='cpu' or p.dtype!=torch.float32 or not p.requires_grad for p in parameters):
        raise ValueError('trainable float32 CPU classifier required')
    budget=SamplingBudget(len(population.images),len(population.images),batch_size,epochs)
    batches=iter(paired_batches(budget,hand_seed=train_seed,font_seed=validation_seed))
    first=next(batches)
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=False)
    optimizer=torch.optim.AdamW(parameters,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.01,amsgrad=False)
    half=budget.total_steps*.5
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1. if step<half else (budget.total_steps-step)/half)
    history=[];best_score=0.;best=None
    def checkpoint(name):
        partial=directory/(name+'.partial')
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},partial)
        partial.replace(directory/name)
        return dict(file=name,sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest())
    def logits(images):
        result=model(images)
        if result.shape!=(len(images),14784) or not torch.isfinite(result).all():
            raise ValueError('finite14784-way classifier output required')
        return result
    for epoch in range(epochs):
        model.train();losses=[];validation=[]
        for offset in range(budget.steps_per_epoch):
            pair=first if epoch==0 and offset==0 else next(batches)
            indices=pair['hand_indices'].numpy();validation.append(pair['font_indices'].numpy())
            images=crop_resize(population.images[indices])
            images=normalize_rgb(np.stack([augment_pretraining(image,sampler.draw()) for image in images]))
            labels=torch.from_numpy(population.labels[indices].copy())
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(logits(images),labels)
            loss.backward();optimizer.step();scheduler.step()
            value=float(loss.detach())
            if not math.isfinite(value):raise ValueError('nonfinite classifier training loss')
            losses.append(value)
        model.eval();correct=0;count=0
        with torch.no_grad():
            for indices in validation:
                pred=logits(prepare_handwriting(population.images[indices])).argmax(1)
                labels=torch.from_numpy(population.labels[indices].copy())
                correct+=int((pred==labels).sum());count+=len(labels)
        accuracy=correct/count
        if accuracy>best_score:
            best_score=accuracy;best=checkpoint('best.pt')
        last=checkpoint('last.pt')
        history.append(dict(epoch=epoch,training_loss=sum(losses)/len(losses),same_population_accuracy=accuracy))
        partial=directory/'history.json.partial';partial.write_text(json.dumps(history,indent=2)+'\n');partial.replace(directory/'history.json')
    if next(batches,None) is not None:raise ValueError('unconsumed classifier training budget')
    return dict(completed_epochs=len(history),optimizer_updates=budget.total_steps,best_checkpoint=best,
                last_checkpoint=last,history=history,approved=False,evaluation_is_held_out=False)
