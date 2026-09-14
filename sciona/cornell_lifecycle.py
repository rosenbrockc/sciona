"""Thirteen full Cornell outputs and four explicit model-only continuations."""
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from sciona.cornell_augmentation import Augmentation,manifest,PINS
from sciona.cornell_model import SourceModel
from sciona.cornell_population import prepare_records,check_split,batches
from sciona.cornell_training import train_phase
from sciona.cornell_voting import vote

INVENTORY='1659594ea11a2631debf3af85cf4eb7e7f938de39640a06d0bd07eed8ed65d2f'


def predict(model,waveform):
    values=np.asarray(waveform)
    if values.ndim!=1 or not len(values) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError('Expected finite mono inference waveform at32000Hz')
    clips=[];frames=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(values),960000):
            chunk=np.zeros(960000,dtype=np.float32);selected=values[start:start+960000]
            chunk[:len(selected)]=selected
            # Preserve all ten source evaluation copies and their NumPy mean.
            inputs=torch.from_numpy(chunk)[None,None].expand(1,10,-1)
            output=model((inputs,None))
            clips.append(output['clipwise_output'].cpu().numpy()[0].mean(axis=0))
            frames.append(output['framewise_output'].cpu().numpy()[0].mean(axis=0))
    return np.stack(clips),np.stack(frames)


def execute(source_dir,dependency_dir,*,populations,background,short_noises,inference,
            epochs=50,batch_size=2,seed=0,backbone_state=None,synthetic=False,progress=None):
    """Explicit fold populations and schedules; all audio is pre-resampled32kHz.

    Epoch/batch controls are explicit execution settings, not historical recipe
    parity. Temporary checkpoints stay private. Selected best/latest-tie models
    replace hand-picked historical checkpoints under this corrected contract.
    """
    inventory=manifest('competition_cornell_training_inventory.json',INVENTORY)
    sourcepins=manifest('competition_cornell_source_pins.json',PINS)
    hashes={p['software_path']:p['sha256'] for p in sourcepins['pins']}
    prepared={}
    required={('five' if m['ordinal'] in range(4,9) else 'four',m['parameters']['fold']) for m in inventory['members']}
    if set(populations)!=required:raise ValueError('Expected exactly fourfold and fivefold populations')
    if type(batch_size) is not int or batch_size<2 or batch_size%2:raise ValueError('Expected positive even training batch size')
    for key,split in populations.items():
        train=prepare_records(split['training'],training=True);valid=prepare_records(split['validation'],training=False)
        check_split(train,valid)
        if len(train)<batch_size:raise ValueError('Each fold requires a complete training batch')
        prepared[key]=(train,valid)
    source=SourceModel(source_dir);all_clips=[];all_frames=[];reports=[]
    with tempfile.TemporaryDirectory(prefix='cornell-private-lifecycle-') as temporary:
        for member in inventory['members']:
            ordinal=member['ordinal'];params=member['parameters'];path=member['software_configuration']
            if hashlib.sha256((Path(source_dir)/path).read_bytes()).hexdigest()!=member['configuration_sha256']:
                raise ValueError('Cornell member configuration changed')
            train,valid=prepared[('five' if ordinal in range(4,9) else 'four',params['fold'])]
            model=source.build(seed=seed+ordinal,backbone_state=backbone_state,synthetic=synthetic)
            phases=[('initial',.001),('continuation',params['lr'])] if params['apply_mixup'] else [('initial',params['lr'])]
            if params['apply_mixup']:
                initial=path.replace('_2.py','.py')
                if hashlib.sha256((Path(source_dir)/initial).read_bytes()).hexdigest()!=hashes[initial]:
                    raise ValueError('Cornell continuation initializer changed')
            for phase,peak in phases:
                augmentation=Augmentation(source_dir,dependency_dir,variant='background' if params['aug_name']=='secondary_default' else 'default',background=background,short_noises=short_noises,seed=seed+ordinal)
                def training(epoch):return batches(train,batch_size=batch_size,training=True,seed=seed+ordinal*1000+epoch,augmentation=augmentation)
                def validation():return batches(valid,batch_size=1,training=False,seed=seed)
                result=train_phase(source,model,training_batches=training,validation_batches=validation,
                    steps_per_epoch=len(train)//batch_size,epochs=epochs,peak=peak,mixup=params['apply_mixup'],
                    augmented_loss=params['criterion_name'].endswith('_augd'),seed=seed+ordinal,
                    checkpoint_directory=Path(temporary)/f'{ordinal}-{phase}')
                reports.append(dict(member=ordinal,phase=phase,selected_epoch=result['selected_epoch'],selected_score=result['selected_score']))
                if progress:progress(dict(member=ordinal,phase=phase,completed=True))
            clip,frame=predict(model,inference);all_clips.append(clip);all_frames.append(frame)
            del model;gc.collect()
    combined=vote(np.stack(all_clips),np.stack(all_frames),duration_seconds=len(inference)/32000)
    return dict(models_completed=13,training_phases=reports,**combined)
