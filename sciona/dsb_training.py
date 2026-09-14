"""Candidate runtime-array training assembly. No datasets or model states embedded."""
import numpy as np

from sciona.dsb_components import (sample_proposals, crop_classifier_proposal,
                                    assign_anchor_labels, _integer)
from sciona.dsb_detector_crop import crop_detection
from sciona.dsb_augmentation import augment_detection, augment_classifier


def detector_indexed_training_sample(index, volumes, boxes_by_image, table, eligible_image_indices,
                                    *, rng=None, **kwargs):
    """Connect sample selection to crop/augment/labels with explicit associations."""
    from sciona.dsb_sampling import select_detector_sample
    if len(volumes)!=len(boxes_by_image):raise ValueError('volume and box collections must align')
    rng=np.random.RandomState() if rng is None else rng
    selected=select_detector_sample(index,table,eligible_image_indices,image_count=len(volumes),rng=rng)
    image=selected['image_index']
    options=dict(kwargs)
    if selected['independent_image']:options['scale']=False
    return detector_training_sample(volumes[image],selected['target'],boxes_by_image[image],
                                    random_crop=selected['random_crop'],rng=rng,**options)


def shared_training_models(topk=5):
    """Source alternating training shares the exact feature network instance."""
    from sciona.dsb_network import Net, CaseNet
    detector = Net()
    return detector, CaseNet(topk, nodulenet=detector)


def detector_training_sample(volume, target, boxes, *, crop_size=128, bound_size=12,
                             scale=True, random_crop=False, flip=True, rotate=True,
                             swap=True, num_neg=800, rng=None, label_seed=None, label_rng=None):
    """Shared NumPy RNG for crop/augment; independent Python seed for label sampling."""
    rng = np.random.RandomState() if rng is None else rng
    sample, selected, transformed_boxes, coords = crop_detection(volume, target, boxes,
        crop_size=crop_size, bound_size=bound_size, scale=scale, random_crop=random_crop, rng=rng)
    if not random_crop:
        sample, selected, transformed_boxes, coords = augment_detection(sample, selected,
            transformed_boxes, coords, flip=flip, rotate=rotate, swap=swap, rng=rng)
    labels = assign_anchor_labels(sample.shape[1:], None if np.isnan(selected[0]) else selected,
                                  transformed_boxes, num_neg=num_neg, random_state=label_seed, rng=label_rng)
    # Alternating-training loader passes raw float32 values; root detector
    # inference loader has a different normalization contract.
    return sample.astype(np.float32), labels, coords


def classifier_training_sample(volume, proposals, known_proposals, *, topk=5, crop_size=96,
                               temperature=1., scale=False, flip=True, rotate=False,
                               swap=False, rng=None):
    """Sequential sampling/crop/augmentation with a single NumPy RNG stream.

    Proposals and their indicators must already be filtered together. Returned
    zero slots and raw crop intensity conventions match source training batches.
    """
    rng = np.random.RandomState() if rng is None else rng
    proposals, known = np.asarray(proposals), np.asarray(known_proposals)
    topk, crop_size = _integer(topk,'topk'), _integer(crop_size,'crop_size')
    if (proposals.ndim != 2 or proposals.shape[1] != 5 or not np.all(np.isfinite(proposals))
            or np.any(proposals[:,4] <= 0) or known.shape != (len(proposals),)
            or not np.all((known == 0) | (known == 1)) or crop_size % 4):
        raise ValueError('invalid proposals, indicators or crop size')
    chosen = sample_proposals(proposals[:,0], topk, temperature, rng=rng)
    crops = np.zeros((topk,1,crop_size,crop_size,crop_size), dtype=np.float32)
    coords = np.zeros((topk,3,crop_size//4,crop_size//4,crop_size//4), dtype=np.float32)
    indicators = np.zeros(topk, dtype=np.float32)
    for slot,index in enumerate(chosen):
        crop,coordinate = crop_classifier_proposal(volume, proposals[index,1:], crop_size=crop_size,
                                                   phase='train', scale_enabled=scale, rng=rng)
        crop,coordinate = augment_classifier(crop,coordinate,flip=flip,rotate=rotate,swap=swap,
                                             filling_value=0,rng=rng)
        crops[slot], coords[slot], indicators[slot] = crop, coordinate, known[index]
    return crops, coords, indicators, chosen


def run_training_epoch(detector, classifier, detector_optimizer, classifier_optimizer,
                       batch_factory, *, epoch, start_epoch, classifier_variant=3,
                       freeze_batchnorm=False, learning_rate_override=None, after_pass=None):
    """Execute source-ordered training passes using caller-supplied batch factories.

    batch_factory(task, augmentation_profile) is called anew for each pass, so
    the initial debug pass cannot exhaust a one-shot loader for the later pass.
    Detector batches are (images, labels, coordinates); classifier batches are
    (images, coordinates, known_proposals, case_labels). Validation/checkpoint
    saving remain separate operations; returned evidence is aggregate only.
    """
    import torch
    from sciona.dsb_schedule import CLASSIFIER_PROFILES, epoch_steps, training_mode
    from sciona.dsb_losses import detector_loss, classifier_loss
    if classifier.NoduleNet is not detector:
        raise ValueError('alternating training requires the same feature network instance')
    for model, optimizer in [(detector,detector_optimizer),(classifier,classifier_optimizer)]:
        parameters=[p for group in optimizer.param_groups for p in group['params']]
        if len(parameters)!=len({id(p) for p in parameters}) or {id(p) for p in parameters}!={id(p) for p in model.parameters()}:
            raise ValueError('optimizer parameters do not match the source task network')
    report=[]
    for step in epoch_steps(epoch,start_epoch,classifier_variant,learning_rate_override):
        task=step['task']
        model,optimizer=(detector,detector_optimizer) if task=='detector' else (classifier,classifier_optimizer)
        profile=(dict(flip=True,swap=False,scale=True,rotate=False) if task=='detector'
                 else {key:CLASSIFIER_PROFILES[classifier_variant][key] for key in ['flip','swap','scale','rotate']})
        training_mode(model,freeze_batchnorm)
        for group in optimizer.param_groups:group['lr']=step['learning_rate']
        parameter=next(model.parameters())
        total,examples,batches=0.,0,0
        for index,batch in enumerate(batch_factory(task,profile)):
            if step['max_batches'] is not None and index>=step['max_batches']:break
            values=[torch.as_tensor(v,device=parameter.device,dtype=parameter.dtype) for v in batch]
            # Source-era PyTorch zero_grad cleared existing buffers rather than
            # replacing them with None; this matters for shared unused heads.
            optimizer.zero_grad(set_to_none=False)
            if task=='detector':
                images,labels,coordinates=values
                _,prediction=model(images,coordinates)
                loss=detector_loss(prediction,labels,num_hard=2,training=True)['total']
            else:
                images,coordinates,known,labels=values
                _,case,each=model(images,coordinates)
                loss=classifier_loss(case,each,labels.reshape(-1),known)['total']
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite training loss')
            loss.backward();optimizer.step()
            count=images.shape[0]
            total+=float(loss.detach())*count;examples+=count;batches+=1
        report.append(dict(task=task,learning_rate=step['learning_rate'],batches=batches,
                           examples=examples,mean_loss=total/examples if examples else None))
        if after_pass is not None and step['max_batches'] is None:
            after_pass(task,model)
    return report


def run_epoch_lifecycle(detector,classifier,detector_optimizer,classifier_optimizer,
                        training_factory,validation_factory,*,epoch,start_epoch,
                        save_frequency=1,**kwargs):
    """Source-ordered training, validation and optional in-memory checkpoint.

    Validation factory keys distinguish detector, classifier held-out, and
    classifier training-set evaluation. Caller owns private array loading and
    checkpoint storage; no filenames or data provenance enter returned metrics.
    """
    from sciona.dsb_validation import validate_detector,validate_classifier
    from sciona.dsb_checkpoint import checkpoint_bytes
    save_frequency=_integer(save_frequency,'save_frequency')
    validation=[]
    def after_pass(task,model):
        if task=='detector':
            validation.append(dict(task=task,metrics=validate_detector(model,validation_factory('detector'))))
        else:
            for key in ['classifier_validation','classifier_training_validation']:
                validation.append(dict(task=key,metrics=validate_classifier(model,validation_factory(key))))
    training=run_training_epoch(detector,classifier,detector_optimizer,classifier_optimizer,
        training_factory,epoch=epoch,start_epoch=start_epoch,after_pass=after_pass,**kwargs)
    checkpoint=checkpoint_bytes(classifier,epoch) if epoch%save_frequency==0 else None
    return dict(training=training,validation=validation,checkpoint=checkpoint)
