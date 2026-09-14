"""Two-font marginal prediction from the pinned reduced submission notebook."""
import torch


def marginal_predictions(logits_a,logits_b):
    for logits in (logits_a,logits_b):
        if (not isinstance(logits,torch.Tensor) or logits.dtype!=torch.float32
                or logits.device.type!='cpu' or logits.ndim!=2 or logits.shape[0]<1
                or logits.shape[1]!=14784 or not torch.isfinite(logits).all()):
            raise ValueError('finite float32 CPU N14784 logits required')
    if logits_a.shape!=logits_b.shape:
        raise ValueError('aligned two-font predictions required')
    # Preserve source reduction order: marginalize each model, then add.
    a=logits_a.softmax(1).reshape(-1,168,11,8)
    b=logits_b.softmax(1).reshape(-1,168,11,8)
    root=(a.sum((2,3))+b.sum((2,3))).argmax(1)
    vowel=(a.sum((1,3))+b.sum((1,3))).argmax(1)
    consonant=(a.sum((1,2))+b.sum((1,2))).argmax(1)
    components=torch.stack((root,vowel,consonant),dim=1)
    submission_components=components.clone()
    submission_components[:,2]=torch.where(consonant==7,2,consonant)
    return dict(joint_classes=(root*11+vowel)*8+consonant,
                components=components,submission_components=submission_components)
