"""Select corrected shared fold membership before entering any family trainer."""
from sciona.cassava_fold_contract import members


def train_planned_fold(family, plan, encoded_images, fold, **settings):
    train, valid = members(plan, fold)
    if (not isinstance(encoded_images, (list, tuple)) or len(encoded_images) != len(plan.keys)
            or not all(isinstance(image, bytes) and image for image in encoded_images)):
        raise ValueError('Encoded image population must align with the validated plan')
    if family == 'vit':
        from sciona.cassava_vit_fold import train_fold
    elif family == 'resnext':
        from sciona.cassava_resnext_fold import train_fold
    elif family == 'efficientnet':
        from sciona.cassava_efficientnet_training import train_fold
    else:
        raise ValueError('Unknown trainable ensemble family')
    return train_fold([encoded_images[i] for i in train], [plan.labels[i] for i in train],
                      [encoded_images[i] for i in valid], [plan.labels[i] for i in valid], **settings)
