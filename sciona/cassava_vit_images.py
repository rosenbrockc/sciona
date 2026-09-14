"""Distinct ViT training, stochastic validation and deterministic inference views."""


def transformations(stage):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    if stage == 'train':
        operations = [A.RandomResizedCrop(384, 384), A.Transpose(p=.5),
                      A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.ShiftScaleRotate(p=.5),
                      A.HueSaturationValue(hue_shift_limit=.2, sat_shift_limit=.2, val_shift_limit=.2, p=.5),
                      A.RandomBrightnessContrast(brightness_limit=(-.1, .1), contrast_limit=(-.1, .1), p=.5)]
    elif stage in ('validation', 'inference'):
        operations = [A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5)] if stage == 'validation' else []
        operations += [A.CenterCrop(384, 384, p=1.), A.Resize(384, 384)]
    else:
        raise ValueError('Explicit train, validation or inference stage required')
    operations += [A.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225], max_pixel_value=255., p=1.)]
    if stage == 'train':
        # Source holes are zero in normalized space, not black raw pixels.
        operations += [A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=.5),
                       A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=.5)]
    return A.Compose([*operations, ToTensorV2(p=1.)], p=1.)
