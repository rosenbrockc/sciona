"""ResNeXt decoded-image transformations in the Albumentations 1.3 reference."""
import cv2
import numpy as np


def decode_rgb(encoded):
    """Decode the same OpenCV color path used by source imread/BGR-to-RGB."""
    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError('Nonempty encoded image bytes required')
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Image decode failed')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def transformations(*, training):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    if not isinstance(training, bool):
        raise ValueError('Explicit training boolean required')
    if training:
        spatial = [A.RandomResizedCrop(512, 512, scale=(.08, 1.), ratio=(.75, 4/3), interpolation=cv2.INTER_LINEAR),
                   A.Transpose(p=.5), A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5),
                   A.ShiftScaleRotate(shift_limit=.0625, scale_limit=.1, rotate_limit=45,
                                      interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=.5)]
    else:
        spatial = [A.Resize(512, 512, interpolation=cv2.INTER_LINEAR)]
    return A.Compose([*spatial, A.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225], max_pixel_value=255.), ToTensorV2()])


def transform_rgb(image, transform):
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
        raise ValueError('Nonempty uint8 RGB pixels required')
    return transform(image=pixels)['image']
