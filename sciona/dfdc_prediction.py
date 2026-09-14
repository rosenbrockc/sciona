"""DFDC inference from decoded RGB using explicit detector and classifier models.

Source: MIT 2020 Selim Seferbekov, commit
89c6290490bac96b29193a4061b3db9dd3933e36; docs/licenses/DFDC-MIT.txt.
The CPU execution device is an explicit adaptation. Default classifier precision
remains source float16; float32 is a caller-selected numerical adaptation.
"""

import numpy as np
import torch
from torchvision.transforms import Normalize

from sciona.dfdc_faces import extract_faces
from sciona.dfdc_frame_sampling import select_frames
from sciona.dfdc_inference_primitives import (
    confident_strategy, isotropically_resize_image, put_to_center,
)


def predict_video(decoded_rgb, detector, models, *, frames_per_video=32, precision='float16'):
    """Return score and explicit source-fallback status, without media identifiers.

    Models must already be in evaluation mode on CPU with the requested dtype.
    Input validation and nonfinite output rejection are stricter than source.
    No file codec, JPEG recompression, model loading or download is implicit.
    """
    if precision not in ('float16', 'float32'):
        raise ValueError('precision must be float16 or float32')
    models = tuple(models)
    if not models:
        raise ValueError('at least one classifier is required')
    dtype = getattr(torch, precision)
    for model in models:
        if not isinstance(model, torch.nn.Module) or model.training:
            raise ValueError('classifiers must be evaluation-mode Torch modules')
        for parameter in model.parameters():
            if parameter.device.type != 'cpu' or parameter.dtype != dtype:
                raise ValueError('classifier parameters must match CPU/requested precision')
    selected = select_frames(decoded_rgb, frames_per_video)
    result = {'score': .5, 'status': 'source_fallback', 'reason': 'no_frames',
              'selected_frames': 0, 'classified_faces': 0, 'models': len(models),
              'precision': precision}
    if selected is None:
        return result
    result['selected_frames'] = len(selected[0])
    try:
        faces = extract_faces(selected[0], detector)
        capacity = frames_per_video * 4
        x = np.zeros((capacity, 380, 380, 3), dtype=np.uint8)
        count = 0
        for frame in faces:
            for face in frame['faces']:
                # Source prepares even excess faces before checking capacity.
                prepared = put_to_center(isotropically_resize_image(face, 380), 380)
                if count + 1 < capacity:
                    x[count] = prepared
                    count += 1
        result['classified_faces'] = count
        if count == 0:
            result['reason'] = 'no_faces'
            return result
        x = torch.tensor(x, device='cpu').float().permute(0, 3, 1, 2)
        normalize = Normalize([.485, .456, .406], [.229, .224, .225])
        for i in range(len(x)):
            x[i] = normalize(x[i] / 255.)
        scores = []
        with torch.no_grad():
            for model in models:
                logits = model(x[:count].to(dtype=dtype))
                probabilities = torch.sigmoid(logits.squeeze())
                # Preserve the source's one-face scalar-index error/fallback.
                values = probabilities[:count].cpu().numpy()
                scores.append(confident_strategy(values))
        score = float(np.mean(scores))
    except Exception:
        # Source returns .5 on execution errors; expose that status without logs
        # that could contain input names or content.
        result['reason'] = 'execution_error'
        return result
    if not np.isfinite(score):
        raise ValueError('classifier ensemble returned a nonfinite score')
    result.update(score=score, status='predicted', reason=None)
    return result
