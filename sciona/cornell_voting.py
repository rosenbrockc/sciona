"""Array-based realization of the Cornell winner's threshold/voting method.

This implementation operates on model probabilities without notebook code.
Time intervals are half-open: padded frames and the final endpoint are excluded.
One-frame events and empty detections are valid. The winner's second vote across
already-voted five-second intervals is retained for whole-record predictions.
"""
import numpy as np


def vote(clip_probabilities, frame_probabilities, *, duration_seconds):
    """Combine all thirteen models; retain a fixed 264-class source head.

    Clip probabilities: [13, chunks, 264]. Frame probabilities:
    [13, chunks, 3001, 264], from 30-second chunks at a 32000/320 hop.
    All ten source evaluation copies have already been averaged per model.
    """
    if isinstance(duration_seconds, (bool, np.bool_)) or not np.isscalar(duration_seconds):
        raise ValueError('Expected positive finite duration')
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError('Expected positive finite duration')
    chunks = int(np.ceil(duration_seconds / 30))
    clips = np.asarray(clip_probabilities)
    frames = np.asarray(frame_probabilities)
    if clips.shape != (13, chunks, 264) or frames.shape != (13, chunks, 3001, 264):
        raise ValueError('Expected thirteen complete source model outputs')
    for values in (clips, frames):
        if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError('Expected finite probabilities')
        tolerance=8*np.finfo(np.float32).eps
        if np.any((values < -tolerance) | (values > 1+tolerance)):
            raise ValueError('Probabilities outside unit interval')
    windows = int(np.ceil(duration_seconds / 5))
    counts = np.zeros((windows, 264), dtype=np.int16)
    for window in range(windows):
        chunk, section = divmod(window, 6)
        start = section * 500
        # Integer centisecond coordinates avoid decimal floor boundary drift.
        remaining = duration_seconds * 100 - chunk * 3000
        end = min(start + 500, int(np.ceil(remaining)), 3000)
        active = (frames[:, chunk, start:end] >= .3).any(axis=1)
        active &= clips[:, chunk] >= .3
        counts[window] = active.sum(axis=0)
    decisions = counts >= 4
    return {'window_votes': counts, 'window_decisions': decisions,
            'whole_record_decisions': decisions.sum(axis=0) >= 4}
