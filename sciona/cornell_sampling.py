"""Private waveform sampling with the pinned Cornell source boundaries."""
import numpy as np

SAMPLE_RATE = 32000
PERIOD_SAMPLES = SAMPLE_RATE * 30


def sample_waveform(waveform, *, training, rng):
    """Sample after length-preserving augmentation; no I/O or global RNG use.

    Training uses one random thirty-second crop or insertion. As in the source,
    the final possible random offset is excluded. Validation takes the first
    two consecutive thirty-second clips and right-pads short waveforms.
    """
    values = np.asarray(waveform)
    if values.ndim != 1 or not values.size or values.dtype.kind not in 'fiu':
        raise ValueError('Expected a nonempty real mono waveform')
    if not np.isfinite(values).all():
        raise ValueError('Waveform must be finite')
    if type(training) is not bool or not isinstance(rng, np.random.RandomState):
        raise ValueError('Expected boolean training and caller RandomState')
    length = len(values)
    if training:
        if length < PERIOD_SAMPLES:
            offset = rng.randint(PERIOD_SAMPLES-length)
            result = np.zeros(PERIOD_SAMPLES, dtype=values.dtype)
            result[offset:offset+length] = values
        elif length > PERIOD_SAMPLES:
            offset = rng.randint(length-PERIOD_SAMPLES)
            result = values[offset:offset+PERIOD_SAMPLES]
        else:
            result = values
        result = result[None]
    else:
        result = np.zeros(2*PERIOD_SAMPLES, dtype=values.dtype)
        count = min(length, len(result))
        result[:count] = values[:count]
        result = result.reshape(2, PERIOD_SAMPLES)
    with np.errstate(over='ignore'):
        result = np.array(result, dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError('Waveform exceeds float32 range')
    return result
