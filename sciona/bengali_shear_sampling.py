"""imgaug 0.3.0 affine parameter draw order for the notebook's fixed settings.

Accepts an already derived per-image RNG. Wrapper seeding and restoration are
separate requirements. Constants still advance the historical RNG.
"""
import numpy as np


def _advance(rng):
    if isinstance(rng,np.random.RandomState):
        state=list(rng.get_state());state[-2]=0;rng.set_state(tuple(state))
        rng.uniform()
    else:
        state=rng.bit_generator.state;state['has_uint32']=0;rng.bit_generator.state=state
        rng.random()


def draw_shear(rng, *, limit=5):
    if type(limit) is not int or limit not in (5,20):
        raise ValueError('source shear limit must be5 or20 degrees')
    if not (isinstance(rng,np.random.RandomState) or
            isinstance(rng,np.random.Generator) and isinstance(rng.bit_generator,np.random.SFC64)):
        raise ValueError('historical RandomState or SFC64 generator required')
    # scale Uniform(1,1): two endpoint draws plus outer draw; translation: one;
    # rotation Uniform(0,0): three; shear endpoints: two.
    for _ in range(9):_advance(rng)
    shear=float(rng.uniform(-float(limit),float(limit),size=(1,)).astype(np.float32)[0])
    # Shear outer draw, then cval, border mode and interpolation constants.
    for _ in range(4):_advance(rng)
    return shear
