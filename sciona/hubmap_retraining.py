"""Source positive-pseudo population append before balanced retraining."""
import numpy as np
from sciona.hubmap_raw_population import prepare_population
from sciona.hubmap_raw_training import prepare_fold
from sciona.hubmap_range import run_training_range


def append_pseudo(training,pseudo_populations):
    """Append ordered pseudo sources, retaining rounded-bin-positive tiles only."""
    result=dict(images_bgr=list(training['images_bgr']),rles=list(training['rles']),
                bins=list(training['bins']),present=list(training['present']))
    for population in pseudo_populations:
        count=len(population['rles'])
        if any(len(population[key])!=count for key in ['images_bgr','bins','present']):
            raise ValueError('Aligned pseudo population required')
        bins=np.asarray(population['bins'])
        if bins.ndim!=1 or bins.dtype.kind not in 'iu' or (bins<0).any():
            raise ValueError('Nonnegative integer pseudo bins required')
        if not np.array_equal(population['present'],bins>0):
            raise ValueError('Pseudo presence must equal rounded bin positivity')
        for i in np.flatnonzero(bins>0):
            for key in result:result[key].append(population[key][i])
    result['bins']=np.asarray(result['bins'],dtype=int)
    result['present']=np.asarray(result['present'],dtype=bool)
    return result


def prepare_retraining(slides,groups,validation_groups,pseudo_sources,*,tile_size=1024,multiplier_bin=20):
    """Pseudo sources are ordered collections of runtime RGB/binary-mask pairs.

    Each source contributes its unshifted and half-shifted populations in source
    order. No pseudo slide or pseudo mask enters labeled validation.
    """
    training,validation=prepare_fold(slides,groups,validation_groups,tile_size=tile_size,multiplier_bin=multiplier_bin)
    pseudo=(prepare_population(source,tile_size=tile_size,multiplier_bin=multiplier_bin) for source in pseudo_sources)
    return append_pseudo(training,pseudo),validation


def run_retraining(model,optimizer,scheduler,slides,groups,validation_groups,pseudo_sources,
                   *,tile_size=1024,multiplier_bin=20,**training_options):
    training,validation=prepare_retraining(slides,groups,validation_groups,pseudo_sources,
                                         tile_size=tile_size,multiplier_bin=multiplier_bin)
    return run_training_range(model,optimizer,scheduler,training,validation,**training_options)
