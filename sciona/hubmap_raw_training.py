"""Group-held-out raw slide preparation followed by the source training range."""
import numpy as np
from sciona.hubmap_raw_population import prepare_population
from sciona.hubmap_range import run_training_range


def prepare_fold(slides, groups, validation_groups, *, tile_size=1024, multiplier_bin=20):
    """Keep every slide of a held-out group exclusively in validation.

    Group labels are runtime inputs. Filtering and within-slide sorting commute
    with this slide-level partition; each partition retains original slide order
    within the unshifted and half-shifted grids.
    """
    slides = list(slides)
    groups = list(groups)
    validation_groups = list(validation_groups)
    if len(slides) != len(groups) or not slides:
        raise ValueError('Aligned nonempty slides and group labels required')
    if any(not isinstance(group, (str, int, np.integer)) or isinstance(group, bool)
           for group in groups + validation_groups):
        raise ValueError('String or integer group labels required')
    held_out = set(validation_groups)
    train = [slide for slide, group in zip(slides, groups) if group not in held_out]
    valid = [slide for slide, group in zip(slides, groups) if group in held_out]
    if not train or not valid:
        raise ValueError('Both training and validation must contain slides')
    training = prepare_population(train, tile_size=tile_size, multiplier_bin=multiplier_bin)
    validation = prepare_population(valid, tile_size=tile_size, multiplier_bin=multiplier_bin)
    if not len(training['rles']) or not len(validation['rles']):
        raise ValueError('Both partitions must retain selected tiles')
    return training, {key: validation[key] for key in ['images_bgr', 'rles']}


def run_raw_training(model, optimizer, scheduler, slides, groups, validation_groups,
                     *, tile_size=1024, multiplier_bin=20, **training_options):
    training, validation = prepare_fold(slides, groups, validation_groups,
                                        tile_size=tile_size, multiplier_bin=multiplier_bin)
    return run_training_range(model, optimizer, scheduler, training, validation, **training_options)
