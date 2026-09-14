"""Callable stage boundaries for the source-compared HuBMAP execution graph."""
from sciona.hubmap_lifecycle import _train_stages
from sciona.hubmap_raw_inference import load_inference_model,predict_slide


def initial_training(slides,groups,initial_stages,initial_selectors,tile_size=1024):
    return _train_stages(initial_stages,initial_selectors,slides,groups,None,tile_size)


def restore_models(checkpoints,input_resolution=320):
    if not checkpoints:raise ValueError('Nonempty ordered checkpoint ensemble required')
    return [load_inference_model(payload,input_resolution=input_resolution) for payload in checkpoints]


def _geometry(options):
    result=dict(resolution=1024,input_resolution=320,pad_size=256,batch_size=12)
    if options is not None:
        if set(options)-set(result):raise ValueError('Only inference geometry overrides allowed')
        result.update(options)
    return result


def pseudo_labels(models,pseudo_sources,pseudo_rng,inference_geometry=None):
    geometry=_geometry(inference_geometry)
    return [[(image,predict_slide(models,image,tta=4,threshold=.5,**geometry,**pseudo_rng))
             for image in source] for source in pseudo_sources]


def retraining(slides,groups,retraining_stages,final_selectors,pseudo_labeled_sources,tile_size=1024):
    return _train_stages(retraining_stages,final_selectors,slides,groups,pseudo_labeled_sources,tile_size)


def final_prediction(models,final_images,final_rng,inference_geometry=None):
    geometry=_geometry(inference_geometry)
    return [predict_slide(models,image,tta=3,threshold=.5,**geometry,**final_rng) for image in final_images]
