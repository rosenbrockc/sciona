"""Initial folds, pseudo labels, fresh retraining folds and final inference."""
from dataclasses import dataclass
import torch
from sciona.hubmap_network import UNET_SERESNEXT101
from sciona.hubmap_schedule import CosineLR
from sciona.hubmap_raw_training import run_raw_training
from sciona.hubmap_retraining import run_retraining
from sciona.hubmap_raw_inference import load_inference_model,predict_slide


@dataclass
class FoldStage:
    """Explicit runtime encoder state, group holdout and independent stage RNGs."""
    encoder_state: dict
    model_seed: int
    validation_groups: list
    options: dict


def train_fold(stage,slides,groups,*,pseudo_sources=None,tile_size=1024):
    """Construct fresh decoder, Adam and Cosine state for either training stage.

    The required encoder state comes from runtime configuration. No pretrained
    download, prior segmentation checkpoint or optimizer resume is implicit.
    """
    if not isinstance(stage,FoldStage) or not isinstance(stage.encoder_state,dict) or not stage.encoder_state:
        raise ValueError('FoldStage with explicit encoder state required')
    options=dict(stage.options)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stage.model_seed)
        model=UNET_SERESNEXT101((options['input_side'],options['input_side']),True,True,None,
                              load_weights=False,encoder_state=stage.encoder_state)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=1e-5)
    scheduler=CosineLR(optimizer,step_size_min=1e-6,t0=19,tmult=1)
    if pseudo_sources is None:
        return run_raw_training(model,optimizer,scheduler,slides,groups,stage.validation_groups,
                                tile_size=tile_size,**options)
    return run_retraining(model,optimizer,scheduler,slides,groups,stage.validation_groups,pseudo_sources,
                          tile_size=tile_size,**options)


def _train_stages(stages,selectors,slides,groups,pseudo_sources,tile_size):
    selected={};metrics=[]
    for fold,stage in enumerate(stages):
        result=train_fold(stage,slides,groups,pseudo_sources=pseudo_sources,tile_size=tile_size)
        metrics.append(result['epochs'])
        for selected_fold,role in selectors:
            if selected_fold==fold:
                if role not in result['checkpoints']:
                    raise ValueError('Requested checkpoint role was not produced')
                selected[(fold,role)]=result['checkpoints'][role]
    return [selected[tuple(key)] for key in selectors],metrics


def run_lifecycle(slides,groups,pseudo_sources,final_images,*,initial_stages,retraining_stages,
                  initial_selectors,final_selectors,pseudo_rng,final_rng,tile_size=1024,
                  inference_geometry=None):
    """Execute ordered runtime source collections and explicit fold ensembles.

    Selectors are ordered (fold index, checkpoint role) pairs. Fresh retraining
    stages use their own encoder states/RNGs; initial checkpoints only generate
    pseudo masks. Source pseudo TTA4 and final notebook TTA3 remain distinct.
    """
    slides=list(slides);groups=list(groups)
    pseudo_sources=[list(source) for source in pseudo_sources];final_images=list(final_images)
    initial_stages=list(initial_stages);retraining_stages=list(retraining_stages)
    initial_selectors=list(initial_selectors);final_selectors=list(final_selectors)
    for stages,selectors in [(initial_stages,initial_selectors),(retraining_stages,final_selectors)]:
        if not stages or not selectors:
            raise ValueError('Nonempty training stages and checkpoint selectors required')
        for fold,role in selectors:
            if isinstance(fold,bool) or not isinstance(fold,int) or not 0<=fold<len(stages) or not isinstance(role,str):
                raise ValueError('Valid ordered checkpoint selectors required')
    geometry=dict(resolution=1024,input_resolution=320,pad_size=256,batch_size=12)
    if inference_geometry is not None:
        if set(inference_geometry)-set(geometry):raise ValueError('Only inference geometry overrides allowed')
        geometry.update(inference_geometry)
    initial,initial_metrics=_train_stages(initial_stages,initial_selectors,slides,groups,None,tile_size)
    models=[load_inference_model(payload,input_resolution=geometry['input_resolution']) for payload in initial]
    pseudo_masks=[]
    for source in pseudo_sources:
        pseudo_masks.append([(image,predict_slide(models,image,tta=4,threshold=.5,**geometry,**pseudo_rng)) for image in source])
    del models,initial
    retrained,retraining_metrics=_train_stages(retraining_stages,final_selectors,slides,groups,pseudo_masks,tile_size)
    models=[load_inference_model(payload,input_resolution=geometry['input_resolution']) for payload in retrained]
    predictions=[predict_slide(models,image,tta=3,threshold=.5,**geometry,**final_rng) for image in final_images]
    return dict(initial_epochs=initial_metrics,retraining_epochs=retraining_metrics,
                checkpoints=retrained,predictions=predictions)
