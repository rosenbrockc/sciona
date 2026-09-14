import numpy as np
import pytest
import torch
from torch import nn
from sciona.bengali_classifier_fit import fit_font_classifier
from sciona.bengali_population import ImagePopulation
from sciona.bengali_pretraining_sampling import PretrainingParameterSampler


@pytest.mark.parametrize('target,has_best',[(7,True),(6,False)])
def test_source_checkpoint_selection_and_complete_fit(tmp_path,target,has_best):
    torch.set_num_threads(2)
    model=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,14784))
    with torch.no_grad():
        model[-1].weight.zero_();model[-1].bias.zero_();model[-1].bias[7]=100
    data=ImagePopulation(np.random.default_rng(73).integers(0,256,(4,137,236),dtype=np.uint8),[target]*4)
    result=fit_font_classifier(model,data,epochs=2,batch_size=2,train_seed=5,validation_seed=6,
        sampler=PretrainingParameterSampler(python_seed=7,image_seed=8,backend='sfc64'),output_directory=tmp_path/'fit')
    assert result['optimizer_updates']==4 and result['completed_epochs']==2
    assert (result['best_checkpoint'] is not None)==has_best
    assert not result['evaluation_is_held_out'] and not result['approved']
    assert [r['same_population_accuracy'] for r in result['history']]==[float(has_best)]*2
    state=torch.load(tmp_path/'fit'/result['last_checkpoint']['file'],weights_only=True)
    for name,value in model.state_dict().items():torch.testing.assert_close(state[name],value,rtol=0,atol=0)
    assert not model.training
