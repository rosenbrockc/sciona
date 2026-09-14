import hashlib
import json
import numpy as np
import pytest
import torch
from torch import nn

from sciona.bengali_fit import fit_population
from sciona.bengali_font_sampling import FontParameterSampler
from sciona.bengali_gan_training import CycleGANTraining
from sciona.bengali_population import ImagePopulation
from sciona.bengali_sampling import SamplingBudget


def trainer(steps):
    return CycleGANTraining(generator_a=nn.Conv2d(3,3,1),generator_b=nn.Conv2d(3,3,1),
        discriminator_a=nn.Conv2d(3,1,1),discriminator_b=nn.Conv2d(3,1,1),
        classifier=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,14784)),
        classifier_weight=1.,total_steps=steps,replay_seed_a=3,replay_seed_b=4)


def population():
    return ImagePopulation(np.random.default_rng(61).integers(0,256,(4,137,236),dtype=np.uint8),[0,1,2,14783])


def sampler():return FontParameterSampler(python_seed=19,image_seed=38,backend='sfc64')


def test_complete_declared_fit_saves_reloadable_epoch_generators(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(95)
    model=trainer(4);data=population();directory=tmp_path/'fit'
    result=fit_population(model,data,data,SamplingBudget(4,4,2,2),hand_seed=3,font_seed=4,
                          font_sampler=sampler(),output_directory=directory)
    assert result['completed_steps']==4 and result['completed_epochs']==2 and not result['approved']
    assert json.loads((directory/'history.json').read_text())==result['history']
    for entry in result['history']:
        path=directory/entry['generator_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['generator_sha256']
        fresh=nn.Conv2d(3,3,1)
        fresh.load_state_dict(torch.load(path,weights_only=True))
    inputs=torch.rand(1,3,224,224)
    torch.testing.assert_close(fresh(inputs),model.generator_b(inputs),rtol=0,atol=0)
    assert all(not m.training for m in [model.generator_a,model.generator_b,model.discriminator_a,model.discriminator_b])


def test_budget_mismatch_precedes_files_and_training(tmp_path):
    data=population();model=trainer(3);directory=tmp_path/'fit'
    with pytest.raises(ValueError,match='budget'):
        fit_population(model,data,data,SamplingBudget(4,4,2,2),hand_seed=3,font_seed=4,font_sampler=sampler(),output_directory=directory)
    assert model.completed_steps==0 and not directory.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    data=population();model=trainer(2);marker=tmp_path/'keep';marker.write_text('synthetic marker')
    with pytest.raises(FileExistsError):
        fit_population(model,data,data,SamplingBudget(4,4,2,1),hand_seed=3,font_seed=4,font_sampler=sampler(),output_directory=tmp_path)
    assert marker.read_text()=='synthetic marker' and model.completed_steps==0
