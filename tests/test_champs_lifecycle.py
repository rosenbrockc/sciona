from types import SimpleNamespace

import pytest
import torch

import sciona.champs_lifecycle as lifecycle
from sciona.champs_training import TrainingOptions


def test_selection_requires_all_eight_finite_types():
    assert lifecycle.selected_metric((torch.tensor(1.),torch.tensor(-2.),torch.full((8,),-2.)))==-2.
    for values in (torch.full((7,),-2.),torch.full((8,),float('nan'))):
        with pytest.raises(ValueError,match='all eight'):
            lifecycle.selected_metric((torch.tensor(1.),torch.tensor(-2.),values))


def test_best_checkpoint_retains_earlier_epoch_on_tie(monkeypatch,tmp_path):
    events=[]
    class Controller:
        def __init__(self,*args):
            self.index=0
            self._namespace={}
        def epoch(self,batches,training=True):
            if training: return (torch.tensor(1.),None,None)
            metric=[2.,1.,1.][self.index]
            self.index+=1
            return (torch.tensor(1.),torch.tensor(metric),torch.full((8,),metric))
        def validation_schedule_step(self,metric):
            events.append(('schedule',metric))
    monkeypatch.setattr(lifecycle,'ChampsTrainer',Controller)
    monkeypatch.setattr(lifecycle,'write_checkpoint',lambda *a,epoch,metric:events.append(('save',epoch,metric)))
    runtime=SimpleNamespace(create_model=lambda *a,**kw:None)
    packed=tuple(torch.zeros(4,1) for _ in range(10))
    report=lifecycle.fit_variant(runtime,'model_H',packed,None,TrainingOptions(batch_size=2),
        epochs=3,checkpoint_path=tmp_path/'synthetic.pt',validation=packed)
    assert report['selected_epoch']==1
    assert [r['selected'] for r in report['history']]==[True,True,False]
    assert events==[('schedule',2.),('save',0,2.),('schedule',1.),('save',1,1.),('schedule',1.)]


def test_validation_driven_schedule_requires_validation(tmp_path):
    packed=tuple(torch.zeros(4,1) for _ in range(10))
    with pytest.raises(ValueError,match='requires a validation'):
        lifecycle.fit_variant(None,'model_H',packed,None,TrainingOptions(batch_size=2,scheduler='dev_perf'),
            epochs=1,checkpoint_path=tmp_path/'synthetic.pt')
    assert not (tmp_path/'synthetic.pt').exists()
