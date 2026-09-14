"""Hash-verified full Cornell DenseNet121 SED and source loss construction."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from sciona.cornell_augmentation import manifest,PINS


class SourceModel:
    def __init__(self,source):
        pins=manifest('competition_cornell_source_pins.json',PINS)
        expected={p['software_path']:p['sha256'] for p in pins['pins']}
        self.namespace=dict(torch=torch,nn=nn,F=F,np=np,librosa=librosa,EPSILON_FP16=1e-5)
        for name in ('augmentations/mixup.py','helpers/sed_audio_utils.py','models/sed_models.py',
                     'loss/sed_scaled_pos_neg_focal_loss.py','loss/sed_scaled_pos_neg_focal_loss_augd.py'):
            path='src/'+name;raw=(Path(source)/path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=expected[path]:raise ValueError('Cornell model source hash mismatch')
            text=raw.decode()
            if name=='helpers/sed_audio_utils.py':
                before='librosa.util.pad_center(fft_window, n_fft)'
                if text.count(before)!=1:raise ValueError('Unreviewed librosa compatibility change')
                text=text.replace(before,'librosa.util.pad_center(fft_window, size=n_fft)')
            nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),self.namespace)

    def build(self,*,seed,backbone_state=None,synthetic=False):
        """Require explicit complete backbone weights or synthetic initialization.

        No weight downloads. A supplied state must match the complete torchvision
        DenseNet121 including its classifier before source takes its features.
        Provenance/identity of supplied weights belongs to the lifecycle contract.
        """
        if type(synthetic) is not bool or (backbone_state is None and not synthetic):
            raise ValueError('Complete backbone state or explicit synthetic initialization required')
        def backbone(pretrained):
            model=models.densenet121(weights=None)
            if backbone_state is not None:model.load_state_dict(backbone_state,strict=True)
            return model
        self.namespace['models']=SimpleNamespace(densenet121=backbone)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model=self.namespace['PANNsDense121Att'](sample_rate=32000,window_size=1024,hop_size=320,
                mel_bins=64,fmin=50,fmax=14000,classes_num=264,apply_aug=True,top_db=None)
        return model

    def loss(self,*,augmented):
        if type(augmented) is not bool:raise ValueError('Expected boolean augmented loss')
        cls=self.namespace['SedScaledPosNegFocalLossAugd' if augmented else 'SedScaledPosNegFocalLoss']
        return cls(gamma=0.,alpha_0=1.,alpha_1=1.,secondary_factor=1.)

    def mix(self,values,weights):
        return self.namespace['do_mixup'](values,weights)
