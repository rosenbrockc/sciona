"""Probe the complete winning DenseNet SED topology on synthetic waveforms."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.source/'manifest.json').read_text())
    for pin in manifest['pins']:
        assert hashlib.sha256((args.source/pin['software_path']).read_bytes()).hexdigest()==pin['sha256']
    torch.set_num_threads(2);torch.manual_seed(1729)
    ns=dict(torch=torch,nn=nn,F=F,np=np,librosa=librosa,
        models=SimpleNamespace(densenet121=lambda pretrained:models.densenet121(weights=None)))
    for name in ('augmentations/mixup.py','helpers/sed_audio_utils.py','models/sed_models.py','loss/sed_scaled_pos_neg_focal_loss.py'):
        text=(args.source/'src'/name).read_text()
        if name=='helpers/sed_audio_utils.py':
            assert text.count('librosa.util.pad_center(fft_window, n_fft)')==1
            text=text.replace('librosa.util.pad_center(fft_window, n_fft)','librosa.util.pad_center(fft_window, size=n_fft)')
        tree=ast.parse(text)
        selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=selected,type_ignores=[]),name,'exec'),ns)
    ns['EPSILON_FP16']=1e-5
    model=ns['PANNsDense121Att'](sample_rate=32000,window_size=1024,hop_size=320,
        mel_bins=64,fmin=50,fmax=14000,classes_num=264,apply_aug=True,top_db=None)
    waveform=torch.randn(2,1,64000)*.1
    original=waveform.clone()
    spec=model.spectrogram_extractor(waveform[:,0])[:,0]
    reference=torch.stft(waveform[:,0],n_fft=1024,hop_length=320,win_length=1024,
        window=torch.hann_window(1024),center=True,pad_mode='reflect',return_complex=True).abs().square().transpose(1,2)
    torch.testing.assert_close(spec,reference,rtol=2e-4,atol=2e-4)
    mel=model.logmel_extractor(spec[:,None])[:,0]
    weights=torch.tensor(librosa.filters.mel(sr=32000,n_fft=1024,n_mels=64,fmin=50,fmax=14000).T)
    reference_mel=10*torch.log10(torch.clamp(reference@weights,min=1e-10))
    torch.testing.assert_close(mel,reference_mel,rtol=2e-4,atol=2e-4)
    model.train()
    lam=torch.tensor([.25,.75])
    output=model((waveform,lam))
    assert output['clipwise_output'].shape==(1,1,264)
    assert output['framewise_output'].shape==(1,1,201,264)
    labels=torch.zeros((2,1,264));labels[0,0,0]=1;labels[1,0,1]=1
    labels=ns['do_mixup'](labels,lam)
    torch.testing.assert_close(labels[0,0,:2],torch.tensor([.25,.75]))
    loss,_=ns['SedScaledPosNegFocalLoss'](gamma=0,secondary_factor=1)(output['clipwise_output'],
        dict(all_labels=labels,secondary_labels=torch.zeros_like(labels)))
    expected=F.binary_cross_entropy(output['clipwise_output'].clamp(1e-5,1-1e-5),labels)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    loss.backward()
    assert model.densenet_features.conv0.weight.grad is not None
    assert model.att_block.att.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        first=model((waveform[:1],None))
        state={k:v.clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state,strict=True)
        second=model((waveform[:1],None))
    for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    torch.testing.assert_close(waveform,original,rtol=0,atol=0)
    report=dict(source_commit=manifest['commit'],complete_densenet121_topology=True,classes=264,
        stft_reference_passed=True,logmel_reference_passed=True,
        max_abs_stft_error=float((spec-reference).abs().max()),max_abs_logmel_error=float((mel-reference_mel).abs().max()),
        mixed_waveform_forward_backward=True,source_gamma_zero_loss_matches_bce=True,
        finite_backbone_attention_gradients=True,strict_state_reload_exact=True,caller_input_preserved=True,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Synthetic two-second waveforms; no complete training schedule or thirteen-model lifecycle claim.',
          'Full DenseNet121 initialized without pretrained weights explicitly for offline topology/gradient evidence; no pretrained or competition accuracy parity.',
          'Only modern librosa pad_center keyword compatibility changed; source classes executed through AST to avoid unrelated launchers and logging.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
