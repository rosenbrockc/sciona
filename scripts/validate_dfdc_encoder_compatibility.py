"""Historical/current full B7 parity measurement with identical synthetic state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import warnings

import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--historical-root',type=Path,required=True)
    args=parser.parse_args()
    pins_path=ROOT/'docs/reviews/competition_dfdc_timm_dependency_pin.json'
    pins=json.loads(pins_path.read_text())
    for path,digest in pins['files'].items():
        assert hashlib.sha256((args.historical_root/path).read_bytes()).hexdigest()==digest
    torch.set_num_threads(2)
    with patch('torch.load',side_effect=AssertionError('implicit checkpoint load')), warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        sys.path.insert(0,str(args.historical_root))
        import timm as historical
        assert historical.__version__=='0.4.12'
        torch.manual_seed(717)
        old=historical.create_model('tf_efficientnet_b7_ns',pretrained=False,drop_path_rate=.2)
        sys.path.pop(0)
        for key in tuple(sys.modules):
            if key=='timm' or key.startswith('timm.'):del sys.modules[key]
        # Remove old timm-only TorchScript overload registrations before loading
        # another version under the same module name. B7 uses eager execution.
        import torch._jit_internal as jit_internal
        for key in tuple(jit_internal._overloaded_methods):
            if 'timm.' in key: del jit_internal._overloaded_methods[key]
        for key in tuple(jit_internal._overloaded_method_class_fileno):
            if 'timm.' in key[0]: del jit_internal._overloaded_method_class_fileno[key]
        import timm as current
        assert current.__version__=='0.9.2'
        new=current.create_model('tf_efficientnet_b7_ns',pretrained=False,drop_path_rate=.2)
        a,b=old.state_dict(),new.state_dict()
        assert a.keys()==b.keys()
        assert all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a)
        new.load_state_dict(a,strict=True)
        del a,b
        torch.manual_seed(719)
        x=torch.rand(1,3,380,380)
        old.eval();new.eval()
        with torch.no_grad():
            left=old.forward_features(x);right=new.forward_features(x)
        eval_exact=torch.equal(left,right)
        eval_error=float((left-right).abs().max())
        del left,right
        old.train();new.train()
        torch.manual_seed(720)
        left=old.forward_features(x)
        torch.manual_seed(720)
        right=new.forward_features(x)
        train_exact=torch.equal(left,right)
        train_error=float((left.detach()-right.detach()).abs().max())
        del left,right
        # Isolate non-stochastic computation by disabling only stochastic depth.
        old.load_state_dict(new.state_dict(),strict=True)
        for model in (old,new):
            for module in model.modules():
                if type(module).__name__=='DropPath':module.drop_prob=0.
                if hasattr(module,'drop_path_rate'):module.drop_path_rate=0.
        lx=x.clone().requires_grad_();rx=x.clone().requires_grad_()
        left=old.forward_features(lx);left.square().mean().backward()
        right=new.forward_features(rx);right.square().mean().backward()
        deterministic_exact=torch.equal(left,right)
        input_grad_exact=torch.equal(lx.grad,rx.grad)
        oldparams=dict(old.named_parameters());newparams=dict(new.named_parameters())
        assert oldparams.keys()==newparams.keys()
        grads=0;grad_mismatches=0
        for key,value in oldparams.items():
            other=newparams[key]
            if value.grad is None:
                assert other.grad is None
            else:
                grads+=1
                if not torch.equal(value.grad,other.grad):grad_mismatches+=1
        files={}
        for module in tuple(sys.modules.values()):
            name=getattr(module,'__name__','');path=getattr(module,'__file__',None)
            if name.startswith('timm') and path and path.endswith('.py'):
                files[name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report={'format':'dfdc-encoder-compatibility.v1','result':'measured',
            'harness_adaptation':'Removed old timm-only TorchScript overload registrations to import both versions; all encoder execution eager.',
            'historical_timm':'0.4.12','installed_timm':current.__version__,
            'checks':{'state_keys_shapes_dtypes_exact':True,'evaluation_features_exact':eval_exact,
                      'evaluation_max_abs_error':eval_error,'training_same_seed_features_exact':train_exact,
                      'training_max_abs_error':train_error,'stochastic_depth_disabled_features_exact':deterministic_exact,
                      'stochastic_depth_disabled_input_gradient_exact':input_grad_exact,
                      'parameter_gradients_compared':grads,'parameter_gradient_mismatches':grad_mismatches},
            'finding':'Historical stochastic depth uses floor(keep+uniform) and divides activations before masking; installed version uses Bernoulli and scales masks. Same-seed training equivalence cannot be assumed.',
            'limits':'Actual full B7 feature extractor at380, synthetic random shared state, current Torch CPU float32. Historical Torch/CUDA/AMP runtime and pretrained quality are not verified. No pipeline code changed by this measurement.',
            'installed_imported_python_sha256':files,
            'sha256':{'scripts/validate_dfdc_encoder_compatibility.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'docs/reviews/competition_dfdc_timm_dependency_pin.json':hashlib.sha256(pins_path.read_bytes()).hexdigest()}}
    (ROOT/'docs/reviews/competition_dfdc_encoder_compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
