"""Execute both pinned Cornell augmentation recipes using synthetic noise banks."""
import ast
import functools
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def namespace(seed):
    cache=Path('/private/tmp/sciona_cornell_dependencies/audiomentations')
    doc=json.loads((ROOT/'docs/reviews/competition_cornell_historical_dependencies.json').read_text())
    manifest=next(d for d in doc if d['wheel'].startswith('audiomentations-'))
    pins={p['software_path']:p['sha256'] for p in manifest['pins']}
    numpy=SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if k!='random'},random=np.random.RandomState(seed))
    noise=(.15*np.sin(np.arange(32000)*.13)+.04).astype(np.float32)
    ns=dict(np=numpy,random=random.Random(seed),functools=functools,
            get_file_paths=lambda _:['synthetic'])
    for name in ['core/transforms_interface.py','core/composition.py','core/utils.py','augmentations/transforms.py']:
        path='audiomentations/'+name;raw=(cache/path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pins[path]
        definitions=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and getattr(n,'name','')!='get_file_paths']
        exec(compile(ast.Module(body=definitions,type_ignores=[]),path,'exec'),ns)
    for name in ('AddBackgroundNoise','AddShortNoises'):
        setattr(ns[name],'_'+name+'__load_sound',staticmethod(lambda path,rate:(noise.copy(),rate)))
    return ns,noise


def main():
    cache=Path('/private/tmp/sciona_cornell_source')
    manifest=json.loads((cache/'manifest.json').read_text());pins={p['software_path']:p['sha256'] for p in manifest['pins']}
    waveform=(np.sin(np.arange(960000)*.021)*.2).astype(np.float32)
    before=waveform.copy();reports=[]
    for variant in ('default','background'):
        path='src/augmentations/sed_'+variant+'_augment.py';raw=(cache/path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pins[path]
        tree=ast.parse(raw);defs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        activated=set();names=None
        for seed in range(24):
            outputs=[]
            for repeat in range(2):
                ns,_=namespace(seed)
                exec(compile(ast.Module(body=defs,type_ignores=[]),path,'exec'),ns)
                args=dict(bckgrd_aug_dir='synthetic')
                if variant=='background':args['secondary_bckgrd_aug_dir']='synthetic'
                transforms=ns['get_transforms'](**args)
                sequence=transforms['train'].__closure__[0].cell_contents
                names=[type(t).__name__ for t in sequence.transforms]
                output=transforms['train'](waveform.copy(),32000)
                assert output.shape==waveform.shape and np.isfinite(output).all()
                np.testing.assert_array_equal(transforms['valid'](waveform.copy(),32000),waveform)
                activated.update(i for i,t in enumerate(sequence.transforms) if t.parameters['should_apply'])
                outputs.append(output)
            np.testing.assert_array_equal(outputs[0],outputs[1])
        assert activated==set(range(len(names)))
        reports.append(dict(variant=variant,sequence=names,seeds=24,all_positions_activated=True,exact_seed_replay=True))
    ns,noise=namespace(2)
    background=ns['AddBackgroundNoise']('synthetic')
    background.parameters=dict(noise_file_path='synthetic',noise_start_index=0,noise_end_index=len(waveform),snr_in_db=12)
    actual=background.apply(waveform,32000)
    rms=lambda x:np.sqrt(np.mean(x*x))
    scaled=noise*(rms(waveform)/(10**(12/20))/rms(noise))
    expected=waveform+np.tile(scaled,int(np.ceil(len(waveform)/len(scaled))))[:len(waveform)]
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    short=ns['AddShortNoises']('synthetic')
    short.parameters=dict(sounds=[dict(start=.5,end=1.5,fade_in_time=.01,fade_out_time=.02,file_path='synthetic',snr_in_db=6)])
    actual=short.apply(waveform,32000)
    gain=np.ones_like(noise);gain[:320]=np.linspace(0,1,320);gain[-640:]=np.linspace(1,0,640)
    faded=noise*gain;expected=waveform.copy()
    expected[16000:48000]+=faded*(rms(waveform[16000:48000])/(10**(6/20))/rms(faded))
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    # Source edge failures must remain visible before a guarded runtime wraps them.
    ns['AddBackgroundNoise']._AddBackgroundNoise__load_sound=staticmethod(lambda path,rate:(np.zeros_like(noise),rate))
    with np.errstate(divide='ignore',invalid='ignore'):
        assert not np.isfinite(background.apply(waveform,32000)).all()
    short.parameters['sounds'][0]['fade_out_time']=0
    try:short.apply(waveform,32000)
    except ValueError:pass
    else:raise AssertionError('Zero fade-out source failure not reproduced')
    np.testing.assert_array_equal(before,waveform)
    report=dict(status='passed',source_commit=manifest['commit'],recipes=reports,
        background_rms_repeat_reference=True,short_noise_fade_rms_reference=True,
        silent_background_nonfinite_reproduced=True,zero_fade_out_failure_reproduced=True,
        original_input_preserved=True,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Original audiomentations0.11.0 classes and complete source recipes. Private I/O replaced with in-memory synthetic bank; caller RNG namespaces isolate randomness. No full training or resampling equivalence claim.')
    (ROOT/'docs/reviews/competition_cornell_augmentation_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
