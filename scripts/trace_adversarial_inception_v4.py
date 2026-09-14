"""Fixed full-size inference DAG from pinned public Inception-v4 source."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from trace_adversarial_inception_v3 import ROOT, Tensor, Trace


class ShapeTensor(Tensor):
    def __getitem__(self, index): return self.shape[index]


class V4Trace(Trace):
    def __init__(self):
        super().__init__()
        self.nn = SimpleNamespace(softmax=lambda x,name:self.softmax(x,scope=name))

    def add(self, *args, **kwargs):
        t = super().add(*args, **kwargs)
        return ShapeTensor(t.name,t.shape)

    def dropout(self, x, keep_prob=.8, **kwargs):
        assert keep_prob==.8
        return super().dropout(x, **kwargs)

    def flatten(self, x, scope=None):
        return self.add('flatten', [x], [x.shape[0], math.prod(x.shape[1:])], scope)

    def fully_connected(self, x, channels, *, activation_fn, scope):
        assert activation_fn is None and len(x.shape)==2
        return self.add('linear', [x], [x.shape[0],channels], scope,
                        in_features=x.shape[1], out_features=channels)


def trace():
    pins=json.loads((ROOT/'docs/reviews/competition_adversarial_source_pins.json').read_text())
    pin=next(s for s in pins['sources'] if s['branch']=='non_targeted')
    p=Path('/private/tmp/sciona_adversarial_source/non_targeted/nets/inception_v4.py')
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==next(f['sha256'] for f in pin['files'] if f['path']=='nets/inception_v4.py')
    body=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)]
    t=V4Trace();ns={'tf':t,'slim':t}
    exec(compile(ast.Module(body=body,type_ignores=[]),'<pinned-inception-v4-topology>','exec'),ns)
    logits, ends=ns['inception_v4'](ShapeTensor('input',[1,299,299,3]),num_classes=1001,is_training=False)
    return dict(format='adversarial-inception-v4-topology.v1',source_sha256=digest,
                nodes=t.nodes,logits=logits.name,endpoints={k:v.name for k,v in ends.items()})


if __name__=='__main__':
    result=trace()
    (ROOT/'sciona/adversarial_inception_v4_topology.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'operations':len(result['nodes']),'convolutions':sum(n['op']=='conv' for n in result['nodes'])}))
