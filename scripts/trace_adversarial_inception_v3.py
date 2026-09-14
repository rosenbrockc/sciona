"""Trace pinned public Slim source into a fixed full-size inference operation DAG.

Software-derived topology only. No TensorFlow installation, execution, or weights.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, name, shape): self.name, self.shape = name, list(shape)
    def get_shape(self): return self
    def as_list(self): return list(self.shape)


class Trace:
    def __init__(self):
        self.nodes, self.defaults, self.scope = [], {}, ''

    @contextmanager
    def variable_scope(self, name, *args, **kwargs):
        old = self.scope
        self.scope = name.name if isinstance(name, SimpleNamespace) else '/'.join(filter(None, (old, name)))
        try: yield SimpleNamespace(name=self.scope)
        finally: self.scope = old

    @contextmanager
    def arg_scope(self, functions, **kwargs):
        old = self.defaults.copy()
        for fn in functions: self.defaults[fn.__name__] = {**self.defaults.get(fn.__name__, {}), **kwargs}
        try: yield
        finally: self.defaults = old

    def add(self, op, inputs, shape, scope=None, **params):
        name = '/'.join(filter(None, (self.scope, scope or op+str(len(self.nodes)))))
        assert name not in {n['name'] for n in self.nodes}
        self.nodes.append(dict(name=name, op=op, inputs=[x.name for x in inputs], shape=list(shape), **params))
        return Tensor(name, shape)

    def spatial(self, op, x, kernel, opts, channels=None):
        stride, padding = opts.get('stride', 1 if op == 'conv' else 2), opts.get('padding', 'SAME' if op == 'conv' else 'VALID')
        out = [(n+stride-1)//stride if padding == 'SAME' else (n-k)//stride+1 for n,k in zip(x.shape[1:3], kernel)]
        assert min(out)>0
        return [x.shape[0], *out, channels or x.shape[3]], dict(kernel=kernel, stride=stride, padding=padding)

    def conv2d(self, x, channels, kernel, **kwargs):
        opts = {**self.defaults.get('conv2d', {}), **kwargs}
        assert set(opts) <= {'stride','padding','scope','activation_fn','normalizer_fn','weights_initializer'}
        shape, params = self.spatial('conv', x, kernel, opts, channels)
        return self.add('conv', [x], shape, opts.get('scope'), in_channels=x.shape[3], out_channels=channels,
                        normalized=opts.get('normalizer_fn', True) is not None,
                        activation=opts.get('activation_fn', True) is not None, **params)

    def pool(self, mode, x, kernel, kwargs):
        opts = {**self.defaults.get(mode+'_pool2d', {}), **kwargs}
        assert set(opts) <= {'stride','padding','scope'}
        shape, params = self.spatial('pool', x, kernel, opts)
        return self.add('pool', [x], shape, opts.get('scope'), mode=mode, **params)

    def max_pool2d(self, x, kernel, **kwargs): return self.pool('max', x, kernel, kwargs)
    def avg_pool2d(self, x, kernel, **kwargs): return self.pool('avg', x, kernel, kwargs)
    def batch_norm(self, *args, **kwargs): raise AssertionError('only conv normalizer allowed')
    def dropout(self, x, **kwargs):
        assert self.defaults['dropout']['is_training'] is False
        return self.add('identity', [x], x.shape, kwargs.get('scope'))
    def softmax(self, x, **kwargs): return self.add('softmax', [x], x.shape, kwargs.get('scope'))
    def concat(self, *, axis, values):
        assert axis==3 and all(v.shape[:3]==values[0].shape[:3] for v in values)
        return self.add('concat', values, [*values[0].shape[:3], sum(v.shape[3] for v in values)])
    def squeeze(self, x, axes, name=None):
        assert axes==[1,2] and x.shape[1:3]==[1,1]
        return self.add('squeeze', [x], [x.shape[0], x.shape[3]], name)


def trace(branch='non_targeted'):
    pins = json.loads((ROOT/'docs/reviews/competition_adversarial_source_pins.json').read_text())
    pin = next(s for s in pins['sources'] if s['branch']==branch)
    p = Path('/private/tmp/sciona_adversarial_source')/branch/'nets/inception_v3.py'
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest == next(f['sha256'] for f in pin['files'] if f['path']=='nets/inception_v3.py')
    body = [n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef)]
    assert {n.name for n in body} == {'inception_v3_base','inception_v3','_reduced_kernel_size_for_small_input'}
    t = Trace()
    ns = {'tf':t, 'slim':t, 'trunc_normal':lambda std:None}
    exec(compile(ast.Module(body=body,type_ignores=[]), '<pinned-inception-v3-topology>', 'exec'),ns)
    logits, ends = ns['inception_v3'](Tensor('input', [1,299,299,3]), num_classes=1001, is_training=False)
    return dict(format='adversarial-inception-v3-topology.v1', source_sha256=digest,
                nodes=t.nodes, logits=logits.name, endpoints={k:v.name for k,v in ends.items()})


if __name__=='__main__':
    result = trace()
    other = trace('targeted')
    assert result['nodes']==other['nodes'] and result['endpoints']==other['endpoints']
    (ROOT/'sciona/adversarial_inception_v3_topology.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'nodes':len(result['nodes']), 'convolutions':sum(n['op']=='conv' for n in result['nodes']), 'both_branches_identical':True}))
