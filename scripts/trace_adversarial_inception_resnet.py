"""Full Inception-ResNet-v2 topology from both pinned competition branches."""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from trace_adversarial_inception_v4 import ROOT, ShapeTensor, V4Trace


class ResidualTensor(ShapeTensor):
    def __init__(self, name, shape, trace):
        super().__init__(name,shape);self.trace=trace
    def __rmul__(self, scale):
        return self.trace.add('scale',[self],self.shape,factor=scale)
    def __add__(self, other):
        assert self.shape==other.shape
        return self.trace.add('add',[self,other],self.shape)


class ResidualTrace(V4Trace):
    def __init__(self):
        super().__init__();self.default_scopes={}
        self.nn.relu=lambda x:self.add('relu',[x],x.shape)

    def add(self,*args,**kwargs):
        t=super().add(*args,**kwargs)
        return ResidualTensor(t.name,t.shape,self)

    @contextmanager
    def variable_scope(self,name,*args,**kwargs):
        if name is None:
            default=args[0];key=(self.scope,default)
            count=self.default_scopes.get(key,0);self.default_scopes[key]=count+1
            name=default+('_'+str(count) if count else '')
        with super().variable_scope(name,*args,**kwargs) as scope: yield scope

    def repeat(self,x,repetitions,layer,**kwargs):
        with self.variable_scope(None,'Repeat'):
            for i in range(repetitions):x=layer(x,scope=layer.__name__+'_'+str(i+1),**kwargs)
        return x

    def conv2d(self,x,channels,kernel,**kwargs):
        old=self.defaults.copy()
        opts=self.defaults.get('conv2d',{}).copy()
        assert opts.pop('rate',1)==1
        self.defaults['conv2d']=opts
        try:return super().conv2d(x,channels,[kernel,kernel] if isinstance(kernel,int) else kernel,**kwargs)
        finally:self.defaults=old

    def pool(self,mode,x,kernel,kwargs):
        return super().pool(mode,x,[kernel,kernel] if isinstance(kernel,int) else kernel,kwargs)
    def concat(self,values,axis):return super().concat(values=values,axis=axis)


def trace(branch='non_targeted'):
    pins=json.loads((ROOT/'docs/reviews/competition_adversarial_source_pins.json').read_text())
    pin=next(s for s in pins['sources'] if s['branch']==branch)
    p=Path('/private/tmp/sciona_adversarial_source')/branch/'nets/inception_resnet_v2.py'
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==next(f['sha256'] for f in pin['files'] if f['path']=='nets/inception_resnet_v2.py')
    body=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)]
    t=ResidualTrace();ns={'tf':t,'slim':t}
    exec(compile(ast.Module(body=body,type_ignores=[]),'<pinned-inception-resnet-topology>','exec'),ns)
    logits,ends=ns['inception_resnet_v2'](ResidualTensor('input',[1,299,299,3],t),num_classes=1001,is_training=False)
    return dict(format='adversarial-inception-resnet-topology.v1',source_sha256=digest,
                nodes=t.nodes,logits=logits.name,endpoints={k:v.name for k,v in ends.items()})


if __name__=='__main__':
    result=trace();other=trace('targeted')
    assert result['nodes']==other['nodes'] and result['endpoints']==other['endpoints']
    (ROOT/'sciona/adversarial_inception_resnet_topology.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'operations':len(result['nodes']),'convolutions':sum(n['op']=='conv' for n in result['nodes'])}))
