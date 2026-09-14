"""Stateful CPU controller for the reviewed CHAMPS source epoch.

Retains source optimizers, loss, cutout, clipping, and scheduler behavior.
Chunked evaluation uses the separately reviewed correction. Runtime-owned
random states and structured checkpoints replace launcher globals/pickles.
"""
import ast
import copy
from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace

import numpy as np
import torch

from sciona.champs_epoch_corrections import correct_chunked_evaluation
from sciona.champs_source_runtime import EXECUTION_VERSION


@dataclass(frozen=True)
class TrainingOptions:
    optim: str = 'Adam'
    lr: float = .001
    scheduler: str = 'cosine'
    warmup_step: int = 5000
    max_step: int = 250
    eta_min: float = 1e-7
    decay_rate: float = .5
    patience: int = 5
    lr_min: float = 0.
    clip: float = .25
    batch_size: int = 48
    batch_chunk: int = 1
    champs_loss: bool = False
    cutout: float = 0.
    max_bond_count: int = 250
    seed: int = 1111

    def validate(self):
        if self.optim not in ('Adam','SGD','Adagrad','RAdam') or self.scheduler not in ('cosine','inv_sqrt','dev_perf','constant'):
            raise ValueError('Unknown CHAMPS optimizer or scheduler')
        for key in ('max_step','batch_size','batch_chunk','max_bond_count'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError('CHAMPS positive integer option required')
        for key in ('warmup_step','patience','seed'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError('CHAMPS nonnegative integer option required')
        if self.seed>=2**32 or self.max_bond_count>406 or self.batch_size%self.batch_chunk:
            raise ValueError('Invalid CHAMPS seed, padding limit or chunk division')
        if self.champs_loss and self.batch_chunk>1:
            raise ValueError('Source log-type loss does not support training chunks')
        for key in ('lr','eta_min','decay_rate','lr_min','clip','cutout'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Finite CHAMPS numeric option required')
        if self.lr<=0 or self.clip<=0 or not 0<=self.cutout<=1 or not 0<self.decay_rate<1 or min(self.eta_min,self.lr_min)<0:
            raise ValueError('CHAMPS numeric option outside valid range')


class ChampsTrainer:
    def __init__(self,runtime,variant,model,options: TrainingOptions):
        options.validate()
        self.options=options
        self.variant=variant
        self.commit=runtime.commit
        self.model=model
        if any(p.device.type!='cpu' for p in model.parameters()):
            raise ValueError('This CHAMPS controller requires CPU parameters')
        use_quad=bool(runtime.model_config(variant).get('use_quad',False))
        if use_quad and options.cutout:
            raise ValueError('Original CHAMPS cutout does not support quadruplets')
        ns={'torch':torch,'np':np,'NUM_BOND_ORIG_TYPES':8,'MAX_BOND_COUNT':options.max_bond_count,
            'APEX_AVAILABLE':False,'logging':lambda *_:None,'para_model':model,'train_step':0,
            'args':SimpleNamespace(**asdict(options),use_quad=use_quad,log_interval=2**62)}
        for path,names in [('src/graph_transformer.py',{'sqdist'}),
                           ('src/utils/filters.py',{'subgraph_filter'}),('src/train.py',{'loss','epoch'})]:
            source=runtime.sources[path]
            if path=='src/train.py':
                source=correct_chunked_evaluation(source,runtime.hashes[path])
            nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
            if len(nodes)!=len(names):
                raise ValueError('Missing CHAMPS training definitions')
            exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
        if options.optim=='RAdam':
            optimizer_ns={}
            exec(compile(runtime.sources['src/utils/radam.py'],'<verified-source-radam>','exec'),optimizer_ns)
            optimizer_type=optimizer_ns['RAdam']
        else:
            optimizer_type=getattr(torch.optim,options.optim)
        self.optimizer=optimizer_type(model.parameters(),lr=options.lr)
        self.scheduler=None
        if options.scheduler=='cosine':
            self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,options.max_step,eta_min=options.eta_min)
        elif options.scheduler=='inv_sqrt':
            def lr_lambda(step):
                if step==0 and options.warmup_step==0:
                    return 1.
                return 1./step**.5 if step>options.warmup_step else step/options.warmup_step**1.5
            self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)
        elif options.scheduler=='dev_perf':
            self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,factor=options.decay_rate,patience=options.patience,min_lr=options.lr_min)
        ns['scheduler']=self.scheduler
        self._namespace=ns
        self._torch_rng=torch.Generator().manual_seed(options.seed).get_state()
        self._numpy_rng=np.random.RandomState(options.seed).get_state()

    def epoch(self,batches,*,training=True):
        iterator=iter(batches)
        try:
            first=next(iterator)
        except StopIteration:
            raise ValueError('CHAMPS epoch requires at least one full batch') from None
        def validate(batch):
            if len(batch)!=10 or any(t.shape[0]!=self.options.batch_size for t in batch):
                raise ValueError('CHAMPS source epoch requires full consistent batches')
            types=batch[3][:,:self.options.max_bond_count,0]
            if not torch.any((types>0)&(types<=8)):
                raise ValueError('CHAMPS batch has no supervised couplings')
        validate(first)
        def batches():
            yield first
            for batch in iterator:
                validate(batch)
                yield batch
        numpy_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(self._torch_rng)
                np.random.set_state(self._numpy_rng)
                result=self._namespace['epoch'](batches(),self.model,self.optimizer if training else None)
                self._torch_rng=torch.random.get_rng_state().clone()
                self._numpy_rng=np.random.get_state()
        finally:
            np.random.set_state(numpy_state)
        if not torch.isfinite(result[0]):
            raise ValueError('CHAMPS epoch produced a nonfinite aggregate loss')
        return result

    def validation_schedule_step(self,metric):
        if not math.isfinite(float(metric)):
            raise ValueError('CHAMPS validation metric must be finite')
        if self.options.scheduler=='dev_perf':
            self.scheduler.step(float(metric))

    def state_dict(self):
        """Private runtime state; tensor/basic types support weights_only loading."""
        nr=self._numpy_rng
        return copy.deepcopy({'execution_version':EXECUTION_VERSION,'commit':self.commit,
            'variant':self.variant,'options':asdict(self.options),'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),'scheduler':None if self.scheduler is None else self.scheduler.state_dict(),
            'radam_buffer':getattr(self.optimizer,'buffer',None),'train_step':self._namespace['train_step'],
            'torch_rng':self._torch_rng,'numpy_rng':[nr[0],nr[1].tolist(),nr[2],nr[3],nr[4]]})

    def load_state_dict(self,state):
        if (state['execution_version']!=EXECUTION_VERSION or state['commit']!=self.commit or
            state['variant']!=self.variant or state['options']!=asdict(self.options)):
            raise ValueError('CHAMPS training state identity mismatch')
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.options.optim=='RAdam':
            self.optimizer.buffer=copy.deepcopy(state['radam_buffer'])
        self._namespace['train_step']=state['train_step']
        self._torch_rng=state['torch_rng'].clone()
        nr=state['numpy_rng']
        self._numpy_rng=(nr[0],np.asarray(nr[1],dtype=np.uint32),nr[2],nr[3],nr[4])
