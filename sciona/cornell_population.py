"""Private Cornell populations and source-balanced epoch batching."""
from dataclasses import dataclass
import numpy as np
import torch
from sciona.cornell_sampling import sample_waveform


@dataclass(frozen=True)
class Record:
    key: str
    waveform: np.ndarray
    primary: int | None
    secondary: tuple


def prepare_records(records,*,training):
    if type(training) is not bool or not isinstance(records,(list,tuple)) or not records:
        raise ValueError('Expected a nonempty population and boolean training')
    result=[];keys=set()
    for raw in records:
        if not isinstance(raw,dict) or set(raw)!={'key','waveform','sample_rate','primary','secondary'}:
            raise ValueError('Unexpected record fields')
        key=raw['key']
        if not isinstance(key,str) or not key or key in keys:raise ValueError('Record keys must be unique')
        keys.add(key)
        if type(raw['sample_rate']) is not int or raw['sample_rate']!=32000:
            raise ValueError('Provision mono audio at32000Hz')
        primary=raw['primary'];secondary=raw['secondary']
        if primary is not None and (type(primary) is not int or not 0<=primary<264):raise ValueError('Invalid primary class')
        if not isinstance(secondary,(list,tuple)) or any(type(v) is not int or not 0<=v<264 for v in secondary):
            raise ValueError('Invalid secondary classes')
        if len(set(secondary))!=len(secondary) or primary in secondary:raise ValueError('Labels must be distinct')
        if primary is None and (training or secondary):raise ValueError('Unlabeled records allowed only in validation')
        values=np.asarray(raw['waveform'])
        if values.ndim!=1 or values.size==0 or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError('Expected nonempty finite mono waveform')
        values=np.array(values,copy=True);values.flags.writeable=False
        result.append(Record(key,values,primary,tuple(secondary)))
    return tuple(result)


def check_split(training,validation):
    if {r.key for r in training}&{r.key for r in validation}:raise ValueError('Training/validation records overlap')


def balanced_indices(records,seed):
    labels=[r.primary for r in records]
    if not labels or None in labels:raise ValueError('Balanced training requires primary labels')
    counts={label:labels.count(label) for label in set(labels)}
    weights=torch.tensor([1/counts[label] for label in labels],dtype=torch.float64)
    return torch.multinomial(weights,len(labels),replacement=True,generator=torch.Generator().manual_seed(seed)).tolist()


def batches(records,*,batch_size,training,seed,augmentation=None):
    """Source train drop-last, validation keep-last, with caller-owned seeds."""
    if type(batch_size) is not int or batch_size<1:raise ValueError('Invalid batch size')
    if type(training) is not bool or not records:raise ValueError('Invalid population')
    if training and (augmentation is None or len(records)<batch_size):raise ValueError('Training requires augmentation and a complete batch')
    indices=balanced_indices(records,seed) if training else list(range(len(records)))
    if training:indices=indices[:len(indices)//batch_size*batch_size]
    rng=np.random.RandomState(seed);clips=1 if training else 2
    for start in range(0,len(indices),batch_size):
        selected=[records[i] for i in indices[start:start+batch_size]]
        waveforms=[];labels=np.zeros((len(selected),clips,264),dtype=np.float32);secondary=np.zeros_like(labels)
        for i,record in enumerate(selected):
            values=augmentation(record.waveform,training=True) if training else record.waveform
            waveforms.append(sample_waveform(values,training=training,rng=rng))
            if record.primary is not None:labels[i,:,record.primary]=1
            for label in record.secondary:labels[i,:,label]=1;secondary[i,:,label]=1
        yield np.stack(waveforms),labels,secondary
