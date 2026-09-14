"""Bounded private JSONL streams and immutable execution snapshots.

File paths, source digests and event values stay runtime-only. Exceptions contain
no source text. Snapshots use temporary files, not in-memory population copies.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from sciona.temporal_sparse_features import validate_event


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def events(stream,*,labeled,max_line_bytes):
    if type(max_line_bytes) is not int or max_line_bytes<1:raise ValueError('Positive line limit required')
    last=None
    while line:=stream.readline(max_line_bytes+1):
        if len(line)>max_line_bytes:raise ValueError('Event line exceeds configured limit')
        try:
            event=json.loads(line,object_pairs_hook=_object)
            validate_event(event,labeled=labeled)
            if labeled and event['target']<0:raise ValueError('Negative target')
        except (ValueError,TypeError,UnicodeError,RecursionError,OverflowError):
            raise ValueError('Invalid event record') from None
        if last is not None and event['time']<last:raise ValueError('Events must be ordered by time')
        last=event['time']
        yield event


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    labeled: bool
    rows: int
    first_time: float
    last_time: float
    max_line_bytes: int
    max_bytes: int


def _copy(source,target,max_bytes):
    digest=hashlib.sha256();total=0
    while block:=source.read(min(65536,max_bytes-total+1)):
        total+=len(block)
        if total>max_bytes:raise ValueError('Source exceeds configured size limit')
        digest.update(block);target.write(block)
    target.seek(0)
    return digest.hexdigest()


def inspect_source(path,*,labeled,max_line_bytes=65536,max_bytes=1073741824):
    if type(path) is not str or not path or type(labeled) is not bool:raise ValueError('Invalid source descriptor')
    if type(max_bytes) is not int or max_bytes<1:raise ValueError('Positive source size limit required')
    with tempfile.TemporaryFile(mode='w+b') as temporary:
        try:
            with open(path,'rb') as source:digest=_copy(source,temporary,max_bytes)
        except OSError:raise ValueError('Cannot read configured source') from None
        count=0;first=None;last=None
        for event in events(temporary,labeled=labeled,max_line_bytes=max_line_bytes):
            if first is None:first=event['time']
            last=event['time'];count+=1
        if not count:raise ValueError('Source is empty')
    return Source(str(Path(path).absolute()),digest,labeled,count,first,last,max_line_bytes,max_bytes)


@contextmanager
def snapshot(descriptor):
    if not isinstance(descriptor,Source):raise ValueError('Inspected source required')
    with tempfile.TemporaryFile(mode='w+b') as temporary:
        try:
            with open(descriptor.path,'rb') as source:digest=_copy(source,temporary,descriptor.max_bytes)
        except OSError:raise ValueError('Cannot read configured source') from None
        if digest!=descriptor.sha256:raise ValueError('Source changed after inspection')
        yield temporary


def validate_time_splits(training,validation,query,*,gap):
    if type(gap) not in (int,float) or not math.isfinite(gap) or gap<0:raise ValueError('Finite nonnegative gap required')
    if not training.labeled or not validation.labeled or query.labeled:raise ValueError('Invalid split label roles')
    if validation.first_time<=training.last_time+gap:raise ValueError('Validation must follow training with gap')
    if query.first_time<=validation.last_time+gap:raise ValueError('Query must follow validation with gap')
