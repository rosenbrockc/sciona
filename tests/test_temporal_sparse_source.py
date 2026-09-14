import json
import pytest
from sciona.temporal_sparse_source import inspect_source,snapshot,events,validate_time_splits


def write(path,times,labeled=True):
    rows=[dict(entity='synthetic',time=t,value=1.,category='synthetic') for t in times]
    if labeled:
        for row in rows:row['target']=2.
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_snapshot_immutable_during_source_mutation(tmp_path):
    path=write(tmp_path/'input.jsonl',[0.,1.]);source=inspect_source(str(path),labeled=True)
    assert source.rows==2 and source.first_time==0 and source.last_time==1
    with snapshot(source) as stream:
        write(path,[8.,9.])
        assert [e['time'] for e in events(stream,labeled=True,max_line_bytes=65536)]==[0.,1.]
    assert stream.closed
    with pytest.raises(ValueError,match='changed'): 
        with snapshot(source):pass

@pytest.mark.parametrize('content',[
    '', '{"entity":"a","entity":"b"}\n', '{not json}\n',
    '{"entity":"a","time":NaN,"value":1,"category":"x","target":2}\n',
    '[]\n','\n',
])
def test_invalid_records_without_source_echo(tmp_path,content):
    path=tmp_path/'input.jsonl';path.write_text(content)
    with pytest.raises(ValueError) as error:inspect_source(str(path),labeled=True)
    assert str(path) not in str(error.value)
    if content:assert content.strip() not in str(error.value) or not content.strip()


def test_line_and_total_limits(tmp_path):
    path=write(tmp_path/'input.jsonl',[0.,1.])
    with pytest.raises(ValueError,match='line'):inspect_source(str(path),labeled=True,max_line_bytes=10)
    with pytest.raises(ValueError,match='size'):inspect_source(str(path),labeled=True,max_bytes=10)


def test_chronological_splits(tmp_path):
    train=inspect_source(str(write(tmp_path/'a.jsonl',[0.,1.])),labeled=True)
    valid=inspect_source(str(write(tmp_path/'b.jsonl',[3.,4.])),labeled=True)
    query=inspect_source(str(write(tmp_path/'c.jsonl',[6.,7.],False)),labeled=False)
    validate_time_splits(train,valid,query,gap=1.)
    with pytest.raises(ValueError):validate_time_splits(train,valid,query,gap=2.)
    with pytest.raises(ValueError):inspect_source(str(write(tmp_path/'d.jsonl',[2.,1.])),labeled=True)


def test_invalid_gap_rejected_before_split_access():
    with pytest.raises(ValueError,match='gap'):validate_time_splits(None,None,None,gap=float('nan'))
