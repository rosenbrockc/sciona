"""Synthetic threshold, voting and temporal-boundary contracts."""
import numpy as np
import pytest
from sciona.cornell_voting import vote


def outputs(chunks=1):
    return (np.zeros((13,chunks,264),dtype=np.float32),
            np.zeros((13,chunks,3001,264),dtype=np.float32))


def test_four_votes_both_thresholds_and_singletons():
    clips,frames=outputs()
    clips[:4,0,:3]=.3
    frames[:4,0,100,0]=.3
    frames[:3,0,100,1]=.8
    frames[:4,0,100,2]=np.nextafter(np.float32(.3),np.float32(0))
    result=vote(clips,frames,duration_seconds=30)
    assert result['window_votes'][0,:3].tolist()==[4,3,0]
    assert result['window_decisions'].sum()==1
    assert not result['whole_record_decisions'].any()


def test_half_open_chunk_and_partial_audio_boundaries():
    clips,frames=outputs(2);clips[:4,:,0]=1
    frames[:4,0,500,0]=1  # exactly five seconds belongs to the next window
    frames[:4,0,3000,0]=1 # thirty-second endpoint is not an extra window
    frames[:4,1,99,0]=1   # valid partial-chunk frame
    frames[:4,1,100:,1]=1 # padded output cannot create detections
    clips[:4,:,1]=1
    result=vote(clips,frames,duration_seconds=31)
    assert np.flatnonzero(result['window_decisions'][:,0]).tolist()==[1,6]
    assert not result['window_decisions'][:,1].any()


def test_whole_record_requires_four_voted_windows():
    clips,frames=outputs();clips[:4,0,:2]=1
    for t in (0,500,1000,1500):frames[:4,0,t,0]=1
    for t in (0,500,1000):frames[:4,0,t,1]=1
    result=vote(clips,frames,duration_seconds=30)
    assert result['whole_record_decisions'][:2].tolist()==[True,False]


def test_empty_predictions_have_stable_shapes():
    result=vote(*outputs(),duration_seconds=1)
    assert result['window_decisions'].shape==(1,264)
    assert not result['window_decisions'].any()


@pytest.mark.parametrize('duration',[0,-1,float('nan'),float('inf'),True])
def test_invalid_duration(duration):
    with pytest.raises(ValueError):vote(*outputs(),duration_seconds=duration)


def test_missing_models_and_invalid_probabilities():
    clips,frames=outputs()
    with pytest.raises(ValueError):vote(clips[:12],frames[:12],duration_seconds=30)
    clips[0,0,0]=float('nan')
    with pytest.raises(ValueError):vote(clips,frames,duration_seconds=30)
    clips[0,0,0]=1.1
    with pytest.raises(ValueError):vote(clips,frames,duration_seconds=30)
