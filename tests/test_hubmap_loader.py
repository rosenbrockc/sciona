import random
import numpy as np
import pytest
import torch
from sciona.hubmap_loader import decode_rle,ordered_batches


def test_column_major_runs_and_empty_mask():
    np.testing.assert_array_equal(decode_rle('2 3',(2,3)),[[0,1,0],[1,1,0]])
    assert not decode_rle('',(2,3)).any()


@pytest.mark.parametrize('rle',['0 1','1 7','1','1 -1','2 2 3 1','x 2'])
def test_invalid_runs_reject(rle):
    with pytest.raises(ValueError):decode_rle(rle,(2,3))


def test_train_drops_tail_without_drawing_augmentation_for_it():
    images=[np.zeros((32,32,3),dtype=np.uint8)]*3;rles=['']*3
    a=random.Random(3);b=random.Random(3)
    first=ordered_batches(images,rles,[0,1,2],batch_size=2,input_side=32,training=True,python_rng=a,numpy_rng=np.random.RandomState(1))
    second=ordered_batches(images,rles,[0,1],batch_size=2,input_side=32,training=True,python_rng=b,numpy_rng=np.random.RandomState(1))
    assert a.getstate()==b.getstate() and len(first)==len(second)==1
    assert all(torch.equal(x,y) for x,y in zip(first[0],second[0]))
    valid=ordered_batches(images,rles,[0,1,2],batch_size=2,input_side=32,training=False,python_rng=a,numpy_rng=np.random.RandomState(1))
    assert [len(batch[0]) for batch in valid]==[2,1]
