import copy
import numpy as np
import pytest
import torch
from sciona.video_multilabel_training import fit


def sample():
    labels=[[i%2,(i//2)%2] for i in range(8)]
    videos=[[[float(2*a-1)+.01*j,float(2*b-1)] for j in range(1+i%3)] for i,(a,b) in enumerate(labels)]
    sparse=[{f'a{a}':1.,f'b{b}':1.} for a,b in labels]
    return videos,sparse,labels


def test_full_training_and_equal_fusion():
    videos,sparse,labels=sample();state=torch.random.get_rng_state().clone()
    fitted=fit(videos,sparse,labels)
    assert torch.equal(state,torch.random.get_rng_state())
    assert len(fitted.history)==81 and fitted.history[-1]<fitted.history[0]*.25
    neural,linear=fitted.components(videos,sparse)
    np.testing.assert_allclose(fitted.predict(videos,sparse),(neural+linear)/2)
    assert neural.shape==linear.shape==(8,2)
    assert fitted.vectorizer.transform(sparse).format=='csr'
    assert set(fitted.vectorizer.vocabulary_)=={'a0','a1','b0','b1'}


def test_unknown_sparse_and_query_batch_isolation():
    videos,sparse,labels=sample();fitted=fit(videos,sparse,labels,epochs=2)
    before=copy.deepcopy(fitted.vectorizer.vocabulary_)
    known=fitted.predict(videos[:1],sparse[:1])
    batch=fitted.predict([videos[0],[[999.,-999.]]],[sparse[0],{'new':900}])
    np.testing.assert_allclose(known,batch[:1],atol=1e-14)
    assert fitted.vectorizer.vocabulary_==before
    assert fitted.vectorizer.transform([{'new':900}]).nnz==0


def test_repeat_and_bounded_padding(monkeypatch):
    from sciona import video_multilabel_training as training
    videos,sparse,labels=sample();original=training.pad_sequences;observed=[]
    def checked(sequences):
        observed.append(len(sequences));return original(sequences)
    monkeypatch.setattr(training,'pad_sequences',checked)
    first=fit(videos,sparse,labels,epochs=3,batch_size=3)
    second=fit(videos,sparse,labels,epochs=3,batch_size=3)
    assert max(observed)==3
    assert first.history==second.history
    np.testing.assert_array_equal(first.predict(videos,sparse),second.predict(videos,sparse))

@pytest.mark.parametrize('case',['empty_vocabulary','negative_sparse','one_class','width','boolean_control'])
def test_invalid_inputs(case):
    videos,sparse,labels=sample();controls={}
    if case=='empty_vocabulary':sparse=[{} for _ in sparse]
    if case=='negative_sparse':sparse[0]={'x':-1}
    if case=='one_class':labels=[[0,1] for _ in labels]
    if case=='width':videos[0][0].append(3.)
    if case=='boolean_control':controls['epochs']=True
    with pytest.raises(ValueError):fit(videos,sparse,labels,**controls)
