import copy
import numpy as np
import pytest
from sciona.image_tabular_features import image_features,FusedFeatures


def test_constant_channel_descriptor_formula():
    image=[[[0.,.5,1.]]*2 for _ in range(3)]
    actual=image_features([image])[0]
    assert len(actual)==31
    np.testing.assert_allclose(actual[:10],[0,0,0,0,1,0,0,0,0,0])
    np.testing.assert_allclose(actual[10:20],[.5,0,.5,.5,0,0,1,0,0,0])
    np.testing.assert_allclose(actual[20:30],[1,0,1,1,0,0,0,1,0,0])
    assert actual[-1]==pytest.approx(2/3)


def test_histogram_edges_and_spatial_gradients():
    image=[[[v,v,v] for v in [0.,.25,.5,.75,1.]]]
    actual=image_features([image])[0]
    np.testing.assert_allclose(actual[4:8],[.2,.2,.2,.4])
    assert actual[8]==.25 and actual[9]==0


def test_training_only_imputation_and_unknown_categories():
    images=[[[[0.,0.,0.]]],[[[1.,1.,1.]]]]
    fitted=FusedFeatures().fit(images,[[0.,None],[10.,None]],[['a'],[None]])
    np.testing.assert_allclose(fitted.imputer.statistics_,[5.,0.])
    before=copy.deepcopy(fitted.scaler.mean_)
    one=fitted.transform(images[:1],[[None,None]],[['new']])
    batch=fitted.transform(images,[[None,None],[999.,999.]],[['new'],['m:']])
    np.testing.assert_array_equal(one,batch[:1])
    np.testing.assert_array_equal(fitted.scaler.mean_,before)
    assert fitted.encoder.transform([['v:new']]).sum()==0
    assert one.shape[1]==35

@pytest.mark.parametrize('images',[
    [],[[[[True,0.,0.]]]],[[[[1.1,0.,0.]]]],[[[[0.,0.]]]],[[[[float('nan'),0.,0.]]]],
])
def test_invalid_rgb(images):
    with pytest.raises(ValueError):image_features(images)


def test_metadata_width_mismatch():
    images=[[[[0.,0.,0.]]]]
    fitted=FusedFeatures().fit(images,[[1.]],[[]])
    with pytest.raises(ValueError):fitted.transform(images,[[1.,2.]],[[]])
