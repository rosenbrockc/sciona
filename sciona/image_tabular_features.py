"""Independent RGB descriptors and training-only image/metadata preparation."""
import math
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


def image_features(images):
    """31 fixed descriptors: per-channel moments, histogram, gradients, aspect.

RGB values are already in [0,1]; no raw file loading or learned image backbone.
Four histogram bins include 1 in the final bin. Gradients are mean absolute
adjacent-pixel differences, with zero for an absent spatial direction.
"""
    if type(images) is not list or not images:raise ValueError('Nonempty RGB image list required')
    result=[]
    for image in images:
        if type(image) is not list or not image or any(type(row) is not list or not row for row in image):raise ValueError('Nonempty RGB row lists required')
        if len({len(row) for row in image})!=1:raise ValueError('Rectangular RGB image required')
        if any(type(pixel) is not list or len(pixel)!=3 or any(type(v) not in (int,float) for v in pixel) for row in image for pixel in row):raise ValueError('Three numeric RGB channels required')
        try:x=np.asarray(image,dtype=np.float64)
        except (ValueError,OverflowError) as error:raise ValueError('Invalid RGB values') from error
        if not np.isfinite(x).all() or (x<0).any() or (x>1).any():raise ValueError('RGB values must be finite in [0,1]')
        features=[]
        for channel in range(3):
            values=x[:,:,channel]
            features.extend([values.mean(),values.std(),values.min(),values.max()])
            features.extend(np.histogram(values,bins=[0,.25,.5,.75,1.])[0]/values.size)
            features.extend([np.abs(np.diff(values,axis=1)).mean() if values.shape[1]>1 else 0.,np.abs(np.diff(values,axis=0)).mean() if values.shape[0]>1 else 0.])
        features.append(x.shape[1]/x.shape[0]);result.append(features)
    return np.asarray(result,dtype=np.float64)


def metadata(numeric,categorical,count):
    if type(numeric) is not list or type(categorical) is not list or len(numeric)!=count or len(categorical)!=count:raise ValueError('Aligned metadata rows required')
    for matrix in (numeric,categorical):
        if any(type(row) is not list for row in matrix) or len({len(row) for row in matrix})!=1:raise ValueError('Rectangular metadata required')
    if not len(numeric[0])+len(categorical[0]):raise ValueError('At least one metadata column required')
    for row in numeric:
        for v in row:
            if v is not None:
                if type(v) not in (int,float):raise ValueError('Finite numeric metadata or null required')
                try:valid=math.isfinite(v)
                except OverflowError:valid=False
                if not valid:raise ValueError('Finite numeric metadata required')
    if any(v is not None and type(v) is not str for row in categorical for v in row):raise ValueError('String categories or null required')
    values=np.asarray([[np.nan if v is None else v for v in row] for row in numeric],dtype=np.float64)
    tokens=np.asarray([['m:' if v is None else 'v:'+v for v in row] for row in categorical],dtype=object)
    return values,tokens


class FusedFeatures:
    def fit(self,images,numeric,categorical):
        visual=image_features(images);values,tokens=metadata(numeric,categorical,len(visual))
        self.widths=(values.shape[1],tokens.shape[1])
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(values) if values.shape[1] else None
        self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64).fit(tokens) if tokens.shape[1] else None
        self.scaler=StandardScaler().fit(self._joined(visual,values,tokens))
        return self

    def _joined(self,visual,values,tokens):
        if (values.shape[1],tokens.shape[1])!=self.widths:raise ValueError('Metadata widths differ from fitted state')
        pieces=[visual]
        if self.imputer is not None:pieces.append(self.imputer.transform(values))
        if self.encoder is not None:pieces.append(self.encoder.transform(tokens))
        return np.concatenate(pieces,axis=1)

    def transform(self,images,numeric,categorical):
        if not hasattr(self,'scaler'):raise ValueError('Fitted preprocessing required')
        visual=image_features(images);values,tokens=metadata(numeric,categorical,len(visual))
        result=self.scaler.transform(self._joined(visual,values,tokens))
        if not np.isfinite(result).all():raise ValueError('Nonfinite fused features')
        return result
