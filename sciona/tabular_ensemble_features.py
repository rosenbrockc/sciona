"""Fold-fitted mixed-table cleaning and encoding for a generic ensemble."""
from collections import Counter
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def validate_table(numeric,categorical):
    if not isinstance(numeric,list) or not isinstance(categorical,list) or not numeric or len(numeric)!=len(categorical):
        raise ValueError('Expected aligned nonempty table rows')
    for matrix in (numeric,categorical):
        if any(not isinstance(row,list) for row in matrix) or len({len(row) for row in matrix})!=1:
            raise ValueError('Expected rectangular list matrices')
    if len(numeric[0])+len(categorical[0])==0:raise ValueError('Table has no columns')
    for row in numeric:
        for value in row:
            if value is not None and (type(value) not in (int,float) or not np.isfinite(value)):
                raise ValueError('Expected finite numeric values or null')
    for row in categorical:
        if any(value is not None and type(value) is not str for value in row):raise ValueError('Expected string categories or null')
    numbers=np.asarray([[np.nan if v is None else v for v in row] for row in numeric],dtype=np.float64)
    tokens=np.asarray([['m:' if v is None else 'v:'+v for v in row] for row in categorical],dtype=object)
    return numbers,tokens


class TabularFeatures:
    """Training-only median/quantile cleaning, category and frequency features.

All-null numeric columns use zero then drop as constants. Constant and duplicate
numeric columns are removed after clipping. Categories use a distinct missing
sentinel, unknown one-hot values are zero, and unknown frequency is zero. No
labels are consumed. Dense output; wide/high-cardinality resource use unqualified.
"""
    def fit(self,numeric,categorical,*,clip_low=.01,clip_high=.99):
        if type(clip_low) not in (int,float) or type(clip_high) not in (int,float) or not 0<=clip_low<clip_high<=1:
            raise ValueError('Invalid clipping quantiles')
        numbers,tokens=validate_table(numeric,categorical)
        self.widths=(numbers.shape[1],tokens.shape[1]);self.medians=[]
        for column in numbers.T:
            observed=column[~np.isnan(column)]
            self.medians.append(float(np.median(observed)) if len(observed) else 0.)
        self.medians=np.asarray(self.medians)
        filled=np.where(np.isnan(numbers),self.medians,numbers)
        self.lower=np.quantile(filled,clip_low,axis=0) if numbers.shape[1] else np.array([])
        self.upper=np.quantile(filled,clip_high,axis=0) if numbers.shape[1] else np.array([])
        clipped=np.clip(filled,self.lower,self.upper)
        self.kept=[]
        for col in range(clipped.shape[1]):
            if np.all(clipped[:,col]==clipped[0,col]):continue
            if any(np.array_equal(clipped[:,col],clipped[:,other]) for other in self.kept):continue
            self.kept.append(col)
        self.counts=[Counter(column) for column in tokens.T]
        self.rows=len(numbers)
        self.onehot=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64).fit(tokens) if tokens.shape[1] else None
        if not self.kept and self.onehot is None:raise ValueError('No varying numeric or categorical features remain')
        self.transform(numeric,categorical)
        return self

    def transform(self,numeric,categorical):
        if not hasattr(self,'onehot'):raise ValueError('Feature transformer not fitted')
        numbers,tokens=validate_table(numeric,categorical)
        if (numbers.shape[1],tokens.shape[1])!=self.widths:raise ValueError('Table column counts differ from fitted schema')
        filled=np.where(np.isnan(numbers),self.medians,numbers)
        numeric_features=np.clip(filled,self.lower,self.upper)[:,self.kept]
        pieces=[numeric_features]
        if self.onehot is not None:
            pieces.append(self.onehot.transform(tokens))
            pieces.append(np.asarray([[self.counts[j][v]/self.rows for j,v in enumerate(row)] for row in tokens]))
        result=np.concatenate(pieces,axis=1)
        if not np.isfinite(result).all():raise ValueError('Nonfinite transformed features')
        return result
