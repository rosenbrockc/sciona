"""Entirely synthetic population for lifecycle validation."""
def payload():
    return dict(version=1,
        training=dict(numeric=[[float(i),float(i%3)] for i in range(12)],
            categorical=[[f'c{i%2}'] for i in range(12)],labels=[i%2 for i in range(12)],
            folds=[i//4 for i in range(12)],groups=[f'train{i//2}' for i in range(12)]),
        calibration=dict(numeric=[[1.,0.],[4.,1.],[7.,2.],[10.,0.]],
            categorical=[['c0'],['c1'],['c0'],['new']],labels=[0,1,0,1],groups=[f'cal{i}' for i in range(4)]),
        query=dict(numeric=[[3.,None],[99.,1.]],categorical=[['c0'],['unseen']],groups=['query0','query1']),
        controls=dict(seed=42,trees=64,max_depth=6,min_leaf=1,clip_low=.01,clip_high=.99))
