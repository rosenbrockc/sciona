"""Synthetic normalized RGB and metadata populations only."""
def payload():
    def population(prefix,count,offset=0.,labeled=True):
        labels=[i%2 for i in range(count)]
        result=dict(images=[[[[float(y)*(1-offset)+offset/2,.2,.3] for _ in range(2)] for _ in range(2)] for y in labels],
            numeric=[[float(y)+offset,None] for y in labels],categorical=[[f'c{y}'] for y in labels],groups=[f'{prefix}{i//2}' for i in range(count)])
        if labeled:result['labels']=labels
        return result
    training=population('train',12);training['folds']=[i//4 for i in range(12)]
    return dict(version=1,training=training,calibration=population('cal',4,.05),query=population('query',2,.1,False),
        controls=dict(seed=12,hidden=12,epochs=40,learning_rate=.02,dropout=.5,smoothing=.1))
