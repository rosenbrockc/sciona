"""Synthetic independent video feature populations; no raw or observed data."""
def payload():
    def population(prefix,count,offset=0.,labeled=True):
        labels=[[i%2,(i//2)%2] for i in range(count)]
        videos=[[[float(2*a-1)+.01*j+offset,float(2*b-1)-offset] for j in range(1+i%3)] for i,(a,b) in enumerate(labels)]
        result=dict(videos=videos,sparse=[{f'a{a}':1.,f'b{b}':1.} for a,b in labels],groups=[f'{prefix}{i}' for i in range(count)])
        if labeled:result['labels']=labels
        return result
    return dict(version=1,training=population('train',8),calibration=population('cal',4,.05),query=population('query',2,.1,False),
        controls=dict(seed=12,hidden=8,epochs=80,batch_size=4,learning_rate=.02))
