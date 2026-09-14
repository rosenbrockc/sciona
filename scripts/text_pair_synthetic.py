"""Synthetic text-pair fixture with invented phrases only."""
def data():
    train=[['red bird','bird red'],['blue lake','lake blue'],['green tree','tree green'],['small cat','cat small'],
        ['red bird','wide river'],['blue lake','small cat'],['green tree','blue lake'],['wide river','tree green']]
    calibration=[['wide river','river wide'],['small cat','small kitten'],['red bird','green tree'],['lake blue','cat small']]
    query=[['green tree','green trees'],['purple cloud','small cat']]
    return train,[1,1,1,1,0,0,0,0],calibration,[1,1,0,0],query

def payload():
    train,y,cal,cy,query=data()
    return dict(version=1,training_pairs=train,training_labels=y,calibration_pairs=cal,
        calibration_labels=cy,query_pairs=query,controls=dict(components=2,seed=42,trees=64,max_depth=6,min_leaf=1))
