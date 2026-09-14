"""Synthetic planar observations only; no real locations or clock provenance."""
def payload():
    return dict(version=1,reference=dict(coordinates='planar',distance_unit='m',time_unit='s',shared_origin=True),
        observations=[dict(x=float(i%2),y=0.,time=float(i),target=float(1+i%2)) for i in range(9)],
        context=[dict(x=0.,y=0.,time=float(i),value=float(i%3)) for i in range(12)],blocks=[i//3 for i in range(9)],
        query=[dict(x=0.,y=0.,time=10.),dict(x=1.,y=0.,time=10.)],
        controls=dict(radius=3.,lookback=10.,period=12.,neighbors=2,gap=0.,seed=12,trees=32,max_depth=5,min_leaf=1,smoothing=.2))
