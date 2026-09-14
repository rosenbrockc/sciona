"""Synthetic constrained-route fixture, no external records."""
def payload():
    triples=[('direct','s','a',1,3),('detour','s','b',2,1),('join','b','a',0,0),('finish','a','g',1,2)]
    return dict(nodes=['s','a','b','g'],edges=[dict(id=i,source=s,target=t,cost=c,resources=[r]) for i,s,t,c,r in triples],
        start='s',goal='g',budgets=[3],blocked_nodes=[],blocked_edges=[],max_expansions=100)
