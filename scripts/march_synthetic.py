"""Generated box scores and explicit generic feature controls only."""
CONTROLS=dict(free_throw_weight=.5,elo_initial=1500.,elo_k=32.,elo_scale=400.,pagerank_alpha=.85,pagerank_tolerance=1e-13,pagerank_iterations=1000)


def game(order,a,b,pa,pb,season=1):
    def box(p):return dict(points=p,field_goal_attempts=50,free_throw_attempts=20,offensive_rebounds=5,turnovers=10)
    return dict(season=season,order=order,a=a,b=b,a_box=box(pa),b_box=box(pb))


def payload():
    games=[]
    for season in range(1,5):
        games += [game(0,0,1,80,70,season),game(1,1,2,75,70,season),game(2,2,0,60,80,season)]
    def row(s,a,b,y):return dict(season=s,order=10+a,a=a,b=b,outcome=y)
    return dict(version=1,games=games,training=[row(1,0,1,1),row(2,1,2,0)],
        calibration=[row(3,0,1,1),row(3,1,2,1)],
        prediction=[dict(season=4,order=10,a=0,b=1),dict(season=4,order=10,a=1,b=0)],
        feature_controls=dict(CONTROLS),regularization=1.,max_iterations=1000,tolerance=1e-8,seed=12)
