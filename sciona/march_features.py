"""Explicit generic basketball season statistics, PageRank and Elo.

Private caller game records only; no historical competition recipe or data.
Game order is an explicit unique integer within each season. Edges transfer
rank from the losing team to the winner, with one unit per game.
"""
import math
import numpy as np
import networkx as nx

_BOX={'points','field_goal_attempts','free_throw_attempts','offensive_rebounds','turnovers'}
_FIELDS={'season','order','a','b','a_box','b_box'}


def season_features(games,*,free_throw_weight,elo_initial,elo_k,elo_scale,pagerank_alpha,pagerank_tolerance,pagerank_iterations):
    for v in (free_throw_weight,elo_initial,elo_k,elo_scale,pagerank_alpha,pagerank_tolerance):
        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('Finite explicit season controls required')
    if not 0<=free_throw_weight<=1 or elo_k<=0 or elo_scale<=0 or not 0<pagerank_alpha<1 or pagerank_tolerance<=0:
        raise ValueError('Invalid season controls')
    if type(pagerank_iterations) is not int or pagerank_iterations<1:raise ValueError('Positive PageRank iteration limit required')
    if not isinstance(games,list) or not games:raise ValueError('Nonempty game list required')
    records=[];identities=set()
    for g in games:
        if not isinstance(g,dict) or set(g)!=_FIELDS:raise ValueError('Exact game fields required')
        if any(type(g[k]) is not int or g[k]<0 for k in ('season','order','a','b')) or g['a']==g['b']:
            raise ValueError('Distinct nonnegative team slots and season/order required')
        key=g['season'],g['order']
        if key in identities:raise ValueError('Ambiguous game chronology')
        identities.add(key);boxes=[]
        for name in ('a_box','b_box'):
            b=g[name]
            if not isinstance(b,dict) or set(b)!=_BOX or any(type(v) is not int or v<0 for v in b.values()):raise ValueError('Nonnegative integer box totals required')
            possessions=b['field_goal_attempts']+free_throw_weight*b['free_throw_attempts']-b['offensive_rebounds']+b['turnovers']
            if possessions<=0 or not math.isfinite(possessions):raise ValueError('Positive estimated possessions required')
            boxes.append((b['points'],possessions,b['turnovers']))
        if boxes[0][0]==boxes[1][0]:raise ValueError('Completed non-drawn games required')
        records.append((g,boxes))
    seasons={}
    for g,boxes in sorted(records,key=lambda r:(r[0]['season'],r[0]['order'])):
        state=seasons.setdefault(g['season'],dict(totals={},elo={},graph=nx.DiGraph(),last_order=-1))
        a,b=g['a'],g['b'];state['last_order']=g['order']
        for team,own,opponent in [(a,boxes[0],boxes[1]),(b,boxes[1],boxes[0])]:
            totals=state['totals'].setdefault(team,np.zeros(5))
            totals+=np.array([own[0],own[1],own[2],opponent[0],opponent[1]])
            state['elo'].setdefault(team,float(elo_initial));state['graph'].add_node(team)
        winner,loser=(a,b) if boxes[0][0]>boxes[1][0] else (b,a)
        graph=state['graph'];weight=graph.get_edge_data(loser,winner,{}).get('weight',0)+1
        graph.add_edge(loser,winner,weight=weight)
        # Stable logistic form of base-10 Elo expectation.
        delta=(state['elo'][b]-state['elo'][a])/elo_scale*math.log(10)
        expected=math.exp(-np.logaddexp(0.,delta))
        update=elo_k*(float(winner==a)-expected)
        state['elo'][a]+=update;state['elo'][b]-=update
    result={}
    for season,state in seasons.items():
        # Canonical insertion order removes row-order effects in numerical iteration.
        graph=nx.DiGraph();graph.add_nodes_from(sorted(state['graph']))
        graph.add_weighted_edges_from(sorted((a,b,d['weight']) for a,b,d in state['graph'].edges(data=True)))
        ranks=nx.pagerank(graph,alpha=pagerank_alpha,tol=pagerank_tolerance,max_iter=pagerank_iterations,weight='weight')
        teams={}
        for team,total in state['totals'].items():
            vector=np.array([total[0]/total[1],total[3]/total[4],total[2]/total[1],ranks[team],state['elo'][team]],dtype=float)
            if not np.isfinite(vector).all():raise ValueError('Nonfinite season features')
            teams[team]=vector
        result[season]=dict(teams=teams,last_order=state['last_order'])
    return result


def matchup_features(seasons,season,a,b):
    if any(type(v) is not int for v in (season,a,b)) or a==b:raise ValueError('Distinct integer matchup team slots required')
    try:left=seasons[season]['teams'][a];right=seasons[season]['teams'][b]
    except KeyError:raise ValueError('Unseen season or team') from None
    return np.concatenate([left,right,left-right])
