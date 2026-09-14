"""Complete constrained-route realization of the generic planning topology.

Nonnegative integer costs and resource consumption; exact expanded-state search.
No historical competition or arbitrary simulation support is implied.
"""
from dataclasses import dataclass
import heapq


@dataclass(frozen=True)
class Edge:
    identity: str
    source: str
    target: str
    cost: int
    resources: tuple


@dataclass(frozen=True)
class State:
    nodes: tuple
    edges: tuple
    start: str
    goal: str
    budgets: tuple
    blocked_nodes: frozenset
    blocked_edges: frozenset
    max_expansions: int


@dataclass(frozen=True)
class Candidates:
    state: State
    edges: tuple


@dataclass(frozen=True)
class SearchResult:
    candidates: Candidates
    status: str
    path: tuple
    cost: int
    resources: tuple
    expansions: int


def _identities(values):
    if not isinstance(values,list) or any(type(x) is not str or not x for x in values) or len(set(values))!=len(values):
        raise ValueError('Expected unique nonempty string identities')
    return tuple(values)


def _nonnegative(values):
    if not isinstance(values,list) or any(type(x) is not int or x<0 for x in values):
        raise ValueError('Expected nonnegative integer resources')
    return tuple(values)


def encode_state(payload):
    fields={'nodes','edges','start','goal','budgets','blocked_nodes','blocked_edges','max_expansions'}
    if not isinstance(payload,dict) or set(payload)!=fields: raise ValueError('Invalid planning fields')
    nodes=_identities(payload['nodes'])
    if not nodes or payload['start'] not in nodes or payload['goal'] not in nodes: raise ValueError('Unknown boundary node')
    budgets=_nonnegative(payload['budgets'])
    if not budgets: raise ValueError('At least one resource budget is required')
    if type(payload['max_expansions']) is not int or payload['max_expansions']<1: raise ValueError('Invalid expansion limit')
    blocked_nodes=frozenset(_identities(payload['blocked_nodes']))
    blocked_edges=frozenset(_identities(payload['blocked_edges']))
    if not blocked_nodes.issubset(nodes): raise ValueError('Unknown blocked node')
    if not isinstance(payload['edges'],list): raise ValueError('Expected edge list')
    edges=[];ids=set()
    for item in payload['edges']:
        if not isinstance(item,dict) or set(item)!={'id','source','target','cost','resources'}: raise ValueError('Invalid edge fields')
        identity=item['id']
        if type(identity) is not str or not identity or identity in ids: raise ValueError('Invalid edge identity')
        if item['source'] not in nodes or item['target'] not in nodes: raise ValueError('Unknown edge node')
        if type(item['cost']) is not int or item['cost']<0: raise ValueError('Expected nonnegative integer edge cost')
        resources=_nonnegative(item['resources'])
        if len(resources)!=len(budgets): raise ValueError('Resource dimension mismatch')
        ids.add(identity);edges.append(Edge(identity,item['source'],item['target'],item['cost'],resources))
    if not blocked_edges.issubset(ids): raise ValueError('Unknown blocked edge')
    return State(nodes,tuple(edges),payload['start'],payload['goal'],budgets,blocked_nodes,blocked_edges,payload['max_expansions'])


def generate_candidates(state):
    edges=tuple(e for e in state.edges if e.identity not in state.blocked_edges
        and e.source not in state.blocked_nodes and e.target not in state.blocked_nodes
        and all(r<=b for r,b in zip(e.resources,state.budgets)))
    return Candidates(state,edges)


def search_routes(candidates):
    state=candidates.state;zero=(0,)*len(state.budgets)
    def finish(status,path=(),cost=0,resources=zero,expansions=0):
        return SearchResult(candidates,status,path,cost,resources,expansions)
    if state.start in state.blocked_nodes or state.goal in state.blocked_nodes: return finish('infeasible')
    adjacency={n:[] for n in state.nodes}
    for edge in candidates.edges: adjacency[edge.source].append(edge)
    initial=(state.start,zero);best={initial:0};parents={};queue=[(0,0,initial)];serial=0;expanded=0
    while queue:
        cost,_,key=heapq.heappop(queue)
        if best.get(key)!=cost: continue
        node,used=key
        if node==state.goal:
            path=[];current=key
            while current!=initial:
                current,edge=parents[current];path.append(edge)
            return finish('optimal',tuple(reversed(path)),cost,used,expanded)
        if expanded>=state.max_expansions: return finish('search_limit',expansions=expanded)
        expanded+=1
        for edge in adjacency[node]:
            resources=tuple(a+b for a,b in zip(used,edge.resources))
            if any(r>b for r,b in zip(resources,state.budgets)):continue
            target=(edge.target,resources);new_cost=cost+edge.cost
            if target not in best or new_cost<best[target]:
                best[target]=new_cost;parents[target]=(key,edge.identity);serial+=1
                heapq.heappush(queue,(new_cost,serial,target))
    return finish('infeasible',expansions=expanded)


def validate_route(result):
    """Independently enforce path continuity, closures, totals and budgets."""
    if result.status not in {'optimal','infeasible','search_limit'}: raise ValueError('Unknown search status')
    if result.status!='optimal':
        if result.path: raise ValueError('Unexpected unverified route')
        return result
    state=result.candidates.state;edges={e.identity:e for e in state.edges}
    node=state.start;cost=0;used=[0]*len(state.budgets)
    if node in state.blocked_nodes: raise ValueError('Blocked start')
    for identity in result.path:
        if identity not in edges or identity in state.blocked_edges: raise ValueError('Invalid route edge')
        edge=edges[identity]
        if edge.source!=node or edge.target in state.blocked_nodes: raise ValueError('Invalid route continuity')
        node=edge.target;cost+=edge.cost;used=[a+b for a,b in zip(used,edge.resources)]
    if node!=state.goal or cost!=result.cost or tuple(used)!=result.resources or any(r>b for r,b in zip(used,state.budgets)):
        raise ValueError('Route constraints or reported totals differ')
    return result


def select_plan(result):
    result=validate_route(result)
    return dict(status=result.status,edge_ids=list(result.path) if result.status=='optimal' else [],
        cost=result.cost if result.status=='optimal' else None,
        resources=list(result.resources) if result.status=='optimal' else None,expansions=result.expansions)


def plan(payload):
    return select_plan(validate_route(search_routes(generate_candidates(encode_state(payload)))))
