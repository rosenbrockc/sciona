"""Strict private replay-to-next-action boundary for the corrected Santa method."""
from dataclasses import dataclass, field
import hashlib
import json
from sciona.santa_agent_features import BanditHistory
from sciona.santa_boosting import fit_threshold_models
from sciona.santa_replay_training import validate_replay


@dataclass(frozen=True, repr=False)
class Prepared:
    configuration: str = field(repr=False)
    fingerprint: str = field(repr=False)


def _history(query):
    if type(query) is not dict or set(query) != {'episode_id', 'history'}:
        raise ValueError('Invalid query fields')
    if type(query['episode_id']) is not str or not query['episode_id']:
        raise ValueError('Opaque query episode identity required')
    if type(query['history']) is not list or len(query['history']) >= 2000:
        raise ValueError('Active bounded query history required')
    history = BanditHistory()
    for row in query['history']:
        if type(row) is not dict or set(row) != {'own_action','opponent_action','reward'}:
            raise ValueError('Invalid visible query history')
        history.observe(row['own_action'],row['opponent_action'],row['reward'])
    return history


def prepare(payload):
    if type(payload) is not dict or set(payload) != {'version','training','validation','query','controls'}:
        raise ValueError('Invalid execution fields')
    if type(payload['version']) is not int or payload['version'] != 1:
        raise ValueError('Unsupported execution version')
    controls = payload['controls']
    fields = {'seed','decision_seed','max_rows','num_boost_round','stopping_rounds','num_leaves'}
    if type(controls) is not dict or set(controls) != fields:
        raise ValueError('Invalid execution controls')
    for name, value in controls.items():
        minimum = 0 if name in ('seed','decision_seed') else 2 if name == 'num_leaves' else 1
        maximum = 2**32-1 if name in ('seed','decision_seed') else 131072 if name == 'num_leaves' else 1000000
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError('Execution control outside allowed range')
    identities = set()
    for name in ('training','validation'):
        population = payload[name]
        if type(population) is not list or not 1 <= len(population) <= 1000:
            raise ValueError('Bounded nonempty replay population required')
        for replay in population:
            validate_replay(replay)
            if replay['episode_id'] in identities:
                raise ValueError('Episode identities overlap')
            identities.add(replay['episode_id'])
    _history(payload['query'])
    if payload['query']['episode_id'] in identities:
        raise ValueError('Query episode overlaps fitting or validation')
    try:
        encoded = json.dumps(payload,allow_nan=False,sort_keys=True,separators=(',',':'))
    except (ValueError,TypeError,OverflowError):
        raise ValueError('Finite JSON execution input required') from None
    if len(encoded.encode()) > 64*1024*1024:
        raise ValueError('Execution input exceeds byte limit')
    return Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())


def execute(prepared):
    if type(prepared) is not Prepared or type(prepared.configuration) is not str:
        raise ValueError('Prepared execution required')
    if hashlib.sha256(prepared.configuration.encode()).hexdigest() != prepared.fingerprint:
        raise ValueError('Prepared execution changed')
    try:
        payload = json.loads(prepared.configuration)
    except (ValueError,TypeError):
        raise ValueError('Invalid prepared JSON') from None
    if prepare(payload) != prepared:
        raise ValueError('Prepared execution is not canonical')
    history = _history(payload['query'])
    controls = dict(payload['controls'])
    decision_seed = controls.pop('decision_seed')
    models = fit_threshold_models(payload['training'],payload['validation'],**controls)
    result = models.predict(history,seed=decision_seed)
    result.update(query_step=history.step,models=2,training_rows=models.rows['training'],
                  validation_rows=models.rows['validation'],
                  model_evaluation={name:{k:v for k,v in e.items() if k!='iteration_rmse'}
                                    for name,e in models.evaluation.items()})
    json.dumps(result,allow_nan=False)
    return result
