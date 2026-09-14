"""Compare pinned NAB scoring on generated inputs and expose boundary failures."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    cache=Path('/private/tmp/sciona_nab_source')
    manifest=json.loads((ROOT/'docs/reviews/competition_nab_source_pins.json').read_text())
    pins={p['software_path']:p['sha256'] for p in manifest['pins']}
    source=cache/'nab/sweeper.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==pins['nab/sweeper.py']
    assert hashlib.sha256((cache/'LICENSE.txt').read_bytes()).hexdigest()==pins['LICENSE.txt']
    spec=importlib.util.spec_from_file_location('pinned_nab_sweeper',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    costs=dict(tpWeight=1.,fpWeight=.11,fnWeight=1.)
    sweeper=module.Sweeper(probationPercent=0.,costMatrix=costs)
    def curve(x):return -math.tanh(2.5*x)
    cases=[[],[0],[2],[3],[4],[5],[7],[2,3,4],[0,2,7]]
    for active in cases:
        scores=[float(i in active) for i in range(8)]
        _,actual=sweeper.scoreDataSet(range(8),scores,[(2,4)],'synthetic',.5)
        expected=-1.
        for i in active:
            if 2<=i<=4:expected=max(expected,curve(-(5-i)/3)/curve(-1.))
        for i in active:
            if i<2:expected-=.11
            elif i>4:expected+=.11*curve((i-4)/2)
        assert math.isclose(actual.score,expected,abs_tol=1e-14)
    # Core source score selection does not select a row below all observed scores.
    _,below=sweeper.scoreDataSet(range(8),[.8]*8,[(2,4)],'synthetic',.2)
    assert below is None
    singleton=False
    try:sweeper.scoreDataSet(range(8),[.5]*8,[(2,2)],'synthetic',.5)
    except ZeroDivisionError:singleton=True
    assert singleton
    report=dict(status='passed',synthetic_only=True,source_commit=manifest['commit'],
        independent_scalar_comparison_cases=len(cases),threshold_below_minimum_returns_no_score=True,
        singleton_window_followed_by_samples_divides_by_zero=True,
        semantics=['Scoring depends on observation index, not elapsed time. Window endpoints are inclusive.',
            'The best active score per window contributes; repeated detections inside a window do not add multiple rewards.',
            'Confusion counts are per point; false-negative score penalty is per undetected scorable window.',
            'Post-window false-positive penalty depends on preceding window width; probation uses a fraction with a 5000-row cap.'],
        source_sha256=pins['nab/sweeper.py'],validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Scoring component evidence only. No detector, full streaming pipeline, historical result or catalog approval claim.')
    (ROOT/'docs/reviews/competition_nab_scoring_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',independent_cases=len(cases),reproduced_source_boundary_failures=2)))


if __name__=='__main__':main()
