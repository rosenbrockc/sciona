"""TrackML event/commit/follow scheduler; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
Supply runtime events and disable analysis to avoid source diagnostic file output.
"""
from collections import OrderedDict
from contextlib import nullcontext
import pprint
import numpy as np
import pandas as pd
from sciona.trackml_runtime import TrackRuntime
from sciona.trackml_candidates import Candidates


def score_event(*args, **kwargs):
    from trackml.score import score_event as source_score
    return source_score(*args, **kwargs)

class TrackLifecycle(TrackRuntime):

    @property
    def indent(self):
        return nullcontext()

    def nonPhysicalPostprocessOddHits(self, run, used, submission_df, k=4):
        """Add best-guess unused hits to candidates with an even number of hits.
        This is a pure score optimization which is not physically motivated. It exploits
        artifacts of the scoring function used in the trackml competition:
          - The score is a monotonically rising function of the summed weight of hits.
          - The score is a monotonically falling function of floor(nhits / 2).
        Due to the second property, adding a hit to a candidate track with an even number
        of hits can only ever increase the candidate's score, never decrease it.
        This fact was originally observed by Grzegorz Sionkowski and published on the
        kaggle discussion forum:
        https://www.kaggle.com/c/trackml-particle-identification/discussion/60638#354053
        """
        tracks_df = submission_df.loc[submission_df['event_id'] == run.event.event_id].copy()
        tracks_df['nhits'] = tracks_df.groupby('track_id', sort=False)['hit_id'].transform('count')
        tracks_df = tracks_df.loc[tracks_df['nhits'] % 2 == 0]
        unused_hit_ids, = np.where(~used)
        if len(unused_hit_ids) and unused_hit_ids[0] == 0:
            unused_hit_ids = unused_hit_ids[1:]
        unused_hit_coords = run.event.hitCoordinatesById(unused_hit_ids)
        unused_hit_layers = (run.hit_in_cyl[unused_hit_ids], run.hit_layer_id[unused_hit_ids])
        nb_df = run.full_neighbors.findIntersectionNeighborhoodK(*unused_hit_coords, *unused_hit_layers, k=k)
        nb_df['unused_hit_id'] = unused_hit_ids[nb_df['vind'].values]
        nb_df.rename(columns={'nb_hit_id': 'hit_id'}, inplace=True)
        nb_df = nb_df.merge(tracks_df, on='hit_id', how='inner', sort=False)
        nb_df.drop(columns='hit_id', inplace=True)
        nb_df = nb_df.sort_values('nb_dist')
        nb_df = nb_df.groupby('unused_hit_id', sort=False, as_index=False).first()
        nb_df = nb_df.groupby('track_id', sort=False, as_index=False).first()
        nb_df.rename(columns={'unused_hit_id': 'hit_id'}, inplace=True)
        add_df = nb_df[['event_id', 'hit_id', 'track_id']].sort_values('hit_id')
        assert not np.any(used[add_df['hit_id'].values])
        used[add_df['hit_id'].values] = True
        return add_df

    def createOrAppendSubmissionFile(self, submission_filename, submission_df, append=False):
        """Create a submission file from the given dataframe, or append to it.
        Args:
            submission_filename (str): path of the submission file
            submission_df (pd.DataFrame): submission dataframe
            append (bool): If True, append to the submission file, which is assumed
                to exist. If False, create or overwrite the submission file
        """
        submission_columns = np.array(['event_id', 'hit_id', 'track_id'])
        if append:
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True
        assert np.all(submission_df.columns == submission_columns)
        submission_df.to_csv(submission_filename, index=False, mode=mode, header=header)

    def findTracks(self, supervisor, events_test, submission_filename=None, analysis=True, score_intermediate=True, score_final=True):
        """
        Args:
            supervisor: XXX remove this argument? XXX
            events_test (list of data.Event): The test data to work on.
            submission_filename (str or None): If given, write the submission
                to the given path
            analysis (bool): If True, store data for off-line analysis.
            score_intermediate (bool): If True, calculate score at the end of each
                track extension iteration (if ground truth is available).
            score_final (bool): If True, calculate score for each processed
                event (if ground truth is available).
        Returns:
            scores (list): list of scores for the given events, if ground
                truth was available and score_final==True,
                otherwise an empty list.
        """
        self.log('params:')
        with self.indent:
            self.log(pprint.PrettyPrinter(indent=4).pformat(self.params))
        if self.layer_functions is not None:
            self.log('layer_functions: ' + ', '.join(self.layer_functions.functions))
        scores = []
        for i_event, event_test in enumerate(events_test):
            submission_df = None
            score = None
            event_test.open()
            self.log('event_test: ', event_test.summary())
            with self.timed('event %d' % event_test.event_id):
                used = np.zeros(1 + event_test.max_hit_id, dtype=np.bool)
                submission_df_parts = []
                min_track_id = 1
                sunset = False
                first_run = None
                for i_commit in range(int(self.params['commit__niter']) + 1):
                    with self.timed('round %d (%6d hits)' % (i_commit, np.sum(~used) - 1) + (' (sunset)' if sunset else '')):
                        run = self.setupRun(event_test, i_commit=i_commit, sunset=sunset, used=used, first_run=first_run, supervisor=supervisor)
                        if i_commit == 0:
                            first_run = run
                        with self.indent:
                            self.log(pprint.PrettyPrinter(indent=4).pformat(run.params))
                        nhits = float(event_test.max_hit_id)
                        ntop_quadratic = run.params['rank__ntop_qu'] * nhits ** 2
                        ntop_linear = run.params['rank__ntop_li'] * nhits
                        self.log('ntop_quadratic = %.0f, ntop_linear = %.0f' % (ntop_quadratic, ntop_linear))
                        with self.timed('chooseLikelyFirstHits'):
                            nh = self.chooseLikelyFirstHits(run)
                        with self.timed('chooseLikelySecondHits'):
                            secnh = self.chooseLikelySecondHits(run, nh['hit_id'])
                            if secnh is None or secnh.empty:
                                submission_df_parts.append(Candidates(event_test).submit(fill=False))
                                break
                            nseeds = len(secnh)
                            self.log('chosen candidates for second hits: ', nseeds)
                        xf, yf, zf = run.event.hitCoordinatesById(secnh['nb_hit_id'])
                        df = pd.DataFrame(data=OrderedDict([('xf', xf), ('yf', yf), ('zf', zf), ('dphi', np.zeros(nseeds, dtype=np.float64)), ('hel_s', np.zeros(nseeds, dtype=np.float64)), ('nskipped', np.zeros(nseeds, dtype=np.int8)), ('donePairs', np.zeros(nseeds, dtype=np.int8))]))
                        run.candidates.addSeeds([secnh['hit_id'], secnh['nb_hit_id']], df)
                        self.filterInvalidTrackCandidates(run, step=0)
                        with self.timed('fitting'):
                            self.fitTracks(run, step=-1)
                        with self.timed('follow tracks'):
                            for i in range(int(run.params['follow__niter'])):
                                if run.candidates.n == 0:
                                    break
                                with self.timed('chooseLikelyNextHits'):
                                    self.chooseLikelyNextHits(run, k=int(run.params['follow__weird_k']) if i == 0 and run.params['follow__weird_triples'] else 4, nmax_per_crossing=4, step=5 * i, nhits_min_keep=2 if i < run.params['follow__nskip_max'] else 3)
                                    self.filterInvalidTrackCandidates(run, step=50 * i + 1)
                                    self.log('number of candidates: ', run.candidates.n)
                                with self.timed('fitting'):
                                    self.fitTracks(run, step=5 * i)
                                with self.timed('findPairs'):
                                    self.findPairs(run, step=5 * i + 1, k=int(run.params['follow__pairs_k']))
                                    if run.hasLayerFunction('pair_theta'):
                                        self.findPairs(run, for_ncross=-1, step=5 * i + 3, k=int(run.params['follow__pairs_k']))
                                if run.candidates.n == 0:
                                    break
                                with self.timed('evaluating'):
                                    value, eval_df = self.evaluateTracks(run, analysis=analysis)
                                    self.log('value ', self.shortStats(value))
                                with self.timed('ranking'):
                                    ranking = np.argsort(-value)
                                    nip = int(run.params['follow__niter']) - 1
                                    xip = i / nip
                                    yip = (nip - i) / nip
                                    ntop = xip * ntop_linear + yip * ntop_quadratic
                                    ntop = int(min(ntop, int(run.params['rank__ntop'])))
                                    self.log('ntop[%2d] = %6d' % (i, ntop))
                                    ranking = ranking[:ntop]
                                    run.candidates.permute(ranking)
                                    if eval_df is not None:
                                        eval_df = eval_df.iloc[ranking].reset_index()
                                with self.timed('dropRedundantTracks'):
                                    if i >= run.params['follow__drop_start']:
                                        eval_df = self.dropRedundantTracks(run, eval_df, nlayers=3 if i < run.params['follow__nskip_max'] else 2)
                                if eval_df is not None:
                                    run.candidates.df['%d_rank' % i] = np.arange(run.candidates.n)
                                    cand_df = run.candidates.analysisDataframe()
                                    eval_df = pd.concat([eval_df, cand_df], axis=1)
                                    eval_df.to_csv('eval-%d.dat' % (i - 1), index=False)
                                if score_intermediate and event_test.has_truth:
                                    submission_df = run.candidates.submit()
                                    score = score_event(event_test.truth_df, submission_df)
                                    self.log('')
                                    self.log('score: %.4f' % score, ' nhits = ', self.shortStats(run.candidates.nHits(), fmt='%.1f'))
                        overfull = run.candidates.n > int(run.params['commit__nmax'])
                        last_round = sunset or i_commit == int(self.params['commit__niter']) - 1 or (not overfull)
                        if overfull and (not last_round):
                            run.candidates.permute(np.arange(int(run.params['commit__nmax'])))
                        if last_round and (not sunset) and any((x.startswith('sunset__') for x in self.params.keys())):
                            last_round = False
                            sunset = True
                        for desperation in range(1 + last_round):
                            submission_df_part = run.candidates.submit(fill=False, used=used, min_track_id=min_track_id, min_nhits=3 if desperation else run.params['commit__min_nhits'], max_nloss=None if desperation else run.params['commit__max_nloss'], max_loss_fraction=1.0 if desperation else run.params['commit__max_loss_fraction'], reserve_skipped=False)
                            min_track_id += run.candidates.n
                            used[submission_df_part['hit_id'].values] = True
                            submission_df_parts.append(submission_df_part)
                        if last_round:
                            break
            submission_df = pd.concat(submission_df_parts, axis=0, ignore_index=True)
            if self.params['post__nonphys_odd']:
                add_df = self.nonPhysicalPostprocessOddHits(run, used, submission_df)
                submission_df = pd.concat([submission_df, add_df], axis=0, ignore_index=True)
            fill_df = Candidates(event_test).submit(fill=True, used=used, min_track_id=min_track_id)
            submission_df = pd.concat([submission_df, fill_df], axis=0, ignore_index=True)
            if submission_filename is not None:
                self.createOrAppendSubmissionFile(submission_filename, submission_df, append=i_event > 0)
            if score_final and event_test.has_truth:
                score = score_event(event_test.truth_df, submission_df)
                scores.append(score)
                self.log('')
                self.log('event %4d score: %.4f' % (event_test.event_id, score))
            event_test.close()
        return scores
