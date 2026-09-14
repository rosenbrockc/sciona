"""TrackML crossing revisit/pair scheduler; BSD-2-Clause source adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
"""
from string import ascii_lowercase
import numpy as np
import pandas as pd
from sciona.trackml_ranking import RankedTrackExtension

class PairedTrackExtension(RankedTrackExtension):

    def findPairs(self, run, k=4, for_ncross=3, step=0, origin_coords=(0.0, 0.0, 0.0)):
        """Find paired hits for layer crossings.
        Note:
            With "paired hits" we mean the set of (1 to 4) hits created by a track crossing
            and interacting with one layer of the detector.
        Args:
            run (Run): the Run object holding the data structures for this round
            k (int): number of neighbors to find for each intersection and pick the
                paired hits from
            for_ncross (int == 2, 3, or -1):
                2...find paired hits for the first crossing in tracks with exactly
                        two crossings
                3...find paired hits for the first two crossings in tracks with exactly
                        three crossings
                -1...find paired hits for the most recent crossing in all tracks with
                        at least three crossings
            origin_coords (3-tuple of coordinates): coordinates to assume as the first
                 point in helix fitting if only two points are known.
            step (int): algorithm step index for logging and off-line analysis
        """
        if for_ncross == -1:
            mask_seeds = (run.candidates.ncross >= 3) & (run.candidates.df['donePairs'].values < run.candidates.ncross)
        else:
            mask_seeds = (run.candidates.ncross == for_ncross) & (run.candidates.df['donePairs'].values == 0)
        nseeds = np.sum(mask_seeds)
        self.log('step %d: number of seeds to find pairs for: ' % step, nseeds)
        if nseeds == 0:
            return
        nmax_pairs = min(k, run.candidates.nmax_per_crossing)
        extend_hit_cols = ['extend_hit_id'] + ['extend_hit_id_' + ch for ch in ascii_lowercase[1:nmax_pairs]]
        for i_step in range(1 if for_ncross == -1 else 2):
            if for_ncross == -1:
                crossing = -1
                pitch_from = (1, 2)
            else:
                crossing = -(for_ncross - 1) - i_step
                pitch_from = (4 - for_ncross, 3 - for_ncross)
            last_coords = run.candidates.hitCoordinates(crossing, mask=mask_seeds)
            has_intersection, xi, yi, zi, _, hel_s, nb_df = self.findHelixIntersectionNeighbors(run, -3, -2, -1, *last_coords, k=k, nmax_per_crossing=nmax_pairs, pitch_from=pitch_from, revisit=True, mask=mask_seeds, force_cyl_closer=run.hit_in_cyl[run.candidates.hitIds(crossing, mask=mask_seeds)], origin_coords=origin_coords, step=step + i_step)
            keep_mask = np.full(run.candidates.n, True)
            if nb_df is not None:
                existing_hit_id = run.candidates.hitIds(crossing, mask=mask_seeds)
                cand_df = pd.DataFrame(data={'existing_hit_id': existing_hit_id})
                nb_df = nb_df.join(cand_df, on='extend_index', how='left')
                extend_index = nb_df['extend_index'].values
                candidate_index = np.nonzero(mask_seeds)[0][extend_index]
                has_any_neighbor = np.full(run.candidates.n, False)
                has_any_neighbor[candidate_index] = True
                existing_is_any = np.full(run.candidates.n, False)
                for col in extend_hit_cols:
                    existing_is_this = nb_df['existing_hit_id'].values == nb_df[col].values
                    existing_is_any[candidate_index] = existing_is_any[candidate_index] | existing_is_this
                self.log('existing is not any neighbor found: ', np.sum(mask_seeds & ~existing_is_any))
                self.log('existing is not any neighbor found (but neighbors found): ', np.sum(has_any_neighbor & ~existing_is_any))
                keep_mask[mask_seeds & ~has_any_neighbor] = False
                for i_pair, col in enumerate(extend_hit_cols):
                    hit_to_set = np.zeros(run.candidates.n, dtype=np.int32)
                    hit_to_set[candidate_index] = nb_df[col].values
                    run.candidates.setHitIds(crossing, hit_to_set, pair=i_pair, mask=has_any_neighbor)
            self.log('dropping bad candidates: ', np.sum(~keep_mask))
            run.candidates.update(keep_mask=keep_mask)
            mask_seeds = mask_seeds[keep_mask]
            keep_mask = self.filterInvalidTrackCandidates(run, step=10 * step + i_step, mask=mask_seeds)
            mask_seeds = mask_seeds[keep_mask]
        if for_ncross == -1:
            run.candidates.df.loc[mask_seeds, 'donePairs'] = run.candidates.ncross[mask_seeds]
        else:
            run.candidates.df.loc[mask_seeds, 'donePairs'] = for_ncross - 1
        self.log('remaining candidates: ', run.candidates.n)
