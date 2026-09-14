"""TrackML round setup and parameter scheduling; BSD-2-Clause adaptation.

Derived from edwinst/trackml_solution; see docs/licenses/TrackML-BSD-2-Clause.txt.
"""
from collections import OrderedDict
from contextlib import nullcontext
import re
import sys
import numpy as np
from sciona.trackml_pairing import PairedTrackExtension
from sciona.trackml_neighbors import Neighbors
from sciona.trackml_candidates import Candidates
from sciona.trackml_cells import CellFeatures

class TrackRuntime(PairedTrackExtension):

    def __init__(self, spec, intersector, params, layer_functions=None):
        super().__init__(spec, intersector)
        self.params = params.copy()
        self.layer_functions = layer_functions

    def timed(self, *args):
        return nullcontext()

    class Run:
        """Represents one run (i.e. the application to one event) of the algorithm.
        Note: If the algorithm does multiple iterations of the commit loop ("rounds"),
              a new Run object will be created for each round.
        Note: The Algorithm class does some direct attribute setting on the Run object.
        """

        def __init__(self, algo, supervisor, event, params, layer_functions=None, used=None, hit_in_cyl=None, hit_layer_id=None, fit_columns=None):
            self.algo = algo
            self.supervisor = supervisor
            self.event = event
            self.params = params
            self.neighbors = Neighbors(self.algo.spec, params=self.params, layer_functions=layer_functions)
            available_hits_df = self.event.hits_df
            if used is not None:
                available_hits_df = available_hits_df.loc[~used[available_hits_df['hit_id']]]
            self.available_hits_fraction = len(available_hits_df) / len(self.event.hits_df)
            self.neighbors.fit(available_hits_df, hit_in_cyl=hit_in_cyl, hit_layer_id=hit_layer_id)
            self.hit_in_cyl = hit_in_cyl
            self.hit_layer_id = hit_layer_id
            self.candidates = Candidates(self.event, fit_columns=fit_columns)
            self.layer_functions = layer_functions

        def hasLayerFunction(self, function):
            return self.layer_functions is not None and function in self.layer_functions.functions

    def paramsForRun(self, i_commit=0, sunset=False):
        """Determine parameters to use for the given commit round.
        This function implements a kind of mini-language in the system for
        hyper-parameters that allows things like:
            *) override/modify specific parameters only for the "sunset" round
            *) override/modify specific parameters "upto" or starting "from"
                a given round index
        Args:
            i_commit (int): zero-based index of the commit round
            sunset (bool): if True, this is the last ("sunset") round, in which
                we take desparate measures
        """
        params = self.params.copy()
        conditions = OrderedDict([('upto', lambda match, i=i_commit: 1 if i <= int(match.group(2)) else 0), ('from', lambda match, i=i_commit: 1 if i >= int(match.group(2)) else 0), ('sunset', lambda match, sunset=sunset: 2 if sunset else 0)])
        operations = OrderedDict([('ADD__', lambda params, key, value: params.__setitem__(key, params[key] + value)), ('SUB__', lambda params, key, value: params.__setitem__(key, params[key] - value)), ('', lambda params, key, value: params.__setitem__(key, value))])
        regex = '(' + '|'.join(conditions.keys()) + ')(\\d*)__' + '(' + '|'.join(operations.keys()) + ')(.*)'
        buckets = [{op: OrderedDict() for op in operations.keys()} for i in range(3)]
        for key, value in params.items():
            match = re.match(regex, key)
            if match:
                bucket = conditions[match.group(1)](match)
                name = match.group(4)
                if name not in params:
                    print("warning: '%s' adds new param '%s'" % (key, name), file=sys.stderr)
                buckets[bucket][match.group(3)][name] = value
        for bucket in buckets[1:]:
            for op, executor in operations.items():
                for key, value in bucket[op].items():
                    executor(params, key, value)
        return params

    def setupRun(self, event, i_commit=0, sunset=False, used=None, first_run=None, fit_columns=None, supervisor=None, cell_details=None):
        """Setup data structures for running a round of the algorithm.
        Args:
            event (data.Event): the event for which to run the algorithm
            i_commit (int): commit round index (0 for first round)
            sunset (bool): True if this will be the sunset round
            used (None or bool array(1 + max_hit_id,)): if given, used[hit_id] == True for
                hit_ids which have been committed already in earlier rounds
            first_run (None or Run): None for setting up the first round, otherwise
                the Run object for the first round
            fit_columns (None or list of str): If given, this list defines the
                fit parameters that will be stored in the candidates list.
            supervisor (supervised.Supervisor): supervisor object (XXX remove?)
            cell_details (None or SimpleNamespace): If given, this is filled with some
                detailed data about cell features for off-line analyis.
                (Only if cell features are actually calculated in this call.)
        Returns:
            run (Run): the Run object holding the data structures for this round
        """
        if i_commit == 0:
            hit_in_cyl = np.zeros(1 + event.max_hit_id, dtype=np.bool)
            hit_layer_id = np.full(1 + event.max_hit_id, -1, dtype=np.int8)
        else:
            hit_in_cyl = None
            hit_layer_id = None
        run = self.Run(self, supervisor, event, params=self.paramsForRun(i_commit=i_commit, sunset=sunset), hit_in_cyl=hit_in_cyl, hit_layer_id=hit_layer_id, layer_functions=self.layer_functions, used=used, fit_columns=fit_columns)
        if i_commit == 0:
            run.full_neighbors = run.neighbors
            cell_features = None
            if event.has_cells:
                cell_features = CellFeatures(self, run, details=cell_details)
            run.cell_features = cell_features
        else:
            assert first_run is not None
            run.hit_in_cyl = first_run.hit_in_cyl
            run.hit_layer_id = first_run.hit_layer_id
            run.full_neighbors = first_run.neighbors
            run.cell_features = first_run.cell_features
        return run
