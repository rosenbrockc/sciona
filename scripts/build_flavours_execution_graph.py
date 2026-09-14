"""Unpublished corrected graph, bound to the exact original competition intake."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus


def build_graph():
    ports = [
        ('training_inputs', 'list', 'Six aligned arrays:46ordered base features,three daughter momenta,three transverse momenta,parent transverse momentum,flight distance,lifetime; finite physical magnitudes'),
        ('labels', 'list', 'Aligned binary labels; every outer training fold must contain at least200rows per class'),
        ('mass', 'list', 'Finite aligned training-only mass regression targets'),
        ('query_inputs', 'list', 'Same physical representation as training; nonempty; caller establishes disjoint identity and identical base-column order'),
        ('controls', 'dict', 'Exact agreement_a,agreement_b,correlation physical populations; aligned weights_a,weights_b,correlation_mass and training quality. Correlation has at least200rows; all evaluation metadata excluded from fitting'),
        ('excluded_column', 'int', 'Source-prescribed restricted-feature position in the caller-ordered46base columns, integer0..45'),
    ]
    node = AlgorithmicNode(node_id='execute', name='flavours_execute',
        description='Physical feature construction,35mass-regression fits,30classifier fits,50neural fits,source nonlinear blend and three evaluation diagnostics',
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.flavours_execution.flavours_execute',
        inputs=[IOSpec(name=n,type_desc=t,constraints=c) for n,t,c in ports],
        outputs=[IOSpec(name='result',type_desc='dict',constraints='Private ordered query scores,control metrics,fit histories and exact neural/classifier checkpoint replay evidence')])
    return CDGExport(nodes=[node],edges=[],metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft','target_tier':3,
        'source_version_ids':['6256d5b1-d742-53b1-9ff4-c2c7265ce981'],
        'source_content_hash':'1f7003883cd8067a1fcaa170601336d99e2911e2024a0590eb41425757dc0539',
        'scope':'Complete independently implemented training and evaluated prediction reference for the winning Flavours method',
        'choices':['Notebook direct-mass repeated cross-fitting and full classifier refits override conflicting PDF prose',
                   'Modern CPU exact-tree XGBoost,base_score0.5; NumPy float64 neural implementation and modern stratified validation allocation',
                   'Fixed111458tree rounds,150000neural epochs; no tuning against control diagnostics'],
        'limitations':['No historical library numerical parity or competition accuracy claim',
                       'OOF discrimination is diagnostic: mass features are not nested within downstream classifier folds',
                       'Caller establishes source column semantics,physical units and population identity/disjointness',
                       'A200row correlation population yields a single whole-population window and a vacuous zero CvM',
                       'Automated Tier3 target only; original intake remains draft'],
        'num_nodes':1,'num_edges':0})
