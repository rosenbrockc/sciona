"""Finite-distribution realization of the corrected variance identity."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.variance_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.weighted_population_variance.weighted_population_variance'


def build_variance_execution():
    return CDGExport(nodes=[AlgorithmicNode(node_id='variance',name='Evaluate normalized population moments',
        description='Corrected normalized-expectation variance identity, realized on a finite weighted real distribution.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='values',type_desc='numpy.ndarray',dim_signature='1',constraints='Finite real dimensionless (...,N), N>0; converts to float64.'),
                IOSpec(name='weights',type_desc='numpy.ndarray',dim_signature='1',constraints='Finite nonnegative real, identical shape; positive total weight in every batch. No broadcasting.')],
        outputs=[IOSpec(name='mean',type_desc='numpy.ndarray',dim_signature='1',constraints='Finite float64 batch shape; independently rounded exact normalized mean.'),
                 IOSpec(name='population_variance',type_desc='numpy.ndarray',dim_signature='1',constraints='Nonnegative finite float64 batch shape; exact centered second moment before rounding, using exact mean.')])],edges=[],
        metadata={'artifact_source':'physics_reconstructed_numerical_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'physical_regime':'Normalized finite nonnegative weighted distribution; population variance, not unbiased sample estimator or continuum quadrature guarantee.'})
