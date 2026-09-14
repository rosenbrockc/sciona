"""Two-output real quadratic numerical realization of the corrected proof."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.quadratic_corrected_proof import SOURCE_VERSION
PRIMITIVE='sciona.atoms.physical_quantities.quadratic_roots.real_quadratic_roots'


def build_quadratic_execution():
    return CDGExport(nodes=[AlgorithmicNode(node_id='roots',name='Solve real quadratic with ordered roots',
        description='Complete explicitly corrected real quadratic solver; preserves both alternative roots.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name=name,type_desc='numpy.ndarray',constraints='finite real coefficients, identical nonempty shapes; a != 0; discriminant >= 0; consistent polynomial units') for name in ['a','b','c']],
        outputs=[IOSpec(name=name,type_desc='numpy.ndarray',constraints='finite real root in coefficient shape; lower <= upper; repeated roots retained twice') for name in ['lower_root','upper_root']])],edges=[],
        metadata={'artifact_source':'corrected_physics_numerical_realization','publication_status':'draft',
            'source_version_id':SOURCE_VERSION,'source_parity_claim':False,'num_nodes':1,'num_edges':0})
