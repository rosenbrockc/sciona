"""Sine-squared identity and full proof certificate."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.sine_squared_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.sine_squared.sine_squared'


def build_sine_squared_execution():
    inputs=[IOSpec(name='angle_srepr',type_desc='str',constraints='Guarded serialized finite real scalar angle in radians; symbolic real expressions accepted.')]
    outputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('sine_squared_srepr','Symbolic sin(angle)^2.'),
        ('half_angle_srepr','Symbolic (1-cos(2*angle))/2; no floating-point subtraction performed.'),
        ('certificate_json','JSON13step certificate with Euler and independent endpoint ODE checks; no sine-sign inference.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='formula',name='Sine-squared half-angle identity',
        description='Construct both squared-identity expressions and full certificate for arbitrary real angle.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_symbolic_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'execution_phase':'symbolic synthesis',
                  'physical_regime':'Finite real dimensionless angle in radians; global real-line identity with endpoint ODE certificate; no principal square-root claim.'})
