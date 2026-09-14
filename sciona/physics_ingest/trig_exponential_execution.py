"""Synthesis-time Cartesian curl-curl realization."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,IOSpec,NodeStatus
from sciona.physics_ingest.trig_exponential_proof import SOURCE_VERSION,SOURCE_HASH

PRIMITIVE='sciona.atoms.physics.trig_exponential.trig_exponential'


def build_trig_exponential_execution():
    inputs=[IOSpec(name='angle_srepr',type_desc='str',constraints='Guarded serialized finite real scalar angle in radians; symbolic real expressions accepted.')]
    outputs=[IOSpec(name=n,type_desc='str',constraints=c) for n,c in [
        ('cosine_exponential_srepr','Exact cosine exponential representation.'),
        ('sine_exponential_srepr','Exact sine exponential representation.'),
        ('certificate_json','JSON ODE/initial-value certificate for general real angle, specialized to input; no complex log.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='inverse_formula',name='Sine and cosine exponential representations',
        description='Construct both inverse representations with checked Euler premise and linear-system certificate.',
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=PRIMITIVE,inputs=inputs,outputs=outputs)],edges=[],
        metadata={'artifact_source':'physics_reconstructed_symbolic_realization','publication_status':'draft',
                  'source_version_id':SOURCE_VERSION,'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                  'num_nodes':1,'num_edges':0,'execution_phase':'symbolic synthesis',
                  'physical_regime':'Finite real dimensionless angle in radians; global real-line ODE proof with initial value at zero.'})
