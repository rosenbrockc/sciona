"""Executable sine double-angle identity."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.sine_double_angle_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.sine_double_angle.sine_double_angle'


def build_sine_double_angle_execution():
    return CDGExport(nodes=[AlgorithmicNode(
        node_id='double_angle', name='Sine double-angle identity',
        description='Evaluate the six-step exponential derivation of sin(2*x)=2*sin(x)*cos(x).',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='angle', type_desc='numpy.ndarray', dim_signature='1',
                       constraints='Nonempty finite real angles in radians; scalar arrays supported. No float64 angle doubling.')],
        outputs=[IOSpec(name='double_angle_sine', type_desc='numpy.ndarray', dim_signature='1',
                        constraints='Finite float64 sin(2*x) via product identity at 450-digit precision, input shape preserved; subnormal results supported.')])],
        edges=[], metadata={'artifact_source':'physics_reconstructed_numerical_realization',
                            'publication_status':'draft','source_version_id':SOURCE_VERSION,
                            'source_content_hash':SOURCE_HASH,'literal_source_parity':False,
                            'num_nodes':1,'num_edges':0,
                            'physical_regime':'Real dimensionless angles in radians; exact imaginary unit interpretation in source proof.'})
