"""Execution realization of the reconstructed real-3D momentum derivation."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.momentum_vector_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physical_quantities.momentum_recoil.momentum_recoil'


def build_momentum_execution():
    return CDGExport(nodes=[AlgorithmicNode(node_id='recoil', name='Compute recoil momentum and squared norm',
        description='Real 3D momentum difference and squared Euclidean norm under explicit conservation premises.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name=name, type_desc='numpy.ndarray', constraints='finite real momentum in kg*m/s, nonempty shape (...,3); same shape and orthonormal frame for both inputs', dim_signature='M1 L1 T-1')
                for name in ['incoming_momentum', 'outgoing_momentum']],
        outputs=[IOSpec(name='recoil_momentum', type_desc='numpy.ndarray', constraints='finite float64 kg*m/s; input shape preserved', dim_signature='M1 L1 T-1'),
                 IOSpec(name='squared_momentum', type_desc='numpy.ndarray', constraints='nonnegative finite float64 kg^2*m^2/s^2; input shape without final axis', dim_signature='M2 L2 T-2')])], edges=[],
        metadata={'artifact_source': 'physics_reconstructed_numerical_realization', 'publication_status': 'draft',
                  'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH, 'literal_source_parity': False,
                  'physical_regime': 'Momentum conservation in a common orthonormal frame; no energy or scattering-event validity claim.',
                  'num_nodes': 1, 'num_edges': 0})
