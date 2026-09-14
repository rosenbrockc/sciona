"""Executable normalized finite-dimensional Hermitian expectation."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.hermitian_expectation_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.hermitian_expectation.hermitian_expectation'


def build_hermitian_execution():
    return CDGExport(nodes=[AlgorithmicNode(node_id='expectation', name='Compute normalized Hermitian expectation',
        description='Finite-dimensional quadratic form divided by positive state norm; explicit reconstruction of source adjoint proof.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=[IOSpec(name='operator', type_desc='numpy.ndarray', constraints='Finite numeric (...,N,N), N>0; exactly Hermitian after complex128 conversion; common orthonormal basis.'),
                IOSpec(name='state', type_desc='numpy.ndarray', constraints='Finite numeric (...,N), same batch shape; each state nonzero; no broadcasting.')],
        outputs=[IOSpec(name='expectation', type_desc='numpy.ndarray', constraints='Finite float64 batch shape; normalized expectation in operator units. Nonzero underflow to zero rejected.')])], edges=[],
        metadata={'artifact_source': 'physics_reconstructed_numerical_realization', 'publication_status': 'draft',
                  'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH,
                  'literal_source_parity': False, 'num_nodes': 1, 'num_edges': 0,
                  'physical_regime': 'Finite-dimensional pure-state expectation in an orthonormal basis; state normalization explicit through positive norm division.'})
