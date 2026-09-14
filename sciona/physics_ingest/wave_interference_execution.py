"""Two normalized scalar waves, including fixed phase and incoherent mean."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.wave_interference_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.wave_interference.wave_interference'


def build_wave_interference_execution():
    inputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature='1', constraints=c) for n, c in [
        ('amplitude_a', 'Finite nonnegative normalized scalar magnitude; common polarization/mode and scale.'),
        ('amplitude_b', 'Finite nonnegative normalized magnitude; amplitudes may not both be zero.'),
        ('relative_phase', 'Finite real phase difference in radians; literal converted float64 value.')]]
    outputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature='1', constraints=c) for n, c in [
        ('intensity', 'Fixed-phase normalized squared magnitude of the sum.'),
        ('incoherent_mean', 'Fixed-amplitude ensemble mean with vanishing averaged cosine.'),
        ('constructive', 'Maximum normalized intensity over phase, (a+b)^2.'),
        ('destructive', 'Minimum normalized intensity over phase, (a-b)^2.'),
        ('interference', 'Signed cross term 2*a*b*cos(relative_phase).'),
        ('ratio', 'Fixed-phase intensity divided by incoherent mean, before independent rounding.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='interference', name='Normalized two-wave interference',
        description='Evaluate fixed-phase intensity, incoherent mean, phase extrema, cross term and their ratio.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=inputs, outputs=outputs)], edges=[], metadata={
        'artifact_source': 'physics_reconstructed_numerical_realization', 'publication_status': 'draft',
        'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH, 'literal_source_parity': False,
        'num_nodes': 1, 'num_edges': 0,
        'physical_regime': 'Two normalized scalar waves in a common mode/polarization. Fixed nonnegative amplitudes, not both zero. Coherent fixed phase is distinct from constructive alignment. Incoherent mean assumes vanishing averaged cosine with fixed amplitudes. Factor two requires equal amplitudes and constructive phase; independent output rounding.'})
