"""Level-ground projectile range with explicit fixed-speed maximum."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.projectile_range_proof import SOURCE_VERSION, SOURCE_HASH
PRIMITIVE = 'sciona.atoms.physics.projectile_range.projectile_range'


def build_projectile_range_execution():
    inputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('launch_speed', 'L T^-1', 'Positive finite fixed speed; identical nonempty shapes.'),
        ('gravity', 'L T^-2', 'Positive finite constant downward acceleration.'),
        ('launch_angle', '1', 'Radians in [0,float64(pi/2)]; zero is limiting flight-root extension.')]]
    outputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('flight_time', 'T', 'Time to equal launch height; zero-angle limiting value is zero.'),
        ('horizontal_range', 'L', 'Horizontal displacement, not path length.'),
        ('maximizing_angle', '1', 'Rounded mathematical pi/4 radians for fixed speed.'),
        ('maximum_range', 'L', 'Exact optimum v²/g independently rounded.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='flight', name='Level-ground projectile range',
        description='Compute flight and range and the global fixed-speed range maximum.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=inputs, outputs=outputs)], edges=[], metadata={
            'artifact_source': 'physics_reconstructed_numerical_realization', 'publication_status': 'draft',
            'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH,
            'literal_source_parity': False, 'num_nodes': 1, 'num_edges': 0,
            'physical_regime': 'Fixed launch speed, positive constant gravity, no drag, equal launch/landing height; angle interpreted as actual float64 radians, zero-angle limiting extension.'})
