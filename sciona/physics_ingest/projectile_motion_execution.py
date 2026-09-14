"""Full ideal-projectile state at a specified elapsed time."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.projectile_motion_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.projectile_motion.projectile_motion'


def build_projectile_motion_execution():
    inputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('initial_x', 'L', 'Finite signed initial horizontal position; identical nonempty shapes.'),
        ('initial_y', 'L', 'Finite signed initial upward-positive position.'),
        ('initial_speed', 'L T^-1', 'Finite nonnegative speed including zero.'),
        ('launch_angle', '1', 'Any finite real angle in radians from positive x; literal float64 value.'),
        ('gravity', 'L T^-2', 'Finite nonnegative constant downward acceleration, including inertial zero.'),
        ('elapsed_time', 'T', 'Finite nonnegative elapsed time from initial state.')]]
    outputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('x', 'L', 'Signed horizontal position; no horizontal-velocity division.'),
        ('y', 'L', 'Signed upward-positive position; no terrain cutoff.'),
        ('vx', 'L T^-1', 'Signed horizontal velocity.'),
        ('vy', 'L T^-1', 'Signed vertical velocity, including descending motion.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='motion', name='Time-based projectile state',
        description='Evaluate full signed position and velocity at time t under constant downward gravity.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=inputs, outputs=outputs)], edges=[], metadata={
        'artifact_source': 'physics_reconstructed_numerical_realization', 'publication_status': 'draft',
        'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH, 'literal_source_parity': False,
        'num_nodes': 1, 'num_edges': 0,
        'physical_regime': 'Fixed Cartesian axes, upward-positive y, constant downward gravity, no drag or collision cutoff. All finite launch angles, nonnegative speed/gravity/time; zero limits included. Literal converted angles and independent output rounding.'})
