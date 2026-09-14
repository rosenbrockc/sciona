"""Complete released-from-rest spring motion realization."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.spring_mass_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.spring_mass.spring_mass'


def build_spring_mass_execution():
    inputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c)
              for n, d, c in [
                  ('mass', 'M', 'Positive finite constant mass; identical nonempty input shapes.'),
                  ('stiffness', 'M T^-2', 'Positive finite linear spring stiffness.'),
                  ('initial_displacement', 'L', 'Signed displacement from equilibrium at release; zero initial velocity.'),
                  ('time', 'T', 'Signed finite time since release, in seconds.')]]
    outputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c)
               for n, d, c in [
                   ('angular_frequency', 'T^-1', 'Positive natural frequency sqrt(k/m), radians per second.'),
                   ('displacement', 'L', 'A cos(sqrt(k/m)*t), independently rounded.'),
                   ('velocity', 'L T^-1', '-A sqrt(k/m) sin(sqrt(k/m)*t), independently rounded.'),
                   ('acceleration', 'L T^-2', '-A (k/m) cos(sqrt(k/m)*t), independently rounded.')]]
    return CDGExport(nodes=[AlgorithmicNode(
        node_id='motion', name='Undamped spring motion released from rest',
        description='Correct the source missing square root and evaluate full position, velocity and acceleration.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=inputs, outputs=outputs)], edges=[], metadata={
            'artifact_source': 'physics_reconstructed_numerical_realization',
            'publication_status': 'draft', 'source_version_id': SOURCE_VERSION,
            'source_content_hash': SOURCE_HASH, 'literal_source_parity': False,
            'num_nodes': 1, 'num_edges': 0,
            'physical_regime': 'Constant positive mass and stiffness; signed displacement and time, zero initial velocity; net linear restoring force, no damping or driving.'})
