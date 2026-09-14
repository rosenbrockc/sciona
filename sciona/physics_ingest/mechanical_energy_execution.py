"""Constant-force mechanical energy, with explicit potential and rounding scope."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.mechanical_energy_proof import SOURCE_VERSION, SOURCE_HASH

PRIMITIVE = 'sciona.atoms.physics.mechanical_energy.mechanical_energy'


def build_mechanical_energy_execution():
    inputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('mass', 'M', 'Finite positive constant mass; identical nonempty shapes.'),
        ('force', 'L M T^-2', 'Finite signed constant net conservative force, including zero.'),
        ('initial_position', 'L', 'Finite signed initial position; potential reference is x=0.'),
        ('initial_velocity', 'L T^-1', 'Finite signed initial velocity; reversal allowed.'),
        ('elapsed_time', 'T', 'Finite nonnegative elapsed time, including zero.')]]
    outputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('final_position', 'L', 'Signed position x0+u*t+F*t^2/(2*m).'),
        ('final_velocity', 'L T^-1', 'Signed velocity u+F*t/m.'),
        ('kinetic_energy', 'L^2 M T^-2', 'Nonnegative kinetic energy from the exact model velocity.'),
        ('potential_energy', 'L^2 M T^-2', 'Signed U=-F*x, with zero at x=0.'),
        ('total_energy', 'L^2 M T^-2', 'Exact model K+U rounded once; rounded K+U may differ.'),
        ('work', 'L^2 M T^-2', 'Signed F*(x-x0), equal to the exact kinetic energy change.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='energy', name='Constant-force mechanical energy',
        description='Evaluate constant-force motion, kinetic and potential energy, conserved total energy and work.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=inputs, outputs=outputs)], edges=[], metadata={
        'artifact_source': 'physics_reconstructed_numerical_realization', 'publication_status': 'draft',
        'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH, 'literal_source_parity': False,
        'num_nodes': 1, 'num_edges': 0,
        'physical_regime': 'One-dimensional Newtonian constant net force, positive constant mass, U=-F*x, no other work. Nonnegative duration; zero force and reversal supported. Exact intermediates, independent output rounding; no exact rounded-energy identity claim.'})
