"""Radial infall work and speed with an explicit fixed-central-field domain."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.infall_speed_proof import SOURCE_VERSION, SOURCE_HASH
PRIMITIVE = 'sciona.atoms.physics.infall_speed.infall_speed'


def build_infall_speed_execution():
    inputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('gravitational_constant', 'L^3 M^-1 T^-2', 'Positive finite SI G; identical nonempty input shapes.'),
        ('test_mass', 'M', 'Positive finite test particle mass; caller establishes negligible backreaction.'),
        ('central_mass', 'M', 'Positive finite fixed central mass.'),
        ('radius', 'L', 'Positive finite center-based radius; caller establishes exterior clearance.')]]
    outputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('speed', 'L T^-1', 'Positive speed magnitude sqrt(2GM/r).'),
        ('inward_radial_velocity', 'L T^-1', 'Negative radial velocity for inward motion.'),
        ('gravitational_work', 'M L^2 T^-2', 'Positive work on particle from rest at infinity.'),
        ('potential_energy', 'M L^2 T^-2', 'Negative potential with zero at infinity.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='infall', name='Radial infall from rest at infinity',
        description='Evaluate positive speed, inward velocity, work and potential in the fixed central-field test-mass model.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=inputs, outputs=outputs)], edges=[], metadata={
            'artifact_source': 'physics_reconstructed_numerical_realization', 'publication_status': 'draft',
            'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH,
            'literal_source_parity': False, 'num_nodes': 1, 'num_edges': 0,
            'physical_regime': 'Newtonian fixed central spherical exterior field, negligible test mass, inward radial zero-energy motion. Rest at infinity is asymptotic; no finite travel-time or general two-body claim.'})
