"""Escape-energy threshold reusing the exact approved infall provider."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.physics_ingest.escape_speed_proof import SOURCE_VERSION, SOURCE_HASH
PRIMITIVE = 'sciona.atoms.physics.infall_speed.infall_speed'


def build_escape_speed_execution():
    inputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('gravitational_constant', 'L^3 M^-1 T^-2', 'Positive finite SI G; identical nonempty input shapes.'),
        ('test_mass', 'M', 'Positive finite test particle mass; caller establishes negligible backreaction.'),
        ('central_mass', 'M', 'Positive finite fixed central mass.'),
        ('radius', 'L', 'Positive finite center-based radius; caller establishes exterior clearance.')]]
    outputs = [IOSpec(name=n, type_desc='numpy.ndarray', dim_signature=d, constraints=c) for n, d, c in [
        ('escape_speed', 'L T^-1', 'Positive ideal escape threshold magnitude sqrt(2GM/r).'),
        ('inward_counterpart_velocity', 'L T^-1', 'Negative time-reversed infall counterpart; NOT outward escape velocity.'),
        ('required_launch_energy', 'M L^2 T^-2', 'Positive minimum launch kinetic energy, equal to external work against gravity; outward gravitational work has opposite sign.'),
        ('potential_energy', 'M L^2 T^-2', 'Negative potential with zero at infinity.')]]
    return CDGExport(nodes=[AlgorithmicNode(node_id='escape', name='Ideal escape speed and launch energy',
        description='Reuse zero-energy test-mass solution for positive escape speed, required launch energy and potential; retain explicitly labeled inward counterpart.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE,
        inputs=inputs, outputs=outputs)], edges=[], metadata={
            'artifact_source': 'physics_reconstructed_numerical_realization', 'publication_status': 'draft',
            'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH,
            'literal_source_parity': False, 'reused_provider_version_id': '23b6922d-d058-5dbc-aee2-2276694c7cdd',
            'reused_provider_content_hash': '485f1797fab10da7d52f2ef79af046281f01db9abadff2070b520363cb5bcd04', 'num_nodes': 1, 'num_edges': 0,
            'physical_regime': 'Newtonian fixed central exterior spherical field, negligible test mass, no dissipation. Escape threshold reaches asymptotic rest at infinity. Caller establishes outward collision-free path; inward counterpart output is time-reversed motion only.'})
